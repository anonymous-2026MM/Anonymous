import logging
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from torch import nn
import keras
from models import MFBasedModel, FeatrueAE
from DecDiff import DecoupleModel, DecoupleUtils, DecDiffCDR
import DiffModel as Diff
import json
import os
import tqdm
import random

class Run():
    def __init__(self,config):
        self.config=config
        self.use_cuda = config['use_cuda'] 
        self.base_model = config['base_model']
        self.root = config['root']
        self.ratio = config['ratio']
        self.task = config['task']
        self.src = config['src_tgt_pairs'][self.task]['src']
        self.tgt = config['src_tgt_pairs'][self.task]['tgt']
        self.uid_all = config['src_tgt_pairs'][self.task]['uid']
        self.iid_all = config['src_tgt_pairs'][self.task]['iid']
        self.batchsize_src = config['src_tgt_pairs'][self.task]['batchsize_src']
        self.batchsize_tgt = config['src_tgt_pairs'][self.task]['batchsize_tgt']
        self.batchsize_meta = config['src_tgt_pairs'][self.task]['batchsize_meta']
        self.batchsize_map = config['src_tgt_pairs'][self.task]['batchsize_map']
        self.ratio = [float(self.ratio.split(',')[0][1:]),float(self.ratio.split(',')[1][:-1])]
        self.batchsize_diff =int(self.ratio[0] * 5)*config['src_tgt_pairs'][self.task]['batchsize_diff']

        self.batchsize_test = config['src_tgt_pairs'][self.task]['batchsize_test']
        self.batchsize_diff_test = config['src_tgt_pairs'][self.task]['batchsize_diff_test']

        self.batchsize_aug = self.batchsize_src
        self.epoch = config['epoch']
        self.emb_dim = config['emb_dim']
        self.meta_dim = config['meta_dim']
        self.lr = config['lr']
        self.la_lr = config['la_lr']
        
        self.wd = config['wd']
        

        self.input_root = self.root + 'ready/_' + str(int(self.ratio[0] * 10)) + '_' + str(int(self.ratio[1] * 10)) + \
            '/tgt_' + self.tgt + '_src_' + self.src
        self.src_path = self.input_root + '/train_src.csv'
        self.tgt_path = self.input_root + '/train_tgt.csv'
        
        self.meta_path = self.input_root + '/train_meta.csv'
        print(f"meta_path:{self.meta_path}")
        if not self.config["test_ratio"]:
            self.test_path = self.input_root + '/test.csv'
        else:
            self.test_path = self.root + 'ready/_' + str(10-int(float(self.config["test_ratio"]) * 10)) + '_' + str(int(float(self.config["test_ratio"]) * 10)) + \
            '/tgt_' + self.tgt + '_src_' + self.src + '/test.csv'
        print(f"test_path:{self.test_path}")
        self.warm_tgt_train_path = self.input_root + '/warm_start_tgt_train.csv'
        self.warm_train_path = self.input_root + '/warm_start_train.csv'
        self.warm_test_path = self.input_root + '/warm_start_test.csv'

        self.results = {'tgt_mae': 10, 'tgt_rmse': 10, 
                        'aug_mae': 10, 'aug_rmse': 10, 
                        'diff_mae': 10, 'diff_rmse': 10,  
                        }
        
        self.device = "cuda" if config['use_cuda'] else "cpu"

        self.diff_lr = config['diff_lr']
        self.diff_steps = config['diff_steps']
        self.diff_sample_steps = config['diff_sample_steps']
        self.diff_scale = config['diff_scale']
        self.diff_dim   = config['diff_dim'] 
        self.diff_task_lambda = config['diff_task_lambda'] 
        self.diff_mask_rate = config["diff_mask_rate"]
        self.vis_feat = None
        # ========== logging ==========
        self.logger = logging.getLogger('CDR')
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        if self.config.get('log_file'):
            log_dir = os.path.dirname(self.config['log_file'])
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(self.config['log_file'], mode='a', encoding='utf-8')
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            self.logger.addHandler(file_handler)
        else:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(console_handler)

        self.logger.info("="*60)
        self.logger.info("Experiment Configuration:")
        self.logger.info(f"  Task: {self.task} ({self.src} -> {self.tgt})")
        self.logger.info(f"  Epoch: {self.epoch} | Seed: {self.config.get('seed', 'N/A')}")
        self.logger.info(f"  Ratio: {self.ratio}")
        self.logger.info(f"  Diff_LR/Dec_Diff_LR: {self.diff_lr}")
        self.logger.info(f"  Device: {self.device} | Use CUDA: {self.use_cuda}")    

    def seq_extractor(self, x):
        x = x.rstrip(']').lstrip('[').split(', ')
        for i in range(len(x)):
            try:
                x[i] = int(x[i])
            except:
                x[i] = self.iid_all
        return np.array(x)
    
    def parse_seq(self,seq_str):
        if not isinstance(seq_str, str): return []
        return [int(i) for i in seq_str.strip('[]').split(', ') if i.isdigit()]
    
    def get_asin_list(self, seq_str, idx2asin, valid_asins=None, max_len=30):
        if not isinstance(seq_str, str) or '[' not in seq_str:
            return ['MISSING'] * max_len
        
        ids = self.parse_seq(seq_str)
        asins = []
        for idx in ids[:max_len]:
            asin = idx2asin.get(idx)
            if asin and (valid_asins is None or asin in valid_asins):
                asins.append(asin)
        
        return asins + ['MISSING'] * (max_len - len(asins))

    def read_log_data(self, path, batchsize, history=False, shuffle=True):
        if not history:
            cols = ['uid', 'iid', 'y']
            x_col = ['uid', 'iid']
            y_col = ['y']
            data = pd.read_csv(path, header=None)
            data.columns = cols
            X = torch.tensor(data[x_col].values, dtype=torch.long)
            y = torch.tensor(data[y_col].values, dtype=torch.long)
            if self.use_cuda:
                X = X.cuda()
                y = y.cuda()
            dataset = TensorDataset(X, y)
            data_iter = DataLoader(dataset, batchsize, shuffle=shuffle)
            return data_iter
        else:
            data = pd.read_csv(path, header=None)
            cols = ['uid', 'iid', 'y', 'src_pos_seq', 'tgt_pos_seq']
            x_col = ['uid', 'iid']
            y_col = ['y']
            data.columns = cols
            src_pos_seq = keras.preprocessing.sequence.pad_sequences(data.src_pos_seq.map(self.seq_extractor), maxlen=20, padding='post')
            src_pos_seq = torch.tensor(src_pos_seq, dtype=torch.long)
            id_fea = torch.tensor(data[x_col].values, dtype=torch.long)
            X = torch.cat([id_fea, src_pos_seq], dim=1)
            y = torch.tensor(data[y_col].values, dtype=torch.long)
            if self.use_cuda:
                X = X.cuda()
                y = y.cuda()
            dataset = TensorDataset(X, y)
            data_iter = DataLoader(dataset, batchsize, shuffle=shuffle)
            return data_iter

    def read_map_data(self, data_path):
        cols = ['uid', 'iid', 'y', 'src_pos_seq', 'tgt_pos_seq']
        data = pd.read_csv(data_path, header=None)
        data.columns = cols
        X = torch.tensor(data['uid'].unique(), dtype=torch.long)
        y = torch.tensor(np.array(range(X.shape[0])), dtype=torch.long)
        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, self.batchsize_map, shuffle=True)
        return data_iter

    def read_diff_data(self, data_path, batch_size, shuffle=True):
        meta_uid_seq = pd.read_csv(data_path, header=None,
                                names=['meta_uid', 'iid', 'y', 'src_pos_seq', 'tgt_pos_seq'])

        meta_uid   = torch.tensor(meta_uid_seq['meta_uid'].values, dtype=torch.long)
        iid_input  = torch.tensor(meta_uid_seq['iid'].values,   dtype=torch.long)
        y_input    = torch.tensor(meta_uid_seq['y'].values,     dtype=torch.float)

        tgt_pos_seq = [torch.tensor(self.parse_seq(s)) for s in meta_uid_seq['tgt_pos_seq']]
        tgt_pos_seq = pad_sequence(tgt_pos_seq, batch_first=True, padding_value=0)[:, :30].long()

        src_pos_seq= [torch.tensor(self.parse_seq(s)) for s in meta_uid_seq['src_pos_seq']]
        src_pos_seq= pad_sequence(src_pos_seq, batch_first=True, padding_value=0)[:, :30].long()

        if self.use_cuda:
            meta_uid, iid_input, y_input, src_pos_seq, tgt_pos_seq = \
                meta_uid.cuda(), iid_input.cuda(), y_input.cuda(), src_pos_seq.cuda(), tgt_pos_seq.cuda()

        return DataLoader(TensorDataset(meta_uid, iid_input, y_input, src_pos_seq, tgt_pos_seq),
                        batch_size, shuffle=shuffle)

    def read_aug_data(self, tgt_path ):
        #merge source train , target train 
        cols_train = ['uid', 'iid', 'y']
        x_col = ['uid', 'iid']
        y_col = ['y']
        src = pd.read_csv(self.src_path, header=None)
        src.columns = cols_train
        tgt = pd.read_csv(tgt_path, header=None)
        tgt.columns = cols_train

        X_src = torch.tensor(src[x_col].values, dtype=torch.long)
        y_src = torch.tensor(src[y_col].values, dtype=torch.long)
        X_tgt = torch.tensor(tgt[x_col].values, dtype=torch.long)
        y_tgt = torch.tensor(tgt[y_col].values, dtype=torch.long)
        X = torch.cat([X_src, X_tgt])
        y = torch.cat([y_src, y_tgt])
        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, self.batchsize_aug, shuffle=True)

        return data_iter

    def read_decdiff_data(self, data_path, batch_size, user_vecs, shuffle=True):
        data = pd.read_csv(data_path, header=None, 
                        names=['uid', 'iid', 'y', 'src_pos_seq', 'tgt_pos_seq'])
        
        id_map = json.load(open(f"{self.input_root}/id_map.json"))
        idx2asin_src = {v: k for k, v in id_map['iid_dict_src'].items()}

        dataset = []
        for _, r in data.iterrows():
            uid, iid = int(r['uid']), int(r['iid'])
            if iid >= len(user_vecs['item_tgt']) or uid >= len(user_vecs['user_src']):
                continue

            src_ids = self.parse_seq(r['src_pos_seq'])
            src_asins = []
            for idx in src_ids[:30]:
                asin = idx2asin_src.get(idx)
                if asin:
                    src_asins.append(asin)
            
            if not src_asins:
                continue
                
            dataset.append({
                'user_src': user_vecs['user_src'][uid],
                'user_tgt': user_vecs['user_tgt'][uid],
                'item_text': user_vecs['item_tgt'][iid],
                'y': float(r['y']),
                'src_seq': src_asins
            })
        
        def collate(batch):
            return {
                'user_src': torch.stack([b['user_src'] for b in batch]),
                'user_tgt': torch.stack([b['user_tgt'] for b in batch]),
                'item_text': torch.stack([b['item_text'] for b in batch]),
                'y': torch.tensor([b['y'] for b in batch], dtype=torch.float32, device=self.device),
                'src_seq': [b['src_seq'] for b in batch]
            }
        
        return DataLoader(dataset, batch_size=batch_size, 
                        shuffle=shuffle, collate_fn=collate,
                        num_workers=0, pin_memory=False)

        
    def get_data(self):
        print('========Reading data========')
        data_src = self.read_log_data(self.src_path, self.batchsize_src)
        print('src {} iter / batchsize = {} '.format(len(data_src), self.batchsize_src))

        data_tgt = self.read_log_data(self.tgt_path, self.batchsize_tgt)
        print('tgt {} iter / batchsize = {} '.format(len(data_tgt), self.batchsize_tgt))

        data_meta = self.read_log_data(self.meta_path, self.batchsize_meta, history=True)
        print('meta {} iter / batchsize = {} '.format(len(data_meta), self.batchsize_meta))

        data_map = self.read_map_data(self.meta_path)
        print('map {} iter / batchsize = {} '.format(len(data_map), self.batchsize_map))

        data_diff = self.read_diff_data(self.meta_path,batch_size=self.batchsize_diff)
        print('diff {} iter / batchsize = {} '.format(len(data_diff), self.batchsize_diff))

        data_aug = self.read_aug_data(self.tgt_path)
        print('aug {} iter / batchsize = {} '.format(len(data_aug), self.batchsize_aug))

        data_test = self.read_log_data(self.test_path, self.batchsize_test, history=True,shuffle=False)
        print('test {} iter / batchsize = {} '.format(len(data_test), self.batchsize_test))

        data_diff_test = self.read_diff_data(self.test_path,batch_size=self.batchsize_diff_test,shuffle=False)
        print('diff {} iter / batchsize = {} '.format(len(data_diff_test), self.batchsize_diff_test))

        return data_src, data_tgt, data_meta, data_map, data_diff, data_aug, data_test,data_diff_test

    def get_model(self):
        # MFBasedModel
        model = MFBasedModel(self.uid_all, self.iid_all,
                             self.emb_dim, self.meta_dim)
        return model.to(self.device)

    def freeze_emb(self, model, training_phase=True):
        device = self.device
        emb_dim = self.emb_dim

        id_map_path = os.path.join(self.input_root, 'id_map.json')
        with open(id_map_path) as f:
            mapping = json.load(f)
        src_feat = torch.zeros(self.iid_all + 1, emb_dim, device=device)
        tgt_feat = torch.zeros(self.iid_all + 1, emb_dim, device=device)
        for orig_iid, src_idx in mapping['iid_dict_src'].items():
            if src_idx == 0: continue
            global_idx = mapping['iid_dict_global'][orig_iid] 
            if global_idx < self.vis_feat.size(0):
                src_feat[src_idx] = self.vis_feat[global_idx]
        
        for orig_iid, tgt_idx in mapping['iid_dict_tgt'].items():
            if tgt_idx == 0: continue
            global_idx = mapping['iid_dict_global'][orig_iid]
            if global_idx < self.vis_feat.size(0):
                tgt_feat[tgt_idx] = self.vis_feat[global_idx]
        
        with torch.no_grad():
            model.src_model.iid_embedding.weight.copy_(src_feat)
            model.tgt_model.iid_embedding.weight.copy_(tgt_feat)
            model.aug_model.iid_embedding.weight.copy_(tgt_feat)

        if training_phase:
            model.src_model.iid_embedding.weight.requires_grad = False
            model.tgt_model.iid_embedding.weight.requires_grad = False
            model.aug_model.iid_embedding.weight.requires_grad = False
            model.src_model.uid_embedding.weight.requires_grad = True
            model.tgt_model.uid_embedding.weight.requires_grad = True
            model.aug_model.uid_embedding.weight.requires_grad = True
            
            return model
        
        else:
            with torch.no_grad():
                model.register_buffer('pre_item_vec_src', src_feat)
                model.register_buffer('pre_item_vec_tgt', tgt_feat)
                model.register_buffer('pre_user_vec_src', model.src_model.uid_embedding.weight.detach().clone())
                model.register_buffer('pre_user_vec_tgt', model.tgt_model.uid_embedding.weight.detach().clone())
            for name, param in model.named_parameters():
                if 'iid_embedding' in name:
                    param.requires_grad = False            
            return model
    def get_optimizer(self, model):
        optimizer_src = torch.optim.Adam(params=model.src_model.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer_tgt = torch.optim.Adam(params=model.tgt_model.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer_meta = torch.optim.Adam(params=model.meta_net.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer_aug = torch.optim.Adam(params=model.aug_model.parameters(), lr=self.lr, weight_decay=self.wd)
        
        optimizer_map = torch.optim.Adam(params=model.mapping.parameters(), lr=self.lr, weight_decay=self.wd)
    
        return optimizer_src, optimizer_tgt, optimizer_meta, optimizer_aug, optimizer_map
        
    def eval_mae(self, model, data_loader, stage, exp_part=None, **kwargs):
        targets, predicts = [], []
        loss_fn = nn.L1Loss()
        mse_fn = nn.MSELoss()
            
        with torch.no_grad():
            if exp_part == 'VGD_CDR':           
                for batch in data_loader:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    pred = model.predict(
                        batch['user_src'], batch['user_tgt'], batch['src_seq'],
                        batch['item_text'], batch['y']
                    )
                    
                    targets.extend(batch['y'].tolist())
                    predicts.extend(pred.tolist())
            else:
                for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                    model.eval() 
                    pred = model(X, stage, self.device)
                    targets.extend(y.squeeze(1).tolist())
                    predicts.extend(pred.tolist())
        
        targets = torch.tensor(targets).float()
        predicts = torch.tensor(predicts)
        return loss_fn(targets, predicts).item(), torch.sqrt(mse_fn(targets, predicts)).item()
   
    def train(self, data_loader, model, criterion, optimizer, epoch, stage, mapping=False,diff=False):
        print('Training Epoch {}:'.format(epoch + 1))

        loss_ls = []
        if diff == False:
            for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                if mapping:
                    model.train()
                    src_emb, tgt_emb = model(X, stage, self.device)
                    loss = criterion(src_emb, tgt_emb) 

                    model.zero_grad()
                    loss.backward()
                    optimizer.step()

                else:
                    model.train()
                    pred = model(X, stage, self.device)
                    loss = criterion(pred, y.squeeze().float())

                    model.zero_grad()
                    loss.backward()
                    optimizer.step()

                loss_ls.append(loss.item())
            return torch.tensor(loss_ls).mean()
        elif diff == True:
            task_loss_ls = []
            for X in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                model[1].train()

                loss = model[0]( X ,stage,self.device,diff_model = model[1],is_task=False)
                model[1].zero_grad()
                loss.backward()
                _ = torch.nn.utils.clip_grad_norm_(model[1].parameters(),1.)
                optimizer.step()
                #task
                task_loss = model[0]( X ,stage,self.device,diff_model = model[1],is_task=True)
                model[1].zero_grad()
                task_loss.backward()
                _ = torch.nn.utils.clip_grad_norm_(model[1].parameters(),1.)
                optimizer.step()
                
                loss_ls.append(loss.item())
                task_loss_ls.append(task_loss.item())
            #return torch.tensor(loss_ls).mean()
            return torch.tensor(loss_ls).mean() ,torch.tensor(task_loss_ls).mean() 

    def update_results(self, mae, rmse,  phase):

        if mae < self.results[phase + '_mae'] and rmse < self.results[phase + '_rmse']:
            self.results[phase + '_mae'] = mae
            self.results[phase + '_rmse'] = rmse
        
    def reset_results(self):
        self.results = {'tgt_mae': 10, 'tgt_rmse': 10, 
                        'aug_mae': 10, 'aug_rmse': 10, 
                        'diff_mae': 10, 'diff_rmse': 10
                        }

    def TgtOnly(self, model, data_tgt, data_test, criterion, optimizer):
        print('=========TgtOnly========')
        for i in range(self.epoch):
            loss = self.train(data_tgt, model, criterion, optimizer, i, stage='train_tgt')
            mae, rmse = self.eval_mae(model, data_test, stage='test_tgt')
            self.update_results(mae, rmse, 'tgt')
            print('loss:{} MAE: {} RMSE: {}'.format(loss,mae, rmse))

    def SrcOnly(self, model, data_src, criterion, optimizer_src): 
        print('=====SrcOnly=====')
        for i in range(self.epoch):
            loss = self.train(data_src, model, criterion, optimizer_src, i, stage='train_src')

    def DataAug(self, model, data_aug, data_test, criterion, optimizer):
        print('=========DataAug========')
        for i in range(self.epoch):
            loss = self.train(data_aug, model, criterion, optimizer, i, stage='train_aug')
            mae, rmse = self.eval_mae(model, data_test, stage='test_aug')
            self.update_results(mae, rmse, 'aug')
            print('loss:{} MAE: {} RMSE: {}'.format(loss,mae, rmse))

    def DecDiff_CDR(self, model, train_loader, test_loader, optimizer):    
        for epoch in range(self.epoch):
            model.train()
            total_losses = []
            recon_losses = []
            task_losses = []
            diff_losses = []
            
            train_iter = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1} Training", ncols=100)
            for batch in train_iter:
                batch_asins = set()
                for seq in batch['src_seq']:
                    batch_asins.update([asin for asin in seq if asin != 'MISSING'])
                batch_asins = list(batch_asins)
                
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                losses = model.compute_loss(
                    batch['user_src'], batch['user_tgt'], batch['src_seq'],
                    batch['item_text'], batch['y']
                )
                                
                optimizer.zero_grad()
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()                

                total_losses.append(losses['total'].item())
                recon_losses.append(losses['recon'])
                task_losses.append(losses['task'])
                diff_losses.append(losses['diff'])
                
                train_iter.set_postfix({'total': f"{losses['total'].item():.4f}"})
            
            avg_total = np.mean(total_losses) if total_losses else 0.0
            avg_recon = np.mean(recon_losses) if recon_losses else 0.0
            avg_task = np.mean(task_losses) if task_losses else 0.0
            avg_diff = np.mean(diff_losses) if diff_losses else 0.0
            print(f"Total: {avg_total:.4f} Recon: {avg_recon:.3f} Task: {avg_task:.4f} Diff: {avg_diff:.4f}", end="")
            
            model.eval()
            with torch.no_grad():
                mae, rmse = self.eval_mae(model, test_loader,stage="VGD-Test", exp_part="VGD_CDR")
            
            self.update_results(mae, rmse, 'diff')
            print(f"  - MAE: {mae} | RMSE: {rmse}")
            self.logger.info(f"Epoch {epoch+1}| Total: {avg_total:.4f} Recon: {avg_recon:.3f} Task: {avg_task:.4f} Diff: {avg_diff:.4f} - MAE: {mae} | RMSE: {rmse}")
        
        self.result_print(['diff'])

    def model_save(self,model,path):
        torch.save(model.state_dict(),path)

    def model_load(self, model, path, strict=False):
        state = torch.load(path, map_location=self.device)
        model.load_state_dict(state, strict=strict)

    def result_print(self, phase):
        print_str=''
        for p in phase:
            for m in ['_mae', '_rmse']:
                metric_name = p + m
                # print_str += metric_name + ': {:.10f} '.format(self.results[metric_name])
                print_str += metric_name + ': {} '.format(self.results[metric_name])
        print(print_str)

        self.logger.info(print_str)

    def main(self, exp_part='None_CDR', save_path=None):
        criterion = torch.nn.MSELoss()
        data_src, data_tgt, data_meta, data_map, data_diff, data_aug, data_test, data_diff_test = self.get_data()
        optimizer = optimizer_diff = None

        if exp_part == 'None_CDR':
            model = self.get_model()
            optimizer_src, optimizer_tgt, _, optimizer_aug, _ = self.get_optimizer(model)
            
            self.TgtOnly(model, data_tgt, data_test, criterion, optimizer_tgt)
            self.SrcOnly(model, data_src, criterion, optimizer_src)
            self.DataAug(model, data_aug, data_test, criterion, optimizer_aug)
            self.result_print(['tgt', 'aug'])
            model = self.freeze_emb(model, training_phase=False)
            self.model_save(model, save_path)

        elif exp_part == 'Base_Space':
            id_map_path = os.path.join(self.input_root, 'id_map.json')
            with open(id_map_path) as g:
                mapping = json.load(g)
            global2local = mapping['iid_dict_global']

            src_feat_dict, src_dim = DecoupleUtils.load_feat_dict(self.root, self.src, 'text')
            tgt_feat_dict, tgt_dim = DecoupleUtils.load_feat_dict(self.root, self.tgt, 'text')
            
            raw_feat_dict = {**src_feat_dict, **tgt_feat_dict}
            feat_dim = src_dim or tgt_dim
            
            vis_feat = torch.zeros(len(global2local), feat_dim, device=self.device)
            for item_id, vec in raw_feat_dict.items():
                if item_id in global2local:
                    vis_feat[global2local[item_id]] = vec.to(self.device)
            
            if feat_dim != self.emb_dim:
                ae = FeatrueAE(feat_dim, self.emb_dim).to(self.device)
                self.vis_feat = ae.train_model(vis_feat, self.device, epochs=30, lr=1e-3)
            else:
                self.vis_feat = vis_feat
            
            model = self.get_model()
            torch.nn.init.normal_(model.src_model.uid_embedding.weight, std=1.0)
            torch.nn.init.normal_(model.tgt_model.uid_embedding.weight, std=1.0)
            torch.nn.init.normal_(model.aug_model.uid_embedding.weight, std=1.0)
            
            self.freeze_emb(model, training_phase=True)
            optimizer_src, optimizer_tgt, _, optimizer_aug, _ = self.get_optimizer(model)

            self.TgtOnly(model, data_tgt, data_test, criterion, optimizer_tgt)
            self.SrcOnly(model, data_src, criterion, optimizer_src)
            self.result_print(['tgt'])
            model = self.freeze_emb(model, training_phase=False)
            self.model_save(model, save_path)
        elif exp_part == 'Decouple':
            print("===== Decouple =====")

            decouple_epoch = int(self.config.get('epoch', 6))
            decouple_lr = float(self.config.get('decouple_lr', 1e-4))
            decouple_batch_size = self.batchsize_diff

            src_image_feat, image_input_dim = DecoupleUtils.load_feat_dict(self.root, self.src, 'image')
            tgt_image_feat, _ = DecoupleUtils.load_feat_dict(self.root, self.tgt, 'image')
            all_asins = list(set(src_image_feat.keys()).union(set(tgt_image_feat.keys())))
            asin2idx = {asin: idx for idx, asin in enumerate(all_asins)}
            num_total_asins = len(all_asins)
            global_image_feat = torch.zeros((num_total_asins, image_input_dim), device=self.device)
            for asin, feat in src_image_feat.items():
                idx = asin2idx[asin]
                global_image_feat[idx] = feat.to(self.device)
            for asin, feat in tgt_image_feat.items():
                idx = asin2idx[asin]
                global_image_feat[idx] = feat.to(self.device)
            asin2domain = {asin: 'src' for asin in src_image_feat.keys()}
            asin2domain.update({asin: 'tgt' for asin in tgt_image_feat.keys()})

            decouple_model = DecoupleModel(input_dim=image_input_dim,latent_dim=self.emb_dim).to(self.device)
            optimizer = torch.optim.Adam(decouple_model.parameters(), lr=decouple_lr, weight_decay=self.wd)
            image_loader = DecoupleUtils.build_data_loader(
                meta_path=self.meta_path, 
                input_root=self.input_root,
                src_feat_dict=src_image_feat, 
                tgt_feat_dict=tgt_image_feat,
                batch_size=decouple_batch_size, 
                task_id=self.task, 
                feat_type='image',
                root=self.root, 
                use_cache=True, 
                ratio=str(int(self.ratio[0] * 10))
            )

            for epoch in range(decouple_epoch):
                epoch_losses = []
                epoch_info_losses = []
                epoch_ortho_losses = []
                epoch_recon_losses = []
                
                pbar = tqdm.tqdm(image_loader, desc=f"Epoch {epoch+1}/{decouple_epoch}")
                for batch in pbar:                    
                    uids, sample_types, pos1, pos2, negs = batch
                    
                    # decouple_loss: call batch_encode_asins()
                    total_loss, info_loss, ortho_loss, recon_loss = decouple_model.decouple_loss(
                        (uids, sample_types, pos1, pos2, negs),
                        global_feat=global_image_feat, 
                        asin2idx=asin2idx,
                        asin2domain=asin2domain, 
                        device=self.device
                    )
                    
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(decouple_model.parameters(), max_norm=2.0)
                    optimizer.step()
            
                    epoch_losses.append(total_loss.item())
                    epoch_info_losses.append(info_loss.item())
                    epoch_ortho_losses.append(ortho_loss.item())
                    epoch_recon_losses.append(recon_loss.item())
                    
                    pbar.set_postfix({'total_loss': f'{total_loss.item():.4f}'})
                avg_total_loss = np.mean(epoch_losses) if epoch_losses else 0.0
                avg_info_loss = np.mean(epoch_info_losses) if epoch_info_losses else 0.0
                avg_ortho_loss = np.mean(epoch_ortho_losses) if epoch_ortho_losses else 0.0
                avg_recon_loss = np.mean(epoch_recon_losses) if epoch_recon_losses else 0.0

                print(f"Epoch {epoch+1} | "
                    f"Total Loss: {avg_total_loss:.4f} | "
                    f"Info Loss: {avg_info_loss:.4f} | "
                    f"Ortho Loss: {avg_ortho_loss:.4f} | "
                    f"Recon Loss: {avg_recon_loss:.4f}")
            
                model_dir = os.path.dirname(self.config['save_path'])
                os.makedirs(model_dir, exist_ok=True)
                decouple_model_path = os.path.join(model_dir, f'{self.task}_{str(int(self.ratio[0] * 10))}_epoch{epoch+1}_decouple_model.pth')
                decouple_model.save(decouple_model_path)

        elif exp_part == 'VGD_CDR':
            print("===== VGD-CDR =====")
            id_map_path = os.path.join(self.input_root, 'id_map.json')
            with open(id_map_path) as f:
                id_map = json.load(f)
            asin2src_idx = id_map['iid_dict_src']  # {ASIN: src_idx}
            src_image_feat, image_input_dim = DecoupleUtils.load_feat_dict(self.root, self.src, 'image')
            tgt_image_feat, _ = DecoupleUtils.load_feat_dict(self.root, self.tgt, 'image')
            user_vecs = DecoupleUtils.load_base_vectors(save_path, self.device)
            pre_item_vec_src = user_vecs["item_src"]  # [num_src_items, emb_dim]
            
            all_asins = list(set(src_image_feat.keys()).union(set(tgt_image_feat.keys())))
            asin2idx = id_map['iid_dict_global']#{asin: i for i, asin in enumerate(all_asins)}
            asin2domain = {asin: 'src' for asin in src_image_feat.keys()}
            asin2domain.update({asin: 'tgt' for asin in tgt_image_feat.keys()})
            
            global_image_feat = torch.zeros(len(all_asins), image_input_dim, device=self.device)
            for asin, idx in asin2idx.items():
                if asin in src_image_feat:
                    global_image_feat[idx] = src_image_feat[asin].to(self.device)
                else:
                    global_image_feat[idx] = tgt_image_feat[asin].to(self.device)
            decouple_model_path = os.path.join(os.path.dirname(self.config['save_path']), 
                                            f'{self.task}_{str(int(self.ratio[0] * 10))}_epoch{self.config["disentangle_epoch"]}_decouple_model.pth')

            decouple_model = DecoupleModel.load(decouple_model_path, self.device, input_dim=image_input_dim, latent_dim=self.config["emb_dim"])
            decouple_model.eval()
            for param in decouple_model.parameters():
                param.requires_grad = False
            decouple_model.batch_encode_asins(
                all_asins, 
                global_feat=global_image_feat,
                asin2idx=asin2idx,
                asin2domain=asin2domain,
                device=self.device, 
                training=False
            )
            print(f"pre-encoding items...")
            model = DecDiffCDR(
                num_steps=self.diff_steps, diff_dim=self.diff_dim, input_dim=self.emb_dim,
                c_scale=self.diff_scale, diff_sample_steps=self.diff_sample_steps,
                diff_task_lambda=self.diff_task_lambda, diff_mask_rate=self.diff_mask_rate,
                decouple_model=decouple_model, 
                global_image_feat=global_image_feat,
                asin2idx=asin2idx,
                asin2domain=asin2domain,
                emb_dim=self.emb_dim,
                pre_item_vec_src=pre_item_vec_src,
                asin2src_idx=asin2src_idx,
            ).to(self.device)

            train_loader = self.read_decdiff_data(self.meta_path, self.batchsize_diff, user_vecs, shuffle=True)
            test_loader = self.read_decdiff_data(self.test_path, self.batchsize_diff_test, user_vecs, shuffle=False)
            print(f"data loading complete！train sample: {len(train_loader.dataset)}, test sample: {len(test_loader.dataset)}")
            
            optimizer = torch.optim.Adam([
                {'params': model.diff_model.parameters(), 'lr': self.diff_lr}
            ], weight_decay=self.wd)
            
            self.DecDiff_CDR(model, train_loader, test_loader, optimizer)
            
        else:
            raise ValueError(f'unknown exp_part: {exp_part}')