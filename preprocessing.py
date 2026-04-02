import pandas as pd
import gzip
import json
import tqdm
import random
import os

class DataPreprocessingMid():
    def __init__(self, root, dealing):
        self.root = root
        self.dealing = dealing

    def main(self):
        print('Parsing ' + self.dealing + ' Mid...')
        re = []
        with gzip.open(self.root + 'raw/' + self.dealing + '_5.json.gz', 'rb') as f:
            for line in tqdm.tqdm(f, smoothing=0, mininterval=1.0):
                line = json.loads(line)
                re.append([line['reviewerID'], line['asin'], line['overall']])
        re = pd.DataFrame(re, columns=['uid', 'iid', 'y'])
        print(self.dealing + ' Mid Done.')
        re.to_csv(self.root + 'mid/' + self.dealing + '.csv', index=0)
        return re

class DataPreprocessingReady():
    def __init__(self, root, src_tgt_pairs, task, ratio, seed=2020):
        self.root = root
        self.src = src_tgt_pairs[task]['src']
        self.tgt = src_tgt_pairs[task]['tgt']
        self.ratio = ratio
        self.task = task
        random.seed(seed)

    def read_mid(self, field):
        path = self.root + 'mid/' + field + '.csv'
        re = pd.read_csv(path)
        return re

    def mapper(self, src, tgt):
        print('Source inters: {}, uid: {}, iid: {}.'.format(len(src), len(set(src.uid)), len(set(src.iid))))
        print('Target inters: {}, uid: {}, iid: {}.'.format(len(tgt), len(set(tgt.uid)), len(set(tgt.iid))))
        co_uid = set(src.uid) & set(tgt.uid)
        all_uid = set(src.uid) | set(tgt.uid)
        print('All uid: {}, Co uid: {}.'.format(len(all_uid), len(co_uid)))
        uid_dict = dict(zip(sorted(all_uid), range(len(all_uid))))

        src_items = sorted(set(src.iid))
        tgt_items = sorted(set(tgt.iid))
        iid_dict_src = dict(zip(src_items, range(len(src_items))))
        iid_dict_tgt = dict(zip(tgt_items, range(len(src_items), len(src_items) + len(tgt_items))))
        all_items = src_items + tgt_items
        iid_dict_global = dict(zip(all_items, range(len(all_items))))
        
        self.id_map = {
            'uid_dict': uid_dict,
            'iid_dict_src': iid_dict_src,
            'iid_dict_tgt': iid_dict_tgt,
            'iid_dict_global': iid_dict_global,
            'src_items_count': len(src_items),
            'tgt_items_count': len(tgt_items),
            'total_items': len(all_items)
        }
        
        src.uid = src.uid.map(uid_dict)
        src.iid = src.iid.map(iid_dict_src)
        tgt.uid = tgt.uid.map(uid_dict)
        tgt.iid = tgt.iid.map(iid_dict_tgt)
        return src, tgt

    def get_history(self, data, uid_set):
        pos_seq_dict = {}
        for uid in tqdm.tqdm(uid_set):
            pos = data[(data.uid == uid) & (data.y > 3)].iid.values.tolist()
            pos_seq_dict[uid] = pos
        return pos_seq_dict

    def split(self, src, tgt):
        print('All iid: {}.'.format(len(set(src.iid) | set(tgt.iid))))
        src_users = set(src.uid.unique())
        tgt_users = set(tgt.uid.unique())
        co_users = src_users & tgt_users
        test_users = set(random.sample(sorted(co_users), round(self.ratio[1] * len(co_users))))
        
        train_src = src
        train_tgt = tgt[tgt['uid'].isin(tgt_users - test_users)].copy()
        test = tgt[tgt['uid'].isin(test_users)].copy()
        
        src_pos_seq_dict = self.get_history(src, co_users)
        tgt_pos_seq_dict = self.get_history(tgt, co_users)
        
        train_meta = tgt[tgt['uid'].isin(co_users - test_users)].copy()
        
        train_meta['src_pos_seq'] = train_meta['uid'].map(src_pos_seq_dict)
        train_meta['tgt_pos_seq'] = train_meta['uid'].map(tgt_pos_seq_dict)
        test['src_pos_seq'] = test['uid'].map(src_pos_seq_dict)
        test['tgt_pos_seq'] = test['uid'].map(tgt_pos_seq_dict)
        
        return train_src, train_tgt, train_meta, test

    def save(self, train_src, train_tgt, train_meta, test):
        output_root = self.root + 'ready/_' + str(int(self.ratio[0] * 10)) + '_' + str(int(self.ratio[1] * 10)) + \
                    '/tgt_' + self.tgt + '_src_' + self.src
        if not os.path.exists(output_root):
            os.makedirs(output_root)
        print(output_root)
                
        train_src.to_csv(output_root + '/train_src.csv', sep=',', header=None, index=False)
        train_tgt.to_csv(output_root + '/train_tgt.csv', sep=',', header=None, index=False)
        train_meta.to_csv(output_root + '/train_meta.csv', sep=',', header=None, index=False)
        test.to_csv(output_root + '/test.csv', sep=',', header=None, index=False)
        
        id_map_path = output_root + '/id_map.json'
        with open(id_map_path, 'w') as f:
            json.dump(self.id_map, f, indent=2)
        print(f'ID map save to: {id_map_path}')

    def main(self):
        src = self.read_mid(self.src)
        tgt = self.read_mid(self.tgt)
        src, tgt = self.mapper(src, tgt)
        
        print("\n" + "="*60)
        print(f"task {self.task}: {self.src} → {self.tgt}")
        print(f"ratio {self.ratio[0]}:{self.ratio[1]} complete.")
        print("="*60)
        print("please update config.json：")
        print(f'''   "src_tgt_pairs": {{
      "{self.task}": {{
        "src": "{self.src}",
        "tgt": "{self.tgt}",
        "uid": {len(self.id_map['uid_dict'])},
        "iid": {self.id_map['total_items']},
        ...''')
        print("="*60 + "\n")
        
        train_src, train_tgt, train_meta, test = self.split(src, tgt)
        self.save(train_src, train_tgt, train_meta, test)