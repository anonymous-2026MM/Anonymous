
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver

noise_schedule = NoiseScheduleVP(schedule='linear')

def get_timestep_embedding(timesteps, embedding_dim: int):
    """
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    timesteps = timesteps.to(dtype=torch.float32)

    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    assert embedding_dim % 2 == 0
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32,device=timesteps.device) * -emb)
    # emb = tf.range(num_embeddings, dtype=DEFAULT_DTYPE)[:, None] * emb[None, :]
    #emb = tf.cast(timesteps, dtype=torch.float32)[:, None] * emb[None, :]
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], axis=1)
    #if embedding_dim % 2 == 1:  # zero pad
    #    emb = torch.pad(emb, [0,1])
    assert emb.shape == torch.Size([timesteps.shape[0], embedding_dim])
    return emb
class BehaviorSeq(nn.Module):
    def __init__(self, input_dim=128, output_dim=64, n_heads=8, n_layers=4, max_len=100, dropout=0.2):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, input_dim))
        if input_dim == output_dim:
            self.pool = nn.Sequential(
                # nn.LayerNorm(input_dim),
               nn.Identity()
            )
        else:
            self.pool = nn.Sequential(
                # nn.LayerNorm(input_dim),
                nn.Linear(input_dim, output_dim)
            )

    def forward(self, seq_emb):
        B, L, D = seq_emb.shape
        out = seq_emb #+ self.pos_emb[:, :L, :]
        # out = self.encoder(x)
        hist = out.mean(dim=1)
        return self.pool(hist)
class DiffCDR(nn.Module):
    def __init__(self,num_steps=200, diff_dim=32,input_dim =32,c_scale=0.1,diff_sample_steps=30,diff_task_lambda=0.1,diff_mask_rate=0.1,
                 keep_trace=False,  aggregate_method=None,use_al_mlp=True,diff_speed=0.5):
        super(DiffCDR,self).__init__()

        #-------------------------------------------
        #define params
        self.diff_speed = diff_speed
        self.register_parameter('scale', nn.Parameter(torch.tensor(5.0)))
        self.register_parameter('bias', nn.Parameter(torch.tensor(3.0)))
        self.num_steps = num_steps
        self.betas = torch.linspace(1e-4,0.02 ,num_steps)

        self.alphas = 1-self.betas
        self.alphas_prod = torch.cumprod(self.alphas,0)
        self.alphas_prod_p = torch.cat([torch.tensor([1]).float(),self.alphas_prod[:-1]],0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        self.one_minus_alphas_bar_log = torch.log(1 - self.alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)

        assert self.alphas.shape==self.alphas_prod.shape==self.alphas_prod_p.shape==\
        self.alphas_bar_sqrt.shape==self.one_minus_alphas_bar_log.shape\
        ==self.one_minus_alphas_bar_sqrt.shape

        #-----------------------------------------------
        self.diff_dim = diff_dim
        self.input_dim = input_dim
        self.task_lambda = diff_task_lambda
        self.sample_steps = diff_sample_steps
        self.c_scale = c_scale
        self.mask_rate = diff_mask_rate
        self.mask_rate2 = diff_mask_rate
        #-----------------------------------------------
        self.linears = nn.ModuleList([
            nn.Linear(input_dim, diff_dim),
            nn.Linear(diff_dim, input_dim)
        ])
        
        self.step_emb_linear = nn.Sequential(
            nn.Linear(diff_dim, input_dim)
        )
        self.cond_emb_linear = nn.Sequential(
            nn.Linear(input_dim, input_dim)
        )
        self.cond_emb_linear2 = nn.Sequential(
            nn.Linear(input_dim, input_dim)
        )

        self.behavior_enc = BehaviorSeq(input_dim)
        self.behavior_enc2 = BehaviorSeq(input_dim)

        self.num_layers = 1

        #linear for alm 
        if use_al_mlp:
            self.al_mlp = nn.Sequential(
                nn.Linear(input_dim, input_dim * 2), nn.GELU(),nn.Dropout(0.1),
                nn.Linear(input_dim * 2, input_dim), nn.LayerNorm(input_dim)
            )
        else:
            self.al_mlp = nn.Identity()
        self.norm = nn.LayerNorm(input_dim)
        self.keep_trace = keep_trace

        if keep_trace:            
            self.aggregate_method = aggregate_method
            self.attention_mlp = nn.Sequential(
                nn.Linear(input_dim * 2, input_dim),  # [final_emb; traj_step]
                nn.ReLU(),
                nn.Linear(input_dim, 1)
            )
            
            self.output_proj = nn.Identity()

    def aggregate_trajectory(self, final_emb, trajectory):
        """        
        Args:
            final_emb: tensor, shape [batch_size, input_dim]
            trajectory: tensor, shape [num_steps, batch_size, input_dim]
        """
        method = self.aggregate_method
        k,_,_  = trajectory.shape
        # trajectory shape: [k, batch_size, input_dim]
        
        if not method:
            return final_emb
        
        elif method == 'mean':
            context = trajectory.mean(dim=0)  # [batch_size, input_dim]
            return context
        
        elif method == 'attention':
            # MLP-based attention
            
            batch_size, dim = final_emb.shape
            final_expanded = final_emb.unsqueeze(0).expand(k, -1, -1)  # [k, batch_size, dim]
            combined = torch.cat([final_expanded, trajectory], dim=-1)
            attn_scores = self.attention_mlp(combined)
            attn_weights = F.softmax(attn_scores, dim=0)  # [k, batch_size, 1]
            context = (trajectory * attn_weights).sum(dim=0)  # [batch_size, dim]

            weight=0.9
            enhanced =weight* final_emb +(1-weight)* context
            return self.output_proj(enhanced)
        elif method == 'weighted_attention':
            weights = F.softmax(self.aggregation_weights, dim=0)  # [k]
            context1 = (trajectory * weights.view(-1, 1, 1)).sum(dim=0)  # [batch_size, input_dim]
            
            batch_size, dim = final_emb.shape
            final_expanded = final_emb.unsqueeze(0).expand(k, -1, -1)  # [k, batch_size, dim]
            combined = torch.cat([final_expanded, trajectory], dim=-1)
            attn_scores = self.attention_mlp(combined)
            attn_weights = F.softmax(attn_scores, dim=0)  # [k, batch_size, 1]
            context = (trajectory * attn_weights).sum(dim=0)  # [batch_size, dim]
            
            weight=0.7
            enhanced =weight* final_emb +(1-weight)/2 * context+(1-weight)/2 * context1
            return self.output_proj(enhanced)
        
        else:
            raise ValueError(f"Unknown aggregation method: {method}")


    def forward(self, x, t, cond_emb, cond_mask, cond_emb2=None, cond_mask2=None):
        for idx in range(self.num_layers):
            t_embedding = get_timestep_embedding(t, self.diff_dim)
            t_embedding = self.step_emb_linear(t_embedding)
            if cond_emb is not None:
                cond_embedding = self.cond_emb_linear(cond_emb)
            else:
                cond_embedding = None
            if random.random() < self.diff_speed and cond_emb2 is not None and cond_mask2 is not None:
                cond_embedding2 = self.cond_emb_linear2(cond_emb2)
            else:
                cond_embedding2 = None

            if cond_embedding is not None and cond_embedding2 is not None:
                t_c_emb = t_embedding + \
                        cond_embedding * cond_mask.unsqueeze(-1) + \
                        cond_embedding2 * cond_mask2.unsqueeze(-1)
            elif cond_embedding is not None:
                t_c_emb = t_embedding + cond_embedding * cond_mask.unsqueeze(-1)
            elif cond_embedding2 is not None:
                t_c_emb = t_embedding + cond_embedding2 * cond_mask2.unsqueeze(-1)
            else:
                t_c_emb = t_embedding
            
            residual = x
            x = x + t_c_emb
            x = self.linears[0](x)
            x = F.gelu(x)
            x = self.linears[1](x)
            x = self.norm(x + residual)
        
        return x
        
    def get_al_emb(self,emb):
        return self.al_mlp (emb)
#---------------------------------------------------------
#loss 
def q_x_fn(model,x_0,t,device):
    noise = torch.normal(0,1,size = x_0.size() ,device=device)

    alphas_t = model.alphas_bar_sqrt.to(device)[t]
    alphas_1_m_t = model.one_minus_alphas_bar_sqrt.to(device)[t]

    return (alphas_t * x_0 + alphas_1_m_t * noise),noise

def diffusion_loss_fn(model, user_src, user_tgt, cond_emb, iid_emb, y_input,
                      device, is_task, cond_emb2=None):
    x_0 = user_tgt
    num_steps = model.num_steps
    
    if not is_task:
        batch_size = x_0.shape[0]
        t = torch.randint(0, num_steps, (batch_size // 2,), device=device)
        if batch_size % 2 == 0:
            t = torch.cat([t, num_steps - 1 - t], dim=0)
        else:
            extra_t = torch.randint(0, num_steps, (1,), device=device)
            t = torch.cat([t, num_steps - 1 - t, extra_t], dim=0)
        t = t.unsqueeze(-1)
        
        x, e = q_x_fn(model, x_0, t, device)
        
        cond_mask = (torch.rand(cond_emb.shape[0], device=device) <= model.mask_rate).float()
        cond_mask = 1 - cond_mask.int()
        cond_mask2 = None
        if cond_emb2 is not None:
            cond_mask2 = (torch.rand(cond_emb2.shape[0], device=device) <= model.mask_rate2).float()
            cond_mask2 = 1 - cond_mask2.int()
        output = model(x, t.squeeze(-1), cond_emb, cond_mask, cond_emb2, cond_mask2)
        return F.smooth_l1_loss(e, output)
    
    else:
        final_emb = p_sample_loop(model, cond_emb, user_src, device, cond_emb2)
        recon_loss = F.smooth_l1_loss(final_emb, user_tgt)
        
        final_emb_norm = F.normalize(final_emb, p=2, dim=1)
        item_text_norm = F.normalize(iid_emb, p=2, dim=1)
        y_pred = (final_emb_norm * item_text_norm).sum(1) * model.scale + model.bias
        
        task_loss = (y_pred - y_input.squeeze().float()).square().mean()
        return recon_loss + model.task_lambda * task_loss

#generation fun
def p_sample(model, cond_emb, x, device, cond_emb2=None):
    classifier_scale_para = model.c_scale
    dmp_sample_steps = model.sample_steps
    num_steps = model.num_steps
    model_kwargs = {
        'cond_emb': cond_emb,
        'cond_mask': torch.zeros(cond_emb.size(0), device=device)
    }
    
    if cond_emb2 is not None:
        model_kwargs['cond_emb2'] = cond_emb2
        model_kwargs['cond_mask2'] = torch.zeros(cond_emb2.size(0), device=device)
    
    model_fn = model_wrapper(
        model,
        noise_schedule,
        is_cond_classifier=True,
        classifier_scale=classifier_scale_para,
        time_input_type="1",
        total_N=num_steps,
        model_kwargs=model_kwargs
    )
    
    dpm_solver = DPM_Solver(model_fn, noise_schedule)
    sample_output = dpm_solver.sample(
        x,
        steps=dmp_sample_steps,
        eps=1e-4,
        adaptive_step_size=False,
        fast_version=True,
        keep_trace=model.keep_trace
    )
    
    if model.keep_trace:
        final_x, trajectory = sample_output
        aggregated = model.aggregate_trajectory(final_x,trajectory)
        return model.get_al_emb(aggregated).to(device)
    else:
        return model.get_al_emb(sample_output).to(device)

def p_sample_loop(model, cond_emb, ref_tensor, device, cond_emb2=None, from_noise=True):
    if from_noise:
        cur_x = torch.randn_like(ref_tensor, device=device)
    else:
        cur_x = ref_tensor
    return p_sample(model, cond_emb, cur_x, device, cond_emb2)