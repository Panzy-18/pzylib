import torch
import bmtrain as bmt
import math
import torch.nn.functional as F
from torch import nn

class Linear(bmt.DistributedModule):

    def __init__(self,
                 dim_in : int,
                 dim_out : int,
                 dtype = torch.half,
                 init_gain : float = 0.02,
                 bias : bool = False
                 ) -> None:
        super().__init__()
        self.weight = bmt.DistributedParameter(
            torch.empty((dim_out, dim_in), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.xavier_normal_, gain=init_gain)
        )
        self.bias = bmt.DistributedParameter(
            torch.empty((dim_out,), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.zeros_)
        ) if bias else None
    
    def forward(self, x : torch.Tensor):
        x = F.linear(x, self.weight)
        if self.bias is not None:
            x += self.bias
        return x

class DenseGatedACT(bmt.DistributedModule):

    def __init__(self,
                 dim_in : int,
                 dim_out : int,
                 dtype = torch.half,
                 init_gain : float = 0.02,
                 bias : bool = False,
                 act_fn : str = 'gelu',
                 ) -> None:
        super().__init__()
        self.w_0 = Linear(
            dim_in = dim_in,
            dim_out = dim_out,
            dtype = dtype,
            init_gain = init_gain,
            bias = bias,
        )
        self.w_1 = Linear(
            dim_in = dim_in,
            dim_out = dim_out,
            dtype = dtype,
            init_gain = init_gain,
            bias = bias,
        )
        if act_fn == 'relu':
            self.act_fn = torch.nn.ReLU()
        elif act_fn == 'gelu':
            self.act_fn = torch.nn.GELU()
        else:
            raise NotImplementedError
    
    def forward(self, x : torch.Tensor):
        score = self.act_fn(self.w_0(x))
        x = torch.mul(self.w_1(x), score)
        return x

class RMSNorm(bmt.DistributedModule):

    def __init__(self,
                 dim : int,
                 dtype = torch.half,
                 init_var : float = 0.02,
                 eps : float = 1e-5,
                 ) -> None:
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = bmt.DistributedParameter(torch.ones(dim, dtype=dtype) * init_var)

    def forward(self, x: torch.Tensor):
        assert x.shape[-1] == self.dim

        dtype = x.dtype
        var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        x = (x * torch.rsqrt(var + self.eps)).to(dtype)

        return torch.mul(x, self.weight)


class Embedding(bmt.DistributedModule):

    def __init__(self,
                num_embeddings : int,
                embedding_dim : int,
                dtype = torch.half,
                init_mean : float = 0.0,
                init_std : float= 0.02,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = bmt.DistributedParameter(
            torch.empty(num_embeddings, embedding_dim, dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.trunc_normal_, mean=init_mean, std=init_std*1.137, a=-2*init_std, b=2*init_std)
        )
    
    def forward(self, ids : torch.Tensor):
        embeds = F.embedding(ids, self.weight)
        return embeds
    
    def projection(self, x : torch.Tensor):
        logits = F.linear(x, self.weight)
        return logits


class ClassificationHead(torch.nn.Module):
    def __init__(self, 
                 dim_model : int, 
                 num_classes : int, 
                 init_gain : float,
                 norm_eps : float, 
                 norm_init_var : float,
                 dtype = torch.half
                 ):
        super().__init__()
        self.dense = Linear(
            dim_in = dim_model, 
            dim_out = dim_model, 
            init_gain = init_gain,
            bias = True, 
            dtype = dtype
            )
        self.act_fn = torch.nn.GELU()
        self.norm = RMSNorm(
            dim = dim_model, 
            eps = norm_eps, 
            dtype = dtype,
            init_var = norm_init_var,
            )
        self.decoder = Linear(
            dim_in = dim_model, 
            dim_out = num_classes, 
            init_gain = init_gain,
            bias = True, 
            dtype = dtype
        )
    
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.norm(hidden_states)
        logits = self.decoder(hidden_states)
        return logits


class Pooler(torch.nn.Module):
    def __init__(self,
                 dim_model : int, 
                 init_gain : float, 
                 num_classes : int, 
                 norm_eps : float, 
                 norm_init_var : float,
                 dtype = torch.half,
                 mode = 'mean',
    ) -> None:
        super().__init__()
        self.mode = mode
        self.dense = Linear(
            dim_in = dim_model, 
            dim_out = dim_model, 
            init_gain = init_gain,
            bias = False, 
            dtype = dtype
        )
        self.act_fn = torch.nn.GELU()
        self.norm = RMSNorm(
            dim = dim_model, 
            eps = norm_eps, 
            dtype = dtype,
            init_var = norm_init_var,
            )
        self.decoder = Linear(
            dim_in = dim_model, 
            dim_out = num_classes, 
            init_gain = init_gain,
            bias = True, 
            dtype = dtype
        )
    
    def forward(self,
        hidden_states : torch.Tensor,
        attention_mask : torch.Tensor,
    ):
        if self.mode == 'mean':
            hidden_states = self.dense(hidden_states)
            pooled_output = (hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1).unsqueeze(-1)
        
        elif self.mode == 'cls':
            pooled_output = self.dense(hidden_states[:, 0, :])
        
        else:
            raise NotImplementedError
        
        pooled_output = self.act_fn(pooled_output)
        pooled_output = self.norm(pooled_output)
        logits = self.decoder(pooled_output)

        return logits

class OHEMLoss(nn.Module):

    def __init__(
        self,
        ratio = 0.5,
        criterion = nn.CrossEntropyLoss(reduction = 'none'),
    ) -> None:
        super().__init__()
        self.ratio = ratio
        self.criterion = criterion
    
    def forward(self,
        logits : torch.Tensor,
        labels : torch.Tensor,
    ):
        origin_loss = self.criterion(logits, labels)
        sorted_loss, idx = torch.sort(origin_loss, descending = True)
        keep_num = int(sorted_loss.shape[0] * self.ratio + 0.5)
        if keep_num > origin_loss.shape[0]:
            keep_num = origin_loss.shape[0]
        keep_idx = idx[:keep_num]
        loss = origin_loss[keep_idx]
        loss = loss.sum() / keep_num
        return loss



def gradient_penalty(model, optimizer, loss, eps):
    '''梯度惩罚'''
    loss.backward(retain_graph=True)
    for name, param in model.named_parameters():
        if 'embedding' in name:
            gp = torch.pow(param.grad, 2).sum()
            loss += gp * 0.5 * eps
    optimizer.zero_grad()
    loss.backward()
    return loss
        
