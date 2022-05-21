import math
import numpy as np
from functools import lru_cache
import torch
import bmtrain as bmt
from model_center.model.basemodel import BaseModel
from model_center.model.config.config import Config
from transformers.modeling_outputs import TokenClassifierOutput, SequenceClassifierOutput
import pdb
import json
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
        

@lru_cache(maxsize=128)
def build_relative_position(query_size, key_size):
    q_ids = np.arange(0, query_size)
    k_ids = np.arange(0, key_size)
    rel_pos_ids = q_ids[:, None] - np.tile(k_ids, (q_ids.shape[0],1)) # q相对于k的距离
    rel_pos_ids = torch.tensor(rel_pos_ids, dtype=torch.long)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    return rel_pos_ids

class DisentangledSelfAttention(bmt.DistributedModule):

    def __init__(self, dim_in : int, 
                       dim_head : int,
                       num_heads : int, 
                       dim_out : int = None,
                       dtype = torch.half,
                       init_gain : int = 0.02,
                       bias = False,
                       dropout_p : float = 0,
                       max_seq_len : int = 512,
                       ):
        
        super().__init__()
        if dim_out is None:
            dim_out = dim_in

        self.all_head_size = dim_head * num_heads
        self.dim_in = dim_in
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim_out = dim_out

        self.project_q = Linear(
            dim_in = dim_in,
            dim_out = num_heads * dim_head,
            dtype = dtype,
            init_gain = init_gain,
            bias = bias,
        )
        self.project_k = Linear(
            dim_in = dim_in,
            dim_out = num_heads * dim_head,
            dtype = dtype,
            init_gain = init_gain,
            bias = bias,
        )
        self.project_v = Linear(
            dim_in = dim_in,
            dim_out = num_heads * dim_head,
            dtype = dtype,
            init_gain = init_gain,
            bias = bias,
        )
        self.pos_project_q = Linear(
            dim_in = dim_in,
            dim_out = num_heads * dim_head,
            dtype = dtype,
            init_gain = init_gain,
            bias = bias,
        )
        self.pos_project_k = Linear(
            dim_in = dim_in,
            dim_out = num_heads * dim_head,
            dtype = dtype,
            init_gain = init_gain,
            bias = bias,
        )
        self.attention_out = Linear(
            dim_in = num_heads * dim_head,
            dim_out = dim_out,
            dtype = dtype,
            init_gain = init_gain,
            bias = bias,
        )

        self.max_seq_len = max_seq_len
        if dropout_p:
            self.pos_dropout = torch.nn.Dropout(dropout_p)
            self.attention_dropout = torch.nn.Dropout(dropout_p)
        else:
            self.pos_dropout = None
            self.attention_dropout = None
        
        self.softmax = torch.nn.Softmax(dim=-1)

    def transpose_for_scores(self, x: torch.Tensor):
        batch_size, len_seq, all_head_size = x.shape
        x = x.view(batch_size, len_seq, self.num_heads, self.dim_head).permute(0, 2, 1, 3) # (batch, num_heads, len_seq, dim_head)
        x = x.contiguous().view(batch_size * self.num_heads, len_seq, self.dim_head)
        return x
    
    def forward(self,
        query_features: torch.Tensor,
        key_value_features: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        '''
        if self-attention, query == key_value

            query (:obj:`torch.Tensor` of shape ``(batch, len_q, dim_model)``): Indices of input sequence tokens. It will be embedded by model's internal embedding lookup matrix.
            key_value (:obj:`torch.Tensor` of shape ``(batch, len_k, dim_model)``): Length of input sequence before padding.  
            mask (:obj:`torch.Tensor` of shape ``(batch, len_q, len_k)``): Used to avoid performing attention on padding token indices.
        '''
        batch_size = query_features.shape[0]
        len_q = query_features.shape[-2]
        len_k = key_value_features.shape[-2]
        if len_q != len_k:
            raise NotImplementedError
        
        query_states = self.project_q(query_features)             # (batch, len_q, num_heads * dim_head)
        key_states = self.project_k(key_value_features)         # (batch, len_k, num_heads * dim_head)
        value_states = self.project_v(key_value_features)         # (batch, len_k, num_heads * dim_head)
        query_states = self.transpose_for_scores(query_states)    # (batch * num_heads, len_q, dim_head)
        key_states = self.transpose_for_scores(key_states)    # (batch * num_heads, len_k, dim_head)
        value_states = self.transpose_for_scores(value_states)    # (batch * num_heads, len_k, dim_head)

        # c2c, c2p, p2c
        scale = 1 / math.sqrt(self.dim_head * 3)
        context_attention_scores = torch.matmul(query_states, key_states.transpose(1, 2))
        position_attention_scores = self.position_attention(query_states, key_states, position_embeddings, batch_size)
        score = (context_attention_scores + position_attention_scores) * scale
        score = score.view(batch_size, self.num_heads, score.shape[-2], score.shape[-1])

        score = torch.masked_fill(
            score,
            attention_mask.view(batch_size, 1, len_q, len_k)==False,
            torch.scalar_tensor(-1e4, device=score.device, dtype=score.dtype)
        ) 
        score = self.softmax(score)

        # avoid nan in softmax
        score = torch.masked_fill(
            score,
            attention_mask.view(batch_size, 1, len_q, len_k)==False,
            torch.scalar_tensor(0, device=score.device, dtype=score.dtype)
        ).view(batch_size * self.num_heads, len_q, len_k)

        if self.attention_dropout is not None:
            score = self.attention_dropout(score)
        
        out_hidden_states = torch.matmul(score, value_states)
        out_hidden_states = out_hidden_states.view(batch_size, self.num_heads, len_q, self.dim_head).permute(0, 2, 1, 3).reshape(batch_size, len_q, self.num_heads * self.dim_head)
        out_hidden_states = self.attention_out(out_hidden_states)

        return out_hidden_states
    
    def position_attention(self,
        query_states: torch.Tensor, # (batch * num_heads, len_q, dim_head)
        key_states: torch.Tensor, # (batch * num_heads, len_k, dim_head)
        position_embeddings: torch.Tensor, # [2 * max_seq_len, dim_model]
        batch_size: int
    ):
        len_q = query_states.shape[-2]
        len_k = key_states.shape[-2]

        if self.pos_dropout is not None:
            position_embeddings = self.pos_dropout(position_embeddings)

        relative_pos = build_relative_position(len_q, len_k)
        relative_pos = relative_pos.long().to(query_states.device) # [len_q, len_k]

        position_embeddings = position_embeddings.unsqueeze(0)
        pos_query_states = self.pos_project_q(position_embeddings) # [1, 2 * max_seq_len, all_head_size]
        pos_key_states = self.pos_project_k(position_embeddings) # [1, 2 * max_seq_len, all_head_size]
        
        pos_query_states = self.transpose_for_scores(pos_query_states).repeat(batch_size, 1, 1)   # (batch * num_heads, 2 * max_seq_len, dim_head)
        pos_key_states = self.transpose_for_scores(pos_key_states).repeat(batch_size, 1, 1)   # (batch * num_heads, 2 * max_seq_len, dim_head)

        score = 0
        # c2p
        c2p_attention = torch.matmul(query_states, pos_key_states.transpose(1, 2)) # (batch * num_heads, len_q, 2 * max_seq_len)
        c2p_position_index = torch.clamp(relative_pos + self.max_seq_len, 0, 2 * self.max_seq_len - 1)
        c2p_position_index = c2p_position_index.unsqueeze(0).expand([batch_size * self.num_heads, len_q, len_k])
        c2p_scores = torch.gather(c2p_attention, dim = -1, index = c2p_position_index)
        score += c2p_scores
        # p2c
        p2c_attention = torch.matmul(pos_query_states, key_states.transpose(1, 2))  # (batch * num_heads, 2 * max_seq_len, len_k)
        p2c_position_index = torch.clamp(-relative_pos + self.max_seq_len, 0, 2 * self.max_seq_len - 1)
        p2c_position_index = p2c_position_index.unsqueeze(0).expand([batch_size * self.num_heads, len_q, len_k])
        p2c_scores = torch.gather(p2c_attention, dim = -2, index = p2c_position_index)
        score += p2c_scores

        return score

class DebertaAttention(torch.nn.Module):
    
    def __init__(self,
                 dim_model : int, 
                 num_heads : int, 
                 dim_head : int, 
                 dtype = torch.half,
                 norm_init_var : float = 1.0,
                 norm_eps : float = 1e-5, 
                 bias : bool = False,
                 init_gain : float = 0.02,
                 dropout_p : float = 0,
                 max_seq_len : int = 512
    ):
        super().__init__()

        self.attention = DisentangledSelfAttention(
            dim_in = dim_model, 
            num_heads = num_heads, 
            dim_head = dim_head,
            dim_out = dim_model, 
            dtype = dtype,
            init_gain = init_gain,
            dropout_p = dropout_p,
            max_seq_len = max_seq_len,
            bias = bias,
        )

        self.norm = RMSNorm(
            dim = dim_model,
            dtype = dtype,
            eps = norm_eps, 
            init_var = norm_init_var,
        )

        if dropout_p:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None
    
    def forward(self, 
        hidden_states : torch.Tensor,
        attention_mask : torch.Tensor,
        position_embeddings : torch.Tensor,
    ):
        attention_out = self.attention(
            query_features = hidden_states,
            key_value_features = hidden_states,
            position_embeddings = position_embeddings,
            attention_mask = attention_mask,
        )

        if self.dropout is not None:
            attention_out = self.dropout(attention_out)
        
        output_hidden_states = self.norm(hidden_states + attention_out)
        return output_hidden_states

class DebertaFeedForward(torch.nn.Module):
    def __init__(self,
                 dim_in : int,
                 dim_ff : int,
                 dim_out : int = None,
                 dtype = torch.half,
                 init_gain : float = 0.02,
                 bias : bool = False,
                 act_fn : str = 'gelu',
                 norm_init_var : float = 1.0,
                 norm_eps : float = 1e-5, 
                 dropout_p = 0,
    ) -> None:
        super().__init__()

        if dim_out is None:
            dim_out = dim_in

        self.ff = DenseGatedACT(
            dim_in = dim_in,
            dim_out = dim_ff,
            dtype = dtype,
            init_gain = init_gain,
            bias = bias,
            act_fn = act_fn,
        )
        self.w_out = Linear(
            dim_in = dim_ff,
            dim_out = dim_out,
            dtype = dtype,
            init_gain = init_gain,
            bias = bias,
        )
        self.norm = RMSNorm(
            dim = dim_out, 
            dtype = dtype,
            eps = norm_eps, 
            init_var = norm_init_var,
        )

        if dropout_p:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None
    
    def forward(self, x : torch.Tensor):
        """ 
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of feed-forward module.

        Return:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of feed-forward module.
        """
        short_cut = x
        x = self.ff(x)
        x = self.w_out(x)
        if self.dropout is not None:
            x = self.dropout(x)
        
        x = self.norm(x + short_cut)
        
        return x

class DebertaLayer(torch.nn.Module):

    def __init__(self, 
                 dim_model : int, 
                 dim_ff : int,
                 num_heads : int,
                 dim_head : int,
                 dtype = torch.half, 
                 init_gain : float = 0.02,
                 norm_init_var : float = 1.0,
                 norm_eps : float = 1e-5, 
                 bias : bool = False,
                 ffn_activate_fn : str = "gelu",
                 dropout_p : float = 0,
                 max_seq_len : int = 512,
                ):
        super().__init__()
        self.attention = DebertaAttention(
            dim_model = dim_model, 
            num_heads = num_heads, 
            dim_head = dim_head, 
            dtype = dtype,
            init_gain = init_gain,
            norm_eps = norm_eps, 
            norm_init_var = norm_init_var,
            bias = bias,
            dropout_p = dropout_p,
            max_seq_len = max_seq_len
        )
        self.ffn = DebertaFeedForward(
            dim_in = dim_model,
            dim_ff = dim_ff,
            dtype = dtype, 
            init_gain = init_gain,
            bias = bias,
            norm_eps = norm_eps, 
            norm_init_var = norm_init_var,
            act_fn = ffn_activate_fn,
            dropout_p = dropout_p,
        )

    def forward(self,
        hidden_states : torch.Tensor,
        attention_mask : torch.Tensor,
        position_embeddings: torch.Tensor,
    ):
        """
        Args:
            self_hidden_states (:obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``): Input of transformer block(self-attention block). It can be the raw embedding of a batch of sequences.
            self_attention_mask (:obj:`torch.Tensor` of shape ``(batch, seq_self, seq_self)``): Avoid invalid areas to participate in the calculation of self-attention.  

        Return:
            :obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``: The output of transformer block.

        """
    
        output_hidden_states = self.attention(
            hidden_states = hidden_states,
            attention_mask = attention_mask,
            position_embeddings = position_embeddings
        )

        output_hidden_states = self.ffn(output_hidden_states)
        
        return output_hidden_states


class DebertaEncoder(torch.nn.Module):

    def __init__(self, 
            num_layers : int,
            dim_model : int, 
            dim_ff : int,
            num_heads : int,
            dim_head : int,
            dtype : torch.dtype = torch.half,
            norm_init_var : float = 1.0,
            norm_eps : float = 1e-5, 
            bias : bool = False,
            ffn_activate_fn : str = "gated_gelu",
            dropout_p : float = 0,
            max_seq_len : int = 512,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.layers = bmt.TransformerBlockList([
            bmt.CheckpointBlock(
                DebertaLayer(
                    dim_model = dim_model, 
                    dim_ff = dim_ff,
                    num_heads = num_heads,
                    dim_head = dim_head,
                    dtype = dtype, 
                    norm_eps = norm_eps, 
                    norm_init_var = norm_init_var,
                    bias = bias,
                    ffn_activate_fn = ffn_activate_fn,
                    dropout_p = dropout_p,
                    max_seq_len = max_seq_len
                )
            )
            for _ in range(num_layers)
        ])
    
    def forward(self,
        hidden_states : torch.Tensor,
        attention_mask : torch.Tensor,
        position_embeddings: torch.Tensor,
    ):
        
        hidden_states = self.layers(hidden_states, attention_mask, position_embeddings)
        return hidden_states

class DebertaEmbedding(torch.nn.Module):

    def __init__(self,
        vocab_size : int,
        max_seq_len : int,
        embedding_size : int,
        type_size : int = 1,
        dtype = torch.half,
        init_mean : float = 0.,
        init_std : float = 0.02,
        dropout_p : float = 0,
        norm_init_var : float = 1.0,
        norm_eps : float = 1e-5, 
    ):
        super().__init__()
        self.input_embedding = Embedding(
            num_embeddings = vocab_size,
            embedding_dim = embedding_size,
            dtype = dtype,
            init_mean = init_mean,
            init_std = init_std,
        )
        self.position_embedding = Embedding(
            num_embeddings = max_seq_len * 2,
            embedding_dim = embedding_size,
            dtype = dtype,
            init_mean = init_mean,
            init_std = init_std,
        )
        self.token_type_embedding = Embedding(
            num_embeddings = type_size,
            embedding_dim = embedding_size,
            dtype = dtype,
            init_mean = init_mean,
            init_std = init_std,
        )
        self.norm = RMSNorm(
            dim = embedding_size,
            dtype = dtype,
            eps = norm_eps, 
            init_var = norm_init_var,
        )

        position_ids = torch.arange(0, max_seq_len * 2, dtype = torch.int32, device = self.position_embedding.weight.device)
        self.register_buffer('position_ids', position_ids)

        if dropout_p:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None

    @property
    def position_embeddings(self):
        return self.position_embedding(self.position_ids)
    
    def forward(self, 
        input_ids = None,
        token_type_ids = None,
    ):

        hidden_states = self.input_embedding(input_ids.to(torch.int32))
        token_type_embeds = self.token_type_embedding(token_type_ids.to(torch.int32))
        hidden_states = hidden_states + token_type_embeds

        if self.dropout:
            hidden_states = self.dropout(hidden_states)
        
        hidden_states = self.norm(hidden_states)

        return hidden_states

class DebertaConfig(Config):
    '''
        Notice:
            to ensure no error is raised, dim_model should be eq to dim_head * num_heads
            ffn_activate_fn = 'gelu' or 'relu'
            bias : if all the linear layer in model have bias
            ohem : (recommended) deprecated.
    '''
    def __init__(self,
                 vocab_size = 30,
                 type_size = 1,
                 max_seq_len = 512,
                 dim_model = 768,
                 num_heads = 12,
                 dim_head = 64,
                 dim_ff = 3072,
                 ffn_activate_fn = 'gelu',
                 num_layers = 12,
                 dropout_p = 0.05,
                 norm_init_var = 1.0,
                 norm_eps = 1e-12, 
                 bias = True,
                 half = True,
                 ohem = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.type_size = type_size
        self.max_seq_len = max_seq_len
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.ffn_activate_fn = ffn_activate_fn
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.init_mean = 0
        self.init_std = 0.02
        self.norm_init_var = norm_init_var
        self.norm_eps = norm_eps
        self.bias = bias

        self.ohem = ohem

        self.init_gain = (2 * self.num_layers) ** -0.5
        self.res_coefficient = (2 * self.num_layers) ** 0.5
        self.half = half

        if half: 
            self.dtype = torch.half
        else:
            self.dtype = torch.float
    
    def save_pretrained(self, save_path):
        to_dict = dict(
            vocab_size = self.vocab_size,
            type_size = self.type_size,
            max_seq_len = self.max_seq_len,
            dim_model = self.dim_model,
            num_heads = self.num_heads,
            dim_head = self.dim_head,
            dim_ff = self.dim_ff,
            ffn_activate_fn = self.ffn_activate_fn,
            num_layers = self.num_layers,
            dropout_p = self.dropout_p,
            norm_init_var = self.norm_init_var,
            norm_eps = self.norm_eps, 
            bias = self.bias,
            half = self.half,
            ohem = self.ohem,
        )
        json_string = json.dumps(to_dict, indent=2, sort_keys=True) + "\n"
        with open(save_path, "w", encoding="utf-8") as writer:
            writer.write(json_string)

class Deberta(BaseModel):

    _CONFIG_TYPE = DebertaConfig

    def __init__(self, config: DebertaConfig):
        super().__init__()

        self.embedding_layer = DebertaEmbedding(
            vocab_size = config.vocab_size,
            max_seq_len = config.max_seq_len,
            embedding_size = config.dim_model,
            type_size = config.type_size,
            dtype = config.dtype,
            dropout_p = config.dropout_p,
            norm_init_var = config.norm_init_var,
            norm_eps = config.norm_eps, 
            init_mean = config.init_mean,
            init_std = config.init_std,
        )
        self.encoder = DebertaEncoder(
            num_layers = config.num_layers,
            dim_model = config.dim_model, 
            dim_ff = config.dim_ff,
            num_heads = config.num_heads,
            dim_head = config.dim_head,
            dtype = config.dtype, 
            norm_eps = config.norm_eps, 
            bias = config.bias,
            ffn_activate_fn = config.ffn_activate_fn, 
            norm_init_var = config.norm_init_var,
            dropout_p = config.dropout_p,
            max_seq_len = config.max_seq_len,
        )
    
    def forward(self,
        input_ids = None,
        inputs_embeds = None,
        attention_mask = None,
        token_type_ids = None,
    ):
        assert input_ids is not None or inputs_embeds is not None

        if input_ids is not None:
            batch = input_ids.size(0)
            seq_length = input_ids.size(1)
            device = input_ids.device
        else:
            batch = inputs_embeds.size(0)
            seq_length = inputs_embeds.size(1)
            device = inputs_embeds.device
        
        with torch.no_grad():
            if attention_mask is not None:
                attention_mask = attention_mask.to(torch.bool)
            else:
                attention_mask = torch.ones(seq_length, device=device)[None, :].repeat(batch, 1).to(torch.bool)
            attention_mask = attention_mask.view(batch, seq_length, 1) & attention_mask.view(batch, 1, seq_length)

            if token_type_ids is None:
                token_type_ids = torch.zeros(seq_length, dtype=torch.int32, device=device)[None, :].repeat(batch, 1)
        
        if inputs_embeds is None:
            hidden_states = self.embedding_layer(input_ids.to(torch.int32), token_type_ids)
        
        position_embeddings = self.embedding_layer.position_embeddings

        last_hidden_states = self.encoder(hidden_states, attention_mask, position_embeddings)

        return last_hidden_states

class DebertaForTokenClassification(BaseModel):

    _CONFIG_TYPE = DebertaConfig

    def __init__(self, config: DebertaConfig, num_classes: int):

        super().__init__()
        self.model = Deberta(config)
        self.classifier = ClassificationHead(
            dim_model = config.dim_model,
            num_classes = num_classes,
            norm_eps = config.norm_eps,
            dtype = config.dtype,
            norm_init_var = config.norm_init_var,
            init_gain = config.init_gain,
        )
        if config.ohem is not None:
            self.criterion = OHEMLoss(ratio = config.ohem)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()
        bmt.init_parameters(self.model)
        bmt.init_parameters(self.classifier)
    
    def forward(self,
        input_ids : torch.Tensor = None,
        attention_mask : torch.Tensor = None,
        labels : torch.Tensor  = None,
    ):
        last_hidden_states = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits: torch.Tensor = self.classifier(last_hidden_states)
        loss = self.criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))

        return TokenClassifierOutput(
            loss = loss,
            logits = logits,
        )

class DebertaForSequenceClassification(BaseModel):

    _CONFIG_TYPE = DebertaConfig

    def __init__(self, config: DebertaConfig, num_classes: int, mode = 'mean'):

        super().__init__()
        self.model = Deberta(config)
        self.classifier = Pooler(
            dim_model = config.dim_model,
            init_gain = config.init_gain,
            num_classes = num_classes,
            norm_eps = config.norm_eps,
            norm_init_var = config.norm_init_var,
            dtype = config.dtype,
            mode = mode,
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        bmt.init_parameters(self.model)
        bmt.init_parameters(self.classifier)
    
    def forward(self,
        input_ids : torch.Tensor = None,
        attention_mask : torch.Tensor = None,
        labels : torch.Tensor  = None,
    ):
        last_hidden_states = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(last_hidden_states, attention_mask)
        loss = self.criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))

        return SequenceClassifierOutput(
            loss = loss,
            logits = logits,
        )
