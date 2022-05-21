import math

import torch
from torch import nn
import bmtrain as bmt
from .base_component import Linear, RMSNorm, ClassificationHead, Embedding
import torch.nn.functional as F
from model_center.model.basemodel import BaseModel
from model_center.model.config.config import Config
from transformers.modeling_outputs import TokenClassifierOutput
import pdb

class ScaleOffset(bmt.DistributedModule):
    def __init__(self,
                 head_size : int,
                 dtype = torch.half,
                 init_mean : float = 0.0,
                 init_std : float = 0.02) -> None:
        super().__init__()
        self.dim = head_size
        self.gamma = bmt.DistributedParameter(
            torch.empty((head_size), dtype = dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.trunc_normal_, mean=init_mean, std=init_std*1.137, a=-2*init_std, b=2*init_std)
        )
        self.beta = bmt.DistributedParameter(
            torch.empty((head_size), dtype = dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.trunc_normal_, mean=init_mean, std=init_std*1.137, a=-2*init_std, b=2*init_std)
        )

    def forward(self, x : torch.Tensor):
        x = x.mul(self.gamma.view(1, 1, -1)) + self.beta.view(1, 1, -1)
        return x

class GatedAttention(bmt.DistributedModule):
    def __init__(self,
                 dim_model : int,
                 head_size : int,
                 dtype = torch.half,
                 init_mean : float = 0.0,
                 init_std : float = 0.02,
                 init_gain : float = 0.02,
                 dropout_p : int = 0,
                 bias : bool = False,
                 rope = None) -> None:
        super().__init__()
        self.head_size = head_size
        self.project_head = Linear(
            dim_in = dim_model,
            dim_out = head_size,
            dtype = dtype,
            init_gain = init_gain,
            bias = bias,
        )
        self.project_q = ScaleOffset(
            head_size = head_size,
            dtype = dtype,
            init_mean = init_mean,
            init_std = init_std,
        )
        self.project_k = ScaleOffset(
            head_size = head_size,
            dtype = dtype,
            init_mean = init_mean,
            init_std = init_std,
        )
        self.rope = rope

        if dropout_p:
            self.dropout = torch.nn.Dropout(p=dropout_p)
        else:
            self.dropout = None

    
    def forward(self, 
                x : torch.Tensor, 
                attention_mask : torch.Tensor,
                ): 
        
        seq_len = x.shape[1]
        z = self.project_head(x)

        query_states, key_states = self.project_q(z), self.project_k(z)
        query_states = self.rope(query_states)
        key_states = self.rope(key_states)
        score = torch.matmul(query_states, key_states.transpose(1, 2))
        score = torch.pow(F.relu(score), 2) # [batch_size, seq_len, seq_len]
        score /= (seq_len * self.head_size)

        score = torch.masked_fill(
            score,
            attention_mask == False,
            torch.scalar_tensor(0, device=score.device, dtype=score.dtype) # 没有softmax 直接置0
        )

        if self.dropout:
            score = self.dropout(score)

        return score

class GatedAttentionUnit(bmt.DistributedModule):

    def __init__(self, 
                 dim_model : int, 
                 head_size : int,
                 dim_ff : int,
                 dtype = torch.half, 
                 norm_init_var : float = 1.0,
                 norm_eps : float = 1e-5, 
                 init_mean : float = 0.0, 
                 init_std : float = 0.02,
                 init_gain : float = 0.02,
                 bias : bool = False,
                 dropout_p : float = 0,
                 res_coefficient : float = 1,
                 rope = None,
                ):
        super().__init__()
        self.project_u = Linear(
            dim_in = dim_model,
            dim_out = dim_ff,
            dtype = dtype,
            init_gain = init_gain,
            bias = bias,
        )
        self.project_v = Linear(
            dim_in = dim_model,
            dim_out = dim_ff,
            dtype = dtype,
            init_gain = init_gain,
            bias = bias,
        )
        self.attention = GatedAttention(
            dim_model = dim_model,
            head_size = head_size,
            dtype = dtype,
            init_mean = init_mean,
            init_std = init_std,
            init_gain = init_gain,
            dropout_p = dropout_p,
            bias = bias,
            rope = rope,
        )
        self.project_back = Linear(
            dim_in = dim_ff,
            dim_out = dim_model,
            dtype = dtype,
            init_gain = init_gain,
            bias = bias,
        )
        self.norm = RMSNorm(
            dim = dim_model,
            dtype = dtype,
            eps = norm_eps, 
            init_var = norm_init_var,
        )
        self.res_coefficient = res_coefficient
    
    def forward(self, 
                hidden_states : torch.Tensor, 
                attention_mask : torch.Tensor,
                ):
        # 用post-norm
        shortcut = hidden_states
        u, v = self.project_u(hidden_states), self.project_v(hidden_states)

        hidden_states = torch.mul(u, torch.matmul(self.attention(hidden_states, attention_mask), v))
        hidden_states = self.project_back(hidden_states)
        output_hidden_states = self.norm(hidden_states + self.res_coefficient * shortcut)
        return output_hidden_states
    
# 旋转位置编码
class RoPE(bmt.DistributedModule):

    def __init__(self, 
                 head_size : int,
                 max_seq_len : int = 512,
                 dtype = torch.half,
                 device = 'cuda',
                 ) -> None:
        super().__init__()
        self.head_size = head_size
        self.max_seq_len = max_seq_len

        position = torch.arange(0, max_seq_len).to(torch.float32).unsqueeze(-1) # [max_seq_len, 1]
        half_size = head_size // 2
        freq_seq = torch.arange(0, half_size).to(torch.float32) / half_size
        inv_freq = 10000 ** -freq_seq
        sinusoid = torch.matmul(position, inv_freq.unsqueeze(0))
        self._sin = torch.sin(sinusoid).repeat_interleave(2, dim=-1).to(device).to(dtype) # [max_seq_len, head_size]
        self._cos = torch.cos(sinusoid).repeat_interleave(2, dim=-1).to(device).to(dtype) # [cos(p0), cos(p0), cos(p1), cos(p1) ...]
        
    def forward(self, qk_states: torch.Tensor): # 只有一个head
        assert qk_states.shape[-1] == self.head_size
        seq_len = qk_states.shape[1] 
        h2 = torch.stack((qk_states[..., 1::2] * -1, qk_states[..., ::2])).reshape(qk_states.shape) # [-q1, q0, -q3, q2 ...]
        qk_states_with_pe = torch.mul(qk_states, self._cos[:seq_len, :].unsqueeze(0)) + torch.mul(h2, self._sin[:seq_len, :].unsqueeze(0))
        return qk_states_with_pe

class GAUEncoder(nn.Module):

    def __init__(self, 
                 num_layers : int,
                 dim_model : int, 
                 dim_ff : int,
                 head_size : int,
                 dtype : torch.dtype = torch.half,
                 norm_init_var : float = 1.0,
                 norm_eps : float = 1e-5, 
                 init_mean : float = 0.0, 
                 init_std : float = 0.02,
                 init_gain : float = 0.02,
                 bias : bool = False,
                 dropout_p : float = 0,
                 max_seq_len : int = 512,
                 res_coefficient : float = 1,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers

        self.rope = RoPE(
            head_size = head_size,
            max_seq_len = max_seq_len,
            dtype = dtype,
        )

        self.layers = bmt.TransformerBlockList([
            bmt.CheckpointBlock(
                GatedAttentionUnit(
                    dim_model = dim_model,
                    head_size = head_size,
                    dim_ff = dim_ff,
                    dtype = dtype,
                    norm_init_var = norm_init_var,
                    norm_eps = norm_eps,
                    init_mean = init_mean,
                    init_std = init_std,
                    init_gain = init_gain,
                    bias = bias,
                    dropout_p = dropout_p,
                    rope = self.rope,
                    res_coefficient = res_coefficient,
                )
            )
            for _ in range(num_layers)
        ])

    def forward(self,
               hidden_states : torch.Tensor,
               attention_mask : torch.Tensor,
               ):

        hidden_states = self.layers(hidden_states, attention_mask)

        return hidden_states


class GAUConfig(Config):

    def __init__(self, 
                 vocab_size = 30,
                 type_size = 1,
                 max_seq_len = 512,
                 dim_model = 768,
                 head_size = 128,
                 dim_ff = 1536,
                 num_layers = 30,
                 dropout_p = 0.05,
                 norm_init_var = 1.0,
                 norm_eps = 1e-12, 
                 bias = True,
                 half = True,
                 ):

        super().__init__()

        self.vocab_size = vocab_size
        self.type_size = type_size
        self.max_seq_len = max_seq_len
        self.dim_model = dim_model
        self.head_size = head_size
        self.dim_ff = dim_ff
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.init_mean = 0
        self.init_std = 0.01
        self.norm_init_var = norm_init_var
        self.norm_eps = norm_eps
        self.bias = bias

        self.init_gain = (2 * self.num_layers) ** -0.5
        self.res_coefficient = (2 * self.num_layers) ** 0.5

        if half: 
            self.dtype = torch.half
        else:
            self.dtype = torch.float

class GAUTransformer(BaseModel):

    _CONFIG_TYPE = GAUConfig

    def __init__(self, config: GAUConfig):
        super().__init__()

        self.word_embedding = Embedding(
            num_embeddings = config.vocab_size,
            embedding_dim = config.dim_model,
            dtype = config.dtype,
            init_mean = config.init_mean,
            init_std = config.init_std / math.sqrt(2),
        )
        self.type_embedding = Embedding(
            num_embeddings = config.type_size,
            embedding_dim = config.dim_model,
            dtype = config.dtype,
            init_mean = config.init_mean,
            init_std = config.init_std / math.sqrt(2),
        )
        self.norm = RMSNorm(
            dim = config.dim_model,
            dtype = config.dtype,
            init_var = config.norm_init_var,
            eps = config.norm_eps,
        )

        self.encoder = GAUEncoder(
            num_layers = config.num_layers,
            dim_model = config.dim_model,
            dim_ff = config.dim_ff,
            head_size = config.head_size,
            dtype = config.dtype,
            norm_init_var = config.norm_init_var,
            norm_eps = config.norm_eps,
            init_mean = config.init_mean,
            init_std = config.init_std,
            init_gain = config.init_gain,
            bias = config.bias,
            dropout_p = config.dropout_p,
            max_seq_len = config.max_seq_len,
            res_coefficient = config.res_coefficient,
        )

    def forward(self,
                input_ids = None,
                attention_mask = None,
                token_type_ids = None,
                return_dict = True,
                ):
        assert input_ids is not None

        if input_ids is not None:
            batch = input_ids.size(0)
            seq_length = input_ids.size(1)
            device = input_ids.device
        
        with torch.no_grad():
            if attention_mask is not None:
                attention_mask = attention_mask.to(torch.bool)
            else:
                attention_mask = torch.ones(seq_length, device=device)[None, :].repeat(batch, 1).to(torch.bool)
            attention_mask = attention_mask.view(batch, seq_length, 1) & attention_mask.view(batch, 1, seq_length)

            if token_type_ids is None:
                token_type_ids = torch.zeros(seq_length, dtype=torch.int32, device=device)[None, :].repeat(batch, 1)
        
        word_embeds = self.word_embedding(input_ids)
        type_embeds = self.type_embedding(token_type_ids)

        input_embeds = self.norm(word_embeds + type_embeds)
        last_hidden_states = self.encoder(input_embeds, attention_mask)

        return last_hidden_states

class GAUForTokenClassification(BaseModel):

    _CONFIG_TYPE = GAUConfig

    def __init__(self, config: GAUConfig, num_classes: int):

        super().__init__()
        self.model = GAUTransformer(config)
        self.classifier = ClassificationHead(
            dim_model = config.dim_model, 
            num_classes = num_classes, 
            init_gain = config.init_gain,
            norm_eps = config.norm_eps, 
            norm_init_var = config.norm_init_var,
            dtype = config.dtype,
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
        logits: torch.Tensor = self.classifier(last_hidden_states)
        loss = self.criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))

        return TokenClassifierOutput(
            loss = loss,
            logits = logits,
        )





