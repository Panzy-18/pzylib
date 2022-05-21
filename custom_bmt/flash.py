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


class FLASHLayer(bmt.DistributedModule):
    def __init__(self, 
                 max_seq_len : int,
                 chunk_len : int,
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
        assert max_seq_len % chunk_len == 0

    
    def forward(self):
        pass