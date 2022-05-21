from model_center.model import Bert, BertConfig
from .base_component import ClassificationHead
import torch
import bmtrain as bmt
from transformers.modeling_outputs import TokenClassifierOutput

class BertForTokenClassification(torch.nn.Module):
    def __init__(self, config: BertConfig, num_classes : int):
        super().__init__()
        self.bert : Bert = Bert(config)
        self.classifier = ClassificationHead(
            dim_model = config.dim_model, 
            num_classes = num_classes, 
            init_gain = config.proj_init_std,
            norm_eps = config.norm_eps, 
            norm_init_var = config.norm_init_var,
            dtype = config.dtype,
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        bmt.init_parameters(self.bert)
        bmt.init_parameters(self.classifier)

    def forward(self, 
        input_ids : torch.Tensor = None,
        attention_mask : torch.Tensor = None,
        labels : torch.Tensor  = None,
    ):
        hs = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.classifier(hs)
        loss = self.criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))

        return TokenClassifierOutput(
            loss = loss,
            logits = logits,
        )