import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.normalization_factor = torch.rsqrt(torch.tensor(d))

    def forward(self, q, k, v, mask = None):
        """
            q : Query vector [BatchSize x (_) x Dim]
            k : Key Vector   [BatchSize x Seq Len x Dim], k.t : [BatchSize x Dim x Seq Len]
            v : Value Vector [BatchSize x Seq Len x Dim]

            mask : A mask is required for each sample of shape as  [(_) x Seq Len] 
                   i.e.[BatchSize x (_) x Seq Len]

            unnormalized attention = Q(K.T)
            normalized attention  = Q(K.T) / (d**0.5)
        """
        unnormalized_attn = F.softmax(torch.matmul(q, torch.transpose(k, -2, -1)), dim = -1)
        normalized_attn = unnormalized_attn / self.normalization_factor 
        # normalized_attn : [BatchSize x (_) x Seq Len]
        if (mask is not None):
            # The mask is an batch-wise element multiplication
            normalized_attn = torch.mul(mask, normalized_attn)
        
        attention_value = torch.matmul(normalized_attn, v)
        #attn_value : [BatchSize x (_) x Dim]
        return attention_value, normalized_attn