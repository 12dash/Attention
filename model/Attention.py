import torch
import torch.nn as nn

from model.ScaledDotProductAttention import ScaledDotProductAttention

class Attention(nn.Module):
    def __init__(self, input_dim, value_dim, query_key_dim = 64):
        super().__init__()

        self.query = nn.Linear(input_dim, query_key_dim)
        self.key   = nn.Linear(input_dim, query_key_dim)
        self.value = nn.Linear(input_dim, value_dim)

        self.scaled_attention = ScaledDotProductAttention(value_dim)

    def forward(self, q, k, v, mask = None):
        """
        q : query of shape [batch_size x (_) x input_dim] 
        k : key of shape [batch_size x (seq_len) x input_dim]

        v : value of shape [batch_size x (seq_len) x embedding_dim]

        mask : mask of shape [batch_size x (_) x (seq_len)] or [(_) x (seq_len)]
        """

        # linear transformation of the query, key and value.
        # Most of the time all these are the same vector.
        q = self.query(q)
        k = self.key(k) 
        v = self.value(v)

        attention, attention_weights = self.scaled_attention(q, k, v, mask = mask)
        return attention, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads = 4, query_key_dim = 64):
        super().__init__()
        assert (dim % num_heads == 0), f"embedding dim {dim} must be divisible by num heads {num_heads}"
        
        self.input_dim = dim
        self.num_heads = num_heads
        self.value_dim = dim // num_heads
        self.multihead_attention = nn.ModuleList([(Attention(input_dim=dim, 
                                                             value_dim=self.value_dim, 
                                                             query_key_dim = query_key_dim)) for _ in range(num_heads)])

        self.output_projection = nn.Linear(dim, dim)
    
    def forward(self, q, k, v, mask=None):
        """
        q : query will be of shape [batch_size x (_) x dim]
        k : key will be of shape [batch_size x (seq_len) x dim]
        val : value will be of shape [batch_size x (seq_len) x dim]

        mask : mask of shape [batch_size x (_) x (seq_len)] or [(_) x (seq_len)]
        """
        attention_list, weights_list = [], []

        # This is not the most efficient way of calculating the multi-head since we are making
        # things sequential that could be parallelized
        for attention_head in self.multihead_attention:
            attention, attention_weights = attention_head(q, k, v, mask)
            attention_list.append(attention)
            weights_list.append(attention_weights.unsqueeze(dim = 1))
        
        # Concatenating along the last dim across the multihead attention    
        # Potentially, we can increase the dimension since we take the projection        
        attention = torch.cat(attention_list, dim = -1)

        # Weights from all the multihead-attention are concatenated.  
        weights_list = torch.cat(weights_list, dim = 1)

        # Lastly tranforming this to output dimension
        attention = self.output_projection(attention)
        return attention, weights_list
