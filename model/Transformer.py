import numpy as np

import torch
import torch.nn as nn

from model.Attention import MultiHeadAttention

class Encoder(nn.Module):
    def __init__(self, dim, num_heads, query_key_dim = 64):
        super().__init__()
        self.multi_head_attn = MultiHeadAttention(dim, num_heads, query_key_dim = query_key_dim)
        self.layer_norm_inter = nn.LayerNorm(dim)
        self.fc = nn.Sequential(*[
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        ])
        self.layer_norm_final = nn.LayerNorm(dim)

    def forward(self, x, mask=False):
        """
        x : vector [batch_size x seq_len x dim]
        mask : [batch_size x (_) x seq_len] or [(_) x seq_len]
        (_) is some arbitary dimension depending upon what is the dimension of the query. 
        In the encoder case here, the (_) will be of seq_len.
        """

        attn, attn_weights = self.multi_head_attn(x, x, x, mask = mask)
    
        x = x + attn # Residual connection
        x = self.layer_norm_inter(x) # Layer normalization
        fc_result = self.fc(x) # Fully-connected
        x = x+fc_result # Residual connection
        x = self.layer_norm_final(x) # Layer normalization
        return x, attn_weights

class Decoder(nn.Module):
    def __init__(self, dim, num_heads, query_key_dim = 64):
        super().__init__()
        self.multi_head_attn_1 = MultiHeadAttention(dim, num_heads, query_key_dim = query_key_dim)
        self.layer_norm_1 = nn.LayerNorm(dim)
        self.multi_head_attn_2 = MultiHeadAttention(dim, num_heads, query_key_dim = query_key_dim)
        self.layer_norm_2 = nn.LayerNorm(dim)
        self.fc = nn.Sequential(*[
            nn.Linear(dim, dim), 
            nn.ReLU(), 
            nn.Linear(dim, dim)])
        self.layer_norm_final = nn.LayerNorm(dim)

    def forward(self, x, x_enc, mask_enc = None, mask_dec = None):
        """
        x : shape of [Batch_size x (_) x dim] 
        x_enc : shape of [Batch_size x seq_len x dim] 

        """
        attn, attn_weights_dec = self.multi_head_attn_1(x, x, x, mask = mask_dec)
        x = self.layer_norm_1(x + attn)
        attn, attn_weights_enc = self.multi_head_attn_2(q = x, k = x_enc, v = x_enc, mask = mask_enc)
        x = self.layer_norm_2(x + attn)
        x = self.fc(x)
        x = self.layer_norm_final(x)
        return x, attn_weights_enc, attn_weights_dec

class Transformer(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, 
                 dim, query_key_dim, num_heads, N=4, 
                 device='cpu',
                ):
        """
        input_vocab_size : The number of words in the input-dictionary. 
        output_vocab_size : The number of words in the output-dictionary. 
        dim : The dimension used for all the embeddings linear layers. 
        query_key_dim: dimension of the query and key vectors in attention modules
        num_heads : number of heads in the multi-head attention
        N : Number of encoder-decoder stack
        device : torch.device such 'cpu', 'cuda', 'mps'. For some reason MPS is reallyyyy slowww !!
        """
        super().__init__()
        self.input_dim = dim
        self.input_embeddings = nn.Embedding(input_vocab_size, dim)
        self.output_embeddings = nn.Embedding(output_vocab_size, dim)

        self.encoders = nn.ModuleList([Encoder(dim, num_heads, query_key_dim = query_key_dim) for _ in range(N)])
        self.decoders = nn.ModuleList([Decoder(dim, num_heads, query_key_dim = query_key_dim) for _ in range(N)])

        self.fc_connect = nn.Sequential(*[
            nn.Linear(dim, dim), 
            nn.ReLU(), 
            nn.Linear(dim, output_vocab_size)
        ])        

        self.device = device
        self.positional_embedding = self.get_positional_embedding().to(device)

    def get_positional_embedding(self, sequence_length=10):
        d = self.input_dim
        i = np.arange(0,d//2)
        positional_embedding = []
        for pos in range(sequence_length):
            positional_embedding.append(
                      np.concatenate([np.sin(pos/(10000**((2*i)/d))), 
                                      np.cos(pos/(10000**((2*i+1)/d)))]))
        positional_embedding = np.array(positional_embedding)
        positional_embedding = torch.tensor(positional_embedding, dtype=torch.float32)
        return positional_embedding # Shape : [seq_len x embedding_dim]
    
    def forward(self, x, outputs, mask_enc=None):
        """
        x : [Batch_size x Seq Len x 1] Input is the index of the words
        outputs : [Batch_size x Seq Len x 1] Output that will be shifted right during the teacher forcing
        mask_enc: Additional mask for the encoder used for making PAD in the attention
        """
        x = self.input_embeddings(x)
        outputs = self.output_embeddings(outputs)
        
        positional_embedding = self.positional_embedding.unsqueeze(0)
        positional_embedding = positional_embedding.expand(x.size(0), x.size(1), x.size(2))
        mask_enc = mask_enc.unsqueeze(1).expand(x.size(0),x.size(1),x.size(1))
        
        # Adding positional embedding to the input
        x = x+positional_embedding
 
        encoder_output_list = []
        for idx, encoder in enumerate(self.encoders):
            x, _ = encoder(x, mask = mask_enc)
            encoder_output_list.append(x)
            
        # We need to add the last sequence of the encoder to initial sequence
        h = x[:, -1].unsqueeze(1)  # shape [batch_size x 1 x output_dim]
        outputs = outputs[:, : -1] # shape [batch_size x (_)-1 x output_dim]
        outputs = torch.cat([h, outputs], dim = -2) # shape [batch_size x (_) x output_dim]

        positional_embedding = self.positional_embedding.unsqueeze(0).expand(outputs.size(0), 
                                                                             outputs.size(1), 
                                                                             outputs.size(2))

        # Adding the positional embedding
        outputs = outputs + positional_embedding

        # Maks the future values so that the are not attended in the attention module
        mask_dec = torch.tril(torch.ones(size=(outputs.size(1),x.size(1)), device = self.device))

        for idx, decoder in enumerate(self.decoders):
            outputs, weights_enc, weights_dec = decoder(outputs, 
                                                        encoder_output_list[idx], 
                                                        mask_dec=mask_dec, mask_enc = mask_enc)

        return self.fc_connect(outputs), weights_enc, weights_dec