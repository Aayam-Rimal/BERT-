import torch.nn as nn
import torch
from BertEmbedding import BertEmbedding

class encoder_stack(nn.Module):

    def __init__(self, layers):
        super().__init__()

        self.layers= nn.ModuleList(layers)

    
    def forward(self,x):

        for layer in self.layers:
            x= layer(x)

        return x 


class Transformer(nn.Module):
    
    def __init__(self, encoder, embed_dim,vocab_size):
        super().__init__()

        self.dropout= nn.Dropout(p=0.1)

        self.embedding= BertEmbedding(vocab_size,embed_dim,max_seq_len=512)
        self.dense_layer= nn.Linear(embed_dim,embed_dim)

        self.fc_out= nn.Linear(embed_dim, vocab_size)
        self.activation= nn.GELU()
        self.layernorm= nn.LayerNorm(embed_dim)

        self.encoder= encoder

    def forward(self,src):

        src= self.embedding(src)
        src= self.dropout(src)

        enc_out= self.encoder(src)

        dense_out= self.dense_layer(enc_out)
        activated_out= self.activation(dense_out)
        ln= self.layernorm(activated_out)
        final_out= self.fc_out(ln)

        return final_out


        
    