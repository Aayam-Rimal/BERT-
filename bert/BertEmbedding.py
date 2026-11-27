import torch
import torch.nn as nn

class BertEmbedding(nn.Module):

    def __init__(self, vocab_size,d_model,max_seq_len):
        super().__init__()

        self.input_embed= nn.Embedding(vocab_size,d_model)
        self.positional_embed= nn.Embedding(max_seq_len,d_model)
        self.segment_embed= nn.Embedding(2,d_model)

        self.layernorm= nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self,input_ids,token_type_ids=None):

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        seq_len= input_ids.size(1)

        position_id= torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        token_embeddings = self.input_embed(input_ids)
        positional_embeddings = self.positional_embed(position_id)
        segment_embeddings = self.segment_embed(token_type_ids)

        embeddings = (
            token_embeddings
            + positional_embeddings
            + segment_embeddings
        )

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


        


    


