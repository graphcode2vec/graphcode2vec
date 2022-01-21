import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    r""" position encoding.
    we use the non-parametric cosine based values from the original transformer paper.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        directly add the position embedding to the input embedding (of a sentence)
        x: [sequence length, batch size, embed dim]"""
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        """ 
        ntoken: number of tokens in vcab
        ninp: embedding layer dim
        nhead: num of self attn heads
        nhid: number of hidden dimension in self attention
        """
        super(TransformerModel, self).__init__()
        self.model_type = "Transformer"
        self.src_mask = None  # since we only use encoder, this is always None
        self.pos_encoder = PositionalEncoding(ninp, dropout, max_len=100)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
    
    def init_embeddings(self, embeddings):
        """
        Initialized embedding layer with pre-computed embeddings.

        :param embeddings: pre-computed embeddings
        """
        self.encoder.weight = nn.Parameter(embeddings)
    
    def fine_tune_embeddings(self, fine_tune=False):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: allow?
        """
        for p in self.encoder.parameters():
            p.requires_grad = fine_tune

    def forward(self, src, src_length):
        # src: [max_seq_len x batch_size] batch of sentences
        # following transformer paper, scale the input embedding first
        src = torch.transpose(src, 0, 1)
        src = self.encoder(src) * math.sqrt(self.ninp)
        # add position embedding
        src = self.pos_encoder(src)

        # pass through self attention layers and get outputs
        # [max_seq_len x batch_size x hidden dim]
        output = self.transformer_encoder(src)
        output = torch.transpose(output, 0, 1)
        output = torch.sum( output, 1)
        # TODO: note sure the dimensionality of the SAGE model wants; to double check
        # use another linear layer to adjust the dim if necessary
        return output
