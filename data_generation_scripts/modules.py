"""
This file implements neural network building blocks in pytorch. Additionally,
it monkey patches some layers in pytorch so that they are better compatible
with the C++ codebase implemented in this repository.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


# Monkey Patch the Linear (/ Dense) Layer
def save(self, path):
    utils.write_tensor_to_file(self.weight.T, path + "_weights")
    if self.bias is not None:
        utils.write_tensor_to_file(self.bias, path + "_bias")


nn.Linear.save = save
nn.Dense = nn.Linear


# Monkey Patch the Embedding Layer
def save(self, path):
    utils.write_tensor_to_file(self.weight, path + "_weights")


nn.Embedding.save = save


# Monkey Patch the LayerNorm Layer
def save(self, path):
    utils.write_tensor_to_file(self.weight, path + "_gamma")
    utils.write_tensor_to_file(self.bias, path + "_beta")


nn.LayerNorm.save = save


def init(
    self, normalized_shape, eps=1e-100, elementwise_affine=True, device=None, dtype=None
):
    self.init(
        normalized_shape=normalized_shape,
        eps=eps,
        elementwise_affine=elementwise_affine,
        device=device,
        dtype=dtype,
    )


nn.LayerNorm.init = nn.LayerNorm.__init__
nn.LayerNorm.__init__ = init


#########################
# Define different layers


class PositionwiseFeedForward(nn.Module):
    """
    Source:
    https://nlp.seas.harvard.edu/2018/04/03/attention.html#position-wise-feed-forward-networks
    MIT License - Copyright (c) 2018 Alexander Rush
    (see also: https://github.com/harvardnlp/annotated-transformer/)

    This implementation is based on the original code, but with slight
    modifications.
    """

    def __init__(self, d_model, d_ff, dropout=0.1, dtype=float):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff, dtype=dtype)
        self.w_2 = nn.Linear(d_ff, d_model, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

    def save(self, path):
        self.w_1.save(path + "_d1")
        self.w_2.save(path + "_d2")


class PositionalEncoding(nn.Module):
    """
    Source:
    https://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding
    MIT License - Copyright (c) 2018 Alexander Rush
    (see also: https://github.com/harvardnlp/annotated-transformer/)

    This implementation is based on the original code, but with slight
    modifications.
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
        pe[:, 1::2] = torch.cos(position * div_term)[:, : len(pe[0, 1::2])]
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class MultiheadAttention(nn.Module):
    """
    This is a PyTorch implementation of the MultiheadAttention layer implemented
    in https://github.com/AmritanshuV/Numpy-Transformer/blob/42dd602e9a651d77ddb041790b1d6d513d8b7cf7/transformer/layers/combined/self_attention.py
    """

    def __init__(self, d_model=512, heads_num=8, dropout=0.1, dtype=float):
        super(MultiheadAttention, self).__init__()

        self.d_model = d_model
        self.heads_num = heads_num

        self.d_k = self.d_model // heads_num
        self.scale = self.d_k**0.5

        self.K_linear = nn.Linear(
            self.d_model, self.d_k * heads_num, bias=False, dtype=dtype
        )
        self.Q_linear = nn.Linear(
            self.d_model, self.d_k * heads_num, bias=False, dtype=dtype
        )
        self.V_linear = nn.Linear(
            self.d_model, self.d_k * heads_num, bias=False, dtype=dtype
        )
        self.O_linear = nn.Linear(
            self.d_k * heads_num, self.d_model, bias=True, dtype=dtype
        )

        self.activation = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        batch_size = x.size(0)
        return x.reshape(batch_size, -1, self.heads_num, self.d_k).permute(0, 2, 1, 3)

    def group_heads(self, x):
        batch_size = x.size(0)
        return x.permute(0, 2, 1, 3).reshape(batch_size, -1, self.heads_num * self.d_k)

    def forward(self, query, key, value, mask=None):
        K = self.K_linear(key)
        Q = self.Q_linear(query)
        V = self.V_linear(value)

        self.K = self.split_heads(K)
        self.Q = self.split_heads(Q)
        self.V = self.split_heads(V)

        energy = torch.matmul(self.Q, self.K.permute(0, 1, 3, 2)) / self.scale

        self.mask = mask
        if mask is not None:
            self.mask = self.mask[:, None, ...]
            energy = energy.masked_fill(self.mask == 0, -1e100)

        attention = self.activation(energy)
        self.dropout_attention = self.dropout(attention)

        output = torch.matmul(self.dropout_attention, self.V)
        concat_output = self.group_heads(output)

        O = self.O_linear(concat_output)
        return O, attention

    def save(self, path):
        self.Q_linear.save(path + "_q")
        self.K_linear.save(path + "_k")
        self.V_linear.save(path + "_v")
        self.O_linear.save(path + "_o")


class EncoderLayer(nn.Module):
    """
    This is a PyTorch implementation of the EncoderLayer layer implemented in
    https://github.com/AmritanshuV/Numpy-Transformer/blob/42dd602e9a651d77ddb041790b1d6d513d8b7cf7/transformer/layers/combined/encoder_layer.py
    """

    def __init__(self, d_model, heads_num, d_ff, dropout, data_type=float):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(
            d_model, elementwise_affine=True, dtype=data_type
        )
        self.ff_layer_norm = nn.LayerNorm(
            d_model, elementwise_affine=True, dtype=data_type
        )
        self.self_attention = MultiheadAttention(d_model, heads_num, dropout, data_type)
        self.position_wise_feed_forward = PositionwiseFeedForward(
            d_model, d_ff, dropout
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src, src_mask):
        _src, _ = self.self_attention(src, src, src, src_mask)
        src = self.self_attention_norm(src + self.dropout(_src))

        _src = self.position_wise_feed_forward(src)
        src = self.ff_layer_norm(src + self.dropout(_src))

        return src

    def save(self, path):
        self.self_attention.save(path + "_mha")
        self.self_attention_norm.save(path + "_mha")
        self.position_wise_feed_forward.save(path + "_pwff")
        self.ff_layer_norm.save(path + "_pwff")


class DecoderLayer(nn.Module):
    """
    This is a PyTorch implementation of the DecoderLayer layer implemented in
    https://github.com/AmritanshuV/Numpy-Transformer/blob/42dd602e9a651d77ddb041790b1d6d513d8b7cf7/transformer/layers/combined/decoder_layer.py
    """

    def __init__(self, d_model, heads_num, d_ff, dropout, data_type=float):
        super(DecoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(
            d_model, elementwise_affine=True, dtype=data_type
        )
        self.enc_attn_layer_norm = nn.LayerNorm(
            d_model, elementwise_affine=True, dtype=data_type
        )
        self.ff_layer_norm = nn.LayerNorm(
            d_model, elementwise_affine=True, dtype=data_type
        )
        self.self_attention = MultiheadAttention(d_model, heads_num, dropout, data_type)
        self.encoder_attention = MultiheadAttention(
            d_model, heads_num, dropout, data_type
        )
        self.position_wise_feed_forward = PositionwiseFeedForward(
            d_model, d_ff, dropout
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, trg, trg_mask, src, src_mask):
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.self_attention_norm(trg + self.dropout(_trg))

        _trg, attention = self.encoder_attention(trg, src, src, src_mask)
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        _trg = self.position_wise_feed_forward(trg)
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        return trg, attention

    def save(self, path):
        self.self_attention.save(path + "_mha1")
        self.self_attention_norm.save(path + "_mha1")
        self.encoder_attention.save(path + "_mha2")
        self.enc_attn_layer_norm.save(path + "_mha2")
        self.position_wise_feed_forward.save(path + "_pwff")
        self.ff_layer_norm.save(path + "_pwff")


class Encoder(nn.Module):
    """
    This is a PyTorch implementation of the Encoder layer implemented in
    https://github.com/AmritanshuV/Numpy-Transformer/blob/42dd602e9a651d77ddb041790b1d6d513d8b7cf7/transformer/modules/encoder.py
    """

    def __init__(
        self,
        src_vocab_size,
        heads_num,
        layers_num,
        d_model,
        d_ff,
        dropout,
        max_length=5000,
        data_type=float,
    ):
        super(Encoder, self).__init__()

        self.token_embedding = nn.Embedding(src_vocab_size, d_model, dtype=data_type)
        self.position_embedding = PositionalEncoding(d_model, dropout, max_length)

        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, heads_num, d_ff, dropout, data_type)
                for _ in range(layers_num)
            ]
        )

        self.dropout = nn.Dropout(p=dropout)
        self.scale = np.sqrt(d_model).astype(data_type)

    def forward(self, src, src_mask):
        src = self.token_embedding(src) * self.scale
        src = self.position_embedding(src)

        self.first_layer_input = src
        self.first_layer_input.retain_grad()
        src = self.first_layer_input

        for layer in self.layers:
            src = layer(src, src_mask)

        return src

    def save(self, path):
        self.token_embedding.save(path + "_e")
        for idx, layer in enumerate(self.layers):
            layer.save(path + "_l" + str(idx))


class Decoder(nn.Module):
    """
    This is a PyTorch implementation of the Decoder layer implemented in
    https://github.com/AmritanshuV/Numpy-Transformer/blob/42dd602e9a651d77ddb041790b1d6d513d8b7cf7/transformer/modules/decoder.py
    """

    def __init__(
        self,
        trg_vocab_size,
        heads_num,
        layers_num,
        d_model,
        d_ff,
        dropout,
        max_length=5000,
        data_type=float,
    ):
        super(Decoder, self).__init__()

        self.token_embedding = nn.Embedding(trg_vocab_size, d_model, dtype=data_type)
        self.position_embedding = PositionalEncoding(d_model, dropout, max_length)

        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, heads_num, d_ff, dropout, data_type)
                for _ in range(layers_num)
            ]
        )

        self.fc_out = nn.Linear(d_model, trg_vocab_size, dtype=data_type)
        self.dropout = nn.Dropout(p=dropout)
        self.scale = np.sqrt(d_model).astype(data_type)

        self.activation = nn.Softmax(dim=-1)

    def forward(self, trg, trg_mask, src, src_mask):
        trg = self.token_embedding(trg) * self.scale
        trg = self.position_embedding(trg)

        self.first_layer_input = trg
        self.first_layer_input.retain_grad()
        trg = self.first_layer_input

        for layer in self.layers:
            trg, attention = layer(trg, trg_mask, src, src_mask)

        output = self.fc_out(trg)

        activated_output = self.activation.forward(output)
        return activated_output, attention

    def save(self, path):
        self.token_embedding.save(path + "_e")
        for idx, layer in enumerate(self.layers):
            layer.save(path + "_l" + str(idx))
        self.fc_out.save(path + "_d")


class Transformer(nn.Module):
    """
    This is a PyTorch implementation of the Seq2Seq layer implemented in
    https://github.com/AmritanshuV/Numpy-Transformer/blob/42dd602e9a651d77ddb041790b1d6d513d8b7cf7/transformer/transformer.py#L69C1-L69C17
    """

    def __init__(self, encoder_kw, decoder_kw, pad_idx):
        super(Transformer, self).__init__()

        self.encoder = Encoder(**encoder_kw)
        self.decoder = Decoder(**decoder_kw)
        self.pad_idx = pad_idx

    def get_pad_mask(self, x):
        return (x != self.pad_idx).unsqueeze(1)

    def get_sub_mask(self, x):
        seq_len = x.size(1)
        subsequent_mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1).bool()
        return ~subsequent_mask

    def forward(self, src, trg):
        src_mask = self.get_pad_mask(src)
        trg_mask = self.get_pad_mask(trg) & self.get_sub_mask(trg)

        self.src_mask = src_mask.int()
        self.trg_mask = trg_mask.int()

        self.enc_src = self.encoder(src, self.src_mask)
        self.enc_src.retain_grad()

        out, attention = self.decoder(trg, self.trg_mask, self.enc_src, self.src_mask)
        return out, attention

    def save(self, path):
        self.encoder.save(path + "_en")
        self.decoder.save(path + "_de")
