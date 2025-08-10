"""
This file generates the data used in `test/06_layers.cu`.

Usage:
    `python 06_layers.py <data_folder> <temporary_folder>`
    - `data_folder`: The path to the directory where data files will be stored.
    - `temporary_folder`: The path to a temporary directory for intermediate
    files.
"""

import modules
import numpy as np
from modules import nn, torch

import utils

optim = torch.optim


def dense_without_bias():
    """
    Generates random data for testing the Dense layer without bias.
    """
    path = "layers/dense_without_bias"
    dense = nn.Dense(50, 70, bias=False, dtype=float)
    dense.save(path)

    x = utils.create_tensor(np.random.uniform(-1, 1, size=[50, 50]))
    utils.write_tensor_to_file(x, path + "_input")

    y = dense(x)
    utils.write_tensor_to_file(y, path + "_output")

    error = utils.create_tensor(np.random.uniform(-1, 1, size=y.shape))
    utils.write_tensor_to_file(error, path + "_error")

    y.backward(error)
    utils.write_tensor_to_file(x.grad, path + "_input_gradient")

    lr = 0.5
    utils.write_tensor_to_file(lr, path + "_learning_rate")

    optim.SGD(dense.parameters(), lr=lr).step()
    dense.save(path + "_updated")


def dense_with_bias():
    """
    Generates random data for testing the Dense layer with bias.
    """
    path = "layers/dense_with_bias"
    dense = nn.Dense(70, 50, bias=True, dtype=float)
    dense.save(path)

    x = utils.create_tensor(np.random.uniform(-1, 1, size=[50, 10, 70]))
    utils.write_tensor_to_file(x, path + "_input")

    y = dense(x)
    utils.write_tensor_to_file(y, path + "_output")

    error = utils.create_tensor(np.random.uniform(-1, 1, size=y.shape))
    utils.write_tensor_to_file(error, path + "_error")

    y.backward(error)
    utils.write_tensor_to_file(x.grad, path + "_input_gradient")

    lr = 0.5
    utils.write_tensor_to_file(lr, path + "_learning_rate")

    optim.SGD(dense.parameters(), lr=lr).step()
    dense.save(path + "_updated")


def reshape():
    """
    Generates random data for testing the Reshape layer.
    """
    path = "layers/reshape"
    x = utils.create_tensor(np.random.uniform(-1, 1, size=[6, 10, 20, 4]))
    utils.write_tensor_to_file(x, path + "_input")

    y = torch.reshape(x, [6, 2, 400])
    utils.write_tensor_to_file(y, path + "_output")

    error = utils.create_tensor(np.random.uniform(-1, 1, size=y.shape))
    utils.write_tensor_to_file(error, path + "_error")

    input_gradient = torch.reshape(error, x.shape)
    utils.write_tensor_to_file(input_gradient, path + "_input_gradient")


def upscale():
    """
    Generates random data for testing the Upscale layer.
    """
    path = "layers/upscale"

    interpolate = lambda x, size: torch.nn.functional.interpolate(
        x.unsqueeze(1), size=size, mode="bilinear", align_corners=True
    ).squeeze(1)

    x = utils.create_tensor(np.random.uniform(-1, 1, size=[10, 33, 97]))
    utils.write_tensor_to_file(x, path + "_identity_input")
    utils.write_tensor_to_file(interpolate(x, [33, 97]), path + "_identity_output")

    x = utils.create_tensor([[[0, 1], [2, 3]]])
    utils.write_tensor_to_file(x, path + "_small_input")
    utils.write_tensor_to_file(interpolate(x, [4, 4]), path + "_small_output")

    x = utils.create_tensor(np.random.uniform(-1, 1, size=[10, 28, 28]))
    utils.write_tensor_to_file(x, path + "_square_input")
    utils.write_tensor_to_file(interpolate(x, [1000, 1000]), path + "_square_output")

    x = utils.create_tensor(np.random.uniform(-1, 1, size=[10, 13, 17]))
    utils.write_tensor_to_file(x, path + "_rectangular_input")
    utils.write_tensor_to_file(interpolate(x, [130, 97]), path + "_rectangular_output")


def dropout():
    """
    Generates random data for testing the Dropout layer.
    """
    path = "layers/dropout"

    probability = 0.7
    utils.write_tensor_to_file(probability, path + "_probability")

    x = utils.create_tensor(np.ones([1000, 100]))
    utils.write_tensor_to_file(x, path + "_input")

    layer = nn.Dropout(p=probability)
    layer.train()

    y = layer(x)
    mask = torch.abs(y) > 0.5
    utils.write_tensor_to_file(mask.float(), path + "_mask")

    scaling = (x[mask] / y[mask]).mean()
    utils.write_tensor_to_file(scaling, path + "_train_scaling")

    error = utils.create_tensor(np.random.uniform(-1, 1, size=y.shape))
    utils.write_tensor_to_file(error, path + "_error")

    y.backward(error)
    utils.write_tensor_to_file(x.grad, path + "_train_input_gradient")

    layer.eval()
    x.grad.zero_()
    y = layer(x)
    utils.write_tensor_to_file(y, path + "_eval_output")

    y.backward(error)
    utils.write_tensor_to_file(x.grad, path + "_eval_input_gradient")


def layernorm():
    """
    Generates random data for testing the LayerNorm layer.
    """
    path = "layers/layernorm"

    normalized_shape = [10, 20]
    utils.write_tensor_to_file(normalized_shape, path + "_normalized_shape")

    layer = nn.LayerNorm(normalized_shape, dtype=float)
    layer.save(path)

    x = utils.create_tensor(
        np.random.uniform(-0.5, 1.5, size=[100, 5, *normalized_shape])
    )
    utils.write_tensor_to_file(x, path + "_input")

    x_normalized = (x - torch.mean(x, dim=[-2, -1], keepdim=True)) / torch.std(
        x, dim=[-2, -1], correction=0, keepdim=True
    )
    utils.write_tensor_to_file(x_normalized, path + "_input_normalized")

    y = layer(x)
    utils.write_tensor_to_file(y, path + "_output")

    error = utils.create_tensor(np.random.uniform(-1, 1, size=y.shape))
    utils.write_tensor_to_file(error, path + "_error")

    y.backward(error)
    utils.write_tensor_to_file(x.grad, path + "_input_gradient")

    lr = 0.5
    utils.write_tensor_to_file(lr, path + "_learning_rate")

    optim.SGD(layer.parameters(), lr=lr).step()
    layer.save(path + "_updated")


def embedding():
    """
    Generates random data for testing the Embedding layer.
    """
    path = "layers/embedding"

    num_embeddings = 10
    utils.write_tensor_to_file(num_embeddings, path + "_num_embeddings")

    layer = nn.Embedding(num_embeddings, 30, dtype=float)
    layer.save(path)

    x = utils.create_tensor(np.random.randint(0, num_embeddings, size=[100, 50])).type(
        torch.long
    )
    utils.write_tensor_to_file(x, path + "_input")

    y = layer(x)
    utils.write_tensor_to_file(y, path + "_output")

    error = utils.create_tensor(np.random.uniform(-1, 1, size=y.shape))
    utils.write_tensor_to_file(error, path + "_error")

    y.backward(error)

    lr = 0.5
    utils.write_tensor_to_file(lr, path + "_learning_rate")

    optim.SGD(layer.parameters(), lr=lr).step()
    layer.save(path + "_updated")


def positionwisefeedforward():
    """
    Generates random data for testing the PositionwiseFeedForward layer.
    """
    path = "layers/positionwisefeedforward"

    dropout = 0
    utils.write_tensor_to_file(dropout, path + "_dropout")

    hidden_dim = 30
    utils.write_tensor_to_file(hidden_dim, path + "_hidden_dim")

    layer = modules.PositionwiseFeedForward(
        d_model=20, d_ff=hidden_dim, dropout=dropout
    )
    layer.save(path)

    x = utils.create_tensor(np.random.uniform(-1, 1, size=[50, 10, 20]))
    utils.write_tensor_to_file(x, path + "_input")

    y = layer(x)
    utils.write_tensor_to_file(y, path + "_output")

    error = utils.create_tensor(np.random.uniform(-1, 1, size=y.shape))
    utils.write_tensor_to_file(error, path + "_error")

    y.backward(error)
    utils.write_tensor_to_file(x.grad, path + "_input_gradient")

    lr = 0.5
    utils.write_tensor_to_file(lr, path + "_learning_rate")

    optim.SGD(layer.parameters(), lr=lr).step()
    layer.save(path + "_updated")


def positionalencoding():
    """
    Generates random data for testing the PositionalEncoding layer.
    """
    path = "layers/positionalencoding"

    dropout = 0
    utils.write_tensor_to_file(dropout, path + "_dropout")

    max_batchsize = 500
    utils.write_tensor_to_file(max_batchsize, path + "_max_batchsize")

    layer = modules.PositionalEncoding(
        d_model=20, dropout=dropout, max_len=max_batchsize
    )
    utils.write_tensor_to_file(layer.pe.squeeze(1), path + "_pe")

    x = utils.create_tensor(np.random.uniform(-1, 1, size=[100, 10, 20]))
    utils.write_tensor_to_file(x, path + "_input")

    y = layer(x)
    utils.write_tensor_to_file(y, path + "_output")

    error = utils.create_tensor(np.random.uniform(-1, 1, size=y.shape))
    utils.write_tensor_to_file(error, path + "_error")

    y.backward(error)
    utils.write_tensor_to_file(x.grad, path + "_input_gradient")


def multiheadattention():
    """
    Generates random data for testing the MultiheadAttention layer.
    """
    path = "layers/multiheadattention"

    dropout = 0
    utils.write_tensor_to_file(dropout, path + "_dropout")

    num_heads = 4
    utils.write_tensor_to_file(num_heads, path + "_num_heads")

    batchsize = 50
    query_length = 10
    key_value_length = 12
    embedding_dim = 32

    layer = modules.MultiheadAttention(
        d_model=embedding_dim, heads_num=num_heads, dropout=dropout
    )
    layer.save(path)

    query = utils.create_tensor(
        np.random.uniform(-1, 1, size=[batchsize, query_length, embedding_dim])
    )
    utils.write_tensor_to_file(query, path + "_query")

    key = utils.create_tensor(
        np.random.uniform(-1, 1, size=[batchsize, key_value_length, embedding_dim])
    )
    utils.write_tensor_to_file(key, path + "_key")

    value = utils.create_tensor(
        np.random.uniform(-1, 1, size=[batchsize, key_value_length, embedding_dim])
    )
    utils.write_tensor_to_file(value, path + "_value")

    mask = utils.create_tensor(
        np.random.choice(
            [0, 1], size=[batchsize, query_length, key_value_length], p=[0.1, 0.9]
        )
    )
    utils.write_tensor_to_file(mask, path + "_mask")

    y, attention = layer(query, key, value, mask)
    utils.write_tensor_to_file(y, path + "_output")
    utils.write_tensor_to_file(attention, path + "_attention")

    error = utils.create_tensor(np.random.uniform(-1, 1, size=y.shape))
    utils.write_tensor_to_file(error, path + "_error")

    y.backward(error)
    utils.write_tensor_to_file(query.grad, path + "_query_gradient")
    utils.write_tensor_to_file(key.grad, path + "_key_gradient")
    utils.write_tensor_to_file(value.grad, path + "_value_gradient")

    lr = 0.5
    utils.write_tensor_to_file(lr, path + "_learning_rate")

    optim.SGD(layer.parameters(), lr=lr).step()
    layer.save(path + "_updated")


def encoderlayer():
    """
    Generates random data for testing the EncoderLayer layer.
    """
    path = "layers/encoderlayer"

    dropout = 0
    utils.write_tensor_to_file(dropout, path + "_dropout")

    num_heads = 8
    utils.write_tensor_to_file(num_heads, path + "_num_heads")

    hidden_dim = 36
    utils.write_tensor_to_file(hidden_dim, path + "_hidden_dim")

    batchsize = 50
    source_length = 10
    embedding_dim = 93

    layer = modules.EncoderLayer(
        d_model=embedding_dim, heads_num=num_heads, d_ff=hidden_dim, dropout=dropout
    )
    layer.save(path)

    x = utils.create_tensor(
        np.random.uniform(-1, 1, size=[batchsize, source_length, embedding_dim])
    )
    utils.write_tensor_to_file(x, path + "_input")

    mask = utils.create_tensor(
        np.random.choice(
            [0, 1], size=[batchsize, source_length, source_length], p=[0.3, 0.7]
        )
    )
    utils.write_tensor_to_file(mask, path + "_mask")

    y = layer(x, mask)
    utils.write_tensor_to_file(y, path + "_output")

    error = utils.create_tensor(np.random.uniform(-1, 1, size=y.shape))
    utils.write_tensor_to_file(error, path + "_error")

    y.backward(error)
    utils.write_tensor_to_file(x.grad, path + "_input_gradient")

    lr = 0.5
    utils.write_tensor_to_file(lr, path + "_learning_rate")

    optim.SGD(layer.parameters(), lr=lr).step()
    layer.save(path + "_updated")


def decoderlayer():
    """
    Generates random data for testing the DecoderLayer layer.
    """
    path = "layers/decoderlayer"

    dropout = 0
    utils.write_tensor_to_file(dropout, path + "_dropout")

    num_heads = 8
    utils.write_tensor_to_file(num_heads, path + "_num_heads")

    hidden_dim = 36
    utils.write_tensor_to_file(hidden_dim, path + "_hidden_dim")

    batchsize = 50
    source_length = 10
    target_length = 11
    embedding_dim = 93

    layer = modules.DecoderLayer(
        d_model=embedding_dim, heads_num=num_heads, d_ff=hidden_dim, dropout=dropout
    )
    layer.save(path)

    target = utils.create_tensor(
        np.random.uniform(-1, 1, size=[batchsize, target_length, embedding_dim])
    )
    utils.write_tensor_to_file(target, path + "_target")

    target_mask = utils.create_tensor(
        np.random.choice(
            [0, 1], size=[batchsize, target_length, target_length], p=[0.3, 0.7]
        )
    )
    utils.write_tensor_to_file(target_mask, path + "_target_mask")

    source = utils.create_tensor(
        np.random.uniform(-1, 1, size=[batchsize, source_length, embedding_dim])
    )
    utils.write_tensor_to_file(source, path + "_source")

    source_mask = utils.create_tensor(
        np.random.choice([0, 1], size=[batchsize, 1, source_length], p=[0.3, 0.7])
    )
    utils.write_tensor_to_file(source_mask, path + "_source_mask")

    y, attention = layer(target, target_mask, source, source_mask)
    utils.write_tensor_to_file(y, path + "_output")
    utils.write_tensor_to_file(attention, path + "_attention")

    error = utils.create_tensor(np.random.uniform(-1, 1, size=y.shape))
    utils.write_tensor_to_file(error, path + "_error")

    y.backward(error)
    utils.write_tensor_to_file(target.grad, path + "_target_gradient")
    utils.write_tensor_to_file(source.grad, path + "_source_gradient")

    lr = 0.5
    utils.write_tensor_to_file(lr, path + "_learning_rate")

    optim.SGD(layer.parameters(), lr=lr).step()
    layer.save(path + "_updated")


def encoder():
    """
    Generates random data for testing the Encoder layer.
    """
    path = "layers/encoder"

    dropout = 0
    utils.write_tensor_to_file(dropout, path + "_dropout")

    num_heads = 4
    utils.write_tensor_to_file(num_heads, path + "_num_heads")

    num_layers = 2
    utils.write_tensor_to_file(num_layers, path + "_num_layers")

    num_embeddings = 20
    utils.write_tensor_to_file(num_embeddings, path + "_num_embeddings")

    max_batchsize = 500
    utils.write_tensor_to_file(max_batchsize, path + "_max_batchsize")

    hidden_dim = 64
    utils.write_tensor_to_file(hidden_dim, path + "_hidden_dim")

    batchsize = 50
    input_length = 12
    embedding_dim = 32

    layer = modules.Encoder(
        src_vocab_size=num_embeddings,
        heads_num=num_heads,
        layers_num=num_layers,
        d_model=embedding_dim,
        d_ff=hidden_dim,
        dropout=dropout,
        max_length=max_batchsize,
    )
    layer.save(path)

    x = utils.create_tensor(
        np.random.randint(0, num_embeddings, size=[batchsize, input_length])
    ).type(torch.long)
    utils.write_tensor_to_file(x, path + "_input_1")

    mask = utils.create_tensor(
        np.random.choice(
            [0, 1], size=[batchsize, input_length, input_length], p=[0.1, 0.9]
        )
    )
    utils.write_tensor_to_file(mask, path + "_mask")

    y = layer(x, mask)
    utils.write_tensor_to_file(layer.first_layer_input, path + "_first_layer_input_1")
    utils.write_tensor_to_file(y, path + "_output_1")

    error = utils.create_tensor(np.random.uniform(-1, 1, size=y.shape))
    utils.write_tensor_to_file(error, path + "_error")

    y.backward(error)
    utils.write_tensor_to_file(
        layer.first_layer_input.grad, path + "_first_layer_input_gradient"
    )

    lr = 0.5
    utils.write_tensor_to_file(lr, path + "_learning_rate")

    optim.SGD(layer.parameters(), lr=lr).step()

    x = utils.create_tensor(np.random.randint(0, num_embeddings, size=x.shape)).type(
        torch.long
    )
    utils.write_tensor_to_file(x, path + "_input_2")

    y = layer(x, mask)
    utils.write_tensor_to_file(layer.first_layer_input, path + "_first_layer_input_2")
    utils.write_tensor_to_file(y, path + "_output_2")


def decoder():
    """
    Generates random data for testing the Decoder layer.
    """
    path = "layers/decoder"

    dropout = 0
    utils.write_tensor_to_file(dropout, path + "_dropout")

    num_heads = 4
    utils.write_tensor_to_file(num_heads, path + "_num_heads")

    num_layers = 2
    utils.write_tensor_to_file(num_layers, path + "_num_layers")

    num_embeddings = 22
    utils.write_tensor_to_file(num_embeddings, path + "_num_embeddings")

    max_batchsize = 500
    utils.write_tensor_to_file(max_batchsize, path + "_max_batchsize")

    hidden_dim = 64
    utils.write_tensor_to_file(hidden_dim, path + "_hidden_dim")

    batchsize = 50
    target_length = 10
    source_length = 12
    embedding_dim = 32

    layer = modules.Decoder(
        trg_vocab_size=num_embeddings,
        heads_num=num_heads,
        layers_num=num_layers,
        d_model=embedding_dim,
        d_ff=hidden_dim,
        dropout=dropout,
        max_length=max_batchsize,
    )
    layer.save(path)

    target = utils.create_tensor(
        np.random.randint(0, num_embeddings, size=[batchsize, target_length])
    ).type(torch.long)
    utils.write_tensor_to_file(target, path + "_target_1")

    target_mask = utils.create_tensor(
        np.random.choice(
            [0, 1], size=[batchsize, target_length, target_length], p=[0.1, 0.9]
        )
    )
    utils.write_tensor_to_file(target_mask, path + "_target_mask")

    source = utils.create_tensor(
        np.random.uniform(-1, 1, size=[batchsize, source_length, embedding_dim])
    )
    utils.write_tensor_to_file(source, path + "_source_1")

    source_mask = utils.create_tensor(
        np.random.choice(
            [0, 1], size=[batchsize, target_length, source_length], p=[0.1, 0.9]
        )
    )
    utils.write_tensor_to_file(source_mask, path + "_source_mask")

    y, attention = layer(target, target_mask, source, source_mask)
    utils.write_tensor_to_file(layer.first_layer_input, path + "_first_layer_input_1")
    utils.write_tensor_to_file(y, path + "_output_1")
    utils.write_tensor_to_file(attention, path + "_attention_1")

    error = utils.create_tensor(np.random.uniform(-1, 1, size=y.shape))
    utils.write_tensor_to_file(error, path + "_error")

    y.backward(error)
    utils.write_tensor_to_file(
        layer.first_layer_input.grad, path + "_first_layer_input_gradient"
    )
    utils.write_tensor_to_file(source.grad, path + "_source_gradient")

    lr = 0.5
    utils.write_tensor_to_file(lr, path + "_learning_rate")

    optim.SGD(layer.parameters(), lr=lr).step()

    target = utils.create_tensor(
        np.random.randint(0, num_embeddings, size=target.shape)
    ).type(torch.long)
    utils.write_tensor_to_file(target, path + "_target_2")

    source = utils.create_tensor(np.random.uniform(-1, 1, size=source.shape))
    utils.write_tensor_to_file(source, path + "_source_2")

    y, attention = layer(target, target_mask, source, source_mask)
    utils.write_tensor_to_file(layer.first_layer_input, path + "_first_layer_input_2")
    utils.write_tensor_to_file(y, path + "_output_2")
    utils.write_tensor_to_file(attention, path + "_attention_2")


def transformer():
    """
    Generates random data for testing the Transformer layer.
    """
    path = "layers/transformer"

    dropout = 0
    utils.write_tensor_to_file(dropout, path + "_dropout")

    num_heads = 4
    utils.write_tensor_to_file(num_heads, path + "_num_heads")

    num_layers = 2
    utils.write_tensor_to_file(num_layers, path + "_num_layers")

    num_source_embeddings = 20
    utils.write_tensor_to_file(num_source_embeddings, path + "_num_source_embeddings")

    num_target_embeddings = 22
    utils.write_tensor_to_file(num_target_embeddings, path + "_num_target_embeddings")

    max_batchsize = 500
    utils.write_tensor_to_file(max_batchsize, path + "_max_batchsize")

    hidden_dim = 64
    utils.write_tensor_to_file(hidden_dim, path + "_hidden_dim")

    ignore_index = 0
    utils.write_tensor_to_file(ignore_index, path + "_ignore_index")

    embedding_dim = 32
    utils.write_tensor_to_file(embedding_dim, path + "_embedding_dim")

    batchsize = 50
    target_length = 10
    source_length = 12

    encoder_kw = dict(
        src_vocab_size=num_source_embeddings,
        heads_num=num_heads,
        layers_num=num_layers,
        d_model=embedding_dim,
        d_ff=hidden_dim,
        dropout=dropout,
        max_length=max_batchsize,
    )

    decoder_kw = dict(
        trg_vocab_size=num_target_embeddings,
        heads_num=num_heads,
        layers_num=num_layers,
        d_model=embedding_dim,
        d_ff=hidden_dim,
        dropout=dropout,
        max_length=max_batchsize,
    )

    layer = modules.Transformer(encoder_kw, decoder_kw, ignore_index)
    layer.save(path)

    target = utils.create_tensor(
        np.random.randint(0, num_target_embeddings, size=[batchsize, target_length])
    ).type(torch.long)
    utils.write_tensor_to_file(target, path + "_target_1")

    source = utils.create_tensor(
        np.random.randint(0, num_source_embeddings, size=[batchsize, source_length])
    ).type(torch.long)
    utils.write_tensor_to_file(source, path + "_source_1")

    y, attention = layer(source, target)
    utils.write_tensor_to_file(y, path + "_output_1")
    utils.write_tensor_to_file(attention, path + "_attention_1")
    utils.write_tensor_to_file(layer.src_mask, path + "_source_mask_1")
    utils.write_tensor_to_file(layer.trg_mask, path + "_target_mask_1")
    utils.write_tensor_to_file(layer.enc_src, path + "_encoder_output_1")

    error = utils.create_tensor(np.random.uniform(-1, 1, size=y.shape))
    utils.write_tensor_to_file(error, path + "_error")

    y.backward(error)
    utils.write_tensor_to_file(
        layer.encoder.first_layer_input.grad,
        path + "_encoder_first_layer_input_gradient",
    )
    utils.write_tensor_to_file(layer.enc_src.grad, path + "_decoder_source_gradient")
    utils.write_tensor_to_file(
        layer.decoder.first_layer_input.grad,
        path + "_decoder_first_layer_input_gradient",
    )

    lr = 0.5
    utils.write_tensor_to_file(lr, path + "_learning_rate")

    optim.SGD(layer.parameters(), lr=lr).step()

    target = utils.create_tensor(
        np.random.randint(0, num_target_embeddings, size=target.shape)
    ).type(torch.long)
    utils.write_tensor_to_file(target, path + "_target_2")

    source = utils.create_tensor(
        np.random.randint(0, num_source_embeddings, size=source.shape)
    ).type(torch.long)
    utils.write_tensor_to_file(source, path + "_source_2")

    y, attention = layer(source, target)
    utils.write_tensor_to_file(y, path + "_output_2")
    utils.write_tensor_to_file(attention, path + "_attention_2")
    utils.write_tensor_to_file(layer.src_mask, path + "_source_mask_2")
    utils.write_tensor_to_file(layer.trg_mask, path + "_target_mask_2")
    utils.write_tensor_to_file(layer.enc_src, path + "_encoder_output_2")


if __name__ == "__main__":
    dense_without_bias()
    dense_with_bias()
    reshape()
    upscale()
    dropout()
    layernorm()
    embedding()
    positionwisefeedforward()
    positionalencoding()
    multiheadattention()
    encoderlayer()
    decoderlayer()
    encoder()
    decoder()
    transformer()
