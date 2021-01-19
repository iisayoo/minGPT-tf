"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math

import numpy as np
import tensorflow as tf


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)


kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0,
                                                        stddev=0.02)


class CausalSelfAttention(tf.keras.layers.Layer):
    """
    A vanilla multi-head masked self-attention layer with a projection at the
    end. It is possible to use tf.keras.layers.MultiheadAttention here but I am
    including an explicit implementation here to show that there is nothing too
    scary here.
    """

    def __init__(self, config):
        super(CausalSelfAttention, self).__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads
        self.key = tf.keras.layers.Dense(
            config.n_embd,
            kernel_initializer=kernel_initializer)
        self.query = tf.keras.layers.Dense(
            config.n_embd,
            kernel_initializer=kernel_initializer)
        self.value = tf.keras.layers.Dense(
            config.n_embd,
            kernel_initializer=kernel_initializer)

        # regularization
        self.attn_drop = tf.keras.layers.Dropout(config.attn_pdrop)
        self.resid_drop = tf.keras.layers.Dropout(config.resid_pdrop)

        # output projection
        self.proj = tf.keras.layers.Dense(
            config.n_embd,
            kernel_initializer=kernel_initializer)

        lower_tri = np.tril(
            np.ones([config.block_size, config.block_size])
        ).reshape(1, 1, config.block_size, config.block_size)
        self.mask = tf.Variable(lower_tri, trainable=False)

        self.n_embd = config.n_embd
        self.n_head = config.n_head

    def call(self, x):
        # B: batch size, T: block_size, ie number of tokens, C: n_embd.
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        C = tf.shape(x)[2]

        # calculate query, key, value for each head.
        # make head the next leading dim.
        k = tf.reshape(self.key(x),
                       [B, T, self.n_head, C // self.n_head]) # (B, T, nh, hs)
        k = tf.transpose(k, [0, 2, 1, 3]) # (B, nh, T, hs)
        q = tf.reshape(self.query(x),
                       [B, T, self.n_head, C // self.n_head])
        q = tf.transpose(q, [0, 2, 1, 3])
        v = tf.reshape(self.value(x),
                       [B, T, self.n_head, C // self.n_head])
        v = tf.transpose(v, [0, 2, 1, 3])

        # causal self-attention;
        # Self-attend: (B, nh, T, hs) x (B, nh, T, hs) -> (B, nh, T, T)
        att = tf.matmul(q, tf.transpose(k, [0, 1, 3, 2]))
        att *=  1.0 / math.sqrt(self.n_embd / self.n_head)
        att = tf.where(self.mask[:, :, :T, :T] == 0, float('-inf'), att)

        # apply softmax.
        att = tf.keras.layers.Softmax()(att)
        att = self.attn_drop(att)

        # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = tf.matmul(att, v)
        # re-assemble all head outputs side-by-side
        y = tf.reshape(tf.transpose(y, [0, 2, 1, 3]), [B, T, C])
        y = self.resid_drop(self.proj(y))

        return y


class Block(tf.keras.layers.Layer):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super(Block, self).__init__()

        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.attn = CausalSelfAttention(config)
        self.dense_1 = tf.keras.layers.Dense(
            4 * config.n_embd,
            kernel_initializer=kernel_initializer,
            activation='gelu')
        self.dense_2 = tf.keras.layers.Dense(
            config.n_embd,
            kernel_initializer=kernel_initializer,
            activation='gelu')
        self.drop = tf.keras.layers.Dropout(config.resid_pdrop)

    def call(self, x):
        x1 = x + self.attn(self.ln1(x))
        x2 = self.ln2(x1)
        x2 = self.dense_1(x2)
        x2 = self.dense_2(x2)
        y = x1 + self.drop(x2)
        return y


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, config):
        super(PositionalEncoding, self).__init__()

        self.pos_emb = tf.Variable(
            tf.zeros([1, config.block_size, config.n_embd]),
            trainable=True)

    def call(self, x):
        return x + self.pos_emb[:, :tf.shape(x)[1], :]


class GPT(tf.keras.Model):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super(GPT, self).__init__()

        # input embedding stem
        self.tok_emb = tf.keras.layers.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = PositionalEncoding(config)
        self.drop = tf.keras.layers.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = [Block(config) for _ in range(config.n_layer)]

        # decoder head
        self.ln_f = tf.keras.layers.LayerNormalization()
        self.head = tf.keras.layers.Dense(config.vocab_size, use_bias=False)

        self.block_size = config.block_size

    def get_block_size(self):
        return self.block_size

    def call(self, idx, training=False):
        """forward the GPT model"""

        # each index maps to a (learnable) vector
        token_embeddings = self.tok_emb(idx)

        # each position maps to a (learnable) vector
        x = self.pos_emb(token_embeddings)
        x = self.drop(x)

        for layer in self.blocks:
            x = layer(x)

        x = self.ln_f(x)

        logits = self.head(x) # (batch size, token count, vocab size)

        return logits
