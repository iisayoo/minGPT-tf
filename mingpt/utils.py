import random

import numpy as np
import tensorflow as tf


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def top_k_logits(logits, k):
    v, ix = tf.math.top_k(logits, k)
    out = logits.copy()
    out = tf.where(out < v[:, -1:], float('-inf'), out)
    return out


def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict
    the next token in the sequence, feeding the predictions back into the model
    each time. Clearly the sampling has quadratic complexity unlike an RNN that
    is only linear, and has a finite context window of block_size, unlike an
    RNN that has an infinite context window.
    """

    block_size = model.get_block_size()
    for k in range(steps):
        # crop context if needed
        x_cond = x if x.shape[1] <= block_size else x[:, -block_size:]

        logits = model.predict(x_cond, batch_size=x_cond.shape[0])

        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature

        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)

        # apply softmax to convert to probabilities
        probs = tf.keras.layers.Softmax()(logits)

        # sample from the distribution or take the most likely
        if sample:
            ix = tf.random.categorical(tf.math.log(probs), 1)
        else:
            _, ix = tf.math.top_k(probs, 1)

        # append to the sequence and continue
        x = tf.concat([x, ix], axis=1)

    return x
