"""
Simple training loop; Boilerplate that could apply to any arbitrary neural
network, so nothing in this file really has anything to do with GPT
specifically.
"""

import logging
import math

import tensorflow as tf

logger = logging.getLogger(__name__)


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0

    # learning rate decay params:
    # linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_steps = 20
    final_steps = 1750
    final_decay = 0.1

    # checkpoint settings
    ckpt_path = None

    num_workers = 0

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class CosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, learning_rate=6e-4, warmup_steps=20, final_steps=1750,
                 final_decay=.1, steps_per_epoch=None):
        super(CosineSchedule, self).__init__()

        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.final_steps = final_steps
        self.final_decay = final_decay
        self.steps_per_epoch = steps_per_epoch

        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __call__(self, step):
        if self.steps_per_epoch is not None:
            step += self.epoch * self.steps_per_epoch

        less_than = tf.math.less(step, tf.constant(self.warmup_steps,
                                 dtype=tf.float32))

        lr_mult_warmup = tf.divide(step, self.warmup_steps)

        progress = (
            (step - self.warmup_steps)
            / float(max(1, self.final_steps - self.warmup_steps))
        )
        lr_mult = tf.maximum(
            self.final_decay,
            .5 * (1.0 + tf.math.cos(math.pi * progress))
        )

        return tf.cond(less_than, lambda: self.learning_rate * lr_mult_warmup,
                       lambda: self.learning_rate * lr_mult)


class PrintLRCallback(tf.keras.callbacks.Callback):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def on_train_batch_begin(self, batch, logs=None):
        lr = self.learning_rate(batch)
        print("\nLearning rate:", lr.numpy())


class SetEpochCallback(tf.keras.callbacks.Callback):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def on_epoch_begin(self, epoch, logs=None):
        self.learning_rate.set_epoch(epoch)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

    def train(self):
        if self.config.lr_decay:
            learning_rate = CosineSchedule(
                self.config.learning_rate,
                self.config.warmup_steps,
                self.config.final_steps,
                self.config.final_decay)
        else:
            learning_rate = self.config.learning_rate

        optimizer = tf.keras.optimizers.Adam(
            learning_rate,
            beta_1=self.config.betas[0],
            beta_2=self.config.betas[1],
            clipnorm=self.config.grad_norm_clip)

        self.model.compile(optimizer, 'sparse_categorical_crossentropy')

        if self.config.lr_decay:
            callbacks =[PrintLRCallback(), SetEpochCallback(learning_rate)]
        else:
            callbacks = None

        use_multiprocessing = True if self.config.num_workers > 1 else False
        self.model.fit(self.train_dataset, epochs=self.config.max_epochs,
                       callbacks=callbacks, validation_data=self.test_dataset,
                       workers=self.config.num_workers,
                       use_multiprocessing=use_multiprocessing)
