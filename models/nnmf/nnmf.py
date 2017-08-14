import os
import sys

import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_frame import DataFrame
from tf_base_model import TFBaseModel


class DataReader(object):

    def __init__(self, data_dir):
        data_cols = ['i', 'j', 'V_ij']
        data = [np.load(os.path.join(data_dir, '{}.npy'.format(i)), mmap_mode='r') for i in data_cols]

        df = DataFrame(columns=data_cols, data=data)
        self.train_df, self.val_df = df.train_test_split(train_size=0.9)

        print 'train size', len(self.train_df)
        print 'val size', len(self.val_df)

        self.num_users = df['i'].max() + 1
        self.num_products = df['j'].max() + 1

    def train_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.train_df,
            shuffle=True,
            num_epochs=10000
        )

    def val_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.val_df,
            shuffle=True,
            num_epochs=10000
        )

    def batch_generator(self, batch_size, df, shuffle=True, num_epochs=10000, is_test=False):
        return df.batch_generator(batch_size, shuffle=shuffle, num_epochs=num_epochs, allow_smaller_final_batch=is_test)


class nnmf(TFBaseModel):

    def __init__(self, rank=25, **kwargs):
        self.rank = rank
        super(nnmf, self).__init__(**kwargs)

    def calculate_loss(self):
        self.i = tf.placeholder(dtype=tf.int32, shape=[None])
        self.j = tf.placeholder(dtype=tf.int32, shape=[None])
        self.V_ij = tf.placeholder(dtype=tf.float32, shape=[None])

        self.W = tf.Variable(tf.truncated_normal([self.reader.num_users, self.rank]))
        self.H = tf.Variable(tf.truncated_normal([self.reader.num_products, self.rank]))
        W_bias = tf.Variable(tf.truncated_normal([self.reader.num_users]))
        H_bias = tf.Variable(tf.truncated_normal([self.reader.num_products]))

        global_mean = tf.Variable(0.0)
        w_i = tf.gather(self.W, self.i)
        h_j = tf.gather(self.H, self.j)

        w_bias = tf.gather(W_bias, self.i)
        h_bias = tf.gather(H_bias, self.j)
        interaction = tf.reduce_sum(w_i * h_j, reduction_indices=1)
        preds = global_mean + w_bias + h_bias + interaction

        rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(preds, self.V_ij)))

        self.parameter_tensors = {
            'user_embeddings': self.W,
            'product_embeddings': self.H
        }

        return rmse


if __name__ == '__main__':
    base_dir = './'

    dr = DataReader(data_dir=os.path.join(base_dir, 'data'))

    nnmf = nnmf(
        reader=dr,
        log_dir=os.path.join(base_dir, 'logs'),
        checkpoint_dir=os.path.join(base_dir, 'checkpoints'),
        prediction_dir=os.path.join(base_dir, 'predictions'),
        optimizer='adam',
        learning_rate=.005,
        rank=25,
        batch_size=4096,
        num_training_steps=150000,
        early_stopping_steps=30000,
        warm_start_init_step=0,
        regularization_constant=0.0,
        keep_prob=1.0,
        enable_parameter_averaging=False,
        num_restarts=0,
        min_steps_to_checkpoint=5000,
        log_interval=200,
        num_validation_batches=1,
        loss_averaging_window=200,

    )
    nnmf.fit()
    nnmf.restore()
    nnmf.predict()
