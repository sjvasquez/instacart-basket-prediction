import os

import numpy as np
import tensorflow as tf

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_frame import DataFrame
from tf_utils import lstm_layer, time_distributed_dense_layer
from tf_base_model import TFBaseModel


class DataReader(object):

    def __init__(self, data_dir):
        data_cols = [
            'user_id',
            'history_length',
            'order_size_history',
            'reorder_size_history',
            'order_number_history',
            'order_dow_history',
            'order_hour_history',
            'days_since_prior_order_history',
        ]
        data = [np.load(os.path.join(data_dir, '{}.npy'.format(i)), mmap_mode='r') for i in data_cols]
        self.test_df = DataFrame(columns=data_cols, data=data)

        print self.test_df.shapes()
        print 'loaded data'

        self.train_df, self.val_df = self.test_df.train_test_split(train_size=0.9)

        print 'train size', len(self.train_df)
        print 'val size', len(self.val_df)
        print 'test size', len(self.test_df)

    def train_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.train_df,
            shuffle=True,
            num_epochs=10000,
            is_test=False
        )

    def val_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.val_df,
            shuffle=True,
            num_epochs=10000,
            is_test=False
        )

    def test_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.test_df,
            shuffle=False,
            num_epochs=1,
            is_test=True
        )

    def batch_generator(self, batch_size, df, shuffle=True, num_epochs=10000, is_test=False):
        batch_gen = df.batch_generator(batch_size, shuffle=shuffle, num_epochs=num_epochs, allow_smaller_final_batch=is_test)
        for batch in batch_gen:
            batch['order_dow_history'] = np.roll(batch['order_dow_history'], -1, axis=1)
            batch['order_hour_history'] = np.roll(batch['order_hour_history'], -1, axis=1)
            batch['days_since_prior_order_history'] = np.roll(batch['days_since_prior_order_history'], -1, axis=1)
            batch['order_number_history'] = np.roll(batch['order_number_history'], -1, axis=1)
            batch['next_reorder_size'] = np.roll(batch['reorder_size_history'], -1, axis=1)
            if not is_test:
                batch['history_length'] = batch['history_length'] - 1
            yield batch


class rnn(TFBaseModel):

    def __init__(self, lstm_size=300, **kwargs):
        self.lstm_size = lstm_size
        super(rnn, self).__init__(**kwargs)

    def calculate_loss(self):
        x = self.get_input_sequences()
        loss = self.calculate_outputs(x)
        return loss

    def get_input_sequences(self):
        self.user_id = tf.placeholder(tf.int32, [None])
        self.history_length = tf.placeholder(tf.int32, [None])
        self.label = tf.placeholder(tf.int32, [None])
        self.eval_set = tf.placeholder(tf.int32, [None])

        self.order_size_history = tf.placeholder(tf.int32, [None, 100])
        self.reorder_size_history = tf.placeholder(tf.int32, [None, 100])
        self.order_number_history = tf.placeholder(tf.int32, [None, 100])
        self.order_dow_history = tf.placeholder(tf.int32, [None, 100])
        self.order_hour_history = tf.placeholder(tf.int32, [None, 100])
        self.days_since_prior_order_history = tf.placeholder(tf.int32, [None, 100])
        self.next_reorder_size = tf.placeholder(tf.int32, [None, 100])

        self.keep_prob = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)

        order_dow_history = tf.one_hot(self.order_dow_history, 8)
        order_hour_history = tf.one_hot(self.order_hour_history, 25)
        days_since_prior_order_history = tf.one_hot(self.days_since_prior_order_history, 31)
        order_size_history = tf.one_hot(self.order_size_history, 60)
        reorder_size_history = tf.one_hot(self.reorder_size_history, 50)
        order_number_history = tf.one_hot(self.order_number_history, 101)

        order_dow_history_scalar = tf.expand_dims(tf.cast(self.order_dow_history, tf.float32) / 8.0, 2)
        order_hour_history_scalar = tf.expand_dims(tf.cast(self.order_hour_history, tf.float32) / 25.0, 2)
        days_since_prior_order_history_scalar = tf.expand_dims(tf.cast(self.days_since_prior_order_history, tf.float32) / 31.0, 2)
        order_size_history_scalar = tf.expand_dims(tf.cast(self.order_size_history, tf.float32) / 60.0, 2)
        reorder_size_history_scalar = tf.expand_dims(tf.cast(self.reorder_size_history, tf.float32) / 50.0, 2)
        order_number_history_scalar = tf.expand_dims(tf.cast(self.order_number_history, tf.float32) / 100.0, 2)

        x = tf.concat([
            order_dow_history,
            order_hour_history,
            days_since_prior_order_history,
            order_size_history,
            reorder_size_history,
            order_number_history,
            order_dow_history_scalar,
            order_hour_history_scalar,
            days_since_prior_order_history_scalar,
            order_size_history_scalar,
            reorder_size_history_scalar,
            order_number_history_scalar,
        ], axis=2)

        return x

    def calculate_outputs(self, x):
        h = lstm_layer(x, self.history_length, self.lstm_size, scope='lstm-1')
        h_final = time_distributed_dense_layer(h, 50, activation=tf.nn.relu, scope='dense-1')

        n_components = 3
        params = time_distributed_dense_layer(h_final, n_components*3, scope='dense-2')
        means, variances, mixing_coefs = tf.split(params, 3, axis=2)

        mixing_coefs = tf.nn.softmax(mixing_coefs - tf.reduce_min(mixing_coefs, 2, keep_dims=True))
        variances = tf.exp(variances) + 1e-5

        labels = tf.cast(tf.tile(tf.expand_dims(self.next_reorder_size, 2), (1, 1, n_components)), tf.float32)
        n_likelihoods = 1.0 / (tf.sqrt(2*np.pi*variances))*tf.exp(-tf.square(labels - means) / (2*variances))
        nlls = -tf.log(tf.reduce_sum(mixing_coefs*n_likelihoods, axis=2) + 1e-10)

        sequence_mask = tf.cast(tf.sequence_mask(self.history_length, maxlen=100), tf.float32)
        nll = tf.reduce_sum(nlls*sequence_mask) / tf.cast(tf.reduce_sum(self.history_length), tf.float32)

        # evaluate likelihood at a sample of discrete points
        samples = tf.cast(tf.reshape(tf.range(25), (1, 1, 1, 25)), tf.float32)
        means = tf.tile(tf.expand_dims(means, 3), (1, 1, 1, 25))
        variances = tf.tile(tf.expand_dims(variances, 3), (1, 1, 1, 25))
        mixing_coefs = tf.tile(tf.expand_dims(mixing_coefs, 3), (1, 1, 1, 25))
        n_sample_likelihoods = 1.0 / (tf.sqrt(2*np.pi*variances))*tf.exp(-tf.square(samples - means) / (2*variances))
        sample_nlls = -tf.log(tf.reduce_sum(mixing_coefs*n_sample_likelihoods, axis=2) + 1e-10)

        final_temporal_idx = tf.stack([tf.range(tf.shape(self.history_length)[0]), self.history_length - 1], axis=1)
        final_states = tf.gather_nd(h_final, final_temporal_idx)
        final_sample_nlls = tf.gather_nd(sample_nlls, final_temporal_idx)
        self.final_states = tf.concat([final_states, final_sample_nlls], axis=1)

        self.prediction_tensors = {
            'user_ids': self.user_id,
            'final_states': self.final_states
        }

        return nll


if __name__ == '__main__':
    base_dir = './'

    dr = DataReader(data_dir=os.path.join(base_dir, 'data'))

    nn = rnn(
        reader=dr,
        log_dir=os.path.join(base_dir, 'logs_gmm'),
        checkpoint_dir=os.path.join(base_dir, 'checkpoints_gmm'),
        prediction_dir=os.path.join(base_dir, 'predictions_gmm'),
        optimizer='adam',
        learning_rate=.001,
        lstm_size=300,
        batch_size=128,
        num_training_steps=200000,
        early_stopping_steps=10000,
        warm_start_init_step=0,
        regularization_constant=0.0,
        keep_prob=1.0,
        enable_parameter_averaging=False,
        num_restarts=2,
        min_steps_to_checkpoint=1000,
        log_interval=20,
        num_validation_batches=4,
    )
    nn.fit()
    nn.restore()
    nn.predict()
