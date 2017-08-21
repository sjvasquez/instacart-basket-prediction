import tensorflow as tf
import numpy as np
import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_frame import DataFrame
from tf_utils import lstm_layer, time_distributed_dense_layer, dense_layer, log_loss
from tf_base_model import TFBaseModel


class DataReader(object):

    def __init__(self, data_dir):
        data_cols = [
            'user_id',
            'product_id',
            'aisle_id',
            'department_id',
            'is_ordered_history',
            'index_in_order_history',
            'order_dow_history',
            'order_hour_history',
            'days_since_prior_order_history',
            'order_size_history',
            'reorder_size_history',
            'order_number_history',
            'history_length',
            'product_name',
            'product_name_length',
            'eval_set',
            'label'
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
            batch['next_is_ordered'] = np.roll(batch['is_ordered_history'], -1, axis=1)
            batch['is_none'] = batch['product_id'] == 0
            if not is_test:
                batch['history_length'] = batch['history_length'] - 1
            yield batch


class rnn(TFBaseModel):

    def __init__(self, lstm_size=300, **kwargs):
        self.lstm_size = lstm_size
        super(rnn, self).__init__(**kwargs)

    def calculate_loss(self):
        x = self.get_input_sequences()
        return self.calculate_outputs(x)

    def get_input_sequences(self):
        self.user_id = tf.placeholder(tf.int32, [None])
        self.product_id = tf.placeholder(tf.int32, [None])
        self.aisle_id = tf.placeholder(tf.int32, [None])
        self.department_id = tf.placeholder(tf.int32, [None])
        self.is_none = tf.placeholder(tf.int32, [None])
        self.history_length = tf.placeholder(tf.int32, [None])

        self.is_ordered_history = tf.placeholder(tf.int32, [None, 100])
        self.index_in_order_history = tf.placeholder(tf.int32, [None, 100])
        self.order_dow_history = tf.placeholder(tf.int32, [None, 100])
        self.order_hour_history = tf.placeholder(tf.int32, [None, 100])
        self.days_since_prior_order_history = tf.placeholder(tf.int32, [None, 100])
        self.order_size_history = tf.placeholder(tf.int32, [None, 100])
        self.reorder_size_history = tf.placeholder(tf.int32, [None, 100])
        self.order_number_history = tf.placeholder(tf.int32, [None, 100])
        self.product_name = tf.placeholder(tf.int32, [None, 30])
        self.product_name_length = tf.placeholder(tf.int32, [None])
        self.next_is_ordered = tf.placeholder(tf.int32, [None, 100])

        self.keep_prob = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)

        # product data
        product_embeddings = tf.get_variable(
            name='product_embeddings',
            shape=[50000, self.lstm_size],
            dtype=tf.float32
        )
        aisle_embeddings = tf.get_variable(
            name='aisle_embeddings',
            shape=[250, 50],
            dtype=tf.float32
        )
        department_embeddings = tf.get_variable(
            name='department_embeddings',
            shape=[50, 10],
            dtype=tf.float32
        )
        product_names = tf.one_hot(self.product_name, 2532)
        product_names = tf.reduce_max(product_names, 1)
        product_names = dense_layer(product_names, 100, activation=tf.nn.relu)

        is_none = tf.cast(tf.expand_dims(self.is_none, 1), tf.float32)

        x_product = tf.concat([
            tf.nn.embedding_lookup(product_embeddings, self.product_id),
            tf.nn.embedding_lookup(aisle_embeddings, self.aisle_id),
            tf.nn.embedding_lookup(department_embeddings, self.department_id),
            is_none,
            product_names
        ], axis=1)
        x_product = tf.tile(tf.expand_dims(x_product, 1), (1, 100, 1))

        # user data
        user_embeddings = tf.get_variable(
            name='user_embeddings',
            shape=[207000, self.lstm_size],
            dtype=tf.float32
        )
        x_user = tf.nn.embedding_lookup(user_embeddings, self.user_id)
        x_user = tf.tile(tf.expand_dims(x_user, 1), (1, 100, 1))

        # sequence data
        is_ordered_history = tf.one_hot(self.is_ordered_history, 2)
        index_in_order_history = tf.one_hot(self.index_in_order_history, 20)
        order_dow_history = tf.one_hot(self.order_dow_history, 8)
        order_hour_history = tf.one_hot(self.order_hour_history, 25)
        days_since_prior_order_history = tf.one_hot(self.days_since_prior_order_history, 31)
        order_size_history = tf.one_hot(self.order_size_history, 60)
        reorder_size_history = tf.one_hot(self.reorder_size_history, 50)
        order_number_history = tf.one_hot(self.order_number_history, 101)

        index_in_order_history_scalar = tf.expand_dims(tf.cast(self.index_in_order_history, tf.float32) / 20.0, 2)
        order_dow_history_scalar = tf.expand_dims(tf.cast(self.order_dow_history, tf.float32) / 8.0, 2)
        order_hour_history_scalar = tf.expand_dims(tf.cast(self.order_hour_history, tf.float32) / 25.0, 2)
        days_since_prior_order_history_scalar = tf.expand_dims(tf.cast(self.days_since_prior_order_history, tf.float32) / 31.0, 2)
        order_size_history_scalar = tf.expand_dims(tf.cast(self.order_size_history, tf.float32) / 60.0, 2)
        reorder_size_history_scalar = tf.expand_dims(tf.cast(self.reorder_size_history, tf.float32) / 50.0, 2)
        order_number_history_scalar = tf.expand_dims(tf.cast(self.order_number_history, tf.float32) / 100.0, 2)

        x_history = tf.concat([
            is_ordered_history,
            index_in_order_history,
            order_dow_history,
            order_hour_history,
            days_since_prior_order_history,
            order_size_history,
            reorder_size_history,
            order_number_history,
            index_in_order_history_scalar,
            order_dow_history_scalar,
            order_hour_history_scalar,
            days_since_prior_order_history_scalar,
            order_size_history_scalar,
            reorder_size_history_scalar,
            order_number_history_scalar,
        ], axis=2)

        x = tf.concat([x_history, x_product, x_user], axis=2)

        return x

    def calculate_outputs(self, x):
        h = lstm_layer(x, self.history_length, self.lstm_size, scope='lstm-1')
        h = tf.concat([h, x], axis=2)
        h_final = time_distributed_dense_layer(h, 50, activation=tf.nn.relu, scope='dense-1')

        n_components = 1
        params = time_distributed_dense_layer(h_final, n_components*2, scope='dense-2', activation=None)
        ps, mixing_coefs = tf.split(params, 2, axis=2)

        # this is implemented incorrectly, but it still helped...
        mixing_coefs = tf.nn.softmax(mixing_coefs - tf.reduce_min(mixing_coefs, 2, keep_dims=True))
        ps = tf.nn.sigmoid(ps)

        labels = tf.tile(tf.expand_dims(self.next_is_ordered, 2), (1, 1, n_components))
        losses = tf.reduce_sum(mixing_coefs*log_loss(labels, ps), axis=2)
        sequence_mask = tf.cast(tf.sequence_mask(self.history_length, maxlen=100), tf.float32)
        avg_loss = tf.reduce_sum(losses*sequence_mask) / tf.cast(tf.reduce_sum(self.history_length), tf.float32)

        final_temporal_idx = tf.stack([tf.range(tf.shape(self.history_length)[0]), self.history_length - 1], axis=1)
        self.final_states = tf.gather_nd(h_final, final_temporal_idx)

        self.prediction_tensors = {
            'user_ids': self.user_id,
            'product_ids': self.product_id,
            'final_states': self.final_states
        }

        return avg_loss


if __name__ == '__main__':
    base_dir = './'

    dr = DataReader(data_dir=os.path.join(base_dir, 'data'))

    nn = rnn(
        reader=dr,
        log_dir=os.path.join(base_dir, 'logs_bmm'),
        checkpoint_dir=os.path.join(base_dir, 'checkpoints_bmm'),
        prediction_dir=os.path.join(base_dir, 'predictions_bmm'),
        optimizer='adam',
        learning_rate=.001,
        lstm_size=300,
        batch_size=128,
        num_training_steps=200000,
        early_stopping_steps=20000,
        warm_start_init_step=0,
        regularization_constant=0.0,
        keep_prob=1.0,
        enable_parameter_averaging=False,
        num_restarts=2,
        min_steps_to_checkpoint=5000,
        log_interval=20,
        num_validation_batches=4,
    )
    nn.fit()
    nn.restore()
    nn.predict()
