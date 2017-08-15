import os

import numpy as np
import tensorflow as tf

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_frame import DataFrame
from tf_base_model import TFBaseModel
from tf_utils import dense_layer, log_loss


class DataReader(object):

    def __init__(self, data_dir):
        data_cols = [
            'order_id',
            'product_id',
            'features',
            'label'
        ]
        data = [np.load(os.path.join(data_dir, '{}.npy'.format(i)), mmap_mode='r') for i in data_cols]
        df = DataFrame(columns=data_cols, data=data)
        self.data_dim = df['features'].shape[1]

        print df.shapes()
        print 'loaded data'

        self.test_df = df.mask(df['label'] == -1)
        self.train_df = df.mask(df['label'] != -1)
        self.train_df, self.val_df = self.train_df.train_test_split(train_size=0.9)

        print 'train size', len(self.train_df)
        print 'val size', len(self.val_df)
        print 'test size', len(self.test_df)

        self.feature_means = np.load(os.path.join(data_dir, 'feature_means.npy'))
        self.feature_maxs = np.load(os.path.join(data_dir, 'feature_maxs.npy'))
        self.feature_mins = np.load(os.path.join(data_dir, 'feature_mins.npy'))

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
            batch['features'] = np.nan_to_num((batch['features'] - self.feature_means) / (self.feature_maxs - self.feature_mins))
            yield batch


class nn(TFBaseModel):

    def __init__(self, hidden_units=500, **kwargs):
        self.hidden_units = hidden_units
        super(nn, self).__init__(**kwargs)

    def calculate_loss(self):
        self.order_id = tf.placeholder(tf.int32, [None])
        self.product_id = tf.placeholder(tf.int32, [None])
        self.features = tf.placeholder(tf.float32, [None, self.reader.data_dim])
        self.label = tf.placeholder(tf.int32, [None])

        h = dense_layer(self.features, self.hidden_units, activation=tf.nn.relu, scope='dense1')
        h = tf.concat([h, self.features], axis=1)
        y_hat = tf.squeeze(dense_layer(h, 1, activation=tf.nn.sigmoid, scope='dense2'), 1)
        loss = log_loss(self.label, y_hat)

        self.prediction_tensors = {
            'order_ids': self.order_id,
            'product_ids': self.product_id,
            'predictions': y_hat,
            'labels': self.label
        }

        return loss

if __name__ == '__main__':
    base_dir = './'

    dr = DataReader(data_dir=os.path.join(base_dir, 'data'))

    nn = nn(
        reader=dr,
        log_dir=os.path.join(base_dir, 'logs_nn'),
        checkpoint_dir=os.path.join(base_dir, 'checkpoints_nn'),
        prediction_dir=os.path.join(base_dir, 'predictions_nn'),
        optimizer='adam',
        learning_rate=.005,
        hidden_units=1024,
        batch_size=4096,
        num_training_steps=15000,
        early_stopping_steps=5000,
        warm_start_init_step=0,
        regularization_constant=0.0,
        keep_prob=1.0,
        enable_parameter_averaging=False,
        num_restarts=0,
        min_steps_to_checkpoint=100,
        log_interval=20,
        num_validation_batches=8,
    )
    nn.fit()
    nn.restore()
    nn.predict()
