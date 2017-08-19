import tensorflow as tf


def lstm_layer(inputs, lengths, state_size, keep_prob=1.0, scope='lstm-layer', reuse=False, return_final_state=False):
    """
    LSTM layer.

    Args:
        inputs: Tensor of shape [batch size, max sequence length, ...].
        lengths: Tensor of shape [batch size].
        state_size: LSTM state size.
        keep_prob: 1 - p, where p is the dropout probability.

    Returns:
        Tensor of shape [batch size, max sequence length, state_size] containing the lstm
        outputs at each timestep.

    """
    with tf.variable_scope(scope, reuse=reuse):
        cell_fw = tf.contrib.rnn.DropoutWrapper(
            tf.contrib.rnn.LSTMCell(
                state_size,
                reuse=reuse
            ),
            output_keep_prob=keep_prob
        )
        outputs, output_state = tf.nn.dynamic_rnn(
            inputs=inputs,
            cell=cell_fw,
            sequence_length=lengths,
            dtype=tf.float32
        )
        if return_final_state:
            return outputs, output_state
        else:
            return outputs


def temporal_convolution_layer(inputs, output_units, convolution_width, causal=False, dilation_rate=[1], bias=True,
                               activation=None, dropout=None, scope='temporal-convolution-layer', reuse=False):
    """
    Convolution over the temporal axis of sequence data.

    Args:
        inputs: Tensor of shape [batch size, max sequence length, input_units].
        output_units: Output channels for convolution.
        convolution_width: Number of timesteps to use in convolution.
        causal: Output at timestep t is a function of inputs at or before timestep t.
        dilation_rate:  Dilation rate along temporal axis.

    Returns:
        Tensor of shape [batch size, max sequence length, output_units].

    """
    with tf.variable_scope(scope, reuse=reuse):
        if causal:
            shift = (convolution_width / 2) + (int(dilation_rate[0] - 1) / 2)
            pad = tf.zeros([tf.shape(inputs)[0], shift, inputs.shape.as_list()[2]])
            inputs = tf.concat([pad, inputs], axis=1)

        W = tf.get_variable(
            name='weights',
            initializer=tf.contrib.layers.variance_scaling_initializer(),
            shape=[convolution_width, shape(inputs, 2), output_units]
        )

        z = tf.nn.convolution(inputs, W, padding='SAME', dilation_rate=dilation_rate)
        if bias:
            b = tf.get_variable(
                name='biases',
                initializer=tf.constant_initializer(),
                shape=[output_units]
            )
            z = z + b
        z = activation(z) if activation else z
        z = tf.nn.dropout(z, dropout) if dropout is not None else z
        z = z[:, :-shift, :] if causal else z
        return z


def time_distributed_dense_layer(inputs, output_units, bias=True, activation=None, batch_norm=None,
                                 dropout=None, scope='time-distributed-dense-layer', reuse=False):
    """
    Applies a shared dense layer to each timestep of a tensor of shape [batch_size, max_seq_len, input_units]
    to produce a tensor of shape [batch_size, max_seq_len, output_units].

    Args:
        inputs: Tensor of shape [batch size, max sequence length, ...].
        output_units: Number of output units.
        activation: activation function.
        dropout: dropout keep prob.

    Returns:
        Tensor of shape [batch size, max sequence length, output_units].

    """
    with tf.variable_scope(scope, reuse=reuse):
        W = tf.get_variable(
            name='weights',
            initializer=tf.contrib.layers.variance_scaling_initializer(),
            shape=[shape(inputs, -1), output_units]
        )
        z = tf.einsum('ijk,kl->ijl', inputs, W)
        if bias:
            b = tf.get_variable(
                name='biases',
                initializer=tf.constant_initializer(),
                shape=[output_units]
            )
            z = z + b

        if batch_norm is not None:
            z = tf.layers.batch_normalization(z, training=batch_norm, reuse=reuse)

        z = activation(z) if activation else z
        z = tf.nn.dropout(z, dropout) if dropout is not None else z
        return z


def dense_layer(inputs, output_units, bias=True, activation=None, batch_norm=None, dropout=None,
                scope='dense-layer', reuse=False):
    """
    Applies a dense layer to a 2D tensor of shape [batch_size, input_units]
    to produce a tensor of shape [batch_size, output_units].

    Args:
        inputs: Tensor of shape [batch size, input_units].
        output_units: Number of output units.
        activation: activation function.
        dropout: dropout keep prob.

    Returns:
        Tensor of shape [batch size, output_units].

    """
    with tf.variable_scope(scope, reuse=reuse):
        W = tf.get_variable(
            name='weights',
            initializer=tf.contrib.layers.variance_scaling_initializer(),
            shape=[shape(inputs, -1), output_units]
        )
        z = tf.matmul(inputs, W)
        if bias:
            b = tf.get_variable(
                name='biases',
                initializer=tf.constant_initializer(),
                shape=[output_units]
            )
            z = z + b

        if batch_norm is not None:
            z = tf.layers.batch_normalization(z, training=batch_norm, reuse=reuse)

        z = activation(z) if activation else z
        z = tf.nn.dropout(z, dropout) if dropout is not None else z
        return z


def wavenet(x, dilations, filter_widths, skip_channels, residual_channels, scope='wavenet', reuse=False):
    """
    A stack of causal dilated convolutions with paramaterized residual and skip connections as described
    in the WaveNet paper (with some minor differences).

    Args:
        x: Input tensor of shape [batch size, max sequence length, input units].
        dilations: List of dilations for each layer.  len(dilations) is the number of layers
        filter_widths: List of filter widths.  Same length as dilations.
        skip_channels: Number of channels to use for skip connections.
        residual_channels: Number of channels to use for residual connections.

    Returns:
        Tensor of shape [batch size, max sequence length, len(dilations)*skip_channels].
    """
    with tf.variable_scope(scope, reuse=reuse):

        # wavenet uses 2x1 conv here
        inputs = time_distributed_dense_layer(x, residual_channels, activation=tf.nn.tanh, scope='x-proj')

        skip_outputs = []
        for i, (dilation, filter_width) in enumerate(zip(dilations, filter_widths)):
            dilated_conv = temporal_convolution_layer(
                inputs=inputs,
                output_units=2*residual_channels,
                convolution_width=filter_width,
                causal=True,
                dilation_rate=[dilation],
                scope='cnn-{}'.format(i)
            )
            conv_filter, conv_gate = tf.split(dilated_conv, 2, axis=2)
            dilated_conv = tf.nn.tanh(conv_filter)*tf.nn.sigmoid(conv_gate)

            output_units = skip_channels + residual_channels
            outputs = time_distributed_dense_layer(dilated_conv, output_units, scope='cnn-{}-proj'.format(i))
            skips, residuals = tf.split(outputs, [skip_channels, residual_channels], axis=2)

            inputs += residuals
            skip_outputs.append(skips)

        skip_outputs = tf.nn.relu(tf.concat(skip_outputs, axis=2))
        return skip_outputs


def sequence_log_loss(y, y_hat, sequence_lengths, max_sequence_length, eps=1e-15):
    """
    Calculates average log loss on variable length sequences.

    Args:
        y: Label tensor of shape [batch size, max_sequence_length, input units].
        y_hat: Prediction tensor, same shape as y.
        sequence_lengths: Sequence lengths.  Tensor of shape [batch_size].
        max_sequence_length: maximum length of padded sequence tensor.

    Returns:
        Log loss. 0-dimensional tensor.
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.minimum(tf.maximum(y_hat, eps), 1.0 - eps)
    log_losses = y*tf.log(y_hat) + (1.0 - y)*tf.log(1.0 - y_hat)
    sequence_mask = tf.cast(tf.sequence_mask(sequence_lengths, maxlen=max_sequence_length), tf.float32)
    avg_log_loss = -tf.reduce_sum(log_losses*sequence_mask) / tf.cast(tf.reduce_sum(sequence_lengths), tf.float32)
    return avg_log_loss


def sequence_rmse(y, y_hat, sequence_lengths, max_sequence_length):
    """
    Calculates RMSE on variable length sequences.

    Args:
        y: Label tensor of shape [batch size, max_sequence_length, input units].
        y_hat: Prediction tensor, same shape as y.
        sequence_lengths: Sequence lengths.  Tensor of shape [batch_size].
        max_sequence_length: maximum length of padded sequence tensor.

    Returns:
        RMSE. 0-dimensional tensor.
    """
    y = tf.cast(y, tf.float32)
    squared_error = tf.square(y - y_hat)
    sequence_mask = tf.cast(tf.sequence_mask(sequence_lengths, maxlen=max_sequence_length), tf.float32)
    avg_squared_error = tf.reduce_sum(squared_error*sequence_mask) / tf.cast(tf.reduce_sum(sequence_lengths), tf.float32)
    rmse = tf.sqrt(avg_squared_error)
    return rmse


def log_loss(y, y_hat, eps=1e-15):
    """
    Calculates log loss between two tensors.

    Args:
        y: Label tensor.
        y_hat: Prediction tensor

    Returns:
        Log loss. 0-dimensional tensor.
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.minimum(tf.maximum(y_hat, eps), 1.0 - eps)
    log_loss = -tf.reduce_mean(y*tf.log(y_hat) + (1.0 - y)*tf.log(1.0 - y_hat))
    return log_loss


def rank(tensor):
    """Get tensor rank as python list"""
    return len(tensor.shape.as_list())


def shape(tensor, dim=None):
    """Get tensor shape/dimension as list/int"""
    if dim is None:
        return tensor.shape.as_list()
    else:
        return tensor.shape.as_list()[dim]
