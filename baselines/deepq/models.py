import tensorflow as tf
import tensorflow.contrib.layers as layers


def _mlp(hiddens, inpt, num_actions, scope, reuse=False, layer_norm=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        for hidden in hiddens:
            out = layers.fully_connected(out, num_outputs=hidden, activation_fn=None)
            if layer_norm:
                out = layers.layer_norm(out, center=True, scale=True)
            out = tf.nn.relu(out)
        q_out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return q_out


def mlp(hiddens=[], layer_norm=False):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    hiddens: [int]
        list of sizes of hidden layers

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """
    return lambda *args, **kwargs: _mlp(hiddens, layer_norm=layer_norm, *args, **kwargs)

def _log_mlp(hiddens, inpt, num_actions, scope, reuse=False, layer_norm=False):
    print('log_mlp')
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        count = 0
        for hidden in hiddens:
            size = out.get_shape().as_list()[-1]
            log_w = tf.get_variable('log_w%d' % count, [size, hidden])
            w = tf.exp(log_w)
            b = tf.get_variable('b%d' % count, [hidden])
            out = tf.matmul(out, w) + b
            if layer_norm:
                out = layers.layer_norm(out, center=True, scale=True)
            out = tf.nn.relu(out)
            count += 1
        size = out.get_shape().as_list()[-1]
        log_w = tf.get_variable('log_w%d' % count, [size, num_actions])
        w = tf.exp(log_w)
        b = tf.get_variable('b%d' % count, [num_actions])
        q_out = tf.matmul(out, w) + b
        return q_out


def log_mlp(hiddens=[], layer_norm=False):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    hiddens: [int]
        list of sizes of hidden layers

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """
    return lambda *args, **kwargs: _log_mlp(hiddens, layer_norm=layer_norm, *args, **kwargs)

def _self_play_mlp(hiddens, inpt, num_actions, scope, reuse=False, layer_norm=False):
    # Self-play specific network.
    # This just hardcodes a bunch of Erdos specific stuff and it's awful.
    #
    # Explicitly divides the state in half, and then applies the same weights to
    # both halves, then merges the output into a final output vector.
    # We assume a linear model where all weights are guaranteed to be positive.
    # Otherwise it's hard to define an even-splitting attacker.
    print('self_play_mlp')
    with tf.variable_scope(scope, reuse=reuse):
        size = inpt.get_shape().as_list()[-1]
        log_w = tf.get_variable('log_w', [size // 2, 1])
        w = tf.exp(log_w)
        b = tf.get_variable('b', [1])
        set1, set2 = tf.split(inpt, num_or_size_splits=2, axis=1)
        v1 = tf.matmul(set1, w) + b
        v2 = tf.matmul(set2, w) + b
        q_out = tf.concat([v1, v2], axis=1)
        return q_out


def self_play_mlp(hiddens=[], layer_norm=False):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    hiddens: [int]
        list of sizes of hidden layers

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    w: tf.Tensor
        Tensor for the expoential of the variables.
    """
    return lambda *args, **kwargs: _self_play_mlp(hiddens, layer_norm=layer_norm, *args, **kwargs)


def _cnn_to_mlp(convs, hiddens, dueling, inpt, num_actions, scope, reuse=False, layer_norm=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu)
        conv_out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            action_out = conv_out
            for hidden in hiddens:
                action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
                if layer_norm:
                    action_out = layers.layer_norm(action_out, center=True, scale=True)
                action_out = tf.nn.relu(action_out)
            action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

        if dueling:
            with tf.variable_scope("state_value"):
                state_out = conv_out
                for hidden in hiddens:
                    state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=None)
                    if layer_norm:
                        state_out = layers.layer_norm(state_out, center=True, scale=True)
                    state_out = tf.nn.relu(state_out)
                state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
            q_out = state_score + action_scores_centered
        else:
            q_out = action_scores
        return q_out


def cnn_to_mlp(convs, hiddens, dueling=False, layer_norm=False):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    convs: [(int, int int)]
        list of convolutional layers in form of
        (num_outputs, kernel_size, stride)
    hiddens: [int]
        list of sizes of hidden layers
    dueling: bool
        if true double the output MLP to compute a baseline
        for action scores

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """

    return lambda *args, **kwargs: _cnn_to_mlp(convs, hiddens, dueling, layer_norm=layer_norm, *args, **kwargs)

