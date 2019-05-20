import tensorflow as tf
import numpy as np


class Transformer:
    def __init__(self, hp):
        self.hp = hp
        self.token2idx, self.idx2token = load_vocal(hp.vocab)
        self.embeddings = get_token_embeddings(hp.vocab_size, hp.d_model, zero_pad=True)

    def encode(self, xs, training=True):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            x, seqlens, sents1 = xs

            enc = tf.nn.embedding_lookup(self.embeddings, x)
            enc *= self.hp.d_model # rescale
            enc += positional_encoding(enc, self.hp.maxlen1)
            enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=training)

            # blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope('num_blocks_{}'.format(i), reuse=tf.AUTO_REUSE):
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False)
                    enc = ff(enc, num_units=[self.hp.d_ff, self.hp.d_model])
            memory = enc
            return memory, sents1


def positional_encoding(inputs, maxlen, masking=True, scope='positional_encoding'):
    E = inputs.get_shape().as_list()[-1] # static
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1] # dynamic
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) # (N, T)

        position_enc = np.array([
            [pos / np.power(10000, (i-i%2)/E) for i in range(E)]
            for pos in range(maxlen)
        ])

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])

        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)
        
        return tf.to_float(outputs)


def multihead_attention(queries, keys, values,
                        num_heads=8, 
                        dropout_rate=0, 
                        training=True,
                        causality=False,
                        scope='multihead_attention'):
    d_model = queries.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        Q = tf.layers.dense(queries, d_model, use_bias=False)
        K = tf.layers.dense(keys, d_model, bias=False)
        V = tf.layers.dense(values, d_model, bias=False)

        # split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), aixs=0) # (N*h, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (N*h, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (N*h, T_k, d_model/h)

        # Attention
        outputs = scaled_dot_product_attention(Q_, K_, V_, causality, dropout_rate, training)

        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), aixs=2)

        outputs += queries
        outputs = ln(outputs)
    return outputs


def scaled_dot_product_attention(Q, K, V,
                                 causality=False, dropout_rate=0.,
                                 training=True,
                                 scope='scaled_dot_product_attention'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]

        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
        outputs /= d_k ** 0.5
        
        # key masking
        outputs = mask(outputs, Q, K, type='key')

        if causality:
            outputs = mask(outputs, type='future')
        
        outputs = tf.nn.softmax(outputs)
        attention = tf.transpose(outputs, [0, 2, 1])
        tf.summary.image('attention', tf.expand_dims(attention[:1], -1))

        outputs = mask(outputs, Q, K, type='query')
        outputs= tf.layers.dropout(outputs, rate=dropout_rate, training=training)

        outputs = tf.matmul(outputs, V)
    
    return outputs


def ff(inputs, num_units, scope='positionwise_feedforward'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)

        outputs = tf.layers.dense(outputs, num_units[1])

        outputs += inputs

        outputs = ln(outputs)
    
    return outputs

def ln(inputs, epsilon=1e-8, scope='ln'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1: ]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs
