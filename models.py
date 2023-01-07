import tensorflow as tf
from layers import *
from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []
        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.lossPS = 0
        self.accuracy = 0
        self.optimizer = None
        self.optimizerPS = None
        self.opt_op = None
        self.opt_opPS = None

    def _build(self):
        raise NotImplementedError
    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden= layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]
        self.hidden = self.activations[-2]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}
        self._loss()
        self._accuracy()
        self.opt_op = self.optimizer.minimize(self.loss)
        self.opt_opPS = self.optimizer.minimize(self.lossPS)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)

def dense(x, n1, n2, name):
    """
    Used to create a dense layer.
    :param x: input tensor to the dense layer
    :param n1: no. of input neurons
    :param n2: no. of output neurons
    :param name: name of the entire dense layer.i.e, variable scope name.
    :return: tensor with shape [batch_size, n2]
    """
    with tf.variable_scope(name, reuse=None):
        # np.random.seed(1)
        tf.set_random_seed(1)
        x = tf.nn.dropout(x, 1-FLAGS.dropout)
        weights = tf.get_variable("weights", shape=[n1, n2],
                                  initializer=tf.random_normal_initializer(mean=0., stddev=0.01)
                                  )
        bias = tf.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))
        out = tf.add(tf.matmul(x, weights), bias, name='matmul')

        return out
def dense2(x, n1, n2, name):
        """
        Used to create a dense layer.
        :param x: input tensor to the dense layer
        :param n1: no. of input neurons
        :param n2: no. of output neurons
        :param name: name of the entire dense layer.i.e, variable scope name.
        :return: tensor with shape [batch_size, n2]
        """
        # with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(name, reuse=None):
            # np.random.seed(1)
            x = tf.cast(x, tf.float32)
            weights = tf.get_variable("weights", shape=[n1, n1],
                                             initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
            bias = tf.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))
            weights = tf.cast(weights, dtype=tf.float32)

            xw = tf.cast(tf.sparse_tensor_dense_matmul(x, weights), dtype=tf.float32)
            xprim = tf.cast(tf.sparse.to_dense(tf.sparse.transpose(x)), dtype=tf.float32)
            out = tf.add(tf.matmul(xw, xprim), bias,name='ss')
            return out
class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.localLoss = placeholders['localLoss']
        self.weights = placeholders['weights']
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.optimizerPS = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.build()
        self.predict()

    def _loss(self):

        # Cross entropy error
        self.loss= masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])
        for var in self.layers[0].vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        self.lossPS=masked_softmax_cross_entropy(self.outputs, self.localLoss,
                                                 self.placeholders['labels_mask'],
                                                 weights=tf.cast(self.weights,tf.float32))
        self.loss +=-self.lossPS
    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))
        # self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
        #                                     output_dim=FLAGS.hidden1,
        #                                     placeholders=self.placeholders,
        #                                     act=tf.nn.relu,
        #                                     dropout=True,
        #                                     logging=self.logging))
        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        self.pre=tf.nn.softmax(self.outputs)
        return self.pre

