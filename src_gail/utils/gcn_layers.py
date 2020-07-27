import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = dropout

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolutionLayer(object):
    """Graph convolution layer."""
    uid = 1
    def __init__(self, input_dim, output_dim, support, dropout=0.,
                 act=tf.nn.relu, bias=False, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(self.__class__.uid)
            self.__class__.uid += 1
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.dropout = dropout
        self.act = act
        self.support = support if isinstance(support, tf.Tensor) else tf.constant(support,dtype=tf.float32)
        self.bias = bias

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                                        name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        pre_sup = tf.matmul(x, self.vars['weights'])
        support = tf.matmul(self.support, pre_sup)
        output = support

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class GraphConvolutionLayerBatch(object):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, support, dropout=0.,
                 act=tf.nn.relu, bias=False, reuse=False, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.dropout = dropout
        self.act = act
        self.support = support if isinstance(support, tf.Tensor) else tf.constant(support,dtype=tf.float32)
        self.bias = bias
        self.node_num = self.support.get_shape().as_list()[0]

        with tf.variable_scope(self.name):
            self.vars['weights'] = tf.get_variable(name='weights', shape=[self.node_num, input_dim, output_dim],initializer=tf.glorot_uniform_initializer)
            # self.vars['weights'] = glorot([self.node_num, input_dim, output_dim], name='weights') 
            if self.bias:
                self.vars['bias'] = tf.get_variable(name='bias', shape=[self.node_num, output_dim],initializer=tf.zeros_initializer)
                # self.vars['bias'] = zeros([self.node_num, output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = tf.expand_dims(inputs, -2) #(N, G, 1, in_dim)

        # dropout
        x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        #(N, G, 1, in_dim) x (1, G, in_dim, out_dim) = (N, G, 1, out_dim)
        pre_sup = tf.matmul(x, tf.expand_dims(self.vars['weights'], 0)) 
        #(1, G, G) x (N, G, out_dim) = (N, G, out_dim)
        support = tf.matmul(tf.expand_dims(self.support, axis=0), tf.squeeze(pre_sup, axis=2))
        output = support

        # bias
        if self.bias:
            output += tf.expand_dims(self.vars['bias'], axis=0)

        return self.act(output)


    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class MLP(object):
    def __init__(self, input_dim, hidden_dim, node_num, dropout=0.,
                act=tf.nn.relu, **kwargs):

        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.dropout = dropout
        self.act = act
        self.node_num = node_num
        
        with tf.variable_scope(self.name + '_vars'):
            # (G,M,H)
            self.vars['weights_layer_1'] = tf.get_variable(name='weights_layer_1', shape=[self.node_num, input_dim, hidden_dim],initializer=tf.glorot_uniform_initializer)
            # self.vars['weights_layer_1'] = glorot([self.node_num, input_dim, hidden_dim], name='weights_layer_1')
            self.vars['weights_layer_2'] = tf.get_variable(name='weights_layer_2', shape=[self.node_num, hidden_dim, hidden_dim],initializer=tf.glorot_uniform_initializer)
            # (1,G,1,H)
            self.vars['bias_layer_1'] = tf.get_variable(name='bias_layer_1', shape=[1, self.node_num, 1, hidden_dim],initializer=tf.zeros_initializer)
            # self.vars['bias_layer_2'] = zeros([1, self.node_num, 1, hidden_dim], name='bias_layer_2')
            self.vars['bias_layer_2'] = tf.get_variable(name='bias_layer_2', shape=[1, self.node_num, 1, hidden_dim],initializer=tf.zeros_initializer)

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = tf.expand_dims(inputs, -2) #(N, G, 1, in_dim)

        # dropout
        x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        #(N, G, 1, in_dim) x (1, G, in_dim, H) + (1, G, 1, H)= (N, G, 1, H)
        p_h1 = tf.matmul(x, tf.expand_dims(self.vars['weights_layer_1'], 0)) + self.vars['bias_layer_1']
        p_h1 = tf.nn.dropout(self.act(p_h1), 1-self.dropout)
        #(N, G, 1, H) x (1, G, H, H) + (1, G, 1, H)= (N, G, 1, H)
        p_h2 = tf.matmul(p_h1, tf.expand_dims(self.vars['weights_layer_2'], 0)) + self.vars['bias_layer_2']
        output = self.act(p_h2)
        output = tf.squeeze(output, axis=2)#(N, G, H)
        
        return output 

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])

if __name__ == "__main__":
    support = np.ones((6,6))
    inputs = tf.placeholder(tf.float32, shape=(128,6,2), name="x")
    layers = []
    activations = []

    for i in range(3):
        # layers.append(GraphConvolutionLayer(2,2,support))
        layers.append(GraphConvolutionLayerBatch(2,2,support))


    # Build sequential layer model
    activations.append(inputs)
    for layer in layers:
        hidden = layer(activations[-1])
        activations.append(hidden)
    outputs = activations[-1]
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())   
    feed_dict = dict()
    feed_dict.update({inputs:np.ones((128,6,2))})
    out = sess.run(outputs,feed_dict=feed_dict)    
    pass
