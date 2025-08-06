import tensorflow as tf

class InputLayer(tf.keras.layers.Layer):
    def __init__(self, input_size, reservoir_size, input_scaling=1.0):
        super(InputLayer, self).__init__()
        self.input_weights = self.add_weight(
            shape=(reservoir_size, input_size + 1),
            initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0),
            trainable=False
        )
        self.input_weights.assign(self.input_weights * input_scaling)

    def call(self, inputs):
        inputs = tf.reshape(tf.concat([inputs, tf.ones((1,))], axis=0), (-1, 1))
        return tf.matmul(self.input_weights, inputs)
    

class Reservoir(tf.keras.layers.Layer):
    def __init__(self, reservoir_size, input_size, rho=0.99, leaky_parameter=0.5, sparsity=0.1):
        super(Reservoir, self).__init__()
        self.reservoir_size = reservoir_size
        self.input_size = input_size
        self.rho = rho
        self.leaky_parameter = leaky_parameter
        self.state = tf.zeros((reservoir_size, input_size))

        reservoir_weights = tf.random.uniform((reservoir_size, reservoir_size), minval=0, maxval=1.0)
        sparse_mask = tf.random.uniform((reservoir_size, reservoir_size), minval=-1, maxval=1.0) < sparsity
        sparse_weights = reservoir_weights * tf.cast(sparse_mask, tf.float32)

        # rescale sparse weights
        eigenvalues = tf.linalg.eigvals(sparse_weights)
        weight_spectral_radius = tf.reduce_max(tf.abs(eigenvalues))
        scale_factor = self.rho / weight_spectral_radius
        reservoir_weights = sparse_weights * scale_factor

        self.reservoir_weights = self.add_weight(
            shape=(reservoir_size, reservoir_size),
            initializer=tf.keras.initializers.Constant(
                reservoir_weights
            ),
            trainable=False
        )

    def call(self, x):
        self.state = ((1 - self.leaky_parameter) * self.state + self.leaky_parameter 
                      * tf.tanh(tf.matmul(self.reservoir_weights, self.state) + x))
        return self.state
    
class ReshapeIA(tf.keras.layers.Layer):
    def __init__(self, input_size, reservoir_size, input_scaling=1.0, trainable = True):
        super(ReshapeIA, self).__init__()
        self.input_weights = self.add_weight(
            shape=(reservoir_size, input_size + reservoir_size),
            initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0),
            trainable=trainable
        )
        self.input_weights.assign(self.input_weights * input_scaling)

    def call(self, inputs):
        return tf.matmul(self.input_weights, inputs)
    

class OutputLayer(tf.keras.layers.Layer):
    def __init__(self, output_size, num_reservoirs, reservoir_size, input_scaling=1.0):
        super(OutputLayer, self).__init__()
        self.input_weights = self.add_weight(
            shape=(output_size, num_reservoirs * reservoir_size),
            initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0),
            trainable=True
        )
        self.input_weights.assign(self.input_weights * input_scaling)

    def call(self, inputs):
        return tf.matmul(self.input_weights , inputs)