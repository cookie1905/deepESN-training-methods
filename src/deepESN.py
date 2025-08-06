import tensorflow as tf
import numpy as np

from src.layers import InputLayer, OutputLayer, Reservoir, ReshapeIA

class DeepEsn(tf.keras.Model):
    def __init__(self, input_size, output_size, reservoir_size, num_reservoirs,
                  rho=0.99, leaky_parameter=0.55, sparsity=0.1,  input_scaling=1.0):
        super(DeepEsn, self).__init__()

        self.num_reservoirs = num_reservoirs
        self.reservoir_size = reservoir_size
        
        self.input_layer = InputLayer(input_size, self.reservoir_size, input_scaling)

        reservoir = Reservoir(reservoir_size, input_size, rho, leaky_parameter, sparsity)
        self.reservoirs_layers = [reservoir for _ in range(num_reservoirs)]
        
        self.output_layer = OutputLayer(output_size, num_reservoirs, reservoir_size)

    def call(self, u):
        states = []
        x = self.input_layer(u)
        for reservoir in self.reservoirs_layers:
            x = reservoir(x)
            states.append(x)
        output = self.output_layer(tf.concat(states, axis=0))
        return output
    
    def train(self, X, y, learning_rate, num_epochs ):
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        loss_fn = tf.keras.losses.MeanSquaredError()
        training_loss = []
        for epoch in range(1, num_epochs + 1):
            total_loss = 0.0
            
            with tf.GradientTape() as tape:
                outputs = []
                
                for t in range(len(X)):
                    output = self.call(X[t])
                    outputs.append(output)
                
                outputs = tf.concat(outputs, axis=0)
                total_loss = loss_fn(y, outputs)
            
            grads = tape.gradient(total_loss, self.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.trainable_variables))

            print(f'Epoch {epoch }, Total Loss: {total_loss.numpy()}')
            training_loss.append(total_loss.numpy())

        return training_loss
    
    def test(self, X):
        predicted_outputs = []
        for t in range(len(X)):
            output = self.call(X[t])
            predicted_outputs.append(output)

        return np.array(predicted_outputs)


class DeepEsnIA(DeepEsn):
    def __init__(self, input_size, output_size, reservoir_size, num_reservoirs,
                  rho=0.99, leaky_parameter=0.55, sparsity=0.1,  input_scaling=1.0):
        
        super().__init__(input_size, output_size, reservoir_size, num_reservoirs,
                         rho, leaky_parameter, sparsity, input_scaling)

        self.num_reservoirs = num_reservoirs
        self.reservoir_size = reservoir_size
        
        self.input_layer = InputLayer(input_size, self.reservoir_size, input_scaling)

        reservoir = Reservoir(reservoir_size, input_size, rho, leaky_parameter, sparsity)
        self.reservoirs_layers = [reservoir for _ in range(num_reservoirs)]

        self.reshape_ia = [ReshapeIA(input_size, self.reservoir_size, input_scaling, trainable=False) 
                           for _ in range(num_reservoirs -1)]

        self.output_layer = OutputLayer(output_size, num_reservoirs, reservoir_size)

    def call(self, u):
        states = []
        x = self.input_layer(u)

        for i, reservoir in enumerate(self.reservoirs_layers):
            x = reservoir(x)
            states.append(x)
           
            if i < (self.num_reservoirs - 1):
                x = tf.concat([tf.reshape(u, (1, 1)), x], axis=0)
                x = self.reshape_ia[i](x)
                  
        output = self.output_layer(tf.concat(states, axis=0))
        return output