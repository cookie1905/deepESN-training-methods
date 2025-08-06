import tensorflow as tf
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

from src.layers import InputLayer, OutputLayer, Reservoir, ReshapeIA

class DeepEsnDfa(tf.keras.Model):
    def __init__(self, input_size, output_size, reservoir_size, num_reservoirs,
                  rho=0.99, leaky_parameter=0.55, sparsity=0.1,  input_scaling=1.0):
        super(DeepEsnDfa, self).__init__()
        self.num_reservoirs = num_reservoirs
        self.reservoir_size = reservoir_size
       
        self.reservoir_outs = defaultdict(list)
        self.intermediate_outs = defaultdict(list)
        
        self.input_layer = InputLayer(input_size, self.reservoir_size, input_scaling)

        reservoir = Reservoir(reservoir_size, input_size, rho, leaky_parameter, sparsity)
        self.reservoirs_layers = [reservoir for _ in range(num_reservoirs)]

        self.intermediate_layer = [
            tf.keras.layers.Dense(self.reservoir_size, activation=None, use_bias=False, trainable=True) 
            for _ in range(num_reservoirs - 1)
        ]

        self.output_layer = OutputLayer(output_size, num_reservoirs, reservoir_size)

    def call(self, u):
        x = self.input_layer(u)
        reservoir_states = []

        for i, reservoir in enumerate(self.reservoirs_layers):
            x = reservoir(x)
            reservoir_states.append(x)
            self.reservoir_outs[f"res-{i}"].append(x)

            if i < self.num_reservoirs - 1:
                x = self.intermediate_layer[i](tf.transpose(x))
                x = tf.transpose(x)
                self.intermediate_outs[f"int-{i}"].append(x)
            
        output = self.output_layer(tf.concat(reservoir_states, axis=0))
        return output, reservoir_states
    
    def train(self, X, y, learning_rate, num_epochs):
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        loss_fn = tf.keras.losses.MeanSquaredError()
        training_loss = []
        reservoir_outputs = defaultdict(list)
        intermediate_outputs = defaultdict(list)

        feedback_weights = tf.random.normal([self.reservoir_size, 1])
        for epoch in range(1, num_epochs + 1):
            total_loss = 0.0
            
            with tf.GradientTape() as tape:
                outputs = []
                
                for t in range(len(X)):
                    output, reservoir_state = self.call(X[t])
                    outputs.append(output)
                
                outputs = tf.concat(outputs, axis=0)
                total_loss = loss_fn(y, outputs)

            if epoch % 10 == 0:
                reservoir_outputs[f"epoch-{epoch}"].append(self.reservoir_outs)
                intermediate_outputs[f"epoch-{epoch}"].append(self.intermediate_outs)
            
            self.reservoir_outs = defaultdict(list)
            self.intermediate_outs = defaultdict(list)
            
            grads = tape.gradient(total_loss, self.output_layer.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.output_layer.trainable_variables))

            for i in range(self.num_reservoirs - 1):
                W = self.intermediate_layer[i].weights
                grad = total_loss * feedback_weights
                delta_w = tf.matmul(grad, tf.transpose(reservoir_state[i]))
                W = W - learning_rate * delta_w
                self.intermediate_layer[i].set_weights(W)
            
            print(f'Epoch {epoch}, Total Loss: {total_loss.numpy()}')
            training_loss.append(total_loss.numpy())

        self.reservoir_outs = reservoir_outputs
        self.intermediate_outs = intermediate_outputs
        
        return training_loss
    
    def get_node_outs(self):
        return self.reservoir_outs, self.intermediate_outs
    
    def test(self, X):
        predicted_outputs = []
        for t in range(len(X)):
            output, _ = self.call(X[t])
            predicted_outputs.append(output)

        return np.array(predicted_outputs)
    
class DeepEsnIaDfa(tf.keras.Model):
    def __init__(self, input_size, output_size, reservoir_size, num_reservoirs,
                  rho=0.99, leaky_parameter=0.55, sparsity=0.1,  input_scaling=1.0):
        super(DeepEsnIaDfa, self).__init__()
        self.input_size = input_size
        self.num_reservoirs = num_reservoirs
        self.reservoir_size = reservoir_size
        self.reservoir_outs = defaultdict(list)
        self.intermediate_outs = defaultdict(list)

        self.input_layer = InputLayer(input_size, self.reservoir_size, input_scaling)

        reservoir = Reservoir(reservoir_size, input_size, rho, leaky_parameter, sparsity)
        self.reservoirs_layers = [reservoir for _ in range(num_reservoirs)]

        self.reshape_ia = [ReshapeIA(input_size, self.reservoir_size, input_scaling) 
                           for _ in range(num_reservoirs -1)]

        self.output_layer = OutputLayer(output_size, num_reservoirs, reservoir_size)

    def call(self, u):
        x = self.input_layer(u)
        reservoir_states = []
        

        for i, reservoir in enumerate(self.reservoirs_layers):
            x = reservoir(x)
            reservoir_states.append(x)
            self.reservoir_outs[f"res-{i}"].append(x)

            if i < self.num_reservoirs - 1:
                x = tf.concat([tf.reshape(u, (1, 1)), x], axis=0)
                x = self.reshape_ia[i](x)
                self.intermediate_outs[f"int-{i}"].append(x)
  
        output = self.output_layer(tf.concat(reservoir_states, axis=0))
        return output, reservoir_states
    
    def train(self, X, y, learning_rate, num_epochs):
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        loss_fn = tf.keras.losses.MeanSquaredError()
        training_loss = []
        reservoir_outputs = defaultdict(list)
        intermediate_outputs = defaultdict(list)
        feedback_weights = tf.random.normal([self.reservoir_size + self.input_size, 1])
        for epoch in range(num_epochs):
            total_loss = 0.0
            
            with tf.GradientTape() as tape:
                outputs = []
                
                for t in range(len(X)):
                    output, reservoir_state = self.call(X[t])
                    outputs.append(output)
                
                outputs = tf.concat(outputs, axis=0)
                total_loss = loss_fn(y, outputs)

            if epoch % 10 == 0:
                reservoir_outputs[f"epoch-{epoch}"].append(self.reservoir_outs)
                intermediate_outputs[f"epoch-{epoch}"].append(self.intermediate_outs)
            self.reservoir_outs = defaultdict(list)
            self.intermediate_outs = defaultdict(list)
            grads = tape.gradient(total_loss, self.output_layer.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.output_layer.trainable_variables))

            for i in range(self.num_reservoirs - 1):
                W = self.reshape_ia[i].weights
                grad = total_loss * feedback_weights
                delta_w =  tf.matmul(grad, tf.transpose(reservoir_state[i]))
                W = W - learning_rate * tf.transpose(delta_w)
                self.reshape_ia[i].set_weights(W)

            print(f'Epoch {epoch + 1}, Total Loss: {total_loss.numpy()}')
            training_loss.append(total_loss.numpy())
        
        self.reservoir_outs = reservoir_outputs
        self.intermediate_outs = intermediate_outputs
        return training_loss
    
    def get_node_outs(self):
        return self.reservoir_outs, self.intermediate_outs
    
    def test(self, X):
        predicted_outputs = []
        for t in range(len(X)):
            output, _ = self.call(X[t])
            predicted_outputs.append(output)

        return np.array(predicted_outputs)