import tensorflow as tf
import numpy as np
from collections import defaultdict

from src.layers import InputLayer, OutputLayer, Reservoir, ReshapeIA

class DeepEsnTarget(tf.keras.Model):
    def __init__(self, input_size, output_size, reservoir_size, num_reservoirs,
                  rho=0.99, leaky_parameter=0.55, sparsity=0.1,  input_scaling=1.0):
        super(DeepEsnTarget, self).__init__()
        self.num_reservoirs = num_reservoirs
        self.reservoir_size = reservoir_size
       
        self.reservoir_outs = defaultdict(list)
        self.intermediate_outs = defaultdict(list)
        
        self.input_layer = InputLayer(input_size, self.reservoir_size, input_scaling)

        reservoir = Reservoir(reservoir_size, input_size, rho, leaky_parameter, sparsity)
        self.reservoirs_layers = [reservoir for _ in range(num_reservoirs)]


        self.intermediate_layer = [
            tf.keras.layers.Dense(self.reservoir_size, activation=None, use_bias=True, trainable=True) 
            for _ in range(num_reservoirs - 1)
        ]

        self.intermediate_output_layer = [
            tf.keras.layers.Dense(output_size, use_bias=True, trainable=True)
            for _ in range(num_reservoirs - 1)
        ]

        self.output_layer = OutputLayer(output_size, num_reservoirs, reservoir_size)

    def call(self, u):
        states = []
        outs = []
        x = self.input_layer(u)
        
        for i, reservoir in enumerate(self.reservoirs_layers):
            x = reservoir(x)
            states.append(x)
            self.reservoir_outs[f"res-{i}"].append(x)
            

            if i < (self.num_reservoirs - 1):
                x = self.intermediate_layer[i](tf.transpose(x))
                x = tf.transpose(x)
                self.intermediate_outs[f"int-{i}"].append(x)
                o =  self.intermediate_output_layer[i](tf.transpose(x))
                outs.append(o)
                  
        output = self.output_layer(tf.concat(states, axis=0))
        return output, outs
    
    def train(self, X, y, learning_rate, num_epochs):
        output_optimizer = tf.keras.optimizers.Adam(learning_rate)
        reservoir_optimizer = tf.keras.optimizers.Adam(learning_rate)
        loss_fn = tf.keras.losses.MeanSquaredError()
        training_loss = []
        loss_at_layers = defaultdict(list)
        reservoir_outputs = defaultdict(list)
        intermediate_outputs = defaultdict(list)

        for epoch in range(1, num_epochs + 1):
            final_loss = 0.0
            intermediate_loss = []

            with tf.GradientTape(persistent=True) as tape:
                outs = []
                final_outs = []
                for t in range(len(X)):
                    output, intermediate_output = self.call(X[t])
                    final_outs.append(output)
                    outs.append(intermediate_output)
                
                final_outs = tf.concat(final_outs, axis=0)
                final_loss = loss_fn(y, final_outs)
                loss_at_layers['output'].append(final_loss)

                for i in range(self.num_reservoirs - 1):
                    out = tf.stack([sublist[i] for sublist in outs])
                    loss = loss_fn(y, out)
                    loss_at_layers[f'intermediate_{i}'].append(loss)
                    intermediate_loss.append(loss)
                    
            if epoch % 10 == 0:
                reservoir_outputs[f"epoch-{epoch}"].append(self.reservoir_outs)
                intermediate_outputs[f"epoch-{epoch}"].append(self.intermediate_outs)

            self.reservoir_outs = defaultdict(list)
            self.intermediate_outs = defaultdict(list)
        
            grads = tape.gradient(final_loss, self.output_layer.trainable_variables)
            output_optimizer.apply_gradients(zip(grads, self.output_layer.trainable_variables))

            for i, loss in enumerate(intermediate_loss):

                intermediate_w = self.intermediate_layer[i].trainable_variables
                intermediate_o = self.intermediate_output_layer[i].trainable_variables
                downsample_grads = tape.gradient(loss,  intermediate_w + intermediate_o)
                reservoir_optimizer.apply_gradients(zip(downsample_grads,  intermediate_w + intermediate_o))

            print(f'Epoch {epoch }, Total Loss: {final_loss.numpy()}')
            training_loss.append(final_loss.numpy())

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
    

class DeepEsnIaTarget(tf.keras.Model):
    def __init__(self, input_size, output_size, reservoir_size, num_reservoirs,
                  rho=0.99, leaky_parameter=0.55, sparsity=0.1,  input_scaling=1.0):
        super(DeepEsnIaTarget, self).__init__()
        self.num_reservoirs = num_reservoirs
        self.reservoir_size = reservoir_size
        self.reservoir_outs = defaultdict(list)
        self.intermediate_outs = defaultdict(list)
        
        self.input_layer = InputLayer(input_size, self.reservoir_size, input_scaling)

        reservoir = Reservoir(reservoir_size, input_size, rho, leaky_parameter, sparsity)
        self.reservoirs_layers = [reservoir for _ in range(num_reservoirs)]

        self.reshape_ia = [ReshapeIA(input_size, self.reservoir_size, input_scaling) 
                           for _ in range(num_reservoirs -1)]

        self.intermediate_output_layer = [
            tf.keras.layers.Dense(output_size, use_bias=True, trainable=True)
            for _ in range(num_reservoirs - 1)
        ]

        self.output_layer = OutputLayer(output_size, num_reservoirs, reservoir_size)

    def call(self, u):
        outs = []
        states = []
        x = self.input_layer(u)
       
        for i, reservoir in enumerate(self.reservoirs_layers):
            x = reservoir(x)
            states.append(x)
            self.reservoir_outs[f"res-{i}"].append(x)
        
            if i < (self.num_reservoirs - 1):
                x = tf.concat([tf.reshape(u, (1, 1)), x], axis=0)
                x = self.reshape_ia[i](x)
                self.intermediate_outs[f"int-{i}"].append(x)
                o = self.intermediate_output_layer[i](tf.transpose(x))
                outs.append(o)
                
     
        output = self.output_layer(tf.concat(states, axis=0))
        return output, outs
    
    def train(self, X, y, learning_rate, num_epochs):
        output_optimizer = tf.keras.optimizers.Adam(learning_rate)
        reservoir_optimizer = tf.keras.optimizers.Adam(learning_rate)
        loss_fn = tf.keras.losses.MeanSquaredError()
        training_loss = []
        loss_at_layers = defaultdict(list)
        reservoir_outputs = defaultdict(list)
        intermediate_outputs = defaultdict(list)

        for epoch in range(1, num_epochs + 1):
            final_loss = 0.0
            intermediate_loss = []

            with tf.GradientTape(persistent=True) as tape:
                outs = []
                final_outs = []
                for t in range(len(X)):
                    output, intermediate_output = self.call(X[t])
                    final_outs.append(output)
                    outs.append(intermediate_output)
                
                final_outs = tf.concat(final_outs, axis=0)
                final_loss = loss_fn(y, final_outs)
                loss_at_layers['output'].append(final_loss)

                for i in range(self.num_reservoirs - 1):
                    out = tf.stack([sublist[i] for sublist in outs])
                    loss = loss_fn(y, out)
                    loss_at_layers[f'intermediate_{i}'].append(loss)
                    intermediate_loss.append(loss)
                    
            if epoch % 10 == 0:
                reservoir_outputs[f"epoch-{epoch}"].append(self.reservoir_outs)
                intermediate_outputs[f"epoch-{epoch}"].append(self.intermediate_outs)

            self.reservoir_outs = defaultdict(list)
            self.intermediate_outs = defaultdict(list)
        
            grads = tape.gradient(final_loss, self.output_layer.trainable_variables)
            output_optimizer.apply_gradients(zip(grads, self.output_layer.trainable_variables))

            for i, loss in enumerate(intermediate_loss):

                intermediate_w = self.reshape_ia[i].trainable_variables
                intermediate_o = self.intermediate_output_layer[i].trainable_variables
                downsample_grads = tape.gradient(loss,  intermediate_w + intermediate_o)
                reservoir_optimizer.apply_gradients(zip(downsample_grads,  intermediate_w + intermediate_o))

            print(f'Epoch {epoch }, Total Loss: {final_loss.numpy()}')
            training_loss.append(final_loss.numpy())

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
