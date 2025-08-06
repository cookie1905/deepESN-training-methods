import tensorflow as tf
import numpy as np
from src.deepESN import DeepEsn, DeepEsnIA
from src.deepESN_bptt import DeepEsnBptt, DeepEsnIaBptt
from src.deepESN_target import DeepEsnTarget, DeepEsnIaTarget
from src.deepESN_dfa import DeepEsnDfa, DeepEsnIaDfa
import matplotlib.pyplot as plt

emg = np.load("dataset\emg.npy")
time = np.arange(0, 1600, 1)

data_train = tf.convert_to_tensor(emg[0: 1299], dtype=tf.float32)[:, np.newaxis]
data_target = tf.convert_to_tensor(emg[1: 1300], dtype=tf.float32)[:, np.newaxis]

data_test = tf.convert_to_tensor(emg[1299: 1599], dtype=tf.float32)[:, np.newaxis]
data_test_target = tf.convert_to_tensor(emg[1300: 1600], dtype=tf.float32)[:, np.newaxis]


deep_esn = DeepEsnIaDfa(1, 1, 25, 4)
training_loss = deep_esn.train(data_train, data_target, 0.01, 11)
predicted_data = deep_esn.test(data_test)

plt.figure(figsize=(12, 6))
plt.plot(time[0: 1300], emg[0: 1300], label='Train', color='green')
plt.plot(time[1300: 1600], data_test.numpy(), label='Test', color='blue')
plt.plot(time[1300: 1600], np.transpose(predicted_data).flatten(), label='Predicted', color='red')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
plt.close() 


res,int = deep_esn.get_node_outs()

indices_to_plot = [0, 4, 9, 14, 19, 24]
legend_labels = [i + 1 for i in indices_to_plot]

sequences = []
for idx in indices_to_plot:
    sequence = [tensor[idx, 0] for tensor in res["epoch-10"][0]["res-1"]]
    sequences.append(sequence)

plt.figure(figsize=(12, 6))
for i, sequence in enumerate(sequences):
    plt.plot(sequence, label=f'Node {legend_labels[i ]}')

plt.title("epoch-10 reservoir-2")
plt.xlabel('Timestep')
plt.ylabel('Value')
plt.legend()
plt.legend(loc='upper right')
plt.grid(color='lightgray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.show()

sequences = []
for idx in indices_to_plot:
    sequence = [tensor[idx, 0] for tensor in int["epoch-10"][0]["int-0"]]
    sequences.append(sequence)

plt.figure(figsize=(12, 6))
for i, sequence in enumerate(sequences):
    plt.plot(sequence, label=f'Node {legend_labels[i ]}')

plt.title("epoch-10 intermediate-1")
plt.xlabel('Timestep')
plt.ylabel('Value')
plt.legend(loc='upper right')
plt.grid(color='lightgray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.show()
