import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from tensorflow.keras.datasets import mnist
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(8,10))

for i in range(10):
    idx = (y_train == i).argmax()
    axs[i//2,i%2].imshow(x_train[idx], cmap='gray')
    axs[i//2,i%2].set_title(f"Digit {i}")
    axs[i//2,i%2].axis('off')

fig.tight_layout()

num_classes = 10 
num_features = 784 

x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
x_train, x_test = x_train / 255., x_test / 255.

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
x_valid, x_test, y_valid, y_test = train_test_split(x_valid, y_valid, test_size=0.5, random_state=42)

learning_rate = 0.001
training_iters = 128000
batch_size = 128
display_step = 10

n_input = 784 
n_classes = 10 
dropout = 0.75 

x = tf.compat.v1.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) 

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def conv_net(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)

    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


x_train_full, x_test, y_train_full, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.11, random_state=42)

weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = conv_net(x, weights, biases, keep_prob)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

train_accuracy = []
train_loss = []
val_accuracy = []
val_loss = []

with tf.Session() as sess:
    y_train = sess.run(tf.one_hot(y_train, depth=n_classes))
    y_test = sess.run(tf.one_hot(y_test, depth=n_classes))
    y_val = sess.run(tf.one_hot(y_val, depth=n_classes))

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size <= training_iters:
        # Manual batching
        batch_start = ((step-1) * batch_size) % x_train.shape[0]
        batch_end = batch_start + batch_size
        batch_x = x_train[batch_start:batch_end]
        batch_y = y_train[batch_start:batch_end]
        
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        
        if step % display_step == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
            
            train_accuracy.append(acc)
            train_loss.append(loss)
            
            val_loss_, val_acc_ = sess.run([cost, accuracy], feed_dict={x: x_val, y: y_val, keep_prob: 1.})
            
            val_accuracy.append(val_acc_)
            val_loss.append(val_loss_)

            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc) + ", Validation Loss= " + \
                  "{:.6f}".format(val_loss_) + ", Validation Accuracy= " + \
                  "{:.5f}".format(val_acc_))
        step += 1
    print("Optimization Finished!")

    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: x_test[:256],
                                      y: y_test[:256],
                                      keep_prob: 1.}))
    
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Steps (in tens)')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Loss')
plt.xlabel('Steps (in tens)')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout() 
plt.savefig('Model Performance')
plt.show()