import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn import preprocessing


def loadData():
    # load  data
    total = pd.read_csv('data.csv')
    total_data = total.values
    np.random.shuffle(total_data)

    total_y = total_data[:,1]
    total_X = total_data[:,2:-1]


    #malignant 1 else 0
    for i in range(len(total_y)):
        if total_y[i] == 'M':
            total_y[i] = 1
        else:
            total_y[i] = 0

    # print(total_X)
    total_X = pd.DataFrame(total_X)
    result = total_X.copy()
    for feature_name in total_X.columns:
        max_value = total_X[feature_name].max()
        min_value = total_X[feature_name].min()
        result[feature_name] = (total_X[feature_name] - min_value) / (max_value - min_value)
    print(result)

    total_X = total_X.values


    return total_X, total_y

total_X, total_y = loadData()

# Python optimisation variables
learning_rate = 0.0005
epochs = 1000
batch_size = 100

# input 28 x 28  = 784
x = tf.placeholder(tf.float32, [None, 30])
# output 10 classes
y = tf.placeholder(tf.float32, [None, 2])

# weight and biased from input to hidden layer
weight1 = tf.Variable(tf.random_normal([30, 20], stddev=0.03), name='weight1')
bias1 = tf.Variable(tf.random_normal([20]), name='bias1')
# weight and biased from hidden layer to output
weight2 = tf.Variable(tf.random_normal([20, 2], stddev=0.03), name='weight2')
biase2 = tf.Variable(tf.random_normal([2]), name='biase2')

# calculate the output in hidden layer
h_output = tf.add(tf.matmul(x, weight1), bias1)
h_output = tf.nn.sigmoid(h_output)

# output layer using softmax 
prediction = tf.nn.softmax(tf.add(tf.matmul(h_output, weight2), biase2))
cross_entropy = tf.losses.mean_squared_error(y, prediction)

# use optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

#initialize 
init_op = tf.global_variables_initializer()

# accuracy assesment
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#apply crossvalidation k=5
kf = KFold(n_splits=5)

average_accuracy = []

number = 0
# start session for training
with tf.Session() as sess:
  #define cost list for plotting the learning curve
  costs = []

  for training_id, test_id in kf.split(total_X):
    train_x, test_x = total_X[training_id], total_X[test_id]
    train_y, test_y = total_y[training_id], total_y[test_id]

    train_y = tf.one_hot(indices=train_y, depth=2)
    test_y = tf.one_hot(indices=test_y, depth=2)

    #run iterations
    sess.run(init_op)

    #total batches
    total_batch = int(len(train_x) / batch_size)

    #run through epoches
    for epoch in range(epochs):
      average_cost = 0

      #get costs
      for index, offset in enumerate(range(0, len(train_x), batch_size)):
        b_x, b_y = train_x[offset:offset+batch_size], train_y[offset:offset+batch_size].eval()
        _, cost = sess.run([optimizer, cross_entropy], feed_dict={x: b_x, y: b_y})
        
        average_cost += cost / total_batch
      print("Epoch:", (epoch + 1), "cost function output =", "{:.5f}".format(average_cost))

      #cost curve
      costs.append(average_cost)

    plt.plot(costs)
    plt.xlabel("iterations")
    plt.ylabel("cost")
    plt.savefig("plot.png")

    number = number+1

    acu = sess.run(accuracy, feed_dict={x: test_x, y: test_y.eval()})
    print(acu)
    average_accuracy.append(acu)

#take average of all accuracies  
print(sum(average_accuracy)/len(average_accuracy))
