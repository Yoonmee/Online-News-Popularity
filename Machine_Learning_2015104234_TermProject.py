
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import random


#Professor Kim's minmaxscalar
def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

#read dataset
data = pd.read_csv('OnlineNewsPopularity.csv',usecols=range(1,61), encoding="ISO-8859-1")

#check the dataset
print(data.head())
print(data.shape)



# In[2]:


#transform datset as matrix
xy = data.as_matrix()
print(xy)


# In[3]:


# x_data = xy[:, (7,13,14,15,16,17,18,19,25,26,27,38,39,40,41,42,43,44,45,56,48,49)] : feature selection
#no feature selection
x_data = xy[:,0:-1]
y_data = xy[:, [-1]]

print(x_data)
print(x_data.shape)
print(y_data)
print(y_data.shape)

n_features = 59

x_data = np.reshape(x_data, (-1, n_features))
y_data = np.reshape(y_data, (-1, 1))


# In[4]:


#classify y_Data
y_data = np.where(y_data <= 1400, 0, 1)

print(y_data.shape, y_data)


# In[5]:


#data normalization

x_data = MinMaxScaler(x_data)

print(x_data)


# In[6]:


# train/test split
train_size = int(len(y_data) * 0.7)
test_size = len(y_data) - train_size
trainX, testX = np.array(x_data[0:train_size]), np.array(
    x_data[train_size:len(x_data)])
trainY, testY = np.array(y_data[0:train_size]), np.array(
    y_data[train_size:len(y_data)])

print(trainX, testX)
print(trainY, testY)
print(testX.shape)
print(testY.shape)
print(len(testY))


# In[7]:


# placeholders for a tensor that will be always fed.
X1 = tf.placeholder(tf.float32, shape=[None, n_features])
Y1 = tf.placeholder(tf.float32, shape=[None, 1])
W1 = tf.Variable(tf.random_normal([n_features, 1]), name='weight')
b1 = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis1 = tf.sigmoid(tf.matmul(X1, W1) + b1)
cost1 = -tf.reduce_mean(Y1 * tf.log(tf.clip_by_value(hypothesis1,1e-10,1.0)) + (1 - Y1) * tf.log(tf.clip_by_value(1 - hypothesis1,1e-10,1.0)))
train1 = tf.train.GradientDescentOptimizer(learning_rate=1).minimize(cost1)

# Accuracy computation
# True if hypothesis>0.5 else False
prediction1 = tf.cast(hypothesis1 > 0.5, dtype=tf.float32)
is_correct1 = tf.equal(prediction1, Y1)
accuracy1 = tf.reduce_mean(tf.cast(is_correct1, dtype=tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    feed = {X1: trainX, Y1: trainY}

    for step in range(100):
        sess.run(train1, feed_dict=feed)
        if step % 10 == 0:
            print(step, sess.run(cost1, feed_dict=feed))
            
    #predict
    h, c, a = sess.run([hypothesis1, prediction1, accuracy1], feed_dict={X1: testX, Y1: testY})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)


# In[8]:


# placeholders for a tensor that will be always fed.
X2 = tf.placeholder(tf.float32, shape=[None, n_features])
Y2 = tf.placeholder(tf.float32, shape=[None, 1])

keep_prob = tf.placeholder(tf.float32)


# In[22]:


#value of variables
n_hidden_1 = 120
n_hidden_2 = 100


total_num = len(trainY)

learning_rate2 = 0.007
training_epochs = 100

keep_prob = 0.7


# In[10]:


#shape of layers
with tf.name_scope("layer1") as scope:
    W2_1 = tf.get_variable("W2_1", shape=[n_features, n_hidden_1],
                         initializer=tf.contrib.layers.xavier_initializer())
    b2_1 = tf.Variable(tf.random_normal([n_hidden_1]))
    L2_1 = tf.nn.relu(tf.matmul(X2, W2_1) + b2_1)
    L2_1 = tf.nn.dropout(L2_1, keep_prob=keep_prob)
    w1_hist = tf.summary.histogram("weights1", W2_1)
    b1_hist = tf.summary.histogram("biases1", b2_1)
    layer1_hist = tf.summary.histogram("layer1", L2_1)


with tf.name_scope("layer2") as scope:
    W2_2 = tf.get_variable("W2_2", shape=[n_hidden_1, n_hidden_2],
                         initializer=tf.contrib.layers.xavier_initializer())
    b2_2 = tf.Variable(tf.random_normal([n_hidden_2]))
    L2_2 = tf.nn.relu(tf.matmul(L2_1, W2_2) + b2_2)
    L2_2 = tf.nn.dropout(L2_2, keep_prob=keep_prob)
    w2_hist = tf.summary.histogram("weights2", W2_2)
    b2_hist = tf.summary.histogram("biases2", b2_2)
    layer2_hist = tf.summary.histogram("layer2", L2_2)
    
with tf.name_scope("layer3") as scope:
    W2_3 = tf.get_variable("W2_3", shape=[n_hidden_2, 1],
                         initializer=tf.contrib.layers.xavier_initializer())
    b2_3 = tf.Variable(tf.random_normal([1]))

    hypothesis2 = tf.matmul(L2_2, W2_3) + b2_3

    w3_hist = tf.summary.histogram("weights3", W2_3)
    b3_hist = tf.summary.histogram("biases3", b2_3)
    hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis2)



# In[23]:


# define cost/loss & optimizer
with tf.name_scope("cost") as scope:
    cost2 = tf.reduce_mean(tf.squared_difference(Y2, hypothesis2))
    cost_summ = tf.summary.scalar("cost", cost2)

with tf.name_scope("optimizer") as scope:
    optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate2).minimize(cost2)




# In[25]:


# Test mode
prediction2 = tf.cast(hypothesis2 > 0.5, dtype=tf.float32)
correct_prediction2 = tf.equal(prediction2, Y2)
    
# Calculate accuracy
with tf.name_scope("accuracy") as scope:
    accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, "float"))
    accuracy_summ = tf.summary.scalar("accuracy", accuracy2)    
    
    
with tf.Session() as sess:
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("C:\logs\online_publicity_check_3Layer100_final3/0.007")
    writer.add_graph(sess.graph)
    

    sess.run(tf.global_variables_initializer())
    
    feed = {X2: trainX, Y2: trainY}
    
    for epoch in range(training_epochs):
       
        summary, _ = sess.run([merged_summary, optimizer2], feed_dict = feed)
        writer.add_summary(summary, global_step=epoch)
        
        c, _ = sess.run([cost2, optimizer2], feed_dict=feed)
            
        if epoch % 10 == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost =", "{:.9f}".format(c))
    
    h, p, a = sess.run([hypothesis2, prediction2, accuracy2], feed_dict={X2: testX, Y2: testY})
    print("\nHypothesis: ", h, "\nPrediction (Y): ", p, "\nAccuracy: ", a)
    

