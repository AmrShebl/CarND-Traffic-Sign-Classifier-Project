#STEP0
# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = "./DataSet/train.p"
validation_file ="./DataSet/valid.p"
testing_file = "./DataSet/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

#STEP1.0
### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of validation examples
n_validation = X_valid.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

#STEP1.1
### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
#%matplotlib inline
keys = set(y_train)
train_stats = {key:0 for key in keys}
valid_stats = train_stats.copy()
test_stats = train_stats.copy()
for y in y_train:
    train_stats[y]+=1
for y in y_test:
    test_stats[y]+=1
for y in y_valid:
    valid_stats[y]+=1
f, arrx = plt.subplots(3, sharex=True)
arrx[0].plot(*zip(*sorted(train_stats.items())))
arrx[0].set_title('Statistics of Train Data')
arrx[1].plot(*zip(*sorted(valid_stats.items())))
arrx[1].set_title('Statistics of Validation Data')
arrx[2].plot(*zip(*sorted(test_stats.items())))
arrx[2].set_title('Statistics of Test Data')
#plt.show()

#step2.0
### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.
import numpy as np
def rgb2gray(im):
    im = np.dot(im[..., :3], [0.299, 0.587, 0.114])
    im = im[:,:,:,None]
    return im
X_train = rgb2gray(X_train)
X_valid = rgb2gray(X_valid)
X_test = rgb2gray(X_test)
X_train=(X_train-128)/128
X_valid=(X_valid-128)/128
X_test=(X_test-128)/128

#step2.1
### Define your architecture here.
### Feel free to use as many code cells as needed.
import tensorflow as tf
def leNet(X, keep_prob):
    mu=0
    std = 0.1
    #convolution layer
    my_filter = tf.Variable(tf.truncated_normal((5,5,1,6), mean=mu, stddev= std))
    layer = tf.nn.conv2d(X,my_filter,[1,1,1,1],'VALID')
    layer = tf.nn.relu(layer)
    #output dimensions: 28*28*6

    #max pooling layer
    layer = tf.nn.max_pool(layer,[1,2,2,1],[1,2,2,1],'VALID')
    #output dimensions: 14*14*6

    my_filter = tf.Variable(tf.truncated_normal((5,5,6,16), mean=mu, stddev= std))
    layer = tf.nn.conv2d(layer,my_filter,[1,1,1,1],'VALID')
    layer = tf.nn.relu(layer)
    #output dimensions: 10*10*16
    layer = tf.nn.max_pool(layer,[1,2,2,1],[1,2,2,1], 'VALID')
    #output dimensions: 5*5*16
    layer = tf.contrib.layers.flatten(layer)
    #output dimensions: 400
    weight1=tf.Variable(tf.truncated_normal((400,120), mean=mu, stddev= std))
    bias1=tf.Variable(tf.zeros(120))
    layer=tf.add(tf.matmul(layer,weight1),bias1)
    layer=tf.nn.relu(layer)
    #output dimensions: 120
    layer=tf.nn.dropout(layer,keep_prob)
    weight2=tf.Variable(tf.truncated_normal((120,84), mean=mu, stddev= std))
    bias2=tf.Variable(tf.zeros(84))
    layer=tf.add(tf.matmul(layer,weight2),bias2)
    layer=tf.nn.relu(layer)
    #output dimensions: 84
    weight3=tf.Variable(tf.truncated_normal((84,43), mean=mu, stddev= std))
    bias3=tf.Variable(tf.zeros(43))
    logits=tf.add(tf.matmul(layer,weight3),bias3)
    return logits

#step2.2
### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected,
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
from sklearn.utils import shuffle
x = tf.placeholder(tf.float32, (None,32,32,1))
y = tf.placeholder(tf.int32, (None))
one_hot = tf.one_hot(y,43)
keep_prob = tf.placeholder(tf.float32)

learning_rate=0.001
batch_size=400
epochs=30
logits = leNet(x, keep_prob)
cross_entropy =tf.nn.softmax_cross_entropy_with_logits(labels=one_hot,logits=logits)
loss = tf.reduce_mean(cross_entropy)
optimizer=tf.train.AdamOptimizer(learning_rate)
training_operation= optimizer.minimize(loss)
accuracy_operation= tf.reduce_mean(tf.cast(tf.equal(tf.argmax(one_hot,1),tf.argmax(logits,1)),tf.float32))

def evaluate_accuracy(xIn, yIn):
    sess = tf.get_default_session()
    accuracy = 0
    den = 0
    for start_index in range(0, len(xIn)-batch_size, batch_size):
        batch_x=xIn[start_index:start_index+batch_size]
        batch_y=yIn[start_index:start_index+batch_size]
        accuracy+=sess.run(accuracy_operation, feed_dict={x:batch_x, y:batch_y, keep_prob:1})
        den+=1
    return accuracy/den

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epochs):
        print(i)
        X_train, y_train = shuffle(X_train,y_train)
        for start_index in range(0,n_train-batch_size,batch_size):
            end_index=start_index+batch_size
            batch_x = X_train[start_index:end_index]
            batch_y = y_train[start_index:end_index]
            sess.run(training_operation,feed_dict={x:batch_x,y:batch_y, keep_prob:0.5})
        accuracy = evaluate_accuracy(X_train,y_train)
        print("The accuracy of the training data is {}".format(accuracy))
    accuracy = evaluate_accuracy(X_valid, y_valid)
    print("The accuracy of the validation set is {}".format(accuracy))
