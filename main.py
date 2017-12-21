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
    layer=tf.nn.dropout(layer,keep_prob)
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

learning_rate=0.002
#batch_size=34000
epochs=1200
logits = leNet(x, keep_prob)
probabilities=tf.nn.softmax(logits)
cross_entropy =tf.nn.softmax_cross_entropy_with_logits(labels=one_hot,logits=logits)
loss = tf.reduce_mean(cross_entropy)
optimizer=tf.train.AdamOptimizer(learning_rate)
training_operation= optimizer.minimize(loss)
accuracy_operation= tf.reduce_mean(tf.cast(tf.equal(tf.argmax(one_hot,1),tf.argmax(logits,1)),tf.float32))
selected_class=tf.argmax(probabilities,1)
top_5 = tf.nn.top_k(probabilities,k=5)

def evaluate_accuracy(xIn, yIn):
    sess = tf.get_default_session()
    # accuracy = 0
    # den = 0
    # start_index=0
    # for start_index in range(0, len(xIn)-batch_size, batch_size):
    #     batch_x=xIn[start_index:start_index+batch_size]
    #     batch_y=yIn[start_index:start_index+batch_size]
    #     accuracy+= batch_size * sess.run(accuracy_operation, feed_dict={x:batch_x, y:batch_y, keep_prob:1})
    #     den+=batch_size
    # if start_index==0:
    #     accuracy+=len(xIn)* sess.run(accuracy_operation, feed_dict={x:xIn, y:yIn, keep_prob:1})
    #     den+=len(xIn)
    # else:
    #     batch_x=xIn[start_index+batch_size:]
    #     batch_y=yIn[start_index+batch_size:]
    #     accuracy += (len(xIn)-start_index-batch_size) * sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1})
    #     den += len(xIn)-start_index-batch_size
    # return accuracy/den
    return sess.run(accuracy_operation,feed_dict={x:xIn,y:yIn,keep_prob:1})

saver = tf.train.Saver()
model_file="./final_model.ckpt"

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epochs):
        print(i)
        X_train, y_train = shuffle(X_train,y_train)
        # for start_index in range(0,n_train-batch_size,batch_size):
        #     end_index=start_index+batch_size
        #     batch_x = X_train[start_index:end_index]
        #     batch_y = y_train[start_index:end_index]
        #     sess.run(training_operation,feed_dict={x:batch_x,y:batch_y, keep_prob:0.4})
        # start_index = end_index
        # batch_x = X_train[start_index:]
        # batch_y = y_train[start_index:]
        # sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.4})
        sess.run(training_operation,feed_dict={x:X_train,y:y_train,keep_prob:0.4})
        accuracy = evaluate_accuracy(X_train,y_train)
        print("The accuracy of the training data is {}".format(accuracy))
    accuracy = evaluate_accuracy(X_valid, y_valid)
    print("The accuracy of the validation set is {}".format(accuracy))
    accuracy = evaluate_accuracy(X_test, y_test)
    print("The accuracy of the test set is {}".format(accuracy))
    saver.save(sess,model_file)

#Step 3.0
### Load the images and plot them here.
### Feel free to use as many code cells as needed.
from PIL import Image as pilImage
def PIL2array(img):
    r = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
    return r.tolist()
test_image1 = pilImage.open('./WebImages/TestImage1.jpg')
test_image2 = pilImage.open('./WebImages/TestImage2.jpg')
test_image3 = pilImage.open('./WebImages/TestImage3.jpg')
test_image4 = pilImage.open('./WebImages/TestImage4.jpg')
test_image5 = pilImage.open('./WebImages/TestImage5.jpg')
correct_estimates=np.array([33, 3, 27, 28, 25])
input_images = [test_image1, test_image2, test_image3, test_image4, test_image5]
f, plots = plt.subplots(1, len(input_images))
for i, image in enumerate(input_images):
    plots[i].imshow(image)
    plots[i].set_xticks([], [])
    plots[i].set_yticks([], [])
plt.show()
#step3.1
### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.
test_images = input_images[:]
for i in range(len(test_images)):
    test_images[i]= PIL2array(test_images[i].resize((32,32),pilImage.ANTIALIAS))
test_images=np.array(test_images)
test_images=rgb2gray(test_images)
test_images=test_images-128/128
with tf.Session() as sess:
    saver.restore(sess, model_file)
    final_output = sess.run(selected_class, feed_dict={x: test_images, keep_prob: 1})
import csv
with open('signnames.csv') as csvfile:
    reader=csv.DictReader(csvfile)
    signNames={int(row['ClassId']):row['SignName'] for row in reader}
f, plots = plt.subplots(len(input_images)//2+1,2)
for i, image in enumerate(input_images):
    plots[i//2][i%2].imshow(image)
    plots[i//2][i%2].set_title(signNames[final_output[i]])
    plots[i//2][i%2].set_xticks([], [])
    plots[i//2][i%2].set_yticks([], [])
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()
print(final_output)
final_output=np.array(final_output)
final_test_accuracy = np.average(final_output==correct_estimates)
print("The accuracy of detecting images from the web is {}".format(final_test_accuracy))


with tf.Session() as sess:
    saver.restore(sess, model_file)
    final_top_5 = sess.run(top_5,feed_dict={x:test_images, keep_prob:1})
final_top_5_strings=['']*len(final_top_5.values)
for i in range(len(final_top_5.values)):
    string_i=""
    for value, index in zip(final_top_5.values[i], final_top_5.indices[i]):
        string_i+=signNames[index]+": "+ str(value)+"\n"
    final_top_5_strings[i]=string_i
f, plots = plt.subplots(2,len(input_images)//2+1)
for i, image in enumerate(input_images):
    plots[i%2][i//2].imshow(image)
    plots[i%2][i//2].set_title(final_top_5_strings[i])
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()



