# mount google drive
# to use... just put /content/gdrive/My Drive/
# prerequisite things : training_set & test_set
from google.colab import drive
drive.mount("/content/gdrive/")

# Commented out IPython magic to ensure Python compatibility.
# import
# use tensorflow version 1.x
# %tensorflow_version 1.x
import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
import time
import random
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

def load_train(train_path, image_size, classes):
    # array for loading
    images = []
    labels = []
    img_names = []
    cls = []

    print('Going to read training images')

    # current classes : cats & dogs
    for fields in classes:   
        index = classes.index(fields)
        print('Now going to read {} files (Index: {})'.format(fields, index))
        path = os.path.join(train_path, fields, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl) # read image
            image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR) # resizing to our image_size
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0) # divide 255.0 to standardize in 0 ~ 1 for fast computation
            images.append(image)
            label = np.zeros(len(classes)) # labelling for classification cat or dog
            label[index] = 1.0 # set [1, 0] or [0, 1]
            labels.append(label)
            flbase = os.path.basename(fl) # remove pre-pathnames, only file name
            img_names.append(flbase)
            cls.append(fields) # cats or dogs

    # set numpy array
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)

    return images, labels, img_names, cls


class DataSet(object):

  def __init__(self, images, labels, img_names, cls):
    self._num_examples = images.shape[0]

    self._images = images
    self._labels = labels
    self._img_names = img_names
    self._cls = cls
    self._epochs_done = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def img_names(self):
    return self._img_names

  @property
  def cls(self):
    return self._cls

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_done(self):
    return self._epochs_done

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # After each epoch we update this
      self._epochs_done += 1
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]

def read_test_sets(test_path, image_size, classes):
  class DataSets(object):
    pass
  data_sets = DataSets()

  # load training datas
  images, labels, img_names, cls = load_train(train_path, image_size, classes)
  # shuffle datas
  images, labels, img_names, cls = shuffle(images, labels, img_names, cls)
  test_images = images[:]
  test_labels = labels[:]
  test_img_names = img_names[:]
  test_cls = cls[:]
  data_sets.test = DataSet(test_images, test_labels, test_img_names, test_cls)
  return data_sets

def read_train_sets(train_path, image_size, classes, validation_size):
  # just like shell for storing datasets
  class DataSets(object):
    pass
  data_sets = DataSets()

  # load training datas
  images, labels, img_names, cls = load_train(train_path, image_size, classes)
  # shuffle datas
  images, labels, img_names, cls = shuffle(images, labels, img_names, cls)  

  if isinstance(validation_size, float):
    # shape[0] means the number of images
    validation_size = int(validation_size * images.shape[0])

  # from training sets to  validation sets & training sets by rate of validation_size 0.2(in this case)
  validation_images = images[:validation_size]
  validation_labels = labels[:validation_size]
  validation_img_names = img_names[:validation_size]
  validation_cls = cls[:validation_size]

  train_images = images[validation_size:]
  train_labels = labels[validation_size:]
  train_img_names = img_names[validation_size:]
  train_cls = cls[validation_size:]

  # define data_sets's member variables 'train' & 'valid' by class 'DataSet'
  data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
  data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)

  return data_sets

# Prepare input data
# Cats & dogs 2 classes
classes = os.listdir('/content/gdrive/My Drive/training_set')
num_classes = len(classes)

# 20% of the data will automatically be used for validation
validation_size = 0.2
img_size = 128
num_channels = 3
train_path='/content/gdrive/My Drive/training_set'

# We shall load all the training and validation images and labels into memory using openCV and use that during training
data = read_train_sets(train_path, img_size, classes, validation_size=validation_size)


print("Complete reading input data. print each info")
print("Number of files in Training-set:  \t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))



session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')

## labels
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

# load test_set
test_path='/content/gdrive/My Drive/test_set'
t_data = read_test_sets(test_path, img_size, classes)

import matplotlib.pyplot as plt
batch_size = 100
keep_prob = tf.placeholder(tf.float32)
##Network graph params
filter_size_conv1 = 3 
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64
    
fc_layer_size = 128

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_xavier(shape):
    weights = tf.get_variable("weights", shape, initializer=tf.contrib.layers.xavier_initializer())
    return weights

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))



def create_convolutional_layer(input,
               num_input_channels, 
               conv_filter_size,        
               num_filters):  
    
    ## We shall define the weights that will be trained using create_weights function.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    ## We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)

    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')

    layer += biases

    ## We shall be using max-pooling.  
    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
    ## Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)
    layer = tf.nn.dropout(layer, rate = 1 - keep_prob)

    return layer

    

def create_flatten_layer(layer):
    #We know that the shape of the layer will be [batch_size img_size img_size num_channels] 
    # But let's get it from the previous layer.
    layer_shape = layer.get_shape()

    ## Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
    num_features = layer_shape[1:4].num_elements()

    ## Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input,          
             num_inputs,    
             num_outputs,
             use_relu=True):
    
    #Let's define trainable weights and biases.  
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
        layer = tf.nn.dropout(layer, rate = 1 - keep_prob)

    return layer

# X : [?, 128, 128, 3]
layer_conv1 = create_convolutional_layer(input=x,
               num_input_channels=num_channels,
               conv_filter_size=filter_size_conv1,
               num_filters=num_filters_conv1)

# conv1 : [?, 64, 64, 32] by pooling
layer_conv2 = create_convolutional_layer(input=layer_conv1,
               num_input_channels=num_filters_conv1,
               conv_filter_size=filter_size_conv2,
               num_filters=num_filters_conv2)

# conv2 : [?, 32, 32, 32] by pooling
layer_conv3= create_convolutional_layer(input=layer_conv2,
               num_input_channels=num_filters_conv2,
               conv_filter_size=filter_size_conv3,
               num_filters=num_filters_conv3)

# conv3 : [?, 16, 16, 64] by pooling
layer_flat = create_flatten_layer(layer_conv3)

# flat : [?, 16*16*64]
layer_fc1 = create_fc_layer(input=layer_flat,
                     num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                     num_outputs=fc_layer_size,
                     use_relu=True)

layer_fc2 = create_fc_layer(input=layer_fc1,
                     num_inputs=fc_layer_size,
                     num_outputs=num_classes,
                     use_relu=False) 

y_pred = tf.nn.softmax(layer_fc2,name='y_pred')

y_pred_cls = tf.argmax(y_pred, axis=1)
session.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2,
                                                    labels=tf.stop_gradient(y_true))
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


session.run(tf.global_variables_initializer())

def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss, feed_acc):
    acc = session.run(accuracy, feed_dict=feed_acc)
    tr_loss = session.run(cost, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Training Loss: {3:.3f},  Validation Loss: {4:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, tr_loss, val_loss))

total_iterations = 0
loss_list = []
val_list = []

saver = tf.train.Saver()
def train(num_iteration):
    global total_iterations
    
    for i in range(total_iterations,
                   total_iterations + num_iteration):

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)

        
        feed_dict_tr = {x: x_batch,
                           y_true: y_true_batch, keep_prob:0.7}

        feed_dict_tr_acc = {x: x_batch,
                           y_true: y_true_batch, keep_prob:1}
                         
        feed_dict_val = {x: x_valid_batch,
                              y_true: y_valid_batch, keep_prob:1}

        session.run(optimizer, feed_dict=feed_dict_tr)

        # in this case, may be 200 iteration in 1 epoch
        # cuz we have 6401 training datas
        if i % int(data.train.num_examples/batch_size) == 0: 
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            val_list.append(val_loss)
            epoch = int(i / int(data.train.num_examples/batch_size))
            _, loss = session.run((optimizer, cost), feed_dict_tr)
            loss_list.append(loss)           
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss, feed_dict_tr_acc)
            #saver.save(session, 'dogs-cats-model') 


    total_iterations += num_iteration

train(num_iteration=3000)

# Get one and predict
percent = 0
for i in range(0, 100):
  r = random.randint(0, t_data.test.num_examples - 1)
  la = session.run(tf.argmax(t_data.test.labels[r:r + 1], 1))
  pre = session.run(
      tf.argmax(y_pred, axis=1), feed_dict={x: t_data.test.images[r:r + 1], keep_prob:1})
  if la == pre:
    percent = percent + 1
print(str(percent)+'%')

plt.plot(loss_list)
plt.plot(val_list)
