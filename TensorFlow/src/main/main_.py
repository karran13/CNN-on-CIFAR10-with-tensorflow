'''
Created on Dec 22, 2017

@author: Binki
'''

import tensorflow as tf
import numpy as np
import time
from main.help_func import processHelp
from main.help_data import dataHelper

path=[]
data_set=[]
data_labels=[]
data_set_train=[]
data_set_labels=[]

data_set_test=[]


path.append("C:\\Users\\Binki\\Downloads\\cifar-10-python.tar\\cifar-10-python\\cifar-10-batches-py\\data_batch_1")
path.append("C:\\Users\\Binki\\Downloads\\cifar-10-python.tar\\cifar-10-python\\cifar-10-batches-py\\data_batch_2")
path.append("C:\\Users\\Binki\\Downloads\\cifar-10-python.tar\\cifar-10-python\\cifar-10-batches-py\\data_batch_3")
path.append("C:\\Users\\Binki\\Downloads\\cifar-10-python.tar\\cifar-10-python\\cifar-10-batches-py\\data_batch_4")
path.append("C:\\Users\\Binki\\Downloads\\cifar-10-python.tar\\cifar-10-python\\cifar-10-batches-py\\data_batch_5")


path_test = "C:\\Users\\Binki\\Downloads\\cifar-10-python.tar\\cifar-10-python\\cifar-10-batches-py\\test_batch"

# fix up data

helper = processHelp()

train_data = dataHelper(path[0])

train_data.initialize_main_dict()

data_set_train=train_data.returnDataset()

data_labels_init, num_classes = train_data.returnLabelsNumClasses()

data_set_labels=data_labels_init

for i in range(1,5):

    train_data = dataHelper(path[i])

    train_data.initialize_main_dict()
    
    data_set_train=np.concatenate([data_set_train,np.array(train_data.returnDataset())])

    data_labels_temp, num_classes = train_data.returnLabelsNumClasses()
    
    data_set_labels=np.concatenate([data_set_labels,np.array(data_labels_temp)])


test_data = dataHelper(path_test)

test_data.initialize_main_dict()

data_set_test = test_data.returnDataset()

test_data_labels,_ = test_data.returnLabelsNumClasses()

test_cls=test_data.true_classes

print(np.shape(data_set_test))
print(np.shape(test_data_labels))

# make tensorflow graph






def inference(images):
  """Build the CIFAR-10 model.
  Args:
    images: Images returned from distorted_inputs() or inputs().
  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  with tf.variable_scope('conv1') as scope:
      kernel = helper._variable_with_weight_decay('weights',
                                         shape=[5, 5, 3, 64],
                                         stddev=5e-2,
                                         wd=0.0)
      conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
      biases = helper._variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
      pre_activation = tf.nn.bias_add(conv, biases)
      conv1 = tf.nn.relu(pre_activation, name=scope.name)
      
    

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
      kernel = helper._variable_with_weight_decay('weights',
                                           shape=[5, 5, 64, 64],
                                           stddev=5e-2,
                                           wd=0.0)
      conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
      biases = helper._variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
      pre_activation = tf.nn.bias_add(conv, biases)
      conv2 = tf.nn.relu(pre_activation, name=scope.name)
    
  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [64, -1])
    dim = reshape.get_shape()[1]
    weights = helper._variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = helper._variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

  # local4
  with tf.variable_scope('local4') as scope:
      weights = helper._variable_with_weight_decay('weights', shape=[384, 192],
                                              stddev=0.04, wd=0.004)
      biases = helper._variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
      local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('softmax_linear') as scope:
      weights = helper._variable_with_weight_decay('weights', [192, num_classes],
                                          stddev=1 / 192.0, wd=0.0)
      biases = helper._variable_on_cpu('biases', [num_classes],
                              tf.constant_initializer(0.0))
      softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
  
  return softmax_linear


def create_network(training):

    # Wrap the neural network in the scope named 'network'.
    # Create new variables during training, and re-use during testing.
    with tf.variable_scope('network', reuse=not training):    
        images = x
        images = helper.pre_process(images, training)
        logits = inference(images)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits))
        y_pred = tf.nn.softmax(logits=logits)
    
    return y_pred, loss



def random_batch():
    # Number of images in the training-set.
    num_images = len(data_set_train)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=64,
                           replace=False)

    # Use the random index to select random images and labels.
    x_batch = data_set_train[idx, :, :, :]
    y_batch = data_set_labels[idx, :]

    return x_batch, y_batch


def optimize(num_iterations):
    # Start-time used for printing time-usage below.

    for i in range(num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
 
        x_batch, y_batch = random_batch()

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true:y_batch }

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        # We also want to retrieve the global_step counter.
        i_global, _ = session.run([global_step, optimizer],
                                  feed_dict=feed_dict_train)

        # Print status to screen every 100 iterations (and last).
        if (i_global % 100 == 0) or (i == num_iterations - 1):
            # Calculate the accuracy on the training-batch.
            batch_acc = session.run(accuracy,
                                    feed_dict=feed_dict_train)

            # Print status.
            
            batch_loss=session.run(loss,feed_dict=feed_dict_train)
            
            print(batch_loss)
            
            print("\n")
            
            msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
            print(msg.format(i_global, batch_acc))
            
    

def predict_cls(images, labels, cls_true):
    # Number of images.
    num_images = len(images) - 16
    batch_size = 64
    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)

        # Create a feed-dict with the images and labels
        # between index i and j.
        feed_dict = {x: images[i:j, :],
                     y_true: labels[i:j, :]}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Create a boolean array whether each image is correctly classified.
    
    cls_pred = np.array(cls_pred)
    cls_true = np.array(cls_true)
    correct=[]
    for i in range(len(cls_pred)):
        if(cls_pred[i]==cls_true[i]):
            correct.append(1.0)
        else:
            correct.append(0.0)
    print(correct)
    
    
    
    
    return correct, cls_pred



def classification_accuracy(correct):
    # When averaging a boolean array, False means 0 and True means 1.
    # So we are calculating: number of True / len(correct) which is
    # the same as the classification accuracy.
    
    # Return the classification accuracy
    # and the number of correct classifications.
    return correct.mean(), correct.sum()





x = tf.placeholder(tf.float32, shape=[64, 32, 32, 3], name='x')
 
y_true = tf.placeholder(tf.float32, shape=[64, num_classes])
 
y_true_cls = tf.arg_max(y_true, dimension=1)
 
global_step = tf.Variable(initial_value=0,
                          name='global_step', trainable=False)

_,loss = create_network(training=True)
  
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(loss=loss, global_step=global_step)
 
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step=global_step)
 
y_pred,_ = create_network(training=False)
 
y_pred_cls = tf.arg_max(y_pred, dimension=1)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
session = tf.Session()
 
session.run(tf.global_variables_initializer())
 
optimize(10000)

errors,_=predict_cls(data_set_test, test_data_labels, test_cls)



print("final accuracy: ")

print(np.mean(errors))





