# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 20:05:25 2017

@author: Bhumika
"""
import numpy as np
import tensorflow  as tf
from PIL import Image
import warnings
import zipfile
import os
from sklearn import preprocessing
warnings.filterwarnings("ignore")
filename="img_align_celeba.zip"
label_filename="list_attr_celeba.txt"

#Print details of each team member
print("UBitName\t=\tBhumika Khatwani\t\tSunita Pattanayak")
print("personNumber\t=\t50247656\t\t\t50249134")

# #CNN using tensor flow
def tf_conv2d(x, W):
     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
 
def tf_max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
 
def tf_weight_variable(shape):
     initial = tf.truncated_normal(shape, stddev=0.1)
     return tf.Variable(initial)
 
def tf_bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

#Defining height,width for resizing the images to 178x178
height=28
width=28

#Defining path for extracting dataset zip file
extract_path = "celeb_image_data"

#Defining image,label list
images = []
celeba_train_images=[]
celeba_test_images=[]
img_list = []
img_train_list = []
img_test_list = []
labels=[]
celeba_train_labels=[]
celeba_test_labels=[]


img_name,data = np.loadtxt('list_attr_celeba.txt',dtype={
               'names': (
                   'img_name','eyeglasses_label'), 
               'formats': (
                   'U15',np.int)},usecols=(0,15),skiprows=2,unpack=True)

data_matrix=np.array([img_name,data])
a = data_matrix[1]
labels_array=np.transpose(a)

dataset_count = len(labels_array)
training_count = int(0.8 * dataset_count)
test_count = int(0.2 * dataset_count)

for row in range(len(labels_array)):
    if(row<training_count-1):
        celeba_train_labels.append(labels_array[row])
    else:
        celeba_test_labels.append(labels_array[row])

celeba_train_labels = np.array(celeba_train_labels)
celeba_test_labels = np.array(celeba_test_labels)


label_train = np.array(celeba_train_labels).reshape(-1,1)
label_train_tf = tf.placeholder(np.float32)
label_train[label_train == -1] = 0


label_encoder = preprocessing.LabelEncoder()
integer_encoded = label_encoder.fit_transform(label_train)
onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
label_train = onehot_encoder.fit_transform(integer_encoded)


label_test = np.array(celeba_test_labels).reshape(-1,1)
label_test[label_test == -1] = 0
label_test_tf = tf.placeholder(np.float32)

label_encoder = preprocessing.LabelEncoder()
integer_encoded = label_encoder.fit_transform(label_test)
onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
label_test = onehot_encoder.fit_transform(integer_encoded)

#Extracting given dataset file    
with zipfile.ZipFile(filename, 'r') as zip:
    zip.extractall(extract_path)


#Extracting labels,images array needed for training    
for root, dirs, files in os.walk("."):
    path = root.split(os.sep)
        
    if "img_align_celeba" in path:
        image_files = [fname for fname in files if fname.find(".jpg") >= 0]
        for file in image_files:
            #labels.append(int(path[-1]))
            images.append(os.path.join(*path, file))
            if(len(images)<training_count):
                celeba_train_images.append(os.path.join(*path, file))
            else:
                celeba_test_images.append(os.path.join(*path, file))
  
for idx,imgs in enumerate(celeba_train_images):
#   img_resize = images[i].resize(28,28,3)
    img = Image.open(imgs).convert('L') 
    img = img.resize((height, width), Image.ANTIALIAS)
    img_data = list(img.getdata())
    img_train_list.append(img_data)
        
for idx,imgs in enumerate(celeba_test_images):
#   img_resize = images[i].resize(28,28,3)
    img = Image.open(imgs).convert('L') 
    img = img.resize((height, width), Image.ANTIALIAS)
    img_data = list(img.getdata())
    img_test_list.append(img_data)
                   
#Storing image and labels in arrays to be used for training  
img_train_array = np.array(img_train_list)
img_train_array = np.subtract(255, img_train_array)
img_test_array = np.array(img_test_list)
img_test_array = np.subtract(255, img_test_array)

#initialising hyperparams
# =============================================================================
batch_size=1000
iterations=10

W = tf.Variable(tf.zeros([height*width, 2]))
b = tf.Variable(tf.zeros([2]))

#first convolutional layer
x_p = tf.placeholder(tf.float32, shape=[None , height*width])
W_conv1 = tf_weight_variable([5, 5, 1, 32])
b_conv1 = tf_bias_variable([32])
x_image = tf.reshape(x_p, [-1,height,width,1])
h_conv1 = tf.nn.relu(tf_conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = tf_max_pool_2x2(h_conv1)
#      
#second convolutional layer
W_conv2 = tf_weight_variable([5, 5, 32, 64])
b_conv2 = tf_bias_variable([64])
h_conv2 = tf.nn.relu(tf_conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = tf_max_pool_2x2(h_conv2)
#     
# #Densely Connected Layer
W_fc1 = tf_weight_variable([7 * 7 * 64, 1024])
b_fc1 = tf_bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# #     
#Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#     
#Readout
W_fc2 = tf_weight_variable([1024, 2])
b_fc2 = tf_bias_variable([2])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
#     
#Training
y_p = tf.placeholder(tf.float32, shape=[None , 2])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_p, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_p, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for idx in range(iterations):
        batch_images,batch_labels = next_batch(batch_size,img_train_array,label_train)
        if idx % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x_p: batch_images,y_p: batch_labels,keep_prob: 1.0})            
        train_step.run(feed_dict={x_p: batch_images, y_p: batch_labels,keep_prob: 0.6})
    print("\n \nAccuracy for CNN :")
    print('Dataset test accuracy : %0.2f' % accuracy.eval(feed_dict={x_p: img_test_array,y_p: label_test,keep_prob: 1.0}))       


