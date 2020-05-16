# import libraries 
import tensorflow as tf
from tensorflow.python.keras.applications.vgg19 import VGG19

model = VGG19(
    include_top = False,
    weights = 'imagenet'
)
model.trainable = False

from PIL import Image

from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.applications.vgg19 import preprocess_input 
from tensorflow.python.keras.models import Model

import numpy as np
import matplotlib.pyplot as plt

import sys
import os.path as path

# write utility functions
def load_and_process_image(image_path):
    img = load_img(image_path)
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img,axis=0)
    return img

def deprocess(x):
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x,0,255).astype('uint8')
    return x

def return_deprocessed_image(image):
    if len(image.shape) == 4:
        img = np.squeeze(image,axis=0)
    img = deprocess(img)
    return img

# create content and style models
content_layer = 'block5_conv2'

style_layers = [
    'block1_conv1',
    'block3_conv1',
    'block5_conv1'
]

content_model = Model(
    inputs = model.input,
    outputs = model.get_layer(content_layer).output
)

style_models = [
    Model(
        inputs = model.input,
        outputs = model.get_layer(layer).output
    ) for layer in style_layers
]

# calculate content and style costs
def content_cost(content,generated):
    a_c = content_model(content)
    a_g = content_model(generated)
    cost = tf.reduce_mean(tf.square(a_c - a_g))
    return cost

def gram_matrix(A):
    n_c = int(A.shape[-1])
    a = tf.reshape(A, [-1,n_c])
    n = tf.shape(a)[0]
    G = tf.matmul(a,a,transpose_a=True)
    return G / tf.cast(n,tf.float32)

lam = 1. /len(style_models)
def style_cost(style,generated):
    j_style = 0
    
    for style_model in style_models:
        a_s = style_model(style)
        a_g = style_model(generated)
        gs = gram_matrix(a_s)
        gg = gram_matrix(a_g)
        current_cost = tf.reduce_mean(tf.square(gs-gg))
        j_style += current_cost*lam
    
    return j_style

# the training loop function
import time
generated_images = []

def training_loop(content_path,style_path,iterations=20,alpha=10.,beta=20.):
    content = load_and_process_image(content_path)
    style = load_and_process_image(style_path)
    
    generated = tf.Variable(content, dtype=tf.float32)
    
    opt = tf.optimizers.Adam(learning_rate=7.)
    
    best_cost = 1e12 + 0.1
    best_image = None
    start_time = time.time()
    
    for i in range(iterations):
        with tf.GradientTape() as tape:
            j_content = content_cost(content,generated)
            j_style = style_cost(style,generated)
            j_total = alpha*j_content + beta*j_style
            
        grads = tape.gradient(j_total,generated)
        opt.apply_gradients([(grads,generated)])
        
        if j_total<best_cost:
            best_cost = j_total
            best_image = generated.numpy()
            
        print('Cost at {}: {}. Time elapsed: {} '.format(i,j_total,time.time()-start_time))
        generated_images.append(generated.numpy())
        
    return best_image

# generate the final result
script_dir = path.dirname(path.abspath(__file__))
two_up =  path.abspath(path.join(__file__ ,"../.."))
fin = path.abspath(path.join(two_up,"images"))

content = path.join(script_dir, 'content.jpg')
style = path.join(script_dir, 'style.jpg')

best_image = training_loop(content,style)

best_image = return_deprocessed_image(best_image)

final = Image.fromarray(best_image)
final = final.save(path.join(fin, 'final.jpg'))

print('Done')