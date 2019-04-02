import tensorflow as tf



print(tf.__version__)
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()     #relative to ~/.keras/datasets
print("Shape of training_images before reshaping is {0}".format(training_images.shape))  #(10000, 28, 28)
training_images=training_images.reshape(60000, 28, 28, 1)
print("Shape of training_images after reshaping is {0}".format(training_images.shape))  #(10000, 28, 28)
training_images=training_images / 255.0
print("Shape of test_images before reshaping is {0}".format(test_images.shape))  #(10000, 28, 28)
test_images = test_images.reshape(10000, 28, 28, 1)
print("Shape of test_images after reshaping is {0}".format(test_images.shape))  #(10000, 28, 28)
test_images=test_images/255.0



model=tf.keras.models.load_model("model.keras2")

import matplotlib.pyplot as plt
f, axarr = plt.subplots(3,4)
FIRST_IMAGE=0
SECOND_IMAGE=7
THIRD_IMAGE=26
CONVOLUTION_NUMBER = 1
from tensorflow.keras import models
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
#3 images will make 3 rows and 4 layers will make 4 columnbs 
for x in range(0,4):
  test_image=test_images[FIRST_IMAGE]
  print("Shape of test_image before reshaping is {0}".format(test_image.shape))  #(28, 28 ,1)
  test_image=test_image.reshape(1, 28, 28, 1)
  print("Shape of test_image after reshaping is {0}".format(test_image.shape))  #(1, 28, 28)

  f1 = activation_model.predict(test_image)[x]
  axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[0,x].grid(False)
  test_image=test_images[SECOND_IMAGE].reshape(1, 28, 28, 1)
  f2 = activation_model.predict(test_image)[x]
  axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[1,x].grid(False)
  test_image=test_images[THIRD_IMAGE].reshape(1, 28, 28, 1)
  f3 = activation_model.predict(test_image)[x]
  axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[2,x].grid(False)
plt.show()  