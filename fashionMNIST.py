import numpy as np
from tensorflow import keras
from keras.datasets import fashion_mnist
import   tensorflow as tf


Fashion_MNIST={0:'T-shirt',1:'Trouser',2:'Pullover',3:'Dress',4:'Coat',5:'Sandal',6:'Shirt',7:'Sneaker',8:'Bag',9:'Ankle boot'}
print(tf.__version__)



'''in the Fashion-MNIST data set, 60,000 of the 70,000 images are used to train the network, and then 10,000 images, 
one that it hasn't previously seen'''
(training_images, training_labels), (test_images, test_labels)= fashion_mnist.load_data()
print("[INFO] printing first 10 training labels")
print (training_labels[0:10])

'''
What does these values look like? Let's print a training image, and a training label to see...
Experiment with different indices in the array. For example, also take a look at index 42...
that's a a different boot than the one at index 0
'''
import matplotlib.pyplot as plt
plt.imshow(training_images[0])
print("The intger label of the image is {0}".format(training_labels[0]))
#print(training_images[0])
#plt.show()

'''You'll notice that all of the values in the number are between 0 and 255. 
If we are training a neural network, for various reasons it's easier if we treat all values as between 0 and 1, 
a process called '**normalizing**'...and fortunately in Python it's easy to normalize a list like this without looping. 
You do it like this:'''

training_images  = training_images / 255.0
test_images = test_images / 255.0


'''

**Sequential**: That defines a SEQUENCE of layers in the neural network

**Flatten**: Remember earlier where our images were a square, when you printed them out? Flatten just takes that square and turns it into a 1 dimensional set.

**Dense**: Adds a layer of neurons

Each layer of neurons need an **activation function** to tell them what to do. There's lots of options, but just use these for now. 

**Relu** effectively means "If X>0 return X, else return 0" -- so what it does it it only passes values 0 or greater to the next layer in the network.

**Softmax** takes a set of values, and effectively picks the biggest one, so, for example, if the output of the last layer looks like [0.1, 0.1, 0.05, 0.1, 9.5, 0.1, 0.05, 0.05, 0.05], it saves you from fishing through it looking for the biggest value, and turns it into [0,0,0,0,1,0,0,0,0] -- The goal is to save a lot of coding!

'''

model = tf.keras.models.Sequential( )
model.add(tf.keras.layers.Flatten())
model.add( tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))



model.compile(optimizer = tf.train.AdamOptimizer(),loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)



#get losses and accuracy on test data
model.evaluate(test_images, test_labels)

print("[INFO] The probability that this item is each of the 10 classes")
classificationsProbs = model.predict(test_images)   #the probability that this item is each of the 10 classes  (get an array of class probabilitie)
print(classificationsProbs)
print("*********************************************************")

print("[INFO] The detected classes are:")
classes=model.predict_classes(test_images)
print(classes)

# printing prediction of a single image
print("An image with ground truth label {0} is classified as  {1} with probabilty to belong to 10 classes as  {2}".format(test_labels[0],classes[0],classificationsProbs[0]))
print("*********************************************************")



#testing o a single image
image=test_images[0]    # test image is of shape (28, 28)
image = np.expand_dims(image, axis=0)  #make its shape     (1, 28, 28)

predictedClassID=model.predict_classes(image)
predictedClassID=predictedClassID.item(0)  # turn numpy array to scaler integer value
clothType=Fashion_MNIST[predictedClassID]
print("predictedClassID is {0}  with label {1}".format(predictedClassID,clothType))





