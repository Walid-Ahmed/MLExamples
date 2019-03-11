<<<<<<< HEAD
import tensorflow as tf
import numpy as np

print(tf.__version__)


class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>=0.99):
      print("Reached 99% accuracy so cancelling training!")
      self.model.stop_training = True
callbacks = myCallback()

mnist = tf.keras.datasets.mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

import matplotlib.pyplot as plt
plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])


training_images  = training_images / 255.0
test_images = test_images / 255.0



model = tf.keras.models.Sequential( )
model.add(tf.keras.layers.Flatten())
model.add( tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer = tf.train.AdamOptimizer(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10,callbacks=[callbacks])

#get losses and accuracy on test data
model.evaluate(test_images, test_labels)


classificationsProbs = model.predict(test_images)   #the probability that this item is each of the 10 classes
classes=model.predict_classes(test_images)
print(classificationsProbs[0])
print(classes[0])
print(test_labels[0])
image=test_images[0]
image = np.expand_dims(image, axis=0)
=======
import tensorflow as tf
import numpy as np

print(tf.__version__)


class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>=0.99):
      print("Reached 99% accuracy so cancelling training!")
      self.model.stop_training = True
callbacks = myCallback()

mnist = tf.keras.datasets.mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

import matplotlib.pyplot as plt
plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])


training_images  = training_images / 255.0
test_images = test_images / 255.0



model = tf.keras.models.Sequential( )
model.add(tf.keras.layers.Flatten())
model.add( tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer = tf.train.AdamOptimizer(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10,callbacks=[callbacks])

#get losses and accuracy on test data
model.evaluate(test_images, test_labels)


classificationsProbs = model.predict(test_images)   #the probability that this item is each of the 10 classes
classes=model.predict_classes(test_images)
print(classificationsProbs[0])
print(classes[0])
print(test_labels[0])
image=test_images[0]
image = np.expand_dims(image, axis=0)
>>>>>>> 711c06498c075929513a3e1507f362522a878098
predictedClassID=model.predict_classes(image)