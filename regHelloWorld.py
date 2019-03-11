import tensorflow as tf
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

import numpy as np

from tensorflow import keras
batch_size=1
model = tf.keras.Sequential()
model.add(keras.layers.Dense(units=1,input_shape=(4,)))
sgd=keras.optimizers.SGD(lr=0.0001)

model.compile(loss='mean_squared_error',optimizer=sgd)
x1=[1,2,3,4]
x2=[1,2,3,40]
x3=[1,20,3,4]
x4=[1,2,30,4]
x5=[10,2,3,4]

xs=[]
xs.append(x1)
xs.append(x2)
xs.append(x3)
xs.append(x4)
xs.append(x5)
print(xs)
xs = np.array(xs, dtype=float)
print(xs.shape)

ys = np.array([10,46,28,37,19], dtype=float)

model.fit(xs, ys, epochs=500)
model.summary()

x1=[10,20,30,40]

x1 = np.expand_dims(x1, axis=0)

#exit()

print(model.predict(x1))