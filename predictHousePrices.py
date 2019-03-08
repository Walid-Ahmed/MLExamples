'''
In this exercise you'll try to build a neural network that predicts the price of a house according to a simple formula.

So, imagine if house pricing was as easy as a house costs 50k + 50k per bedroom, so that a 1 bedroom house costs 100k, a 2 bedroom house costs 150k etc.

How would you create a neural network that learns this relationship so that it would predict a 7 bedroom house as costing close to 400k etc.

Hint: Your network might work better if you scale the house price down. You don't have to give the answer 400...it might be better to create something that predicts the number 4, and then your answer is in the 'hundreds of thousands' etc.


'''


import tensorflow as tf
import numpy as np
from tensorflow import keras
model = keras.Sequential()# Your Code Here#
model.add(keras.layers.Dense(units=1, input_shape=[1]))

model.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.array([0,  1, 2, 3, 4, 5], dtype=float)
ys = 50+ 50*xs
model.fit(xs, ys, epochs=500)
input=[7]
price=1000*(model.predict(input))
print(price)
