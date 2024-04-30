import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Sequential
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



x_data = np.array([[73,80,75]
                   [93,88,93]
                   [89,91,90]
                   [80,80,80]
                   [96,98,100]
                   [73,66,70], ])
y_data = np.array([72,88,92,81,100,71])

url = "https://raw.githubusercontent.com/Codecademy/datasets/master/streeteasy/manhattan.csv"
manhattan = pd.read_csv(url)
x = manhattan[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor',
'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman',
'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]
y = manhattan[['rent']]

model = Sequential()
model.add(Dense(1, input_dim = 3, activation = 'linear'))
sgd = optimizers.SGD(learning_rate = 1)
model.compile(loss='mse', optimizer = sgd, metrics=['mse'])

x_train, x_test, y_train, y_test = train_test_split(x,y,trai_size = 0.7, test_size = 0.3)

history = model.fit(x_data, y_data, batch_size = 1, epochs = 1, verbose = 1)

x_test = np.array([[90,88,93],[70,70,70]])

mlr = LinearRegression()
mlr.fit(x_train, y_train)
mlr.coef_
mlr.intercept_

plt.plot(mlr.predict(x_test[:50]))
plt.plot(y_test[:50].values.reshape(-1,1))
plt.legend(["predict","real price"])
