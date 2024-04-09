#준희
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

w_0, w_1 = map(float, input("Enter values for w_0 and w_1: ").split())

fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\5주차\\lin_regression_data_01.csv"
temp_data = pd.read_csv(fold_dir, header=None)
temp_data = temp_data.to_numpy()  



x_data = temp_data[:, 0]
y_data = temp_data[:, 1]

w_0_values = []
w_1_values = []
e_mse_values = []
a = 0.001
repeat = 10000  

def e_mse(y_pred, y_true):
    return np.mean((y_true - y_pred) ** 2)  

def gradient_descent(x_data, y_data, w0, w1, a, repeat):
    for i in range(repeat):
        y_pred = w0 * x_data + w1
        gradient_w0 = np.mean((y_pred - y_data) * x_data)
        gradient_w1 = np.mean((y_pred - y_data))
        w0 = w0 - a * gradient_w0
        w1 = w1 - a * gradient_w1
        w_0_values.append(w0)
        w_1_values.append(w1)
        mse = e_mse(y_pred, y_data)
        e_mse_values.append(mse)
    return w0, w1, w_0_values, w_1_values, e_mse_values

final_w0, final_w1, final_w0_values, final_w1_values, final_e_mse_values = gradient_descent(x_data, y_data, w_0, w_1, a, repeat)

x_arange = np.linspace(min(x_data) - 1, max(x_data) + 1, 50)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(final_w0_values)
plt.plot(final_w1_values)
plt.xlabel('step')
plt.ylabel('w0,w1')
plt.title('Optimal Solution')
plt.grid()
plt.legend(["w0", "w1"], loc='upper left')

plt.subplot(1, 3, 2)
plt.plot(final_e_mse_values)
plt.xlabel('step')
plt.ylabel('MSE')
plt.title('E_mse values')
plt.legend(["e_mse"], loc='upper left')
plt.grid()

plt.subplot(1, 3, 3)
plt.scatter(x_data, y_data)
y = final_w0 * x_arange + final_w1
plt.plot(x_arange, y, c='r')
plt.xlabel('weight')
plt.ylabel('length')
plt.title('total graph')
plt.grid()
plt.legend(["optimal", "traning set"], loc='upper left')
plt.show()