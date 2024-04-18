import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\6주차\\lin_regression_data_02.csv"
temp_data = pd.read_csv(fold_dir)
temp_data = temp_data.to_numpy()

N = len(temp_data[:, 0])

x = temp_data[:, :2]
x = np.hstack((x, np.ones([N, 1])))

y = temp_data[:, 2]
y = np.reshape(y, [N, 1])

w = np.array([[1.0], [2.0], [3.0]])
save_w = np.empty((3, 0))
save_w = np.hstack((save_w, w))

MSE = np.mean((x @ w - y) ** 2)
save_MSE = np.empty((0))
save_MSE = np.hstack((save_MSE, MSE))

x0 = np.arange(min(x[:, 0]) - 1, max(x[:, 0]) + 1, 0.1)
x1 = np.arange(min(x[:, 1]) - 1, max(x[:, 1]) + 1, 0.1)
x0, x1 = np.meshgrid(x0, x1)
x0 = x0.flatten()  # 1차원 배열로 변환
x1 = x1.flatten()  # 1차원 배열로 변환

z = w[0, 0] * x0 + w[1, 0] * x1 + w[2, 0]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(temp_data[:, 0], temp_data[:, 1], temp_data[:, 2], c="r", label="data")
ax.set_xlabel("X0")
ax.set_ylabel("X1")
ax.set_zlabel("Y")
ax.grid(True)
ax.legend(loc="upper left")

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(temp_data[:, 0], temp_data[:, 1], temp_data[:, 2], c="r", label="data")
ax.plot_surface(x0.reshape(x0.shape[0], -1), x1.reshape(x1.shape[0], -1), z.reshape(x0.shape[0], -1), alpha=0.5,
                label="face")
ax.set_xlabel("X0")
ax.set_ylabel("X1")
ax.set_zlabel("Y")
ax.grid(True)
ax.legend(loc="upper left")


def GDM(a, n):
    global x, w, save_w, save_MSE
    for i in range(n):
        w = w - a * ((x.T) @ (x @ w - y)) / len(y)
        save_w = np.hstack((save_w, w))
        mse = np.mean((x @ w - y) ** 2)
        save_MSE = np.hstack((save_MSE, mse))
    return mse


Learning_Rate = 0.1
rep = 100
MSE = GDM(Learning_Rate, rep)

X = np.arange(0, rep + 1)
plt.figure()
plt.plot(X, save_w[0, :], c='r', label="w0")
plt.plot(X, save_w[1, :], c='g', label="w1")
plt.plot(X, save_w[2, :], c='b', label="w2")
plt.title("epoch")
plt.xlabel("step")
plt.ylabel("weight")
plt.grid(True)
plt.legend(loc="upper left")

X = np.arange(0, rep + 1)
plt.figure()
plt.plot(X, save_w[0, :], c='r', label="w0")
plt.plot(X, save_w[1, :], c='g', label="w1")
plt.plot(X, save_w[2, :], c='b', label="w2")
plt.title("epoch")
plt.xlabel("step")
plt.ylabel("weight")
plt.grid(True)
plt.legend(loc="upper left")


plt.figure()
plt.plot(X, save_MSE[:], c='m', label="MSE")
plt.title("epoch")
plt.xlabel("step")
plt.ylabel("Mean Sqaure Error")
plt.grid(True)
plt.legend(loc="upper right")

z = w[0, 0] * x0 + w[1, 0] * x1 + w[2, 0]
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(temp_data[:, 0], temp_data[:, 1], temp_data[:, 2], c='b', marker='o', label="real data")
ax.scatter(x[:, 0], x[:, 1], w[0, 0] * x[:, 0] + w[1, 0] * x[:, 1] + w[2, 0], c='r', marker='s',
           label="y_hat data")
ax.plot_surface(x0.reshape(x0.shape[0], -1), x1.reshape(x1.shape[0], -1), z.reshape(x0.shape[0], -1), alpha=0.5,
                label="face")
ax.set_xlabel("X0")
ax.set_ylabel("X1")
ax.set_zlabel("Y")
ax.grid(True)
ax.legend(loc="upper left")
plt.show()
