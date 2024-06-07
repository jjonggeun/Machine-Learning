import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras

#Keras의 mnist 데이터 set 불러오기
mnist = keras.datasets.mnist
(X_train_full, Y_train_full), (X_test, Y_test) = mnist.load_data()

#Train set, Validation set으로 나누기 (5000개의 데이터만 사용)
X_valid, X_train = X_train_full[:500], X_train_full[500:5000]
Y_valid, Y_train = Y_train_full[:500], Y_train_full[500:5000]

#models.Sequential을 이용해 CNN모델 생성
model = keras.models.Sequential([
    keras.layers.Conv2D(128, (3,3), activation = 'relu',
                        strides=(1,1), input_shape=(28,28,1)),
    keras.layers.MaxPool2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dense(units=10, activation='softmax')
    ])

#모델 요약 및 학습 방식 환경 설정
# model.summary()
# model.compile(optimizer = 'adam',
#               loss='sparse_categorical_crossentropy',
#               metrics['accuracy'])

#학습진행
history = model.fix(X_train, Y_train, epochs = 10, batch_size=64,
                    validataion_data=(X_valid, Y_valid))

#Test set을 이용하여 정확도 확인
model.evaluate(X_test, Y_test)

#학습결과 그래프로 확인
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

#잘못 예측 값과 실제 비교 후 시각화
pred=model.predict(X_test)
pred_class = np.argmax(pred, axis=1)
true_class = np.argmax(Y_test, axis=1)

mis_index = np.where(pred_class != true_class)
mis_index = np.array(mis_index).reshape(-1)

plt.figure(figsize=(10,10), dpi=150)
for i in range(5):
    index = mis_index[i]
    plt.subplot(1,5,i+1)
    plt.axis('off')
    plt.grid(False)
    plt.imshow(X_test[index, :, :], cmap='gray')
    plt.title(f'True: {true_class[index]}, Pred: {pred_class[index]}')
plt.show()