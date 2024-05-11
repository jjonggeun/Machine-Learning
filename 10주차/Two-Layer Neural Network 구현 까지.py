import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# 데이터 불러오기
fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\NN_data.csv"
temp_data = pd.read_csv(fold_dir)
temp_data = temp_data.to_numpy()   

# 데이터 분리
x0 = temp_data[:, 0].reshape(-1,1)
x1 = temp_data[:, 1].reshape(-1,1)  # temp_data 1열을 저장
x2 = temp_data[:, 2].reshape(-1,1)  # temp_data 2열을 저장
y = temp_data[:, 3].reshape(-1,1)   # temp_data 3열을 y로 저장

# 시그모이드 함수를 선언
def Sigmoid(x):
    return 1 / (1 + np.exp(-x))
                
xtotal_data= np.hstack((x0,x1,x2))
dummy_data = np.ones((len(xtotal_data), 1))
x_with_dummy = np.hstack((xtotal_data, dummy_data)) 

y_target = np.zeros((len(x0), len(np.unique(y))))

# y 값을 인덱스로 사용하여 해당 위치의 값만 1로 변경
# 이 for문을 실행하면 y값에 해당하는 위치의 0값이 1로 바뀜
for i in range(len(y)):
    y_target[i, int(y[i])-1] = 1

# 입력 속성 수 추출
M = x_with_dummy.shape[1]

# 출력 클래스 수 추출
output_size = y_target.shape[1]

# hidden layer의 노드 수를 원하는 값으로 바꾸는 부분
hidden_size = 10

# weight 초기화 (np.random.rand를 사용하여 표현)
v = np.random.rand(hidden_size,M)   
w = np.random.rand(output_size,hidden_size+1)  


# bias 초기화 (모든 요소가 1로 설정)
bias_input_hidden = np.ones((1, hidden_size))
bias_hidden_output = np.ones((1, output_size))


A=v@x_with_dummy.T
b=Sigmoid(A)
b_with_dummy = np.vstack([b,np.ones([1,len(xtotal_data)])])
B=w@b_with_dummy
y_hat = Sigmoid(B)
