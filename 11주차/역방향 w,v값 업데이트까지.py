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

# 시그모이드 함수의 미분
def Sigmoid_derivative(x):
    return Sigmoid(x) * (1 - Sigmoid(x))
                
xtotal_data= np.hstack((x0,x1,x2))
dummy_data = np.ones((len(xtotal_data), 1))
x_with_dummy = np.hstack((xtotal_data, dummy_data)) 

# y_target은 One-Hot 인코딩된 코드이고
# 데이터셋에 있는 각 샘플에 대해 하나의 행을, 클래스 개수만큼 열
# 해당된 열의 값은 1로 설정하고, 나머지는 0으로
y_target = np.zeros((len(x0), len(np.unique(y))))

# y 값을 인덱스로 사용하여 해당 위치의 값만 1로 변경
# 이 for문을 실행하면 y값에 해당하는 위치의 0값이 1로 바뀜
for i in range(len(y)):
    y_target[i, int(y[i])-1] = 1

# 입력 속성 수 추출
# .shape[1]은 y_target의 열의 개수를 구해주는 것으로 여기서 속성 수를 나타냄
M = x_with_dummy.shape[1]

# 출력 클래스 수 추출
output_size = y_target.shape[1]

# hidden layer의 노드 수를 원하는 값으로 바꾸는 부분
hidden_size = 10

# weight 초기화 (np.random.rand를 사용하여 표현)
v = np.random.rand(hidden_size,M)   
w = np.random.rand(output_size,hidden_size+1)  
w_no = np.random.rand(output_size,hidden_size)


# bias 초기화 (모든 요소가 1로 설정)
bias_input_hidden = np.ones((1, hidden_size))
bias_hidden_output = np.ones((1, output_size))

# 최종적으로 가중치를 곱한 값을 시그모이드 함수에 2번 적용하여 
# 최종 y_hat출력
A=v@x_with_dummy.T
b=Sigmoid(A)
b_with_dummy = np.vstack([b,np.ones([1,len(xtotal_data)])])
B=w@b_with_dummy
y_hat = Sigmoid(B)

learning_rate=0.01
epochs = 1000

y_targett=y_target.T

v_new = np.zeros_like(v)
w_new = np.zeros_like(w)

mi_y = np.zeros([output_size, 1800]) 

for i in range(1800):
    yhat_qn = y_hat[:, i]  
    ytarget_qn = y_targett[:, i]
    mi_y[:, i] = yhat_qn - ytarget_qn 

wmse=2*(mi_y)*y_hat*(1-y_hat)@b_with_dummy.T
vmse=np.dot(((np.dot(w_no.T, mi_y)) * b*(1-b)),  x_with_dummy)

for i in range(4):
    v = v - learning_rate*vmse
    w = w - learning_rate*wmse
    # v값이 너무 큼 값 조정 필요
# v_new = v - learning_rate*vmse
# w_new = w - learning_rate*wmse
# v_new_2 = v_new - learning_rate*vmse
# w_new_2 = w_new - learning_rate*wmse

        