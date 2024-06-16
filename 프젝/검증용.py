import numpy as np
import pandas as pd

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def select_features(file_path):
    path = directory + "heart_disease_new.csv"
    df = pd.read_csv(path)  # pandas를 사용하여 CSV 파일 읽기
    dataset = np.array(df)  # pandas DataFrame을 numpy 배열로 변환

    # 데이터 섞기
    np.random.shuffle(dataset)
    
    #================== 전처리====================
    
    #path데이터(heart_disease_new.csv)에서 ,를 사용해 구분하고 문자열로 1행 분리
    column_names = np.genfromtxt(path, delimiter=',', dtype=str, max_rows=1)
    
    # 열 이름을 기준으로 인덱스 찾아서 저장
    gender_idx = np.where(column_names == "gender")[0][0]
    neuro_idx = np.where(column_names == "a neurological disorder")[0][0]
    target_idx = np.where(column_names == "heart disease")[0][0]
    age_idx = np.where(column_names == "Age")[0][0]
    hbp_idx = np.where(column_names == "High blood pressure")[0][0]
    chol_idx = np.where(column_names == "Cholesterol")[0][0]
    height_idx = np.where(column_names == "height")[0][0]
    weight_idx = np.where(column_names == "Weight")[0][0]
    bmi_idx = np.where(column_names == "BMI")[0][0]
    bpm_idx = np.where(column_names == "BPMeds")[0][0]
    bloodsugar_idx = np.where(column_names == "blood sugar levels")[0][0]
    meat_idx = np.where(column_names == "meat intake")[0][0]
    smoke_idx = np.where(column_names == "Smoking")[0][0]
    
    # gender: female -> 0, male -> 1
    dataset[:, gender_idx] = np.where(dataset[:, gender_idx] == 'female', 0, 1)
    
    # a neurological disorder: no -> 0, yes -> 1
    dataset[:, neuro_idx] = np.where(dataset[:, neuro_idx] == 'yes', 1, 0)
    
    # heart disease: yes -> 1, no -> 0
    dataset[:, target_idx] = np.where(dataset[:, target_idx] == 'yes', 1, 0)
    
    # 결측치 처리 (각 열의 평균값으로 대체)
    for i in range(dataset.shape[1]):
        col = dataset[:, i].astype(float)
        mean_val = np.nanmean(col)  # 결측치를 제외한 평균값 계산
        col[np.isnan(col)] = mean_val  # 결측치를 평균값으로 대체
        dataset[:, i] = col
    
    # 데이터 형식을 float로 변환
    dataset = dataset.astype(float)
    
    #===========================================================#
    # y_data 추출
    y_data = dataset[:, -1]  # 타겟 값 추출
    
    
    # 특징 추출 (상관관계에 기반하여 조합)
    # 다양한 특징을 시도하였고 최종적으로 1, 5, 12를 사용
    # 키가 상관관계가 매우 커서 사용
    feature_1 = dataset[:, height_idx]
    # 특징 5번 이 데이터에서 구해져있는 bmi는 값이 이상한 것들이 많아 따로 bmi를 구해보았다.
    feature_2 = dataset[:, weight_idx] / ((dataset[:, height_idx]/100)**2) 
    # 특징 12번 데이터에서 흡연이 yes이면 무조건 고기를 섭취하므로 두 값을 더하고, 큰 상관관계인 키/5를 빼주어 특징을 구현
    feature_3 = ((dataset[:,smoke_idx]*10) + (dataset[:,meat_idx] * 10)) - dataset[:,height_idx] / 5
    
    # 선택한 특징들을 결합
    selected_features = np.column_stack((feature_1, feature_2, feature_3))

    
    # 특징 데이터 마지막 열에 y값 추가
    features = np.column_stack((selected_features, y_data))
    
    return features

# p(sigmoid 함수)
def sigmoid(Z):
    return 1/(1+np.exp(-Z))     # sigmoid 계산해 반환

# One-Hot Encoding 구현
def One_Hot(xy_data):
    y=xy_data[:,-1]
    n=len(xy_data)  # 데이터 갯수
    q=2 # output 경우의 수
    one_hot=np.zeros([n,q]) # zeros로 0 행렬 생성
    for i in range(n):
        one_hot[i,int(y[i])]=1  # 해당 위치 1로 변경
    return one_hot

# 분할하는 함수
def Split(SET,tra,val,test):
    np.random.shuffle(SET)  # 데이터 받아서 섞기
    tra_set=SET[0:int(len(SET[:,0])/(tra+val+test)*tra),:]      # Traning 비율 입력만큼 슬라이싱
    val_set=SET[int(len(SET[:,0])/(tra+val+test)*tra):int(len(SET[:,0])/(tra+val+test)*(tra+val)),:]    # Validation 비율 입력만큼 슬라이싱
    test_set=SET[int(len(SET[:,0])/(tra+val+test)*(tra+val)):len(SET[:,0]),:]   # Test 비율 입력만큼 슬라이싱
    return tra_set,val_set,test_set     # 슬라이싱 끝낸 배열들 반환

# 정확도 평가 함수
def ACC(y,y_hat):
    count=0     # 카운트할 변수 0으로 초기화
    for i in range(len(y)):
        if np.argmax(y[i,:])==np.argmax(y_hat[:,i]):  # y의 한 행과 y_hat의 한 열의 1의 위치가 같으면
            count+=1    # 1씩 카운트
    return count/len(y)*100       # 정확도 계산해 반환

# 평균제곱오차 함수
def MSE(y,y_hat):
    return np.mean((y.T-y_hat)**2)  # MSE 계산해 반환

# 순전파 함수
def Two_Layer_Neural_Network(xy_data,v,w):
    x=np.hstack([xy_data[:,:-1],np.ones([len(xy_data),1])]).T
    n=len(xy_data)  # 데이터 갯수
    q=2 # output 경우의 수

    A=v@x       # 알파행렬 계산   # (L, 1) = (L, M+1) @ (M+1, 1)
    b=sigmoid(A)    # b 계산      # (L, 1)
    b=np.vstack([b,np.ones([1,n])]) # b 행렬에 더미데이터 추가    # (L+1, 1)
    B=w@b       # 베타행렬 계산   # (Q, 1) = (Q, L+1) @ (L+1, 1)
    y_hat=sigmoid(B)    # y_hat 계산  # (Q, 1)
    
    Y_hat=np.zeros([q,n])   # max값만 1로 만들기 위해 0 행렬 생성

    for i in range(n):
        if (y_hat[0,i]>=0.5):
            Y_hat[1,i]=1
        elif (y_hat[0,i]<0.5):
            Y_hat[0,i]=1    

    return y_hat,Y_hat

# Confusion Matrix 함수
def Confusion_Matrix(One_hot,Y_hat):
    Q=2    # output 경우의 수
    N=len(One_hot)  # 데이터 개수
    matrix=np.zeros([Q+1,Q+1])  # confusion matrix 베이스가 될 0행렬 만들기
    diagonal_num=0  # 대각 성분 개수 셀 변수 0으로 초기화
    for n in range(N):  # 데이터 개수만큼 반복
        target=One_hot[n,:]     # 실제값 데이터 1개만 뽑아오기
        output=Y_hat[:,n]       # y_hat 데이터 1개만 뽑아오기
        matrix[np.argmax(output),np.argmax(target)]+=1    # 뽑아온 데이터들 1의 위치에 맞춰서 confusion matrix 만들기
    for i in range(Q):  # 개수만 세놓은 confusion matrix 데이터 정리
        if (np.sum(matrix[i,:Q])==0):   # 한 행의 합이 0이면 0으로 나눌 수 없고 nan이 나와서
            matrix[i,Q]=0               # 0으로 변경
        elif (np.sum(matrix[:Q,i])==0): # 한 열의 합이 0이면 0으로 나눌 수 없고 nan이 나와서
            matrix[Q,i]=0               # 0으로 변경
        else:   # 위 경우가 아니면
            matrix[i,Q]=matrix[i,i]/np.sum(matrix[i,:Q])    # 행 데이터 정리
            matrix[Q,i]=matrix[i,i]/np.sum(matrix[:Q,i])    # 열 데이터 정리
            diagonal_num+=matrix[i,i]   # 대각성분 개수 세기
    matrix[Q,Q]=diagonal_num/N          # 전체 데이터에 대해 정확도 계산

    return matrix

# 데이터 불러와서 저장
directory="C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\"
data=select_features(directory)

np.random.shuffle(data)

yes_data_index=np.where(data[:,-1]==1)[0]
yes_data=data[yes_data_index,:] # yes 데이터 추출

no_data_index=np.where(data[:,-1]==0)[0]
no_data=data[no_data_index,:]   # no 데이터 추출

yes=500  # 몇개씩 뽑을지
no= 500
data_set=np.vstack([yes_data[:yes,:],no_data[:no,:]])  # yes:no 비율 검사용 셑 저장

_,_,Test_Set=Split(data_set,0,0,1)  # 데이터 분할

One_hot_Test=One_Hot(Test_Set)  # Test 데이터에 대한 One_hot 생성

# =============================================================================
# # 초기 가중치 생성 함수
# def weight(xy_data,l,spread):
#     x=xy_data[:,:-1].T
#     q=1 # output layer node 수 (Class with)
#     m=len(x)  # input layer node 수 (더미 제외)
# 
#     v=np.random.rand(l,m+1)*2*spread-spread
#     w=np.random.rand(q,l+1)*2*spread-spread
#     
#     return v,w
# 
# w_hidden,w_output=weight(Test_Set,L,0.01)   # 초기 가중치 생성
# =============================================================================
# w_hidden=pd.read_csv(fold_directory+"w_hidden.csv",header=None).to_numpy()
# w_output=pd.read_csv(fold_directory+"w_output.csv",header=None).to_numpy()

# address='0\\'
best_v = pd.read_csv('C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\프젝\\프젝 최종 2020142001 곽종근\\w_hidden.csv', header=None).to_numpy()
best_w = pd.read_csv('C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\프젝\\프젝 최종 2020142001 곽종근\\\w_output.csv', header=None).to_numpy()
w_hidden,w_output=best_v,best_w

y_hat_Test,Y_hat_Test=Two_Layer_Neural_Network(Test_Set,w_hidden,w_output)    # Test Set 순전파 batch size=N 진행해서 y_hat들 받아오기
Test_Acc=ACC(One_hot_Test,Y_hat_Test) # Test 정확도 저장
MSE_Test=MSE(Test_Set[:,-1].reshape(-1,1),y_hat_Test)      # Test 평균제곱오차 저장

Confusion_Matrix_Test=Confusion_Matrix(One_hot_Test,Y_hat_Test)    # best_Y_hat으로 Confusion Matrix 생성
print(Confusion_Matrix_Test)

