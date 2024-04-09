#시현

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

fold_dir="C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\5주차\\lin_regression_data_01.csv"
file_data=pd.read_csv(fold_dir,header=None) # pandas로 파일 읽어들여 저장하기
file_data=file_data.to_numpy()  # numpy로 변환

w0,w1=0.0,0.0   # w0와 w1값 초기화
save_w0=np.array([])    # w0 변동 저장할 array 생성
save_w1=np.array([])    # w1 변동 저장할 array 생성
save_w0=np.append(save_w0,w0)   # w0 초기값 저장
save_w1=np.append(save_w1,w1)   # w1 초기값 저장
save_MSE=np.array([])   # MSE 변동 저장할 array 생성
N=len(file_data[:,0])   # 행 개수 세기
count=0

# w0 미분계수 구하는 함수
def def_w0(x,w11):
    SE_sum=0.0  # 지역변수 초기화
    global N    # 전역변수 사용
    for i in range(N):
        SE_sum+=2*file_data[i,0]*((file_data[i,0]*x+w11)-file_data[i,1])
    return SE_sum/N

# w1 미분계수 구하는 함수    
def def_w1(w00,x):
    SE_sum=0.0  # 지역변수 초기화
    global N    # 전역변수 사용
    for i in range(N):
        SE_sum+=2*(w00*file_data[i,0]+x-file_data[i,1])
    return SE_sum/N
    
# 평균제곱오차 구하는 함수
def MSE(w00,w11):
    SE_sum=0.0  # 지역변수 초기화
    global N    # 전역변수 사용
    for i in range(N):
        SE_sum+=((w00*file_data[i,0]+w11)-file_data[i,1])**2
    return SE_sum/N

# 경사하강법 함수
def GDM(a,c):
    global w0,w1,save_w0,save_w1,save_MSE,count   # 전역변수사용
    save_MSE=np.append(save_MSE,MSE(w0,w1)) # 평균제곱오차 초기값 저장
    while(1):   # 무한반복
        w0=w0-a*def_w0(w0,w1)               # w0 변동
        w1=w1-a*def_w1(w0,w1)               # w1 변동
        save_w0=np.append(save_w0,w0)           # wo 변동값 저장
        save_w1=np.append(save_w1,w1)           # w1 변동값 저장
        save_MSE=np.append(save_MSE,MSE(w0,w1)) # 평균제곱오차 변동값 저장
        count+=1    # 반복 횟수 세기
        if (abs(save_w0[count]-save_w0[count-1])<c)and(abs(save_w1[count]-save_w1[count-1])<c)and(abs(save_MSE[count]-save_MSE[count-1])<c):
            break

Learning_Rate=0.0075    # 학습률 지정
# rep=5000                # 반복횟수 지정
error=1e-10                 # 멈출 변화량 지정
GDM(Learning_Rate,error)    # 경사하강법 진행

# Optimal Solution의 x축 범위 정하기
mini_x=min(file_data[:,0])
maxi_x=max(file_data[:,0])
Xo=np.arange(mini_x-1,maxi_x+1,0.1)

Opt_Sol=w0*Xo+w1     # Optimal Solution

plt.figure(figsize=(10,6))  # figsize=(10,6)의 figure객체 생성
# plt.scatter(file_data[:,0],file_data[:,1],c='r',label="actual value")    # x축: 추의 무게, y축: 늘어난 길이 / 실제값 점찍기
plt.plot(Xo,Opt_Sol,c='b',label="Optimal Solution")   # Analytic solution 그리기
plt.title("Spring data") # title 설정 
plt.xlabel("weight [g]") # xlabel 설정 
plt.ylabel("length [cm]") # ylabel 설정 
plt.grid(True) # grid 생성 
plt.legend(loc="upper left") # legend 사용 그래프 식별

X=np.arange(1,count+2)  # x축 범위 정하기
plt.figure(figsize=(10,6))  # figsize=(10,6)의 figure객체 생성
plt.plot(X,save_w0[:],c='c',label="w0")     # w0 변동 그래프 그리기
plt.plot(X,save_w1[:],c='m',label="w1")     # w1 변동 그래프 그리기
plt.title("epoch") # title 설정 
plt.xlabel("step") # xlabel 설정 
plt.ylabel("w0,w1") # ylabel 설정 
plt.grid(True) # grid 생성 
plt.legend(loc="upper left") # legend 사용 그래프 식별

plt.figure(figsize=(10,6))  # figsize=(10,6)의 figure객체 생성
plt.plot(X,save_MSE[:],c='y',label="MSE")   # MSE 변동 그래프 그리기
plt.title("Mean Square Errer") # title 설정 
plt.xlabel("step") # xlabel 설정 
plt.ylabel("MSE") # ylabel 설정 
plt.grid(True) # grid 생성 
plt.legend(loc="upper right") # legend 사용 그래프 식별