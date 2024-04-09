import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fold_dir="C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\4주차\\problem_2_data.csv"
file_data=pd.read_csv(fold_dir)
file_data=file_data.replace(np.nan,None)

Sam_F=np.zeros([1,5])   # Sampling Frequency 리스트 [1,5] 생성
Sam_N=np.zeros([1,5])   # Sampling Number 리스트 [1,5] 생성

for i in range(5):
    for j in range(300):
        if file_data.loc[j,"Signal"+str(i+1)]!=None:    # low의 값이 None이 아닌 경우
            Sam_N[0,i]+=1                               # Sampling Number에 카운트
    Sam_F[0,i]=Sam_N[0,i]//2                            # Sampling Frequency 구하기(2초니까 /2)

Sam_F.sort()    # Sampling Frequency 오름차순 정렬
Fmin=Sam_F[0,0] # 가장 작은 Fs 저장

for i in range(5):
    Sam_F[0,i]=Sam_N[0,i]//2
    
Sam_Sig=np.zeros([300,5])

for i in range(5):
    for j in range(300):
        if j%(Sam_F[0,i]//Fmin)==0:
            Sam_Sig[j//int(Sam_F[0,i]//Fmin):,i]=file_data.loc[j,"Signal"+str(i+1)]

plt.figure(figsize=(10,6))
for i in range(5):
    plt.plot(np.arange(0,2,1/Fmin),Sam_Sig[0:60,i])
plt.title("Signal Graphs")
plt.xlabel("Time [sec]")
plt.ylabel("Value [V]")
plt.grid(True)
plt.legend(["Signal1","Signal2","Signal3","Signal4","Signal5"],loc="upper right")
