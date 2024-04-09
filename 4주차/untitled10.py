import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

A=np.zeros([100,100])
B=np.zeros([100,100])

# j=0
# fold_dir="C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\4주차\\problem_1_data\\"
# file_name=str(j)+".csv"
# final_str=fold_dir+file_name

# temp_data=pd.read_csv(final_str, header=None)
# temp_data=temp_data.to_numpy()

j=0
i=0
for j in range(0,100,1):
    fold_dir="C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\4주차\\problem_1_data\\"
    file_name=str(j)+".csv"
    final_str=fold_dir+file_name

    temp_data=pd.read_csv(final_str, header=None)
    temp_data=temp_data.to_numpy()
    A[:,j]=temp_data[:,25]
    
# for i in range(0,100,1):
#     A[:,i]=temp_data[:,25]



plt.imshow(A, cmap='viridis')
plt.axis('off')
plt.show

# temp_data의 [:,25]를 추출해서 zeros를 한 A의 1열부터 100열까지 차례로 대입한다.
