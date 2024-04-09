import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

C=np.zeros([100,100])




for j in range(0,100,1):
    for i in range(0,10,1):
        fold_dir="C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\4주차\\problem_1_data\\"
        file_name=str(j)+".csv"
        final_str=fold_dir+file_name

        temp_data=pd.read_csv(final_str, header=None)
        temp_data=temp_data.to_numpy()
        C[i*10:(i*10)+10, i*10:(i*10)+10] = temp_data[70:80, 80:90]
    
    



plt.imshow(C, cmap='viridis')
plt.axis('off')
plt.show()

# 0~9 10~19 20~29 30~39 ㅡㅡ 90~99
# 10~19
# 20~29
# 30~39