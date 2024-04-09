import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

C=np.zeros([100,100])





for j in range(0,10,1):
    for i in range(0,10,1):
    fold_dir="C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\4주차\\problem_1_data\\"
    file_name=str(j)+".csv"
    final_str=fold_dir+file_name

    temp_data=pd.read_csv(final_str, header=None)
    temp_data=temp_data.to_numpy()                        #여기까지가 파일을 0~99번까지 계속 반복해줌
    td=temp_data[70:80, 80:90] 
    

  
   
                0             1            2             3          4                5          6          7                  8                 9                                 
        C[0:10, 0:10] [0:10, 10:20] [0:10, 20:30] [0:10, 30:40] [0:10, 40:50] [0:10, 50:60] [0:10, 60:70] [0:10, 70:80] [0:10, 80:90] [0:10, 90:100]
              10             11             12            13                14          15              16           17              18         19                                
        C[10:20, 0:10] [10:20, 10:20] [10:20, 20:30] [10:20, 30:40] [10:20, 40:50] [10:20, 50:60] [10:20, 60:70] [10:20, 70:80] [10:20, 80:90] [10:20, 90:100]
        C[20:30, 0:10] [20:30, 10:20] [20:30, 20:30] [20:30, 30:40] [20:30, 40:50] [20:30, 50:60] [20:30, 60:70] [20:30, 70:80] [20:30, 80:90] [20:30, 90:100]
    
C[i:i+10, ]


plt.imshow(C, cmap='viridis')
plt.axis('off')
plt.show()

# 0~9 10~19 20~29 30~39 ㅡㅡ 90~99
# 10~19
# 20~29
# 30~39