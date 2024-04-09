#U출력
import numpy as np  #numpy를 np라는 이름으로 사용하겠다.
import pandas as pd #pandas를 pd라는 이름으로 사용하겠다.
import matplotlib.pyplot as plt  #matplotlib.pyplot를 plt라는 이름으로 사용하겠다.

B=np.zeros([100,100]) #B라는 100by100의 zeros배열을 선언해놓는다.





for j in range(0,100,1): #파일이 총 100개가 있기때문에 이 100개의 파일을 모두 사용하기 위해 0~99까지 for문(반복문)을 이용해 전부 선언, 오름차순이므로 0부터
    fold_dir="C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\4주차\\problem_1_data\\"
    file_name=str(j)+".csv" #반복문 j를 j.csv로 선언하여 0~99.csv가 전부 불러와질 수 있도록 선언한다.
    final_file=fold_dir+file_name  #fold_dir과 fie_name을 합쳐 파일이 해당하는 위치의 0~99개 파일을 불러오는 final_str을 최종 선언

    temp_data=pd.read_csv(final_file, header=None) #final_file이라는 최종 파일에 있는 데이터를 읽어오는데 헤더파일을 생성하지 않고 읽어온다. dataframe형태
    temp_data=temp_data.to_numpy() #위에서 불러온 파일은 dataframe형태이므로 슬라이싱이 안되기에 이를 numpy배열로 변환시킨다.
    B[j,:]=temp_data[10,:] #위에서 numpy배열로 변환해주었기때문에 슬라이싱이 가능하고 dataframe형태는 불가능하다.
                            #25번째 col을 행렬 A의 0번부터 99번까지 col에 삽입한다.
    


plt.imshow(B, cmap='viridis')
plt.axis('off')
plt.show


