import numpy as np

a=np.zeros([4,5])

b=np.arange(1,21,1)

x_move_num=len(a[1,:])  #행의 길이를 추출하는 코드
y_move_num=len(a[:,1])  #1번째 열의 길이를 구해라 = 4


x_move_range=np.arange(0,x_move_num,1)
y_move_range=np.arange(0,y_move_num,1)

for y in y_move_range:
    for x in x_move_range:
        a[0,x]=b[x+x_move_num*y]

    