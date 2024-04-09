import numpy as np

# a=2
# b=3
# c=4

# if a>b:                 #false니까 밑에 열 실행 x
#     print("a는 b보다 크다")
# elif a<b:
#     print("a는 b보다 작다")
# else:
#     print("그 외 모든 경우")
# print(a>b)


# b=np.arange(1,5,0.5)

# save_a=np.zeros([10,1])
save_a=np.array([])    #저장공간 사이즈를 모르면 그냥 .array로 빈공간으로 만들기
b=np.arange(3,13,1)

iter_range=np.arange(1,11,1)

# for j in iter_range:   #j라는 것을 b의 순서로 바꾸겠다
#     save_a[j-1,0]=b[j-1]

for j in b:
    save_a[j-3,0]=b[j-3]


for j in b:
    j=3
    save_a=np.append(save_a,j)
    
# j=1
# save_a[0,0]=b[0,0]    #b가 c인지r인지 설정 안하면 0,0이 아니라 0으로 가능
# save_a[1,0]=b[0,0]