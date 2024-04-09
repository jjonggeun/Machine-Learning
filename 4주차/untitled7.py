import numpy as np

monst_HP=100

hit_count=0

while monst_HP>0:
    hit_count=hit_count+1
    hit_dam=np.round(5*np.random.rand(1,1)+20)        #while문 안에 넣어야 대미지값이 바뀌고 밖에 있으면 랜덤값으로 고정 되어서 그 값을 계속 사용함
    monst_HP=monst_HP-hit_dam
    
    print("몬스터를", hit_count,"회 때렸습니다.")
    if monst_HP < 0:                           #조건문에 같다를 넣어야하는지 아닌지 잘 생각하기
        print("현재 몬스터 HP: 0")
    else: 
        print("현재 몬스터 HP: ", monst_HP)
    