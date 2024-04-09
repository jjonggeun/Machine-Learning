import numpy as np
import pandas as pd

a=np.arange(1,31,1)
a=np.reshape(a,[6,5])


save_a=pd.DataFrame(a)

fold_dir="C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\4주차\\"
file_name="bss.csv"
final_str=fold_dir+file_name

# save_a.to_csv(final_str)
save_a.to_csv(final_str, index=None, header=None) #헤더,인덱스를 안쓰고싶으면 이렇게 사용