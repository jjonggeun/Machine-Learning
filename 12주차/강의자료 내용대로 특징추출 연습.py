import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x0_set = np.array([], dtype='float32')
x0_set = np.resize(x0_set, (0, 5))

# 가로축 확률밀도함수로 변환 기대값
def feature_1(input_data):
    return np.mean(np.mean(input_data, axis=1))

# 가로축 확률밀도함수로 변환 분산
def feature_2(input_data):
    return np.mean(np.var(input_data, axis=1))

# 세로축 확률밀도함수로 변환 기대값
def feature_3(input_data):
    return np.mean(np.mean(input_data, axis=0))

# 세로축 확률밀도함수로 변환 분산
def feature_4(input_data):
    return np.mean(np.var(input_data, axis=0))

# 대각원소 배열 추출 밀도함수로 변환 기대값
def feature_5(input_data):
    diagonal = np.diag(input_data)
    return np.mean(diagonal)

for i in range(1, 501):
    temp_name = 'C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\Machine-Learning\\MINIST Data\\' + '0_' + str(i) + '.csv'
    temp_image = pd.read_csv(temp_name, header=None)
    temp_image = temp_image.to_numpy(dtype='float32')
    
    x0 = feature_1(temp_image)
    x1 = feature_2(temp_image)
    x2 = feature_3(temp_image)
    x3 = feature_4(temp_image)
    x4 = feature_5(temp_image)
    
    x0_feature = np.array([x0, x1, x2, x3, x4], dtype='float32').reshape(1, -1)
    x0_set = np.concatenate((x0_set, x0_feature), axis=0)


