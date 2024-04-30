def gbs_mean(X, K, k):
    x_min = X.min()  # 데이터의 최솟값
    x_max = X.max()  # 데이터의 최댓값
    
    mu = x_min + ((x_max - x_min) / (K - 1)) * k # 각 가우시안 함수의 평균 계산
    return mu

def variance(X,K):
    x_min = X.min()  # 데이터의 최솟값
    x_max = X.max()  # 데이터의 최댓값
    
    v = (x_max - x_min) / (K - 1) #모든 가우스 함수의 분산
    return v

def GBF(X, mu, v):
    simple = (X - mu) / v
    G = np.exp((-1/2) * (simple ** 2))
    return G