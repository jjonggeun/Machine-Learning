import pandas as pd

fold_dir = "C:\\Users\\user\\OneDrive - 한국공학대학교\\바탕 화면\\3학년 1학기\\머신러닝실습\\4주차\\problem_2_data.csv"
temp_data = pd.read_csv(fold_dir)

# 1열에는 60행 이후로, 2열에는 240행 이후로, 3열에는 120행 이후로, 5열에는 180행 이후로 NaN 값을 삭제
for col, nan_index in zip(temp_data.columns, [60, 240, 120, 180]):
    temp_data[col] = temp_data[col].apply(lambda x: x if temp_data.index.get_loc(x) <= nan_index else None)

# NaN 값을 삭제
cleaned_temp_data = temp_data.dropna()

print(cleaned_temp_data)
