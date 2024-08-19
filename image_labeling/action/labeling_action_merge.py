
import glob
import pandas as pd

# 두 개의 CSV 파일을 읽어들입니다.
# 'file1.csv'와 'file2.csv'는 합치고 정렬할 CSV 파일의 경로입니다.
df1 = pd.read_csv(glob.glob("*_stand.csv")[0])
df2 = pd.read_csv(glob.glob("*_sit.csv")[0])

# 두 데이터프레임을 합칩니다.
df_combined = pd.concat([df1, df2])

# 'timestamp' 열을 기준으로 데이터를 정렬합니다.
df_sorted = df_combined.sort_values(by='timestamp', ascending=True)

# 정렬된 데이터를 새로운 CSV 파일로 저장합니다.
# 'sorted_combined_data.csv'는 저장할 CSV 파일의 경로입니다.
df_sorted.to_csv('action.csv', index=False)

print("두 CSV 파일이 합쳐져서 정렬되어 저장되었습니다.")
