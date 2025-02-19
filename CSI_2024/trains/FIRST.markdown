# 초기 이미지 전처리 과정(위치, 사람수 라벨링 X)

- CSI_DB : Sit, Stand로만 분류한 RAW 데이터

1. 촬영된 이미지 데이터를 **Sit, Stand** 로 분류하여 CSI_DB 디렉토리에 저장합니다.
```
\data
  \sit\*.jpg
  \stand\*.jpg
```
2. 분류한 이미지(`\data`)를 `CSI_VisionRecog` 디렉토리에 넣고 이미지 라벨링을 합니다. 

- L train3
- R train4
```
conda activate pose
python3 inference_0702.py -p yolov8x-pose -m ./runs/train3/model.h5 --source ./data/sit/
```

3. sit, stand 내부에 결과가 라벨링된 csv가 생겨납니다. 

4. `CSI_preprocess` 디렉토리에서 라벨링된 CSV 파일과 기존 시간에 취득된 CSI 데이터를 가져와 전처리를 진행합니다.
```
# /CSI_preprocess

- amplitude_l_0703.py
- CSI 데이터 값.csv
- 라벨링.csv
```

5. 그럼 생기는 processed_L 과 processed_R 데이터는 학습할 데이터가 됩니다. 

```
python3 amplitude_l_0703.py 
python3 amplitude_R_0703.py
python3 ./processed_L matching.py
python3 ./processed_R matching.py
```
- CSI_DB는 120으로 맞추기 전 파일을 저장하는 공간
- DB는 이미지 파일 저장하는 공간
- 이후에 후반 전처리를 위해 학습 코드에서 0 서브 캐리어를 제거하고 row를 120로 맞추는 작업을 진행하게됩니다. 

6. 모델 학습(cd /home/keti/dongwoo/CSI_modeling/CSI_BI_LSTM)
```
conda activate analysis
python3 run.py --model BiLSTM
```
