# 데이터 안내

원자료는 [Kaggle 수업 mirror](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle)에서 제공한 세 MAT 파일이다.

- 2017-05-12_batchdata_updated_struct_errorcorrect.mat
- 2018-02-20_batchdata_updated_struct_errorcorrect.mat
- 2018-04-12_batchdata_updated_struct_errorcorrect.mat

수 GB 원자료는 배포 패키지에서 제외했다. 원자료 없이도 `work/day2/features_all.csv`로 DAY2 학습을 재현할 수 있다. 원자료가 필요하면 루트에서 `python work/download_data.py`를 실행한다.

Batch 1: 46개 중 비정상 5개 제외 → 41개. 중단·재개 셀 0-4 수명 보정값은 공식 로더의 662/981/1060/208/482 cycles다.

Batch 2: 47개 중 cycle_life 결측 8개 제외 → 39개.

Batch 3: 46개 중 비정상 6개 제외 → 40개.

DAY2 ΔQ 피처는 빈 첫 배열 유무를 확인해 실제 방전 cycle 100-10을 정렬했다. Array index는 Batch 1에서 100/10, Batch 2·3에서 99/9다. 각 셀의 실제 사용 인덱스가 CSV에 기록되어 있다.

원출처의 데이터 이용·재배포 조건을 확인하고 따른다.
