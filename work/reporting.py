"""계산 결과를 README와 제출 형식의 성능표로 내보내는 단일 경로."""
from pathlib import Path
import json
import shutil
import hashlib
import platform
from datetime import datetime,timezone
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def performance_rows(r):
    b2,b3=r['test']['Batch 2'],r['test']['Batch 3']
    return [
        ('Train (Batch 1 CV)',r['CV_MAPE'],'%',f"Development {r['n_development']}셀, nested group CV 4-fold 평균"),
        ('Valid (Batch 1 Hold-out)',r['valid']['MAPE'],'%',f"정책 분리 {r['n_valid']}셀"),
        ('Test (Batch 2)',b2['MAPE'],'%',f"Batch1 전체 학습 후 {r['n_batch2']}셀"),
        ('Gap (Train-Valid)',r['gap_train_valid_pp'],'pp','Valid - Train'),
        ('Gap (Valid-Test)',r['gap_valid_test_pp'],'pp','Test(Batch2) - Valid'),
        ('Gap (Target-Test)',r['gap_target_test_pp'],'pp','Test(Batch2) - 논문 참고 목표 9.1'),
        ('Test (Batch 3)',b3['MAPE'],'%',f"추가 {r['n_batch3']}셀"),
        ('Gap (Batch2-Batch3)',r['gap_batch2_batch3_pp'],'pp','Test(Batch3) - Test(Batch2)'),
        ('Gap (Target-Test, Batch 3)',r['gap_target_batch3_pp'],'pp','Test(Batch3) - 9.1 (공통 참고값; Batch3 전용 논문 기준 아님)'),
    ]

def export_performance(r):
    df=pd.DataFrame(performance_rows(r),columns=['index','value','unit','note'])
    df.to_csv(ROOT/'work/day2/model_performance.csv',index=False)
    (ROOT/'results').mkdir(exist_ok=True)
    df.to_csv(ROOT/'results/model_performance.csv',index=False)

def export_readme(r):
    d=json.loads((ROOT/'work/day1_design_results.json').read_text())
    policies=pd.read_csv(ROOT/'work/day1_policy_mean.csv')
    candidates=pd.read_csv(ROOT/'work/day2/candidate_comparison.csv')
    b2,b3=r['test']['Batch 2'],r['test']['Batch 3']
    summary=d['batch_summary']
    def md(headers,rows):
        return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows])
    perf=md(['구분','MAPE (%) / Gap (%p)','비고'],[(a,f'{v:+.2f}' if unit=='pp' else f'{v:.2f}',note) for a,v,unit,note in performance_rows(r)])
    life=' / '.join(f"{s['batch']}: 중앙값 {s['median']:.1f}, 단수명 {100*s['short']/s['n']:.1f}%, 장수명 {100*s['long']/s['n']:.1f}%" for s in summary)
    dq=' / '.join(f"{s['batch']}: ρ={s['dq_rho']:.3f}" for s in summary)
    ranges=' / '.join(f'{b}: {g.mean_life.min():.1f}~{g.mean_life.max():.1f} cycles' for b,g in policies.groupby('batch'))
    ablation=' / '.join(f"{s}: {candidates[(candidates.model=='ElasticNet')&(candidates.features==s)].CV_MAPE.iloc[0]:.2f}%" for s in ['D1','D2','State','Full'])
    largest=r['largest_errors'][:3]
    errors=' / '.join(f"{e['cell_id']}: 실제 {e['cycle_life']:.0f}, 예측 {e['prediction']:.1f}, APE {e['APE']:.1f}%" for e in largest)
    short=next(s for s in r['subgroups'] if s['batch']=='Batch 2' and s['group']=='short')
    text=f'''# ESS 배터리 수명 예측

초기 100사이클의 열화 신호로 총 Cycle Life를 예측하고, 배치별 일반화 성능과 ESS 운영에 활용할 수 있는 범위를 분석한다.

## 프로젝트 개요

- 데이터셋: MIT-Stanford Battery Dataset (Severson et al., Nature Energy 2019), 수업 Kaggle 데이터.
- 학습 데이터: Batch 1 (2017-05-12), {r['n_batch1']}셀.
- 평가 데이터: Batch 2 (2018-02-20), {r['n_batch2']}셀.
- 추가 평가: Batch 3 (2018-04-12), {r['n_batch3']}셀.
- 태스크: Regression (Cycle Life 예측).
- 정제: 공식 로더의 비정상 셀 제외·Batch1 중단/재개 수명 보정, Batch2 수명 결측 제외.

## 파일 구조

```text
ESS_battery_data analysis/
├── README.md
├── requirements.txt
├── run_project.py                 # 전체 실행
├── 실행방법.md
├── data/
│   └── README.md                  # 원자료 출처·정제·다운로드
├── work/
│   ├── download_data.py
│   ├── extract_features.py        # 전처리·초기 피처 추출
│   ├── day1_design.py             # EDA·DAY1 설계 PDF
│   ├── day2_modeling.py           # CV·튜닝·학습·평가
│   ├── build_report.py            # 공통 PDF 스타일
│   ├── build_day2_report.py
│   ├── project_fonts.py
│   ├── reporting.py               # README·성능표 갱신
│   ├── verify_results.py
│   ├── package_submission.py
│   └── day2/                     # 모델·분할·후보 비교·예측 결과
├── figures/                      # DAY1·DAY2 그래프
├── results/
│   └── model_performance.csv      # 과제 형식 성능표
└── output/
    ├── pdf/                      # DAY1·DAY2 보고서
    └── run_manifest.json         # 코드·결과 해시
```

## 환경 설정

Python 3.11 이상. 프로젝트 루트에서 실행한다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_project.py --mode all
```

원자료가 없으면 `python run_project.py --download --mode all`로 내려받는다(약 8GB). DAY1/2 개별 실행 및 정제 피처만으로 DAY2를 재현하는 방법은 `실행방법.md`를 참고한다. 한글 TTF 폰트가 필요하며 macOS에서는 AppleGothic을 자동 사용한다. 다른 환경은 `BATTERY_FONT_PATH`를 지정한다.

## EDA

### Cycle Life 분포

- {life}. 단수명은 <500, 장수명은 >1000사이클이다.
- 핵심 발견: Batch2에 단수명이 집중된다. Batch1·3에는 <500셀이 없어 배치 내부의 절대 장단수명 직접 비교는 불가능하다.

### 열화 곡선 분석

- 초기10~100사이클보다 관측 후기25%의 Qd 감소 기울기가 더 음수인 셀은 115/115개였다(중단·재개 5셀 제외).
- 두 직선의 SSE를 최소화한 변화점을 knee 후보로 표시했다. 셀별 발생 시점은 `work/day1_design_features.csv`에 있다. 물리적 knee 확정값은 아니다.
- 핵심 발견: 후기 가속을 초기 직선 하나로 설명하기 어렵다. knee·후기 기울기는 미래 정보이므로 예측 피처에서 제외했다.

### ΔQ(V) 곡선 분석

- ΔQ(V)=Qd,100(V)-Qd,10(V). 로그분산-수명 Spearman 상관은 {dq}.
- 핵심 발견: Batch2의 단수명 곡선은 음의 골이 더 깊고, 세 배치 모두 로그분산이 클수록 수명이 짧은 방향이다.

### 충전 속도(C-rate)와 수명의 관계

- 정책별 평균 수명 범위: {ranges}. 정책별 평균·셀 수는 `work/day1_policy_mean.csv`에 있다.
- 핵심 발견: C1·전환SOC·C2 조합과 배치가 얽혀 있어 고속 충전의 인과효과를 단정할 수 없다. 반복 셀이 적은 정책도 있다.

### 초기 피처 상관·다중공선성

- ΔQ 로그분산·최솟값과 정책 변수 사이의 중복을 상관·VIF로 확인했다. 표준화는 다중공선성을 해결하지 않는다.
- 보조 검정: Kruskal-Wallis H={d['kruskal_H']:.3f}, p={d['kruskal_p']:.3g}. 세 배치의 동일 분포 가설에 반하는 근거지만, 셀 독립성을 가정하며 인과효과·단순 중앙값 차이로 해석하지 않는다.

## Modeling

### 피처 엔지니어링 전략

- D1: log10 Var(ΔQ). 전압별 초기 변화의 산포를 요약한다.
- D2: D1+min(ΔQ). 가장 큰 음의 변화의 추가 기여를 확인한다.
- State: D1+초기 Qd·IR 기울기+평균 온도+충전시간 기울기. 상태 변화의 추가 기여를 확인한다.
- Full: State+C1·전환SOC. 정책 추가 효과를 비교한다. C2는 소표본의 복잡도·정책 의존성을 줄이기 위해 추가하지 않았으나, 제외의 우월성을 별도로 입증하지는 않았다.
- ElasticNet 기준 피처 비교 CV MAPE: {ablation}. 추가 이득이 없어 D1을 채택했다. 최종 피처가 하나여서 피처 간 다중공선성은 없고 VIF=1이다. 셀 간 독립성을 보장한다는 뜻은 아니다.

### 모델 선택 및 근거

- 후보 모델: Ridge·ElasticNet·RBF SVR ×4피처 묶음, Full Random Forest·Gradient Boosting(총14조합).
- 최종 모델: 로그 타깃 {r['selected_model']} / {r['selected_feature_set']}.
- 선택 이유: Ridge D1과 ElasticNet D1은 약9.77%로 사실상 동률이다. 개발 CV 최솟값과0.01%p 이내인 Ridge D1은 단순성을 우선한다. 단일 피처에서 L1 변수 선택의 이득이 없기 때문이다.
- 파라미터: Ridge α=[0.01,0.1,1,10,100] 탐색. 개발 자료 선택값={r['selection_params'].get('regressor__model__alpha')}, Batch1 전체 재학습={r['final_params'].get('regressor__model__alpha')}. α는 L2 계수 축소 강도다. 전체 후보 설정·결과는 `work/day2/candidate_comparison.csv`에 기록했다.
- 검증: Batch1 정책 그룹 기준 개발 {r['n_development']}셀·홀드아웃 {r['n_valid']}셀(seed42), 정책 중복0. 개발 자료 바깥4분할/안쪽3분할 GroupKFold로 후보 비교·튜닝한다. 결측 대치·표준화는 폴드 내부에서 적합하고, 로그 예측을 exp 역변환한 원 단위 MAPE로 평가한다. 확정 모델을 Batch1 전체 학습 후 Batch2·3에 적용한다.

## 성능 결과

{perf}

- MAPE는 %, Gap은 %p다. Gap은 뒤 오차-앞 오차로 정의하며 양수는 악화다. Batch3의9.1%는 공통 참고값이지 Batch3 전용 논문 목표가 아니다.
- Batch2 기준 모델 MAPE={r['baseline']['Batch 2']['MAPE']:.2f}%, 최종 MAE={b2['MAE']:.2f}, RMSE={b2['RMSE']:.2f}cycles, R²={b2['R2']:.3f}.
- 목표9.1%에는 미달했다. 공식 논문 Batch2(2017-06-30)와 수업 데이터(2018-02-20)의 표본·분할이 달라 동일 조건 재현 비교는 아니다.
- Train은 학습셋 오차가 아니라 CV 평균이다. Valid/Test는 학습 셀 수가30/41개로 달라 Gap을 순수 배치 효과로만 해석하지 않는다. 이전 파일럿·EDA에서 외부 배치를 탐색했으므로 완전 눈가림 테스트는 아니지만, DAY2 외부 점수로 재선정·사후 보정하지 않았다.

## 오류 분석

- 가장 큰 오류: {errors}.
- 공통점: 주로 Batch2의 짧은 수명을 과대예측했다. Batch2 {r['n_batch2']}셀 중 {b2['overprediction_n']}셀 과대예측, 단수명 {short['n']}셀 MAPE={short['MAPE']:.2f}%다.
- 원인 가설: Batch1 단수명 표본 부족, ΔQ 범위·같은 신호의 수명 수준 이동, 정책·제조 배치 차이. Batch3 오차가Batch2보다 {abs(r['gap_batch2_batch3_pp']):.2f}%p 낮아도 모든 배치에 일반화됐다는 뜻은 아니다.
- 개선 방향: 단수명 셀과 새 배치를 확보하고 독립 보정용·평가용 자료를 분리한다. 외부 오차에 맞춘 사후 보정은 하지 않는다.

## ESS 도메인 해석

- 활용: 동일 chemistry·측정조건에서 셀 선별과 추가 점검 우선순위를 정하는 보조지표.
- 한계: 총 Cycle Life와 잔여 수명을 구분한다(RUL=예측 총수명-100). 과대예측은 교체 지연 위험이 있어 BMS 보호·교체 기준을 자동화하기에는 부족하다.
- 배포 전 필요 사항: 현장 온도·SOC·부하·달력 열화 및 셀→모듈→팩 차이를 반영한 독립 검증, 편향 보정과 예측구간 검증.

## 참고문헌

- Severson et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4, 383–391. [DOI](https://doi.org/10.1038/s41560-019-0356-8)
- [공식 데이터](https://data.matr.io/1/), [공식 코드](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation)
- [수업 Kaggle 데이터](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle)

작성자: 이효준 · 울산3반
'''
    (ROOT/'README.md').write_text(text,encoding='utf-8')

def export_all():
    r=json.loads((ROOT/'work/day2/results.json').read_text())
    export_performance(r)
    export_readme(r)
    for name in ['candidate_comparison.csv','external_predictions.csv','holdout_predictions.csv','split_manifest.csv','nested_cv_folds.csv','holdout_permutation_importance.csv','results.json']:
        shutil.copy2(ROOT/'work/day2'/name,ROOT/'results'/name)
    scripts=['day1_design.py','day2_modeling.py','extract_features.py','download_data.py','project_fonts.py','build_report.py','build_day2_report.py','reporting.py','verify_results.py','package_submission.py']
    paths=[ROOT/'README.md',ROOT/'requirements.txt',ROOT/'run_project.py']+[ROOT/'work'/s for s in scripts]+list((ROOT/'work/day2').glob('*.csv'))+list((ROOT/'output/pdf').glob('*.pdf'))
    manifest={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),'model_run_utc':r['generated_at_utc'],
              'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (ROOT/'output/run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':export_all()
