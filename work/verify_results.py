"""데이터 분할·모델 예측·지표·README/PDF/CSV 일치 검증."""
from pathlib import Path
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
from pypdf import PdfReader
from reporting import performance_rows

ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'work/day2'
    r=json.loads((out/'results.json').read_text())
    split=pd.read_csv(out/'split_manifest.csv')
    assert split.cell_id.is_unique
    dev=split[split.split=='development'];valid=split[split.split=='holdout']
    assert not set(dev.charge_policy)&set(valid.charge_policy)
    assert len(dev)==r['n_development'] and len(valid)==r['n_valid']
    saved=joblib.load(out/'final_model.joblib')
    assert all(c.startswith('b1c') for c in saved['batch1_cells'])
    pred=pd.read_csv(out/'external_predictions.csv')
    assert pred.cell_id.is_unique
    np.testing.assert_allclose(saved['model'].predict(pred[saved['features']]),pred.prediction,rtol=1e-10)
    for batch,g in pred.groupby('batch'):
        score=float((100*np.abs(g.prediction-g.cycle_life)/g.cycle_life).mean())
        np.testing.assert_allclose(score,r['test'][batch]['MAPE'])
    fold=pd.read_csv(out/'nested_cv_folds.csv')
    g=fold[(fold.model==r['selected_model'])&(fold.features==r['selected_feature_set'])]
    assert len(g)==4
    np.testing.assert_allclose(g.MAPE.mean(),r['CV_MAPE'])
    np.testing.assert_allclose(r['gap_train_valid_pp'],r['valid']['MAPE']-r['CV_MAPE'])
    np.testing.assert_allclose(r['gap_batch2_batch3_pp'],r['test']['Batch 3']['MAPE']-r['test']['Batch 2']['MAPE'])
    perf=pd.read_csv(ROOT/'results/model_performance.csv')
    assert perf['index'].tolist()==[x[0] for x in performance_rows(r)]
    np.testing.assert_allclose(perf.value,[x[1] for x in performance_rows(r)])
    readme=(ROOT/'README.md').read_text()
    for heading in ['## 프로젝트 개요','## 환경 설정','## EDA','## Modeling','### 피처 엔지니어링 전략','### 모델 선택 및 근거','## 성능 결과','## 오류 분석','## ESS 도메인 해석','## 참고문헌']:
        assert heading in readme,heading
    for path in (ROOT/'output/pdf').glob('*.pdf'):
        reader=PdfReader(path);assert len(reader.pages)>=10
        text='\n'.join(p.extract_text() for p in reader.pages)
        if 'Model-' in path.name:
            for a,v,unit,note in performance_rows(r):assert f'{v:.2f}' in text,(a,v)
        else:assert '최종 선택' not in text and '9.77%' not in text
    manifest=json.loads((ROOT/'output/run_manifest.json').read_text())
    for relative,digest in manifest['sha256'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest,relative
    print('PASS: 정책 분할, Batch1 학습, 저장 모델 예측, CV/외부 지표, 9행 성능표, README, PDF, 산출물 해시')

if __name__=='__main__':main()
