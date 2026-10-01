"""원본 MAT → 피처 추출 → EDA/검정 → 모델링 → PDF를 실행하는 시작 파일."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
FILES = [
    '2017-05-12_batchdata_updated_struct_errorcorrect.mat',
    '2018-02-20_batchdata_updated_struct_errorcorrect.mat',
    '2018-04-12_batchdata_updated_struct_errorcorrect.mat',
]


def run(script: str, env: dict[str, str]) -> None:
    print(f'\n실행: {script}', flush=True)
    subprocess.run([sys.executable, str(ROOT / 'work' / script)],
                   cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['all', 'day1', 'day2'], default='all',
                        help='all: DAY1 + DAY2 분석과 보고서 생성')
    parser.add_argument('--download', action='store_true',
                        help='원본 MAT 파일이 없으면 Kaggle에서 다운로드')
    args = parser.parse_args()

    for relative in ['data', 'work', 'figures', 'output/pdf', 'tmp/mplconfig']:
        (ROOT / relative).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['MPLCONFIGDIR'] = str(ROOT / 'tmp' / 'mplconfig')
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['OMP_NUM_THREADS'] = '1'
    missing = [name for name in FILES if not (ROOT / 'data' / name).is_file()]
    if missing and args.download:
        run('download_data.py', env)
    elif missing:
        parser.error('data/에 원본 MAT 파일이 없습니다: ' + ', '.join(missing) +
                     '\npython run_project.py --download 로 내려받을 수 있습니다.')

    # 저장된 결과 CSV에 의존하지 않고 원본에서 먼저 피처를 다시 계산한다.
    run('extract_features.py', env)

    if args.mode in ['all', 'day1']:
        run('day1_design.py', env)
    else:
        # DAY2만 실행할 때도 원본으로부터 추출한 유효 셀을 입력으로 사용한다.
        import numpy as np
        import pandas as pd
        frame = pd.concat([pd.read_csv(ROOT / 'work' / f'features_batch_{i}.csv')
                           for i in [1, 2, 3]], ignore_index=True)
        frame[np.isfinite(frame.cycle_life)].to_csv(ROOT / 'work' / 'features_all.csv', index=False)

    if args.mode in ['all', 'day2']:
        run('day2_modeling.py', env)
        run('build_day2_report.py', env)
        run('reporting.py', env)
        run('verify_results.py', env)

    print('\n완료. PDF:', ROOT / 'output' / 'pdf')
    print('피처/통계/예측값:', ROOT / 'work')
    print('그래프:', ROOT / 'figures')


if __name__ == '__main__':
    main()
