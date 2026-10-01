"""현재 코드·최신 결과만 포함하는 GitHub 업로드 준비용 ZIP."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from verify_results import main as verify

ROOT=Path(__file__).resolve().parents[1]

def main():
    verify()
    scripts=['day1_design.py','day2_modeling.py','extract_features.py','download_data.py','project_fonts.py','build_report.py','build_day2_report.py','reporting.py','verify_results.py','package_submission.py']
    files=[ROOT/p for p in ['README.md','requirements.txt','run_project.py','실행방법.md','.gitignore','data/README.md','output/run_manifest.json']]
    files += [ROOT/'work'/s for s in scripts]
    for folder,pattern in [('work/day2','*'),('results','*'),('figures/day2','*.png'),('output/pdf','*.pdf')]:
        files += [p for p in (ROOT/folder).glob(pattern) if p.is_file()]
    files += list((ROOT/'figures').glob('design_*.png'))
    for pattern in ['features_batch_*.csv','curves_batch_*.csv','delta_curves_batch_*.npz','day1_design*.csv','day1_design*.json','day1_vif.csv','day1_policy_mean.csv','features_all.csv','extraction_metadata.json']:
        files += list((ROOT/'work').glob(pattern))
    archive=ROOT/'output/ESS-Battery-Day2-GitHub.zip'
    with ZipFile(archive,'w',compression=ZIP_DEFLATED) as z:
        for p in sorted(set(files)):
            assert p.is_file(),p
            z.write(p,'ess-battery-project/'+str(p.relative_to(ROOT)))
    with ZipFile(archive) as z:
        assert z.testzip() is None
        assert not any(p.endswith('.mat') or '/legacy/' in p or '/__pycache__/' in p for p in z.namelist())
        print('Package verified:',len(z.namelist()),'files; size',archive.stat().st_size)
    print(archive)

if __name__=='__main__':main()
