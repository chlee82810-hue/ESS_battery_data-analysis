from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import BaseDocTemplate,Frame,PageTemplate,PageBreak,Spacer,Image,Table,TableStyle
from build_report import p,section_no,callout,data_table,NAVY,TEAL,MUTED,LINE
from project_fonts import korean_font
from reporting import performance_rows

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'work'/'day2'; FIG=ROOT/'figures'/'day2'
OUT=ROOT/'output'/'pdf'/'DS-MINI-Model-울산_3반-이효준.pdf'
font_path=korean_font();font_manager.fontManager.addfont(font_path)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font_path).get_name(),'axes.unicode_minus':False,'savefig.dpi':220,'font.size':10})

def chrome(c,doc):
    c.setTitle('DS Mini Project DAY 2 - Battery Cycle Life Modeling');c.setAuthor('이효준')
    c.saveState();w,h=A4;c.setFillColor(NAVY);c.rect(0,h-11*mm,w,11*mm,fill=1,stroke=0)
    c.setFillColor(colors.white);c.setFont('KR',7.5);c.drawString(18*mm,h-7.3*mm,'DS MINI PROJECT · DAY 2 MODEL DEVELOPMENT & EVALUATION')
    c.setFillColor(MUTED);c.drawString(18*mm,9*mm,'이효준 · 울산 3반');c.drawRightString(w-18*mm,9*mm,str(doc.page))
    c.setStrokeColor(LINE);c.line(18*mm,13*mm,w-18*mm,13*mm);c.restoreState()

def chart(name,width=171*mm):
    im=Image(str(FIG/name));ratio=im.imageHeight/im.imageWidth;im.drawWidth=width;im.drawHeight=width*ratio;return im

def table(rows,widths):return data_table(rows,[x*mm for x in widths])

def build():
    r=json.loads((WORK/'results.json').read_text());df=pd.read_csv(WORK/'features_all.csv')
    candidates=pd.read_csv(WORK/'candidate_comparison.csv');pred=pd.read_csv(WORK/'external_predictions.csv')
    b2,b3=r['test']['Batch 2'],r['test']['Batch 3'];base=r['baseline']['Batch 2']['MAPE'];valid=r['valid']['MAPE']
    package=joblib.load(WORK/'final_model.joblib');pipe=package['model'].regressor_
    slope=float(pipe.named_steps['model'].coef_[0]);intercept=float(pipe.named_steps['model'].intercept_)
    mean=float(pipe.named_steps['scale'].mean_[0]);scale=float(pipe.named_steps['scale'].scale_[0])
    fig,ax=plt.subplots(figsize=(9,3.4));labs=['Train CV','Valid hold-out','Test Batch 2','Extra Batch 3'];vals=[r['CV_MAPE'],valid,b2['MAPE'],b3['MAPE']]
    bars=ax.bar(labs,vals,color=['#167D87','#75ACB0','#E48B37','#8065A1']);ax.axhline(9.1,ls='--',color='#C95454',label='논문 Target 9.1%')
    for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+.6,f'{v:.2f}%',ha='center')
    ax.set(ylabel='MAPE (%)',ylim=(0,max(vals)*1.22),title='공식 과제 분할에서의 모델 성능');ax.legend();plt.tight_layout();plt.savefig(FIG/'06_performance.png',bbox_inches='tight');plt.close()
    fig,ax=plt.subplots(figsize=(9,3.8))
    for batch,g in df.groupby('batch'):ax.scatter(g.log_delta_q_var,g.cycle_life,label=batch,alpha=.7)
    x=np.linspace(df.log_delta_q_var.min(),df.log_delta_q_var.max(),200)
    ax.plot(x,np.exp(intercept+slope*(x-mean)/scale),color='#24313C',lw=2,label='Batch 1 학습 곡선')
    ax.set(xlabel='log10 Var[ΔQ(V)]',ylabel='Cycle life',title='같은 ΔQ 신호에도 배치별 수명 수준이 달라진다');ax.legend();plt.tight_layout();plt.savefig(FIG/'07_signal_fit.png',bbox_inches='tight');plt.close()
    story=[]
    def add(*items):story.extend(items)
    def page(n,title,sub):
        if story:add(PageBreak())
        story.extend(section_no(n,title,sub))
    # 1
    add(Spacer(1,16*mm),p('초기 열화 신호로 예측한<br/>배터리 사이클 수명','KRTitle'),p('DAY 2 모델 개발 및 평가 보고서<br/>Batch 1 학습 · Batch 2 최종 평가 · Batch 3 추가 검증','KRBody'),Spacer(1,12*mm))
    add(callout('최종 모델',f'log10 Var[ΔQ(V)] 하나를 입력으로 쓰는 <b>로그 타깃 Ridge 회귀</b>를 선정했다. Batch 1 내부 CV에서 ElasticNet과 사실상 동률이었고, 피처 추가의 검증상 이득이 없어 단순성을 우선했다.'),Spacer(1,6*mm))
    add(table([['Train CV','Valid','Test Batch 2','Extra Batch 3'],[f"{r['CV_MAPE']:.2f}%",f'{valid:.2f}%',f"{b2['MAPE']:.2f}%",f"{b3['MAPE']:.2f}%"]],[43,42,43,43]),Spacer(1,9*mm))
    add(p(f'핵심 성과. Batch 2 MAPE를 중앙값 기준모델의 {base:.2f}%에서 {b2["MAPE"]:.2f}%로 낮췄다. 다만 논문 목표 9.1%에는 미달했으며, Batch 2에서 39개 중 {b2["overprediction_n"]}개를 과대예측했다. 성능 저하의 중심은 내부 과적합보다 배치 간 관계 이동으로 보인다.','KRBody'))
    add(p('교수님 평가 기준에 맞춰 그래프마다 확인한 사실과 해석을 연결하고, 모델·파라미터·피처 선정 근거 및 오류 원인을 함께 제시했다.','KRBody'),Spacer(1,12*mm),p('제출자: 이효준 · 울산 3반<br/>작성일: 2026.10.01<br/>태스크: Regression / Cycle life / 초기 cycle 10-100<br/>주 지표: MAPE, 보조 지표: MAE·RMSE·R²','KRBody'))
    # 2
    page('01','데이터 분할과 누수 방지','과제의 필수 분할을 적용하고, 동일 충전정책을 가진 셀을 한 그룹으로 분리했다')
    add(table([['단계','데이터','용도'],['Development',f"Batch 1: {r['n_development']}개 셀 / {r['n_dev_policies']}개 정책",'모델·피처 선택, 4-fold nested group CV'],['Hold-out',f"Batch 1: {r['n_valid']}개 셀 / {r['n_valid_policies']}개 정책",'선정 후 독립 내부 검증'],['Final fit','Batch 1 전체 41개','선정 모델 재튜닝·재학습'],['Test','Batch 2 유효 수명 39개','최종 평가; 튜닝에 사용하지 않음'],['Additional','Batch 3 유효 수명 40개','최종 모델 추가 검증']],[32,60,79]),Spacer(1,5*mm))
    add(p('GroupShuffleSplit(test_size=0.25, random_state=42)로 hold-out 정책을 고정했다. Development와 hold-out 사이 정책 중복은 0이다. CV 역시 GroupKFold를 적용해 같은 정책의 반복 셀이 양쪽에 나뉘지 않도록 했다. 단순 셀 hold-out만으로는 동일 정책의 중복을 막을 수 없기 때문이다.','KRBody'))
    add(p('외부 4-fold CV는 일반화 오차 추정, 내부 3-fold CV는 하이퍼파라미터 선택에 사용한다. 모든 fold 안에서 결측 중앙값 대치와 StandardScaler를 학습한다. 타깃은 자연로그로 변환하고, 예측은 exp로 되돌린 뒤 원래 cycle 단위의 MAPE로 튜닝한다.','KRBody'))
    add(callout('사전 탐색 범위의 한계','DAY1의 교차배치 파일럿에서 Batch 2·3 결과를 이미 확인했다. 따라서 이번 외부 평가를 완전히 미관측한 블라인드 테스트로 부르지 않는다. DAY2의 후보 선택은 Batch 1만으로 고정했고, 외부 점수에 맞춘 재선택이나 보정은 하지 않았다.',colors.HexColor('#FFF2DE')),Spacer(1,4*mm))
    add(p('원자료 정제. Batch 1 비정상 셀 5개, Batch 3 비정상 셀 6개를 공식 로더 규칙으로 제외했다. Batch 2의 cycle_life 결측 8개를 지도학습에서 제외했다. Batch 1 중단·재개 5개 셀의 수명은 공식 continuation 길이로 보정했다. 총 120개가 최종 분석 대상이다.','KRSmall'))
    add(p('사이클 인덱스 정렬. Batch 1 배열 index 0의 빈 placeholder를 확인해 ΔQ는 index 100-10, 실제 측정으로 시작하는 Batch 2·3은 index 99-9를 사용했다. 최신 DAY1과 DAY2는 같은 추출 기준을 사용하며 실제 방전 cycle100과10의 순서를 맞춰 cutoff를 지켰다.','KRSmall'))
    # 3
    page('02','피처 선정과 추가 효과 검증','DAY1의 해석 가능한 후보를 실제 내부 검증 오차로 비교했다')
    add(chart('05_ablation.png'),Spacer(1,4*mm))
    sets=[['피처 집합','변수','ElasticNet CV MAPE']]
    definitions={'D1':'log ΔQ variance','D2':'D1 + ΔQ minimum','State':'D1 + Qd slope, IR slope, 온도 평균, 충전시간 slope','Full':'State + 1단계 C-rate, 전환 SOC'}
    for s,d in definitions.items():sets.append([s,d,f"{candidates.loc[(candidates.model=='ElasticNet')&(candidates.features==s),'CV_MAPE'].iloc[0]:.2f}%"])
    add(table(sets,[24,112,35]),Spacer(1,4*mm))
    add(p('ΔQ 분산은 전압 구간마다 열화량이 얼마나 불균일한지 요약한다. log10 변환은 작은 양수값의 스케일을 안정화한다. ΔQ 최솟값은 가장 큰 용량 감소를, Qd·IR slope는 용량·저항 변화율을, 온도와 정책은 운영조건을 설명하므로 후보에 포함했다.','KRBody'))
    add(callout('채택 결론','ElasticNet 기준 D1 9.77%, D2 10.74%, State 10.72%, Full 11.44%로 추가 피처의 이득이 확인되지 않았다. 최종 모델은 D1 하나를 채택했다. 이때 피처 간 다중공선성은 발생하지 않으며 VIF=1이다. 낮은 상관계수가 통계적 독립성을 증명하는 것은 아니다.'))
    # 4
    page('03','모델 비교와 선택 근거','총 14개 모델-피처 조합을 동일한 그룹 CV로 비교했다')
    add(chart('01_candidates.png'),Spacer(1,4*mm))
    add(table([['대표 후보','피처','CV MAPE ± fold SD'],*[ [row.model,row.features,f'{row.CV_MAPE:.2f}% ± {row.CV_SD:.2f}%p'] for _,row in candidates.head(6).iterrows()]],[48,38,85]),Spacer(1,4*mm))
    ridge_cv=float(candidates[(candidates.model=='Ridge')&(candidates.features=='D1')].CV_MAPE.iloc[0])
    en_cv=float(candidates[(candidates.model=='ElasticNet')&(candidates.features=='D1')].CV_MAPE.iloc[0])
    add(p(f'Ridge D1 {ridge_cv:.5f}%와 ElasticNet D1 {en_cv:.5f}%의 차이는 {abs(ridge_cv-en_cv):.5f}%p로 성능 우열의 근거가 되지 않는다. 개발 CV 최솟값 대비0.01%p 이내에서는 단순 Ridge D1을 우선하는 규칙을 코드에 적용했다. 단일 피처에서는 L1 선택이 필요 없고 α 하나로 규제를 조정할 수 있다. DAY1 후보들을 실제 ablation으로 비교해 단순화했다.','KRBody'))
    add(p('Ridge Full의 평균 CV MAPE는 24,832%로 수치적으로 불안정했다. 특정 그룹의 다변량 외삽이 로그 예측에 크게 반영되고 exp 역변환으로 오차가 폭증했다. 이 실패도 포함해 비교했으며, 수치 안정성과 그룹 일반화가 떨어져 선정에서 제외했다.','KRSmall'))
    # 5
    page('04','튜닝 파라미터와 최종 예측식','파라미터가 어떤 문제를 조절하는지 설명하고 실제 선택값을 기록했다')
    add(table([['모델','탐색 범위','선정 의도'],['Ridge','α: 0.01, 0.1, 1, 10, 100','L2 규제로 기울기 변동 억제'],['ElasticNet','α: 0.0001, 0.001, 0.01, 0.1\nl1_ratio: 0.1, 0.5, 0.9','선택(L1)과 축소(L2) 균형'],['RBF SVR','C: 0.1, 1, 10\nε: 0.03, 0.1 / γ: scale, 0.1','비선형성 및 허용오차 조절'],['Random Forest','150 trees / depth: 3, 5, None\nleaf: 2, 4 / max_features: 0.7, 1','작은 표본의 잎별 암기 억제'],['Gradient Boosting','60, 120 trees / lr: 0.03, 0.1\ndepth: 1, 2 / leaf: 3','얕은 트리로 복잡도 제한']],[36,77,58]),Spacer(1,6*mm))
    add(p('Development에서 최적 Ridge α는 0.1이었고, 선택한 모델을 Batch 1 전체로 재학습할 때 내부 그룹 CV가 선택한 α는 1이었다. 모델 종류와 피처는 고정한 상태에서 학습자료 증가에 따른 규제 강도만 재선정했다.','KRBody'))
    add(callout('저장된 최종 모델의 예측식',f'x = log10 Var[ΔQ(V)]<br/>z = (x - ({mean:.5f})) / {scale:.5f}<br/>예측 cycle life = exp({intercept:.5f} + ({slope:.5f}) × z)'))
    add(Spacer(1,4*mm),p(f'해석. ΔQ 로그분산이 1 표준편차 증가하면 예측 수명은 exp({slope:.3f})={np.exp(slope):.3f}배가 된다. 이는 Batch 1에서 학습한 통계적 관계이며, 충전조건 변화의 인과효과를 뜻하지 않는다.','KRBody'))
    add(p(f'타깃 변환 민감도. 같은 Ridge D1을 원타깃으로 학습한 hold-out MAPE는 {r["raw_target_valid"]["MAPE"]:.2f}%, log 타깃은 {valid:.2f}%였다. 로그 타깃의 상대오차 안정화가 유리했지만 차이는 작아 모든 데이터에 우세하다고 일반화하지 않는다.','KRBody'))
    # 6
    page('05','공식 성능 보고와 논문 비교','MAPE와 세 가지 Gap을 과제의 제출 형식으로 제시한다')
    add(chart('06_performance.png'),Spacer(1,4*mm))
    add(table([['구분','MAPE (%) / Gap (%p)','비고']]+[[a,f'{v:+.2f}' if unit=='pp' else f'{v:.2f}',note] for a,v,unit,note in performance_rows(r)],[61,39,71]),Spacer(1,4*mm))
    add(p('Gap은 양수가 성능 저하를 뜻하도록 위 식으로 정의했다. Train은 학습자료 자체의 오차가 아닌 CV 추정값이다. Valid와 Test 사이에는 학습 셀 수가 30개에서 41개로 늘어나는 재학습 차이도 있으므로 Gap을 순수한 배치 효과만으로 해석하지 않는다.','KRSmall'))
    add(p(f'논문 목표 9.1%에 비해 Batch 2 MAPE는 {r["gap_target_test_pp"]:.2f}%p 높다. 과제 mirror의 Batch 2는 2018-02-20이고 원논문 공식 로더의 Batch 2는 2017-06-30이다. 표본·분할·정책 구성이 달라 동일 조건 재현 비교로 볼 수 없다. 9.1%는 과제의 참고 목표로 사용했다.','KRBody'))
    add(p(f'Batch3 오차는 Batch2보다 {abs(r["gap_batch2_batch3_pp"]):.2f}%p 낮다. 외부 배치마다 오차가 달라 특정 피처·배치 구성에 의존했을 가능성이 있다. Batch3 전용 논문 목표는 안내에 없으므로9.1%는 동일 참고값일 뿐 Batch3 공식 성능 재현 비교는 아니다.','KRSmall'))
    # 7
    page('06','셀 단위 예측과 오류 분석','대각선으로 정확도를, 잔차 방향으로 과대·과소예측을 확인한다')
    add(chart('02_prediction.png'),Spacer(1,3*mm),chart('03_residual.png'),Spacer(1,4*mm))
    short=next(g for g in r['subgroups'] if g['batch']=='Batch 2' and g['group']=='short')
    add(p(f'Batch 2는 39개 중 {b2["overprediction_n"]}개에서 예측이 실제 수명보다 높았다. {short["n"]}개 단수명 셀의 평균 MAPE는 {short["MAPE"]:.2f}%이고 평균 과대예측은 {short["bias_cycles"]:.1f} cycles다. 이는 조기 교체를 늦추는 방향의 오류여서 ESS 의사결정에서 중요하다.','KRBody'))
    add(p(f'Batch 3는 {b3["MAPE"]:.2f}%로 더 낮은 오차를 보였다. Batch 3가 항상 더 어려운 테스트라는 가정은 실제 결과와 맞지 않았다. 내부 데이터와 수명·열화 신호의 관계가 얼마나 비슷한지가 배치 순서보다 중요하다.','KRBody'))
    # 8
    page('07','배치 이동과 실패 사례','강한 상관관계가 있어도 새 배치의 절대 수명은 다르게 예측될 수 있다')
    add(chart('07_signal_fit.png'),Spacer(1,4*mm))
    errors=[['셀','배치','실제','예측','APE (%)']]
    for row in r['largest_errors']:errors.append([row['cell_id'],row['batch'],f"{row['cycle_life']:.0f}",f"{row['prediction']:.1f}",f"{row['APE']:.1f}"])
    add(table(errors,[28,30,33,40,40]),Spacer(1,4*mm))
    max1=df.loc[df.batch=='Batch 1','log_delta_q_var'].max();max2=df.loc[df.batch=='Batch 2','log_delta_q_var'].max()
    add(p(f'Batch 1은 단수명(&lt;500) 셀이 없고, Batch 2는 39개 중 28개가 단수명이다. ΔQ 로그분산 최대값은 Batch 1 {max1:.2f}, Batch 2 {max2:.2f}로 Batch 2 일부가 학습 범위를 벗어난다. 또한 학습곡선보다 낮은 실제 수명이 나타나 조건부 관계 이동도 의심된다.','KRBody'))
    add(callout('원인 가설과 검증 방향','짧은 수명 영역의 학습표본 부족, 충전정책 구성 차이, 제조·실험 배치 차이가 원인 후보다. 추가 배치에서 조기 종료 셀을 확보하고, 새 라벨을 별도 보정용·평가용으로 나눠 검증해야 한다. 현재 Batch 2 오차에 맞춰 계수를 사후 보정한 성능은 보고하지 않는다.',colors.HexColor('#FFF2DE')))
    # 9
    page('08','신뢰 범위와 ESS 운영 해석','측정한 오차를 실제 의사결정의 조건과 연결한다')
    ci2=b2['MAPE_ci95'];ci3=b3['MAPE_ci95']
    add(table([['평가','MAPE','셀 bootstrap 95% CI','MAE / RMSE (cycles)'],['Batch 2',f"{b2['MAPE']:.2f}%",f'{ci2[0]:.2f}-{ci2[1]:.2f}%',f"{b2['MAE']:.1f} / {b2['RMSE']:.1f}"],['Batch 3',f"{b3['MAPE']:.2f}%",f'{ci3[0]:.2f}-{ci3[1]:.2f}%',f"{b3['MAE']:.1f} / {b3['RMSE']:.1f}"]],[31,29,49,62]),Spacer(1,5*mm))
    add(p('CI는 고정 모델의 해당 테스트 셀을 5,000회 재표본한 MAPE 범위다. 모델 재학습 불확실성·미래 배치 불확실성을 포함하지 않고, 정책 내 상관이 있으면 낙관적일 수 있다. 세 배치만으로 임의의 ESS 환경에서 동일 오차를 보장할 수 없다.','KRBody'))
    add(p('활용 가능한 의사결정. 동일 chemistry·측정조건에서 초기 상태가 다른 셀을 비교하고, 추가 점검 대상으로 우선순위를 정하는 보조지표로 사용할 수 있다. cycle life는 총수명이며, 100사이클 시점 RUL은 예측 총수명에서 100을 뺀 값이다. 실제 운영시간으로 바꾸려면 사용빈도·온도·SOC 창과 calendar aging을 반영해야 한다.','KRBody'))
    add(p('배포 전 필요한 것. Batch 2의 과대예측 편향을 줄일 독립 보정 데이터, 현장 chemistry·온도·부하 검증, 셀→모듈→팩 수준의 차이 검증, 예측구간의 coverage 평가가 필요하다. 이 회귀 결과만으로 BMS 보호 기능이나 교체 기준을 자동화하지 않는다.','KRBody'))
    add(callout('최종 판단',f'조기 ΔQ 신호는 유효했고 단순 Ridge는 기준모델보다 Batch 2 MAPE를 {(base-b2["MAPE"])/base*100:.1f}% 상대적으로 줄였다. 내부 hold-out의 작은 Gap보다 외부 Batch 2의 큰 Gap이 핵심 한계다. 다음 개선 목표는 모델 복잡도 증가보다 단수명 영역의 대표성과 배치 이동을 보완하는 것이다.'))
    add(Spacer(1,4*mm),p('DAY1의 배치별 ΔQ-Spearman 관계를 탐색 근거로 사용했다. 14개 후보 비교 뒤 사후 유의성 검정을 추가해 최종 모델이 통계적으로 우월하다고 주장하지 않았다.','KRSmall'))
    # 10
    page('09','재현 산출물과 제출 준비','같은 분할·전처리·파라미터·예측값을 확인할 수 있도록 저장했다')
    add(table([['산출물','내용'],['PDF','DAY2 분석·모델·평가·오류 해석'],['README.md','노션 샘플의 개요·EDA·피처·모델·성능·오류·ESS 해석'],['requirements.txt','분석에 사용한 라이브러리 버전'],['work/day2_modeling.py','고정 seed의 재현 가능한 학습·검증 스크립트'],['split_manifest.csv','개발/hold-out 셀과 정책 목록'],['candidate_comparison.csv','14개 조합의 평균 CV·SD·튜닝값'],['external_predictions.csv','Batch 2·3 셀별 실제·예측·APE·잔차'],['final_model.joblib','최종 모델과 피처 순서'],['results.json / model_performance.csv','최종 지표·Gap·CI·오류 요약']],[55,116]),Spacer(1,5*mm))
    add(p('실행. Python 환경에서 requirements.txt를 설치한 후 프로젝트 루트에서 python work/day2_modeling.py, python work/build_day2_report.py를 차례로 실행한다. 제공된 features_all.csv만으로 모델링을 재현할 수 있다. 원본 MAT 데이터는 Kaggle/공식 저장소에서 별도로 확보하며, 수 GB 파일과 가상환경은 GitHub 패키지에서 제외한다.','KRBody'))
    add(p('제출. 노션 DAY2 요구 산출물은 공개 GitHub 링크다. 사용자의 요청에 맞춰 최종 PDF와 GitHub에 업로드할 수 있는 코드·README 패키지를 준비했다. 공개 저장소 생성과 업로드는 별도의 외부 공개 단계다.','KRBody'))
    add(p('참고문헌','KRH2'),p('[1] Severson et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. Nature Energy 4, 383-391. https://doi.org/10.1038/s41560-019-0356-8','KRSmall'),p('[2] Official data / loader: https://data.matr.io/1/ ; https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation','KRSmall'),p('[3] Course data mirror: https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle','KRSmall'),p('[4] DS Mini Project 노션 DAY2 요구사항, 2026.10.01 확인. 교수님이 전달한 그래프·해석·피처·모델 선정 근거 평가 기준 반영.','KRSmall'))
    doc=BaseDocTemplate(str(OUT),pagesize=A4,leftMargin=18*mm,rightMargin=18*mm,topMargin=19*mm,bottomMargin=17*mm)
    doc.addPageTemplates(PageTemplate(id='main',frames=Frame(doc.leftMargin,doc.bottomMargin,doc.width,doc.height,id='main'),onPage=chrome))
    doc.build(story)
    print(OUT)

if __name__=='__main__':build()
