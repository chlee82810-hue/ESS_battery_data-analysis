"""DAY1 전용: 실제 원자료 EDA와 설계만 생성하며 모델을 학습하지 않는다."""
from pathlib import Path
import json
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.stats import spearmanr, kruskal
from statsmodels.stats.outliers_influence import variance_inflation_factor
from reportlab.platypus import PageBreak, Spacer, Frame, PageTemplate
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from extract_features import BATCH_FILES, deref_array, safe_slope
from build_report import p, section_no, callout, data_table, fig, NumberedDocTemplate, page_chrome
from project_fonts import korean_font

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'work'
FIG = ROOT / 'figures'
BATCHES = ['Batch 1', 'Batch 2', 'Batch 3']
FEATURES = ['log_delta_q_var','delta_q_min','qd_early_slope','ir_early_slope','tavg_early_mean','charge_time_early_slope','c_rate_1','switch_soc','c_rate_2']
LABELS = ['log Var(ΔQ)','min ΔQ','초기 Qd 기울기','초기 IR 기울기','초기 평균 온도','충전시간 기울기','C1','전환 SOC','C2']
plt.rcParams.update({'font.family':FontProperties(fname=korean_font()).get_name(),'axes.unicode_minus':False,'font.size':9})

def corr(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float)
    m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<4 or np.ptp(x[m])==0 or np.ptp(y[m])==0: return float('nan')
    return float(spearmanr(x[m],y[m]).statistic)

def save(name, figure):
    figure.tight_layout()
    figure.savefig(FIG/name,dpi=180,bbox_inches='tight')
    plt.close(figure)

def analyze():
    df=pd.concat([pd.read_csv(WORK/f'features_batch_{i}.csv') for i in [1,2,3]],ignore_index=True)
    df=df[np.isfinite(df.cycle_life)].copy()
    df.to_csv(WORK/'features_all.csv',index=False)
    extra=[]
    for batch_name,file_name in BATCH_FILES.items():
        with h5py.File(ROOT/'data'/file_name,'r') as h:
            b=h['batch']
            for row in df[df.batch==batch_name].itertuples():
                idx=int(row.cell_id.split('c')[1])
                summary=h[b['summary'][idx,0]]
                x=np.asarray(summary['cycle'][0,:],float)
                y=np.asarray(summary['QDischarge'][0,:],float)
                m=np.isfinite(x)&np.isfinite(y);x,y=x[m],y[m]
                cut=np.quantile(x,.75)
                early=safe_slope(x[(x>=10)&(x<=100)],y[(x>=10)&(x<=100)])
                late=safe_slope(x[x>=cut],y[x>=cut])
                g=h[b['cycles'][idx,0]]
                offset=1 if np.allclose(deref_array(h,g['Qdlin'][0,0]),0) else 0
                currents=[]; peaks=[]
                for cycle in [10,50,100]:
                    a=deref_array(h,g['I'][cycle-1+offset,0]).astype(float)
                    pos=a[np.isfinite(a)&(a>0.1)]
                    if len(pos): currents.append(float(pos.mean()));peaks.append(float(pos.max()))
                extra.append({'cell_id':row.cell_id,'late_qd_slope':late,'early_qd_slope_check':early,
                              'charge_current_mean_a':np.mean(currents) if currents else np.nan,
                              'charge_current_peak_a':np.mean(peaks) if peaks else np.nan,
                              'acceleration_observed':bool(late<early and late<0)})
    df=df.merge(pd.DataFrame(extra),on='cell_id')
    df.to_csv(WORK/'day1_design_features.csv',index=False)
    curves=pd.concat([pd.read_csv(WORK/f'curves_batch_{i}.csv') for i in [1,2,3]])
    curves=curves[curves.cell_id.isin(df.cell_id)]
    # 같은 축으로 배치를 비교하며 질문별 세 패널을 나란히 배치한다.
    f,axs=plt.subplots(1,3,figsize=(11,3.6),sharex=True,sharey=True)
    for ax,b in zip(axs,BATCHES):
        d=df[df.batch==b];ax.hist(d.cycle_life,bins=np.arange(150,2351,150),color='#167D87',edgecolor='white')
        ax.axvline(500,color='#C95454',ls='--');ax.axvline(1000,color='#F28E2B',ls='--')
        ax.set(title=f'{b} (n={len(d)})',xlabel='Cycle life',xlim=(150,2300))
    axs[0].set_ylabel('셀 수');save('design_q1.png',f)
    f,axs=plt.subplots(1,3,figsize=(11,3.7),sharex=True,sharey=True)
    for ax,b in zip(axs,BATCHES):
        d=df[(df.batch==b)&(~df.continued)]
        for q,col in zip([.1,.5,.9],['#C95454','#167D87','#486FA5']):
            r=d.iloc[(d.cycle_life-d.cycle_life.quantile(q)).abs().argmin()]
            c=curves[curves.cell_id==r.cell_id]
            ax.plot(c.cycle,c.qd,color=col,label=f'{r.cell_id}: {r.cycle_life:.0f}')
            if np.isfinite(r.knee_cycle_descriptive): ax.axvline(r.knee_cycle_descriptive,color=col,alpha=.25,ls=':')
        ax.set(title=b,xlabel='Cycle',ylim=(.75,1.15));ax.legend(fontsize=7)
    axs[0].set_ylabel('방전 용량 Qd (Ah)');save('design_q2.png',f)
    f,axs=plt.subplots(1,3,figsize=(11,3.7),sharex=True,sharey=True)
    for i,(ax,b) in enumerate(zip(axs,BATCHES),1):
        z=np.load(WORK/f'delta_curves_batch_{i}.npz');d=df[df.batch==b]
        for mask,col,label in [(d.cycle_life<500,'#C95454','단수명 <500'),(d.cycle_life>1000,'#486FA5','장수명 >1000')]:
            ids=d[mask].cell_id.tolist()
            if not ids: ax.plot([],[],color=col,label=f'{label}: n=0');continue
            a=np.stack([z[f'{cell}__delta_q'] for cell in ids]);v=z[f'{ids[0]}__voltage']
            ax.plot(v,np.median(a,axis=0),color=col,label=f'{label}: n={len(ids)}')
            ax.fill_between(v,np.quantile(a,.25,axis=0),np.quantile(a,.75,axis=0),color=col,alpha=.15)
        ax.set(title=b,xlabel='Voltage (V)');ax.legend(fontsize=7)
    axs[0].set_ylabel('ΔQ100-10 (Ah)');save('design_q3.png',f)
    policies=df.groupby(['batch','charge_policy'],as_index=False).agg(mean_life=('cycle_life','mean'),n=('cell_id','size'),c1=('c_rate_1','first'),c2=('c_rate_2','first'),soc=('switch_soc','first'))
    policies.to_csv(WORK/'day1_policy_mean.csv',index=False)
    f,axs=plt.subplots(1,3,figsize=(11,3.8),sharey=True)
    for ax,b in zip(axs,BATCHES):
        d=policies[policies.batch==b].sort_values('mean_life')
        ax.scatter(range(len(d)),d.mean_life,s=d.n*22,color='#167D87')
        ax.set(title=f'{b}: {len(d)} 정책',xlabel='평균 수명 오름차순 정책 번호')
        ax.set_xticks(range(len(d)),[str(i+1) for i in range(len(d))],fontsize=6)
    axs[0].set_ylabel('정책별 평균 Cycle life');save('design_q4.png',f)
    cmat=np.array([[corr(df[df.batch==b][k],df[df.batch==b].cycle_life) for b in BATCHES] for k in FEATURES])
    f,ax=plt.subplots(figsize=(10,4.7));im=ax.imshow(cmat,vmin=-1,vmax=1,cmap='RdBu_r',aspect='auto')
    ax.set_xticks(range(3),BATCHES);ax.set_yticks(range(len(LABELS)),LABELS)
    for i in range(len(FEATURES)):
        for j in range(3):ax.text(j,i,f'{cmat[i,j]:.2f}',ha='center',va='center')
    f.colorbar(im,ax=ax,label='Spearman ρ: feature vs life');save('design_q5.png',f)
    f,axs=plt.subplots(1,3,figsize=(12,4.4))
    vifs=[];pairs=[]
    for ax,b in zip(axs,BATCHES):
        d=df[df.batch==b];c=d[FEATURES].corr(method='spearman')
        ax.imshow(c,vmin=-1,vmax=1,cmap='RdBu_r')
        ax.set_xticks(range(9),['DQv','DQm','Qd','IR','T','time','C1','SOC','C2'],rotation=90,fontsize=7)
        ax.set_yticks(range(9),['DQv','DQm','Qd','IR','T','time','C1','SOC','C2'],fontsize=7);ax.set_title(b)
        for i in range(9):
            for j in range(i+1,9):
                if abs(c.iloc[i,j])>=.8:pairs.append((b,LABELS[i],LABELS[j],float(c.iloc[i,j])))
        x=d[FEATURES].replace([np.inf,-np.inf],np.nan).dropna()
        x=x.loc[:,x.std()>0];z=(x-x.mean())/x.std();a=np.column_stack([np.ones(len(z)),z.values])
        for i,k in enumerate(x.columns,1):vifs.append((b,LABELS[FEATURES.index(k)],float(variance_inflation_factor(a,i))))
    save('design_collinearity.png',f)
    pd.DataFrame(vifs,columns=['batch','feature','vif']).to_csv(WORK/'day1_vif.csv',index=False)
    summaries=[]
    for b in BATCHES:
        d=df[df.batch==b];eligible=d[~d.continued]
        summaries.append({'batch':b,'n':len(d),'median':float(d.cycle_life.median()),'min':float(d.cycle_life.min()),'max':float(d.cycle_life.max()),'short':int((d.cycle_life<500).sum()),'long':int((d.cycle_life>1000).sum()),'accel_n':int(eligible.acceleration_observed.sum()),'accel_total':len(eligible),'dq_rho':corr(d.log_delta_q_var,d.cycle_life),'c1_life_rho':corr(d.c_rate_1,d.cycle_life),'current_late_rho':corr(eligible.charge_current_mean_a,eligible.late_qd_slope)})
    test=kruskal(*[df[df.batch==b].cycle_life for b in BATCHES])
    result={'batch_summary':summaries,'kruskal_H':float(test.statistic),'kruskal_p':float(test.pvalue),'collinear_pairs':pairs,'vifs':vifs}
    (WORK/'day1_design_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return df,policies,result

def build():
    df,policies,res=analyze()
    out=ROOT/'output/pdf/DS-MINI-Design-울산_3반-이효준.pdf'
    doc=NumberedDocTemplate(str(out),pagesize=A4,leftMargin=18*mm,rightMargin=18*mm,topMargin=19*mm,bottomMargin=17*mm)
    doc.addPageTemplates([PageTemplate(id='normal',frames=[Frame(doc.leftMargin,doc.bottomMargin,doc.width,doc.height,id='normal')],onPage=page_chrome)])
    story=[]
    def text(t):story.append(p(t))
    def table(rows,widths):story.append(data_table(rows,[x*mm for x in widths]))
    def page(n,title,sub):
        if story:story.append(PageBreak())
        story.extend(section_no(n,title,sub))
    def image(name,h=75):story.append(fig(name,height=h*mm));story.append(Spacer(1,3*mm))
    def insight(t):story.append(callout('EDA → Modeling Strategy',t))
    page('DAY 1','초기 100사이클 기반 배터리 수명 예측 설계','이효준 · 울산 3반 · 2026.10.01')
    story.append(Spacer(1,18*mm))
    text('세 배치의 수명·열화·충전조건을 비교하고, 예측에 사용할 조기 신호와 검증 방법을 설계한다.')
    story.append(callout('이 보고서의 범위','DAY1은 실제 데이터 EDA와 모델 설계 단계다. 모델 학습 결과, 최종 모델 선정, 최적 파라미터, 예측 성능은 포함하지 않는다.'))
    story.append(Spacer(1,8*mm))
    table([['분석 단위','범위'],['셀 수',f'{len(df)}개 유효 셀 / Batch 1·2·3'],['예측 대상','Cycle life: 수명 종료까지 총 사이클 수'],['예측 시점','100사이클 종료 시점'],['과제 구성','다섯 EDA 질문 → 그래프·해석·시사점 → 모델 후보·검증 설계']],[40,131])
    text('목차: 데이터·분석 원칙 / Q1 수명 분포 / Q2 Qd 열화 / Q3 ΔQ(V) / Q4 충전조건 / Q5 피처 관계·중복 / 피처 선정 / 모델링 전략 / 근거 요약 / 정책별 평균 수명 부록.')
    page('01','데이터와 분석 원칙','제공 파일을 사용한 탐색과, 이후 모델 검증을 구분한다')
    table([['배치','실제 원자료 파일','유효 셀'],['Batch 1',BATCH_FILES['Batch 1'],'41'],['Batch 2',BATCH_FILES['Batch 2'],'39'],['Batch 3',BATCH_FILES['Batch 3'],'40']],[24,125,22])
    text('Kaggle 공개 데이터의 원본 MAT 파일을 직접 읽었다. 2018-04-03 varcharge extra 파일은 사용하지 않았다. 제공 데이터의 배치 명칭을 따른다.')
    text('정제: 저자 로더 기준 Batch1 5개, Batch3 6개 셀 제외; Batch2 수명 결측 8개 제외. Batch1 중단·재개 셀 0~4는 공식 continuation 수명을 합산했다. 이 셀의 저장된 Qd 곡선은 전체 최종 수명을 덮지 않아 후기 기울기·가속 비율 계산에서는 제외했다.')
    text('ΔQ(V)=Qd,100(V)-Qd,10(V). 첫 Qdlin 배열이 더미 0인지 검사해 실제 사이클 번호에 맞췄다. Batch1은 배열 인덱스 10/100, Batch2·3은 9/99를 사용한다. 통계와 그래프 모두 같은 추출값을 사용했다.')
    text('전체 수명의 열화 곡선·knee는 EDA 해석에만 사용한다. 예측 피처는 100사이클까지의 정보만 쓴다. 파일명, 셀 ID, 총 관측 길이, 후기 열화 기울기와 knee는 입력에서 제외한다.')
    insight('과제에 따라 세 배치를 모두 탐색하되, 모델 튜닝은 Batch1 내부에서만 수행하도록 설계한다. Batch2·3을 이미 EDA로 살펴봤으므로 완전히 눈가림된 테스트라고 주장하지 않는다. 미래 성능 확증에는 새 배치가 필요하다.')
    page('Q1','Cycle life 분포와 단수명 셀','히스토그램: 공통 구간 150~2300 / 점선: 500·1000사이클')
    image('design_q1.png')
    table([['배치','범위 / 중앙값','<500','>1000']]+[[s['batch'],f"{s['min']:.0f}~{s['max']:.0f} / {s['median']:.1f}",f"{s['short']}/{s['n']} ({100*s['short']/s['n']:.1f}%)",f"{s['long']}/{s['n']} ({100*s['long']/s['n']:.1f}%)"] for s in res['batch_summary']],[26,65,40,40])
    text('해석: Batch2가 상대적으로 짧은 수명에 집중된다. 배치별 분포가 달라 전체 평균 하나로 설명하면 집단 차이를 놓친다. 500 미만은 과제의 단수명 기준이며 통계적 이상치나 측정 오류를 뜻하지 않는다.')
    low=df.nsmallest(3,'cycle_life')
    table([['단수명 예시','수명','충전정책']]+[[r.cell_id,f'{r.cycle_life:.0f}',r.charge_policy] for r in low.itertuples()],[30,25,116])
    text('원인 검토: 위 셀은 Batch2에 속한다. 정책·초기 ΔQ·온도·IR를 함께 확인할 후보지만 관찰자료만으로 충전조건이 조기 고장을 일으켰다고 단정할 수 없다. 시험환경·제조 차이·종료 기준을 추가 확인해야 한다.')
    insight('총 수명을 예측하는 회귀를 선택한다. 긴 꼬리의 영향을 줄이기 위해 원 타깃과 로그 타깃을 비교할 계획이다. 단수명 셀은 임의로 제거하지 않고 상대오차와 그룹별 오차를 평가한다.')
    page('Q2','Qd 열화: 일정한 감소인가, 가속인가?','배치 내 수명 10·50·90% 분위수에 가까운 셀 / 점선: 기술적 knee 후보')
    image('design_q2.png')
    table([['배치','후기 감소 가속 조건 만족','초기·후기 비교 정의']]+[[s['batch'],f"{s['accel_n']}/{s['accel_total']} ({100*s['accel_n']/s['accel_total']:.1f}%)",'후기 기울기 < 초기 기울기, 후기 < 0'] for s in res['batch_summary']],[26,65,80])
    text('초기 기울기는 cycle10~100, 후기는 저장된 전체 곡선의 마지막 25%에서 선형 적합했다. 비교는 기울기 크기의 기술적 지표이며 가속의 통계적 검정은 아니다. 모든 셀이 일정한 속도로 열화한다고 가정하기 어렵다. 대표 3개 곡선은 전체 비율을 대신하지 않는다.')
    text('Knee 후보: 곡선 길이의 20~90% 구간에서 90개 분할점을 탐색하고, 두 직선의 총 제곱오차가 가장 작은 점을 표시했다. 두 직선의 연속성·기울기 가속을 강제하지 않아 물리적 knee 확정값이 아닌 변화점 근사다. 잡음·관측 종료 위치에 민감하다.')
    insight('초기 Qd 기울기를 후보로 둔다. 전체 곡선으로 구한 knee와 후기 기울기는 미래정보이므로 예측에 사용하지 않는다. 초기 감소가 약한 셀도 구별할 수 있도록 ΔQ와 IR 신호를 함께 검토한다.')
    page('Q3','ΔQ(V): 장수명·단수명의 초기 차이','실선: 그룹 중앙값 / 음영: 25~75% 구간 / 모든 배치 동일 축')
    image('design_q3.png')
    table([['배치','log10 Var(ΔQ) vs 수명 ρ','절대 수명 그룹 비교']]+[[s['batch'],f"{s['dq_rho']:.3f}",'양 그룹 존재' if s['short'] and s['long'] else '단수명 그룹 없음: 직접 비교 불가'] for s in res['batch_summary']],[26,65,80])
    text('해석: 세 배치 모두 ΔQ 로그 분산이 클수록 수명이 짧은 방향의 관계를 보인다. 이는 전체 상관이 배치 차이만으로 생긴 현상인지 확인하는 데 도움을 준다. Batch1·3은 <500 셀이 없어 해당 배치에서 장·단수명 차이가 검증됐다고 말할 수 없다.')
    text('곡선 음영은 셀 간 분포이며 신뢰구간이 아니다. 단수명과 장수명 차이는 Batch2에서 확인하되 두 그룹의 표본 수 차이와 정책 차이도 함께 고려한다. 셀의 ΔQ 곡선은 같은 전압 격자에 정렬되어 있다.')
    text('Batch2의 단수명 중앙 곡선은 장수명보다 음의 골이 깊고 전압에 따른 변화가 크다. 따라서 초기 변화의 산포를 요약하는 분산이 수명과 연결될 수 있다. 단, 이 곡선 모양만으로 열화의 물리적 원인을 확정하지 않는다.')
    insight('곡선 전체를 그대로 넣으면 차원이 높아진다. 분산은 전압 구간별 변화의 산포, 최솟값은 가장 큰 음의 변화를 요약한다. log10 분산을 우선 후보로 두고 최솟값 추가 효과는 Batch1 내부 검증에서 확인할 계획이다.')
    page('Q4','충전정책과 수명·열화의 관계','버블: 정책별 평균 수명 / 크기: 해당 정책의 셀 수 / 번호는 부록과 대응')
    image('design_q4.png')
    table([['배치','C1 vs life ρ','평균 충전 전류 vs 후기 Qd 기울기 ρ']]+[[s['batch'],f"{s['c1_life_rho']:.3f}",f"{s['current_late_rho']:.3f}"] for s in res['batch_summary']],[26,55,90])
    text('전류 패턴 요약: cycle10·50·100 각각에서 I>0.1A인 충전 구간의 평균 전류와 최대 전류를 계산해 세 사이클을 평균했다. 평균은 시간가중이 아닌 기록 샘플 평균이므로 CV 구간 길이·샘플링 영향이 있다. 후기 Qd 기울기는 음수일수록 더 빠른 감소다.')
    text('정책은 C1·전환 SOC·C2 조합으로 분리해 평균 수명을 계산했다. 고속 충전의 영향은 C1 하나로 단정할 수 없다. 정책당 셀이 적고 배치와 다른 조건이 함께 달라, 상관은 충전 조건의 인과효과나 최적 정책을 뜻하지 않는다.')
    insight('C1·SOC·C2와 초기 전류 요약을 조건 피처 후보로 둔다. 다중공선성을 점검하고, ΔQ 단독 대비 이 변수를 추가했을 때 Batch1 내부 검증이 개선되는지 확인한다. 후기 열화 기울기는 비교 설명용으로만 쓴다.')
    page('Q5','초기 신호와 수명의 관계','배치별 Spearman 순위상관 / 동일 컬러 범위 -1~1')
    image('design_q5.png',92)
    text('해석: log Var(ΔQ)는 모든 배치에서 같은 음의 방향을 보여 핵심 조기 신호 후보로 적합하다. 다른 피처는 배치에 따라 상관 크기·방향이 달라질 수 있다. 전체 데이터에서 상관이 큰 피처를 모두 넣는 방식은 집단 차이와 중복 정보를 과대평가할 수 있다.')
    text('표는 단변량·단조 관계의 탐색이며 독립적인 예측 기여나 인과관계를 증명하지 않는다. 최종 피처 선정은 Batch1 내부 교차검증과 피처 묶음 비교로 확인할 계획이다. 테스트 배치의 상관값으로 튜닝하지 않는다.')
    strongest=[]
    for b in BATCHES:
        d=df[df.batch==b];k=max(FEATURES,key=lambda k:abs(corr(d[k],d.cycle_life)))
        strongest.append(f'{b}: {LABELS[FEATURES.index(k)]} (ρ={corr(d[k],d.cycle_life):.3f})')
    text('탐색 후보 중 가장 큰 절대 상관: '+'; '.join(strongest)+'. 이는 단변량 순위이며 최종 선정 순위는 아니다.')
    insight('물리적 설명 가능성, 예측 시점에서의 이용 가능성, 배치 간 방향 안정성, 검증에서의 추가 기여를 함께 기준으로 삼는다. 상관이 낮은 피처도 비선형·조건부 관계가 있을 수 있어 자동 폐기하지 않는다.')
    page('Q5+','피처 중복과 다중공선성','배치별 피처 간 Spearman 상관 / DQv=로그 분산, DQm=최솟값')
    image('design_collinearity.png',80)
    pairs=sorted(res['collinear_pairs'],key=lambda r:abs(r[3]),reverse=True)[:5]
    table([['배치','|ρ|≥0.8 중복 후보 (상위 예시)','ρ']]+[[b,f'{a} / {c}',f'{r:.3f}'] for b,a,c,r in pairs],[26,119,26])
    text('VIF는 아홉 후보를 동시에 사용한 선형 진단으로 계산했다. 배치별 전체 결과는 day1_vif.csv에 저장했다. 정책 조합이 제한되면 C1·C2·SOC가 함께 움직이고 VIF가 커질 수 있다. 표본이 39~41개이므로 수치 안정성에 주의한다.')
    vmax=[]
    for b in BATCHES:
        item=max([r for r in res['vifs'] if r[0]==b],key=lambda r:r[2])
        vmax.append(f'{b}: {item[1]} VIF={item[2]:.1f}')
    text('동시 입력 시 최대 VIF: '+'; '.join(vmax)+'. 큰 값은 해당 묶음을 그대로 사용하기 전에 축소가 필요하다는 신호다.')
    insight('훈련 폴드에서만 결측 대치·표준화·상관/VIF 진단을 적합한다. |ρ|≥0.8 또는 VIF>5는 검토 기준으로 쓰고 의미·안정성을 고려해 대표 변수를 남긴다. Ridge/ElasticNet 규제도 비교한다. 표준화는 스케일을 맞출 뿐 다중공선성을 해결하지 않으며, 낮은 상관도 통계적 독립성을 보장하지 않는다.')
    page('02','피처 선정과 파생의 타당성','선정안이며 최종 확정은 DAY2의 훈련 데이터 내부 비교에서 수행한다')
    table([['후보 묶음','이유','주의·확인 계획'],['D1: log10 Var(ΔQ)','산포와 수명 관계가 세 배치에서 같은 방향','분산은 ddof=1; 양수 확인 후 로그. 가장 단순한 기준 모델'],['D2: D1 + min(ΔQ)','국소적으로 큰 변화 추가 요약','분산과 중복 점검; 추가 성능 없으면 제거'],['상태: D1 + 초기 Qd·IR 기울기 + 평균 온도 + 충전시간 기울기','용량 감소, 저항 변화, 열·충전 행동을 반영','10~100만 사용; 이상값·결측·중복 점검'],['조건: 상태 + C1·SOC·C2 / 전류 요약','충전 정책과 실제 전류 기록의 영향 검토','정책 혼재·공선성·샘플 평균 한계; 별도 추가 비교'],['금지 입력','knee, 후기 기울기, 총 관측 사이클, 셀 ID, 수명','미래정보 누수·식별자 의존을 방지']],[42,65,64])
    text('수명은 총 cycle life로 정의하며 잔여 수명(RUL)과 구분한다. 100사이클 이후 잔여 수명이 필요하면 예측 총 수명에서 100을 빼는 별도 운영 정의가 필요하다. 본 DAY1은 운영 적용의 성능을 입증하지 않는다.')
    insight('독립성을 확보했다는 표현 대신 중복을 줄이고 조건부 추가 기여를 검증한다고 기술한다. 작은 표본에서는 변수 수를 줄인 D1·D2를 우선 비교하며, 피처를 늘린 모델은 실제 개선 여부로 판단한다.')
    page('03','모델 후보와 검증 설계','회귀 선택 / 아직 모델 학습·최종 선정·성능 측정 전')
    table([['후보','선택 이유','탐색할 설정'],['중앙값 기준 모델','초기 신호 없이도 얻는 성능의 기준','Batch1 훈련 타깃 중앙값'],['Ridge','소표본·상관 피처의 계수 변동 억제','alpha: 0.01, 0.1, 1, 10, 100'],['ElasticNet','규제와 변수 축소를 함께 비교','alpha: 0.001~1; l1_ratio: 0.1, 0.5, 0.9'],['SVR(RBF)','ΔQ와 수명의 비선형 관계 비교','C: 1,10,100; epsilon: 0.01,0.1; gamma: scale,0.1'],['Random Forest / Gradient Boosting','조건 간 상호작용 탐색','깊이 2~5·leaf 최소 3~8 / learning_rate 0.03,0.1']],[40,72,59])
    text('원 타깃과 로그 타깃을 비교할 계획이다. 로그 모델도 exp 역변환 후 원 수명 단위에서 MAPE·MAE·RMSE·R²를 계산한다. 로그 공간 오차를 원 단위 MAPE로 혼동하지 않는다.')
    text('분할안: Batch1 충전정책 그룹 기준 약 75% 개발 / 25% 홀드아웃(seed=42). 같은 정책의 셀이 양쪽에 겹치지 않게 한다. 개발 데이터는 바깥 GroupKFold 4분할로 후보 비교, 안쪽 3분할로 파라미터 탐색. 모든 전처리는 폴드 내부에서 수행한다.')
    text('선정안: 개발 교차검증 MAPE를 우선하고 MAE·RMSE·오차 변동도 확인한다. 유사한 성능이면 더 단순하고 설명 가능한 모델을 택한다. 홀드아웃으로 일반화 확인 후 확정 전략으로 Batch1 전체를 학습해 Batch2 필수·Batch3 추가 평가를 계획한다. 외부 결과를 보고 후보를 재선정하지 않는다.')
    insight('DAY1의 결론은 회귀·피처 후보·검증 절차의 설계다. 어떤 모델이 실제로 우수한지와 선택된 파라미터의 근거는 DAY2에서 실험 결과로 제시한다.')
    page('04','통계 해석과 설계 근거 요약','그래프를 나열하지 않고 질문별 관찰을 다음 결정으로 연결한다')
    table([['EDA 질문','관찰·해석','설계 시사점'],['분포','배치 간 수명 분포 차이, 짧은 수명 집중','로그 타깃 비교·단수명 오류 별도 확인'],['열화','셀별 가속 양상·곡선 변화점 차이','초기 기울기 후보 / 후기·knee 제외'],['ΔQ','세 배치 로그 분산-수명 음의 관계','D1 기준 + D2 추가 기여 비교'],['충전조건','정책 조합과 배치가 함께 달라짐','조건 추가 비교·인과 주장 금지'],['피처 관계','신호 강도와 중복이 배치마다 다름','폴드 내부 선정·규제·소수 피처 우선']],[27,73,71])
    text(f"보조 가설검증: 세 배치의 수명 분포가 동일하다는 귀무가설을 Kruskal-Wallis로 탐색했다. H={res['kruskal_H']:.3f}, p={res['kruskal_p']:.3g}. 동일한 분포라는 가설에 반하는 근거다. 분포 모양이 다르면 이를 중앙값 차이 검정으로만 해석할 수 없다.")
    text('전제·한계: 셀 관측의 독립성과 순위 비교 가능성을 가정한다. 동일 정책·시험 장비로 인한 군집 의존성이 남을 수 있다. 유의성은 제조 배치나 충전조건의 인과효과를 증명하지 않으며, 어느 배치 쌍이 다른지는 이 전체 검정만으로 알 수 없다. 필수 검정이 아닌 분포 탐색의 보조 근거로 사용했다.')
    text('자료·재현: Kaggle itshpark/data-driven-prediction-of-battery-cycle (version1); 저자 공개 로더의 정제 규칙; 수업 DAY1 안내와 제출 화면. 원자료 → extract_features.py → day1_design.py의 순서로 통계·그림·이 PDF를 재생성한다. run_project.py --mode day1은 모델을 학습하지 않는다.')
    text('Kaggle: https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle')
    text('공식 로더: https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation')
    for b in BATCHES:
        d=policies[policies.batch==b].sort_values('mean_life').reset_index(drop=True)
        page('부록',f'{b} 충전정책별 평균 수명','Q4 버블 번호 대응 / 평균은 기술통계이며 최적 정책 추천이 아니다')
        table([['번호','정책: C1(전환 SOC)-C2','n','평균 수명']]+[[i+1,r.charge_policy,r.n,f'{r.mean_life:.1f}'] for i,r in enumerate(d.itertuples())],[17,101,17,36])
        text('n=1 정책은 단일 셀의 결과다. 평균 차이를 정책 효과로 판단하려면 반복 셀·동일 조건 비교가 더 필요하다.')
    doc.build(story)
    print(out)

if __name__=='__main__':build()
