"""Batch-1-only model selection, policy-grouped validation, frozen external tests."""
from pathlib import Path
import json, warnings
from datetime import datetime, timezone
import joblib
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor
from project_fonts import korean_font
from reporting import export_performance

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work'/'day2'; OUT.mkdir(exist_ok=True)
FIG=ROOT/'figures'/'day2'; FIG.mkdir(exist_ok=True)
font_path=korean_font();font_manager.fontManager.addfont(font_path)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font_path).get_name(),'axes.unicode_minus':False,'figure.dpi':140,'savefig.dpi':200,'font.size':10})
FEATURES={
 'D1':['log_delta_q_var'],
 'D2':['log_delta_q_var','delta_q_min'],
 'State':['log_delta_q_var','qd_early_slope','ir_early_slope','tavg_early_mean','charge_time_early_slope'],
 'Full':['log_delta_q_var','qd_early_slope','ir_early_slope','tavg_early_mean','charge_time_early_slope','c_rate_1','switch_soc'],
}

def factory(name,log=True):
    if name=='Ridge': model,grid=Ridge(),{'alpha':[.01,.1,1,10,100]}
    elif name=='ElasticNet': model,grid=ElasticNet(max_iter=30000),{'alpha':[.0001,.001,.01,.1],'l1_ratio':[.1,.5,.9]}
    elif name=='SVR': model,grid=SVR(),{'C':[.1,1,10],'epsilon':[.03,.1],'gamma':['scale',.1]}
    elif name=='RandomForest': model,grid=RandomForestRegressor(n_estimators=150,random_state=42,n_jobs=1),{'max_depth':[3,5,None],'min_samples_leaf':[2,4],'max_features':[.7,1.]}
    else: model,grid=GradientBoostingRegressor(random_state=42),{'n_estimators':[60,120],'learning_rate':[.03,.1],'max_depth':[1,2],'min_samples_leaf':[3]}
    pipe=Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',model)])
    ttr=TransformedTargetRegressor(regressor=pipe,func=np.log if log else None,inverse_func=np.exp if log else None)
    return ttr,{'regressor__model__'+k:v for k,v in grid.items()}

def tune(name,df,cols,log=True):
    estimator,grid=factory(name,log)
    cv=GroupKFold(n_splits=3)
    search=GridSearchCV(estimator,grid,scoring='neg_mean_absolute_percentage_error',cv=cv,n_jobs=-1,error_score='raise')
    search.fit(df[cols],df.cycle_life,groups=df.charge_policy)
    return search

def metrics(y,p):
    return {'MAPE':float(mean_absolute_percentage_error(y,p)*100),'MAE':float(mean_absolute_error(y,p)),
            'RMSE':float(root_mean_squared_error(y,p)),'R2':float(r2_score(y,p))}

def savefig(name):
    plt.tight_layout();plt.savefig(FIG/name,bbox_inches='tight',facecolor='white');plt.close()

def main():
    warnings.filterwarnings('ignore',category=UserWarning,module='sklearn')
    corrected=OUT/'features_all.csv'
    fresh=ROOT/'work'/'features_all.csv'
    dates=['2017-05-12','2018-02-20','2018-04-12']
    raw_complete=all((ROOT/'data'/f'{date}_batchdata_updated_struct_errorcorrect.mat').exists() for date in dates)
    source=fresh if raw_complete and fresh.exists() else corrected
    if not source.exists():raise FileNotFoundError('전체 원자료 또는 work/day2/features_all.csv가 필요합니다')
    df=pd.read_csv(source)
    df=df[np.isfinite(df.cycle_life)&(df.cycle_life>0)].copy()
    assert df.cell_id.is_unique
    # Batch 1 has a zero-capacity placeholder at array index 0. Batches 2/3
    # start with a real cycle. Align discharge-cycle ordinal numbers so no
    # external feature uses cycle 101 while claiming a cycle-100 cutoff.
    files={'Batch 1':'2017-05-12','Batch 2':'2018-02-20','Batch 3':'2018-04-12'}
    for batch,date in files.items():
        raw_path=ROOT/'data'/f'{date}_batchdata_updated_struct_errorcorrect.mat'
        if not raw_path.exists():
            if 'cycle100_array_index' not in df:raise FileNotFoundError('Need corrected features or original MAT files')
            continue
        with h5py.File(raw_path) as f:
            a=f['batch']
            for idx,row in df[df.batch==batch].iterrows():
                cell=int(row.cell_id.split('c')[-1]);g=f[a['cycles'][cell,0]]
                first=np.asarray(f[g['Qdlin'][0,0]][()]).ravel()
                offset=1 if np.allclose(first,0) else 0
                i10,i100=9+offset,99+offset
                d=np.asarray(f[g['Qdlin'][i100,0]][()]).ravel()-np.asarray(f[g['Qdlin'][i10,0]][()]).ravel()
                d=d[np.isfinite(d)]
                df.loc[idx,'log_delta_q_var']=np.log10(max(np.var(d,ddof=1),1e-16))
                df.loc[idx,'delta_q_min']=np.min(d)
                df.loc[idx,'cycle10_array_index']=i10;df.loc[idx,'cycle100_array_index']=i100
    df.to_csv(OUT/'features_all.csv',index=False)
    b1=df[df.batch=='Batch 1'].reset_index(drop=True)
    tr,va=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=42).split(b1,groups=b1.charge_policy))
    dev,valid=b1.iloc[tr].copy(),b1.iloc[va].copy()
    assert not set(dev.charge_policy)&set(valid.charge_policy)
    split=pd.concat([dev.assign(split='development'),valid.assign(split='holdout')]).sort_values('cell_id')
    split[['cell_id','charge_policy','cycle_life','split']].to_csv(OUT/'split_manifest.csv',index=False)
    configs=[(m,s) for m in ['Ridge','ElasticNet','SVR'] for s in FEATURES]+[(m,'Full') for m in ['RandomForest','GradientBoosting']]
    rows=[];fold_rows=[]; cv=GroupKFold(n_splits=4)
    for name,fset in configs:
        cols=FEATURES[fset];fold_scores=[]
        for fold,(i,j) in enumerate(cv.split(dev,groups=dev.charge_policy),1):
            train,test=dev.iloc[i],dev.iloc[j]
            assert not set(train.charge_policy)&set(test.charge_policy)
            search=tune(name,train,cols)
            score=metrics(test.cycle_life,search.predict(test[cols]))['MAPE']
            fold_scores.append(score);fold_rows.append({'model':name,'features':fset,'fold':fold,'n':len(test),'MAPE':score})
        search=tune(name,dev,cols)
        rows.append({'model':name,'features':fset,'n_features':len(cols),'CV_MAPE':np.mean(fold_scores),'CV_SD':np.std(fold_scores,ddof=1),'inner_MAPE':-search.best_score_*100,'params':search.best_params_})
        print(name,fset,'nested CV',round(np.mean(fold_scores),2),flush=True)
    table=pd.DataFrame(rows).sort_values('CV_MAPE').reset_index(drop=True)
    # Selection occurs before any holdout or external prediction is evaluated.
    # Predeclared simplicity tie-break, using development CV only.
    best=table.iloc[0]
    simple=table[(table.model=='Ridge')&(table.features=='D1')].iloc[0]
    tie_tolerance_pp=.01
    if simple.CV_MAPE<=best.CV_MAPE+tie_tolerance_pp:best=simple
    name,fset=best['model'],best['features'];cols=FEATURES[fset]
    print('FROZEN WINNER',name,fset,flush=True)
    selection_model=tune(name,dev,cols)
    valid_pred=selection_model.predict(valid[cols]); valid_metrics=metrics(valid.cycle_life,valid_pred)
    # Refit selected configuration family with grouped tuning using all Batch 1.
    final=tune(name,b1,cols)
    joblib.dump({'model':final.best_estimator_,'features':cols,'batch1_cells':b1.cell_id.tolist()},OUT/'final_model.joblib')
    external=[];test_metrics={};baselines={}
    for batch in ['Batch 2','Batch 3']:
        test=df[df.batch==batch].copy();pred=final.predict(test[cols])
        test_metrics[batch]=metrics(test.cycle_life,pred)
        baselines[batch]=metrics(test.cycle_life,np.repeat(b1.cycle_life.median(),len(test)))
        test['prediction']=pred;test['APE']=100*np.abs(pred-test.cycle_life)/test.cycle_life;test['residual']=pred-test.cycle_life
        external.append(test)
    predictions=pd.concat(external); predictions.to_csv(OUT/'external_predictions.csv',index=False)
    pd.DataFrame({'cell_id':valid.cell_id,'actual':valid.cycle_life,'prediction':valid_pred}).to_csv(OUT/'holdout_predictions.csv',index=False)
    table.to_csv(OUT/'candidate_comparison.csv',index=False);pd.DataFrame(fold_rows).to_csv(OUT/'nested_cv_folds.csv',index=False)
    # Permutation importance uses the independent Batch-1 holdout, not training fit.
    importance=permutation_importance(selection_model,valid[cols],valid.cycle_life,scoring='neg_mean_absolute_percentage_error',n_repeats=100,random_state=42,n_jobs=-1)
    imp=pd.DataFrame({'feature':cols,'importance_pp':importance.importances_mean*100,'std_pp':importance.importances_std*100}).sort_values('importance_pp',ascending=False)
    imp.to_csv(OUT/'holdout_permutation_importance.csv',index=False)
    X=StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(b1[cols]));vif={c:float(variance_inflation_factor(X,i)) if len(cols)>1 else 1. for i,c in enumerate(cols)}
    # Prespecified sensitivity: same estimator/feature family, raw target.
    raw=tune(name,dev,cols,False);raw_valid=metrics(valid.cycle_life,raw.predict(valid[cols]))
    result={'selection_rule':'minimum development nested group CV MAPE; prefer Ridge D1 within 0.01 percentage points of minimum',
      'tie_tolerance_pp':tie_tolerance_pp,'minimum_candidate_CV_MAPE':float(table.CV_MAPE.min()),
      'generated_at_utc':datetime.now(timezone.utc).isoformat(),
      'selected_model':name,'selected_feature_set':fset,'features':cols,'n_development':len(dev),'n_valid':len(valid),
      'n_batch1':len(b1),'n_batch2':int((df.batch=='Batch 2').sum()),'n_batch3':int((df.batch=='Batch 3').sum()),'n_dev_policies':dev.charge_policy.nunique(),'n_valid_policies':valid.charge_policy.nunique(),
      'CV_MAPE':float(best.CV_MAPE),'CV_SD':float(best.CV_SD),'valid':valid_metrics,'test':test_metrics,'baseline':baselines,
      'gap_train_valid_pp':valid_metrics['MAPE']-float(best.CV_MAPE),'gap_valid_test_pp':test_metrics['Batch 2']['MAPE']-valid_metrics['MAPE'],
      'gap_target_test_pp':test_metrics['Batch 2']['MAPE']-9.1,
      'gap_batch2_batch3_pp':test_metrics['Batch 3']['MAPE']-test_metrics['Batch 2']['MAPE'],
      'gap_target_batch3_pp':test_metrics['Batch 3']['MAPE']-9.1,
      'selection_params':selection_model.best_params_,'final_params':final.best_params_,
      'vif':vif,'raw_target_valid':raw_valid,'target':9.1}
    for batch,g in predictions.groupby('batch'):
        rng=np.random.default_rng(42);boot=np.mean(rng.choice(g.APE.to_numpy(),(5000,len(g)),replace=True),axis=1)
        result['test'][batch]['MAPE_ci95']=np.percentile(boot,[2.5,97.5]).tolist()
        result['test'][batch]['overprediction_n']=int((g.residual>0).sum())
    result['largest_errors']=predictions.sort_values('APE',ascending=False)[['cell_id','batch','cycle_life','prediction','APE']].head(6).to_dict('records')
    result['subgroups']=[]
    for (batch,label),g in predictions.assign(life_group=np.where(predictions.cycle_life<500,'short',np.where(predictions.cycle_life>1000,'long','middle'))).groupby(['batch','life_group']):
        result['subgroups'].append({'batch':batch,'group':label,'n':len(g),'MAPE':float(g.APE.mean()),'bias_cycles':float(g.residual.mean())})
    (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    export_performance(result)
    # Charts
    visible=table[table.CV_MAPE<100].copy();labels=visible.model+' / '+visible.features
    fig,ax=plt.subplots(figsize=(10,4.8))
    ax.barh(labels,visible.CV_MAPE,xerr=visible.CV_SD,color=['#167D87']+['#92ADB9']*(len(visible)-1),capsize=2)
    ax.invert_yaxis();ax.set(xlabel='Nested grouped CV MAPE (%)',title='Batch 1 후보 비교 (평균 + fold SD)')
    failed=table[table.CV_MAPE>=100]
    ax.text(.01,-.23,'; '.join(f'{r.model}/{r.features}: CV {r.CV_MAPE:,.0f}% (범위 밖)' for r in failed.itertuples()),transform=ax.transAxes,fontsize=9,color='#B44949')
    savefig('01_candidates.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,(batch,g) in zip(axes,predictions.groupby('batch')):
        sc=ax.scatter(g.cycle_life,g.prediction,c=g.APE,cmap='YlOrRd',edgecolor='white',s=48,vmin=0,vmax=70)
        fig.colorbar(sc,ax=ax,label='APE (%)',shrink=.75)
        limit=max(g.cycle_life.max(),g.prediction.max())*1.07;ax.plot([0,limit],[0,limit],'--',color='#536471')
        ax.set(xlim=(0,limit),ylim=(0,limit),xlabel='Actual cycle life',ylabel='Predicted cycle life',title=f'{batch}: MAPE {test_metrics[batch]["MAPE"]:.1f}%')
    savefig('02_prediction.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for batch,g in predictions.groupby('batch'):
        axes[0].scatter(g.cycle_life,g.residual,label=batch,alpha=.75)
        axes[1].scatter(g.log_delta_q_var,g.APE,label=batch,alpha=.75)
    axes[0].axhline(0,ls='--',color='#536471');axes[0].set(xlabel='Actual cycle life',ylabel='Prediction - actual (cycles)',title='양수 잔차: 수명 과대예측')
    axes[1].set(xlabel='log10 Var[ΔQ]',ylabel='Absolute percentage error (%)',title='신호 구간별 오차');axes[0].legend();axes[1].legend();savefig('03_residual.png')
    fig,ax=plt.subplots(figsize=(9,3.6));ax.barh(imp.feature,imp.importance_pp,xerr=imp.std_pp,color='#167D87',capsize=3);ax.invert_yaxis();ax.axvline(0,color='#555',lw=.7)
    ax.set(xlabel='Permutation 후 MAPE 증가 (%p)',title='Batch 1 hold-out 피처 중요도 (100회, 선: 반복 SD)');savefig('04_importance.png')
    fig,ax=plt.subplots(figsize=(9,3.8))
    for model in ['ElasticNet','SVR']:
        g=table[table.model==model].set_index('features').loc[list(FEATURES)]
        ax.plot(g.index,g.CV_MAPE,'o-',label=model)
    ax.set(xlabel='D1: 분산 / D2: +최솟값 / State: +상태 / Full: +정책',ylabel='Nested CV MAPE (%)',title='피처 추가가 항상 성능을 높이는가?');ax.legend();savefig('05_ablation.png')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
