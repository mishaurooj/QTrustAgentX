#!/usr/bin/env python3
"""Reviewer 5: original-source native specialists + phase-matched 2x2 repeated ablation.
Native: URL numeric ExtraTrees, email/SMS TFIDF-SVD-calibrated LinearSVC, QR 68-feature ExtraTrees.
The manuscript's local RC-QTG circuit is used, with 20 Pauli observables and no
post-readout reliability summary (to isolate in-circuit reliability).
Checkpoints each seed. The protocol is a reconstruction from supplied source files
and manuscript description, not a claim to have the unpublished original models.
"""
import argparse, hashlib, itertools, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,matthews_corrcoef,roc_auc_score,brier_score_loss

MODS=('url','email','sms','qr');PAIRS=tuple(itertools.combinations(range(4),2));PHASES=(.5,1.)
def native_model(mod,seed,fast=False):
    if mod in ('url','qr'):
        return make_pipeline(StandardScaler(),ExtraTreesClassifier(
            n_estimators=(80 if fast else (600 if mod=='url' else 500)),
            class_weight='balanced',n_jobs=-1,random_state=seed))
    tf=TfidfVectorizer(ngram_range=(1,2),max_features=(8000 if fast else 25000),
                       min_df=2,sublinear_tf=True,dtype=np.float32)
    svd=TruncatedSVD(n_components=(48 if fast else 256),random_state=seed)
    svm=LinearSVC(class_weight='balanced',random_state=seed,max_iter=3000)
    cal=CalibratedClassifierCV(svm,method='sigmoid',cv=(2 if fast else 3))
    return make_pipeline(tf,svd,StandardScaler(),cal)
def local_zx(score,reliability,beta):
    p=np.clip(np.asarray(score,float),1e-8,1-1e-8);r=float(reliability)
    th=2*np.arcsin(np.sqrt(p));a0=th*np.sqrt(r);a1=1.1*th*np.sqrt(r)
    phi=beta*np.pi*(p-.5)*r; rx=.25*th*r
    a=np.ones_like(p,dtype=complex);b=np.zeros_like(p,dtype=complex)
    def ry(a,b,angle):
        c=np.cos(angle/2);s=np.sin(angle/2);return c*a-s*b,s*a+c*b
    def rz(a,b,angle):return a*np.exp(-.5j*angle),b*np.exp(.5j*angle)
    a,b=ry(a,b,a0);a,b=rz(a,b,phi)
    c=np.cos(rx/2);s=np.sin(rx/2);a,b=c*a-1j*s*b,c*b-1j*s*a
    a,b=ry(a,b,a1);a,b=rz(a,b,phi)
    return np.abs(a)**2-np.abs(b)**2,2*np.real(np.conj(a)*b)
def quantum_features(scores,mods,rho,beta):
    n=len(scores);z=np.ones((n,4));x=np.zeros((n,4));mods=np.asarray(mods)
    for j,m in enumerate(MODS):
        ix=np.flatnonzero(mods==m)
        if len(ix):z[ix,j],x[ix,j]=local_zx(scores[ix],rho[m],beta)
    return np.column_stack([z,x,*[z[:,a]*z[:,b] for a,b in PAIRS],
                             *[x[:,a]*x[:,b] for a,b in PAIRS]]).astype(np.float32)
def ci(vals):
    v=np.asarray(vals,float);mean=float(v.mean());n=len(v)
    if n<2:return mean,np.nan,np.nan
    w=float(student_t.ppf(.975,n-1)*v.std(ddof=1)/np.sqrt(n))
    return mean,mean-w,mean+w
def signflip(v):
    v=np.asarray(v,float);n=len(v);observed=abs(v.mean())
    if n<=16:
        return float(np.mean([abs(np.mean(v*np.array(s)))>=observed-1e-12
                              for s in itertools.product((-1,1),repeat=n)]))
    rng=np.random.default_rng(42);s=rng.choice((-1,1),size=(100000,n))
    return float((1+np.sum(np.abs((s*v).mean(axis=1))>=observed-1e-12))/(len(s)+1))
def get_X(d,ix,mod,native):
    sub=d.iloc[ix];
    if mod in ('url','qr'):return native[mod][sub.native_index.to_numpy(dtype=int)]
    return sub.text.fillna('').astype(str).to_numpy()
def run_seed(seed,d,native,args):
    tr=[];te=[]
    for mod in MODS:
        sub=d[d.modality.eq(mod)]
        split=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed)
        a,b=next(split.split(np.zeros(len(sub)),sub.label,sub.group_id))
        tr.extend(sub.index[a]);te.extend(sub.index[b])
    tr=np.array(sorted(tr));te=np.array(sorted(te))
    assert not (set(d.iloc[tr].group_id)&set(d.iloc[te].group_id))
    p_tr=np.full(len(tr),np.nan);p_te=np.full(len(te),np.nan)
    trloc={int(v):i for i,v in enumerate(tr)};teloc={int(v):i for i,v in enumerate(te)}
    for mod in MODS:
        mt=np.array([i for i in tr if d.at[i,'modality']==mod]);ms=np.array([i for i in te if d.at[i,'modality']==mod])
        part=d.iloc[mt]
        cv=StratifiedGroupKFold(n_splits=(3 if args.fast else args.inner_folds),shuffle=True,random_state=seed+101)
        for fold,(a,b) in enumerate(cv.split(np.zeros(len(mt)),part.label,part.group_id)):
            model=native_model(mod,seed+fold,args.fast)
            model.fit(get_X(d,mt[a],mod,native),d.iloc[mt[a]].label)
            prob=model.predict_proba(get_X(d,mt[b],mod,native))[:,1]
            for i,p in zip(mt[b],prob):p_tr[trloc[int(i)]]=p
        model=native_model(mod,seed,args.fast)
        model.fit(get_X(d,mt,mod,native),d.iloc[mt].label)
        prob=model.predict_proba(get_X(d,ms,mod,native))[:,1]
        for i,p in zip(ms,prob):p_te[teloc[int(i)]]=p
        print(f'  {mod}: train {len(mt)} test {len(ms)}',flush=True)
    if not np.isfinite(p_tr).all() or not np.isfinite(p_te).all():raise RuntimeError('Missing OOF/test specialist scores')
    rho={};train=d.iloc[tr];test=d.iloc[te]
    for mod in MODS:
        idx=train.modality.to_numpy()==mod;y=train.label.to_numpy()[idx];p=p_tr[idx]
        skill=1-np.mean((y-p)**2)/max(np.mean((y-y.mean())**2),1e-8)
        rho[mod]=float(np.clip(skill,.05,1.))
    metrics=[];pred=[]
    for phase in PHASES:
        for reliability in ('estimated','uniform'):
            rr=rho if reliability=='estimated' else dict.fromkeys(MODS,1.)
            xt=quantum_features(p_tr,train.modality,rr,phase)
            xv=quantum_features(p_te,test.modality,rr,phase)
            arb=make_pipeline(StandardScaler(),LogisticRegression(
                class_weight='balanced',solver='lbfgs',max_iter=3000,random_state=seed))
            arb.fit(xt,train.label)
            prob=arb.predict_proba(xv)[:,1];yp=(prob>=.5).astype(int);y=test.label.to_numpy()
            cond=f'beta{phase:.2f}_{reliability}'
            metrics.append(dict(seed=seed,phase=phase,reliability=reliability,condition=cond,
                n_train=len(tr),n_test=len(te),f1=f1_score(y,yp),mcc=matthews_corrcoef(y,yp),
                auc=roc_auc_score(y,prob),brier=brier_score_loss(y,prob)))
            pred.extend(dict(seed=seed,source_id=s,modality=m,condition=cond,
                             y_true=int(yy),y_pred=int(pp),y_score=float(pr))
                        for s,m,yy,pp,pr in zip(test.source_id,test.modality,y,yp,prob))
    splits=[dict(seed=seed,source_id=s,modality=m,split=sp)
            for sp,dd in [('train',train),('test',test)] for s,m in zip(dd.source_id,dd.modality)]
    reli=[dict(seed=seed,modality=m,estimated_rho=r) for m,r in rho.items()]
    return pd.DataFrame(metrics),pd.DataFrame(pred),pd.DataFrame(splits),pd.DataFrame(reli)
def table_tex(summary,paired,out,n):
    lines=[r'\begin{table}[t]',r'\centering',r'\begingroup\color{blue}\scriptsize',
      r'\caption{\textcolor{blue}{Phase-matched RC-QTG reliability ablation: '+str(n)+r' paired repeated outer splits. Entries are mean performance. Paired 95\% confidence intervals and Holm-adjusted significance tests appear below.}}',
      r'\label{tab:reliability_phase_factorial}',r'\resizebox{\linewidth}{!}{%',
      r'\begin{tabular}{llrrrr}',r'\toprule',r'Phase & Reliability & F1 & MCC & ROC-AUC & Brier \\',r'\midrule']
    for _,r in summary.sort_values(['phase','reliability']).iterrows():
        lines.append(f"{r.phase:.2f} & {r.reliability.capitalize()} & {r.f1_mean:.4f} & {r.mcc_mean:.4f} & {r.auc_mean:.4f} & {r.brier_mean:.4f} "+r'\\')
    lines.extend([r'\midrule',r'\multicolumn{6}{l}{Paired differences: estimated minus uniform (95\% CI; Holm-adjusted $p$)} \\'])
    for _,r in paired.sort_values(['phase','metric']).iterrows():
        lines.append(r'\multicolumn{6}{l}{'+f"$\\beta={r.phase:.2f}$, {r.metric.upper()}: $\\Delta={r.estimated_minus_uniform:+.5f}$, 95\\% CI $[{r.ci_low:+.5f},{r.ci_high:+.5f}]$, $p_{{\\mathrm{{Holm}}}}={r.p_holm_8_tests:.4f}$"+r'} \\')
    lines.extend([r'\bottomrule',r'\end{tabular}}',r'\endgroup',r'\end{table}'])
    out.write_text('\n'.join(lines)+'\n',encoding='utf8')
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('Reviewer5_Original_Native'))
    p.add_argument('--output',type=Path,default=Path('Reviewer5_Original_Native/Results'))
    p.add_argument('--seeds',type=int,default=10);p.add_argument('--seed-start',type=int,default=42)
    p.add_argument('--inner-folds',type=int,default=5);p.add_argument('--fast',action='store_true',help='DEBUG ONLY: fewer trees, lower SVD, 3 folds; not manuscript-grade')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(args.data/'NATIVE_SOURCE_MANIFEST.csv',low_memory=False).fillna({'text':'','group_id':''})
    d['source_id']=d.source_id.astype(str)
    if d.source_id.duplicated().any():raise ValueError('Duplicate source IDs')
    required={'url','email','sms','qr'}
    if set(d.modality.unique())!=required:raise ValueError('Missing/extra modalities; use original four-modality source corpus')
    f=np.load(args.data/'NATIVE_FEATURES.npz');native={'url':f['url'],'qr':f['qr']}
    print('CORPUS',len(d),d.modality.value_counts().to_dict(),flush=True)
    if args.fast:print('WARNING: FAST MODE IS DEBUG ONLY; DO NOT USE FOR PAPER',flush=True)
    # Guard against accidentally reusing checkpoints from a different corpus or setting.
    digest=hashlib.sha256((args.data/'NATIVE_SOURCE_MANIFEST.csv').read_bytes()).hexdigest()
    protocol={'manifest_sha256':digest,'seeds_start':args.seed_start,'inner_folds':args.inner_folds,
              'fast':args.fast,'model':'source-native-v2'}
    stamp=args.output/'CHECKPOINT_PROTOCOL.json'
    if stamp.exists():
        previous=json.loads(stamp.read_text(encoding='utf8'))
        if previous!=protocol:raise RuntimeError('Checkpoint protocol changed. Use a new --output directory. '+str(stamp))
    else:
        if list(args.output.glob('SEED_*.npz')):raise RuntimeError('Existing checkpoints have no protocol audit; use a new output directory')
        stamp.write_text(json.dumps(protocol,indent=2),encoding='utf8')
    files=[]
    for seed in range(args.seed_start,args.seed_start+args.seeds):
        checkpoint=args.output/f'SEED_{seed}.npz'
        if checkpoint.exists():
            print('Resume cached',seed,flush=True)
        else:
            print('START seed',seed,flush=True);t=time.time()
            m,pr,sp,rh=run_seed(seed,d,native,args)
            # Save one atomic checkpoint per seed so interruptions do not discard prior work.
            temp=checkpoint.with_suffix('.tmp.npz')
            np.savez_compressed(temp,metrics=m.to_records(index=False),pred=pr.to_records(index=False),
                                splits=sp.to_records(index=False),reli=rh.to_records(index=False))
            temp.replace(checkpoint)
            print('DONE seed',seed,'seconds',round(time.time()-t,1),flush=True)
        files.append(checkpoint)
    def collect(key):
        return pd.concat([pd.DataFrame.from_records(np.load(f,allow_pickle=True)[key]) for f in files],ignore_index=True)
    met=collect('metrics');pred=collect('pred');splits=collect('splits');reli=collect('reli')
    met.to_csv(args.output/'TABLE5_ALL_SEED_METRICS.csv',index=False)
    pred.to_csv(args.output/'TABLE5_TEST_PREDICTIONS.csv',index=False)
    splits.to_csv(args.output/'TABLE5_SPLIT_ASSIGNMENTS.csv',index=False)
    reli.to_csv(args.output/'TABLE5_TRAINING_RELIABILITY.csv',index=False)
    summary=[]
    for (phase,mode),g in met.groupby(['phase','reliability']):
        r=dict(phase=phase,reliability=mode,repeats=len(g))
        for k in ('f1','mcc','auc','brier'):
            mu,lo,hi=ci(g[k]);r.update({k+'_mean':mu,k+'_ci_low':lo,k+'_ci_high':hi})
        summary.append(r)
    summary=pd.DataFrame(summary);summary.to_csv(args.output/'TABLE5_REPEATED_4_CELLS.csv',index=False)
    tests=[]
    for phase in PHASES:
        a=met[(met.phase==phase)&(met.reliability=='estimated')].set_index('seed')
        b=met[(met.phase==phase)&(met.reliability=='uniform')].set_index('seed')
        for k in ('f1','mcc','auc','brier'):
            diff=(a[k]-b[k]).sort_index();mu,lo,hi=ci(diff)
            tests.append(dict(phase=phase,metric=k,estimated_minus_uniform=mu,
                              ci_low=lo,ci_high=hi,p_signflip=signflip(diff),n_pairs=len(diff)))
    tests=pd.DataFrame(tests);idx=np.argsort(tests.p_signflip.to_numpy());pvals=tests.p_signflip.to_numpy();adj=np.empty(len(tests));running=0
    for j,i in enumerate(idx):
        running=max(running,min(1.,(len(tests)-j)*pvals[i]));adj[i]=running
    tests['p_holm_8_tests']=adj;tests.to_csv(args.output/'TABLE5_PAIRED_CI_AND_TESTS.csv',index=False)
    table_tex(summary,tests,args.output/'TABLE5_REVISED_BLUE.tex',args.seeds)
    audit=json.loads((args.data/'SOURCE_AUDIT.json').read_text())
    audit.update({'seeds':list(range(args.seed_start,args.seed_start+args.seeds)),
                  'inner_folds':3 if args.fast else args.inner_folds,'debug_fast':args.fast,
                  'specialists':'native URL numeric ExtraTrees; email/SMS TFIDF SVD calibrated LinearSVC; QR image 68-feature ExtraTrees',
                  'phase':'manuscript local 2-layer RY/RZ/RX/RZ; 20 Pauli features; no post-readout summaries',
                  'uncertainty':'paired resplit t CI, exact sign-flip test, Holm across eight tests; overlapping resplits are dependent',
                  'original_code_reproduced_exactly':False,
                  'warning':'This reconstructs manuscript-specified models from uploaded source archive. Exact original model weights/splits were not provided; do not assert numerical replication.'})
    (args.output/'PROTOCOL_AUDIT.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    print(summary[['phase','reliability','f1_mean','auc_mean','brier_mean']].to_string(index=False),flush=True)
    print('FINISHED',args.output.resolve(),flush=True)
if __name__=='__main__':main()
