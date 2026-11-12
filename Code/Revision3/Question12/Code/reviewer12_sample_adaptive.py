"""QTrustAgent-X Reviewer 12: DED/JMD sample-adaptive reliability benchmark.
Usage: python reviewer12_sample_adaptive.py --ded-root ... --jmd-root ... --output reviewer12_results --seeds 10
Requires: numpy pandas scipy scikit-learn. Quantum-inspired phase/pair reconstruction, NOT exact RC-QTG circuit.
"""
import argparse, itertools, json, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, GroupShuffleSplit
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, brier_score_loss

MODS=['url','email','sms','qr','image']; EPS=1e-6
LABELS=['label','status','class','target','y','binary_label','is_phishing','is_malicious','phishing_label','ground_truth','attack']
TEXT={'url':['url_text','url','website','link','domain'], 'email':['email_text','email_body','body','email_content','text','message'], 'sms':['sms_text','text','message_text','sms','message'], 'qr':['qr_text','decoded_qr','qr_content','qr_url','text'], 'image':['image_text','ocr_text','image_caption','image_description']}
POS={'phishing','phish','malicious','attack','smishing','spam','fraud','positive','true','1'}
NEG={'benign','legitimate','legit','ham','normal','safe','negative','false','0'}

def norm(s): return re.sub(r'[^a-z0-9]','',str(s).lower())
def resolve(s, project):
 p=Path(s).expanduser()
 for q in [p,Path.cwd()/p,project/p,project.parent/p]:
  if q.is_dir(): return q.resolve()
 for base in [project,project.parent,Path.cwd()]:
  if base.is_dir():
   for child in base.iterdir():
    if child.is_dir() and norm(child.name)==norm(p.name): return child.resolve()
 raise FileNotFoundError(f'Cannot locate {s}; project={project}')
def col(df,options):
 lookup={str(x).strip().lower():x for x in df.columns}
 return next((lookup[x] for x in options if x in lookup),None)
def read(path):
 for enc in ['utf-8','utf-8-sig','latin-1']:
  try: return pd.read_csv(path,encoding=enc,low_memory=False,on_bad_lines='skip')
  except UnicodeDecodeError: pass
 raise ValueError(f'Cannot decode {path}')
def labels(series):
 s=series.astype('string').str.strip().str.lower()
 missing=s.isna() | s.isin(['','nan','none','null','<na>'])
 unknown=sorted(set(s[~missing].dropna().tolist())-POS-NEG,key=str)
 if unknown: raise ValueError(f'Unknown label values: {unknown[:12]}')
 out=s.isin(POS).astype(int).to_numpy()
 return out, (~missing).to_numpy(dtype=bool)
def file_label(path):
 n=path.stem.lower()
 if 'phishing' in n or 'smishing' in n: return 1
 if 'legit' in n or 'benign' in n: return 0
 return None

def frame(path, dataset, default=None):
 raw=read(path); lc=col(raw,LABELS)
 if lc is None:
  inferred=file_label(path)
  if inferred is None:
   print(f'SKIP no label: {path.name}; columns={list(raw.columns)[:12]}'); return None
  y=np.full(len(raw),inferred,dtype=int); good=np.ones(len(raw),dtype=bool)
 else:
  try: y,good=labels(raw[lc])
  except ValueError as e:
   print(f'SKIP ambiguous label in {path.name}: {e}'); return None
 if not good.all(): print(f'  {path.name}: dropped {sum(~good)} missing-label rows')
 raw=raw.loc[good].reset_index(drop=True); y=y[good]
 if not len(raw): return None
 out=pd.DataFrame({'label':y})
 for m in MODS:
  candidates=TEXT[m] if dataset=='JMD' or default==m else [x for x in TEXT[m] if x not in ('text','message')]
  c=col(raw,candidates)
  if c is not None:
   val=raw[c].fillna('').astype(str)
   # JMD mask is authoritative; don't turn inactive cells into evidence.
   mc=col(raw,[f'mask_{m}'])
   if mc is not None:
    active=pd.to_numeric(raw[mc],errors='coerce').fillna(0).eq(1)
    val=val.where(active,'')
   out[f'text_{m}']=val
  pc=col(raw,[f'p_{m}',f'{m}_prob',f'{m}_probability',f'{m}_score'])
  if pc is not None: out[f'p_{m}']=pd.to_numeric(raw[pc],errors='coerce')
 if default and f'text_{default}' not in out:
  generic=col(raw,['text','message','content','payload','raw_text'])
  if generic is not None: out[f'text_{default}']=raw[generic].fillna('').astype(str)
 if not any(f'text_{m}' in out or f'p_{m}' in out for m in MODS):
  print(f'SKIP no evidence: {path.name}'); return None
 out['dataset']=dataset; out['source_file']=str(path)
 if dataset=='JMD':
  sc=col(raw,['split']); out['split']=raw[sc].astype(str).str.lower().str.strip() if sc else ''
  ic=col(raw,['incident_id']);out['incident_id']=raw[ic].astype(str) if ic else [f'jmd:{i}' for i in range(len(raw))]
  for m in MODS:
   sid=col(raw,[f'{m}_source_id'])
   if sid is not None: out[f'{m}_source_id']=raw[sid].fillna('').astype(str)
 else:
  out['incident_id']=[f'{path.stem}:{i}' for i in range(len(raw))]
 return out

def load_ded(root):
 frames=[]
 for m,folder in [('url','01_url'),('sms','02_sms'),('email','03_email_human_llm')]:
  for path in sorted((root/folder).rglob('*.csv')):
   f=frame(path,'DED',m)
   if f is not None:
    frames.append(f);print(f'DED {m}: {path.name} ({len(f)} rows)')
 if not frames: raise ValueError('No DED labeled CSV files recognized')
 print('NOTE: DED QR image files are not scored without a trained image/QR specialist.')
 return pd.concat(frames,ignore_index=True)
def load_jmd(root):
 candidates=[p for p in root.rglob('*.csv') if 'multimodal_31_combinations' in p.name.lower()]
 if not candidates: raise FileNotFoundError(f'JMD multimodal_31_combinations CSV not found under {root}')
 p=max(candidates,key=lambda x:x.stat().st_size)
 f=frame(p,'JMD');
 if f is None: raise ValueError('JMD labels/evidence not recognized')
 print(f'JMD: {p.name} ({len(f)} rows), split counts={f.split.value_counts().to_dict()}')
 return f

def specialist():
 return make_pipeline(TfidfVectorizer(analyzer='char',ngram_range=(3,5),max_features=12000,min_df=2,sublinear_tf=True),LogisticRegression(solver='liblinear',max_iter=1000,class_weight='balanced'))
def scores(train,test,seed):
 y=train.label.to_numpy(dtype=int); pt=np.full((len(train),5),np.nan);pe=np.full((len(test),5),np.nan)
 for j,m in enumerate(MODS):
  pc=f'p_{m}'
  if pc in train and pc in test and train[pc].notna().any() and test[pc].notna().any():
   raise ValueError(f'Existing {pc} detected: cannot verify OOF provenance; remove it or supply a verified score pipeline.')
  tc=f'text_{m}'
  if tc not in train: continue
  a=train[tc].fillna('').astype(str);b=test[tc].fillna('').astype(str) if tc in test else pd.Series(['']*len(test))
  ti=np.flatnonzero(a.str.strip().str.len().to_numpy()>0);vi=np.flatnonzero(b.str.strip().str.len().to_numpy()>0)
  if len(ti)<30 or len(np.unique(y[ti]))<2:
   print(f'  {m}: unavailable or one-class source');continue
  folds=min(3,int(np.bincount(y[ti],minlength=2).min()))
  if folds<2: continue
  cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed)
  for fit,val in cv.split(ti,y[ti]):
   model=specialist();model.fit(a.iloc[ti[fit]],y[ti[fit]])
   pt[ti[val],j]=model.predict_proba(a.iloc[ti[val]])[:,1]
  model=specialist();model.fit(a.iloc[ti],y[ti]);
  if len(vi):pe[vi,j]=model.predict_proba(b.iloc[vi])[:,1]
  print(f'  {m}: OOF train={len(ti)}, test={len(vi)}')
 return pt,pe

def rel_features(p):
 valid=np.isfinite(p);safe=np.where(valid,np.clip(p,EPS,1-EPS),.5)
 confidence=2*np.abs(safe-.5)
 entropy=-(safe*np.log(safe)+(1-safe)*np.log(1-safe))/np.log(2)
 count=valid.sum(axis=1,keepdims=True)
 other_count=count-valid.astype(int)
 other_sum=(safe*valid).sum(axis=1,keepdims=True)-safe*valid
 disagreement=np.where(other_count>0,np.abs(safe-other_sum/np.maximum(other_count,1)),0)
 return np.stack([confidence,entropy,disagreement],axis=-1)
def fit_rel(p,y):
 valid=np.isfinite(p);feats=rel_features(p);models=[];fixed=[]
 for j in range(5):
  mask=valid[:,j]
  if not mask.any():models.append(None);fixed.append(.5);continue
  correct=((p[mask,j]>=.5).astype(int)==y[mask]).astype(int)
  fixed.append((correct.sum()+1)/(len(correct)+2))
  if len(correct)<30 or len(np.unique(correct))<2:models.append(None);continue
  mdl=LogisticRegression(solver='liblinear',max_iter=1000)
  mdl.fit(feats[mask,j],correct);models.append(mdl)
 return models,np.array(fixed)
def apply_rel(p,models,fixed):
 valid=np.isfinite(p);f=np.broadcast_to(fixed,p.shape).copy();a=f.copy();feat=rel_features(p)
 for j,mdl in enumerate(models):
  mask=valid[:,j]
  if mdl is not None and mask.any():a[mask,j]=mdl.predict_proba(feat[mask,j])[:,1]
 f[~valid]=0;a[~valid]=0
 return f,a

def features(p,r):
 valid=np.isfinite(p);safe=np.where(valid,np.clip(p,EPS,1-EPS),.5)
 theta=np.pi*(safe-.5);weighted=theta*r
 parts=[valid.astype(float),r,np.sin(weighted),np.cos(weighted)]
 for j,k in itertools.combinations(range(5),2):
  parts.append(((valid[:,j]&valid[:,k])*r[:,j]*r[:,k]*np.sin(theta[:,j]-theta[:,k]))[:,None])
 return np.column_stack(parts)
def ece(y,p,bins=15):
 ids=np.minimum((p*bins).astype(int),bins-1);return float(sum((ids==k).mean()*abs(y[ids==k].mean()-p[ids==k].mean()) for k in range(bins) if (ids==k).any()))
def evaluate(y,p):
 pred=(p>=.5).astype(int)
 return dict(F1=f1_score(y,pred,zero_division=0),MCC=matthews_corrcoef(y,pred),ROC_AUC=roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan,Brier=brier_score_loss(y,p),ECE=ece(y,p))

def split(df,seed,dataset):
 if dataset=='JMD' and df.split.isin(['train','val','test']).all():
  # Preserve the source-defined partition; validation is included only in training.
  tr=np.flatnonzero(df.split.isin(['train','val']).to_numpy());te=np.flatnonzero((df.split=='test').to_numpy())
  if not len(tr) or not len(te):raise ValueError('JMD source split missing train/test')
  # Audit source reuse across split boundaries.
  for m in MODS:
   c=f'{m}_source_id'
   if c in df:
    aa=set(df.iloc[tr][c].dropna())-{''};bb=set(df.iloc[te][c].dropna())-{''}
    if aa&bb:raise ValueError(f'JMD source leakage: {m} has {len(aa&bb)} IDs across train/test')
  return tr,te
 y=df.label.to_numpy(dtype=int)
 # DED: split within each modality/source, preserving class proportions.
 s=StratifiedShuffleSplit(n_splits=1,test_size=.15,random_state=seed)
 return next(s.split(np.zeros(len(df)),y))

def run(df,dataset,seed):
 tr,te=split(df,seed,dataset);train=df.iloc[tr].reset_index(drop=True);test=df.iloc[te].reset_index(drop=True)
 print(f'\n{dataset} seed={seed}: train={len(train)} test={len(test)}')
 pt,pe=scores(train,test,seed)
 oktr=np.isfinite(pt).any(axis=1);okte=np.isfinite(pe).any(axis=1)
 if not oktr.any() or not okte.any():raise ValueError(f'No usable specialist predictions for {dataset}')
 if (~oktr).any() or (~okte).any():print(f'  excluded rows without scores: train={(~oktr).sum()}, test={(~okte).sum()}')
 yt=train.label.to_numpy(dtype=int)[oktr];yv=test.label.to_numpy(dtype=int)[okte];pt=pt[oktr];pe=pe[okte]
 if len(np.unique(yt))<2 or len(np.unique(yv))<2:raise ValueError('Usable subset has only one class')
 # Crossfit reliability for arbiter training to avoid training on its own correctness predictions.
 folds=min(3,int(np.bincount(yt,minlength=2).min()));folds=max(folds,2)
 cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed+101)
 train_fixed=np.zeros_like(pt);train_adapt=np.zeros_like(pt)
 for fit,val in cv.split(pt,yt):
  rm,rf=fit_rel(pt[fit],yt[fit]);a,b=apply_rel(pt[val],rm,rf);train_fixed[val]=a;train_adapt[val]=b
 rm,rf=fit_rel(pt,yt);test_fixed,test_adapt=apply_rel(pe,rm,rf)
 results=[];variation=[]
 for method,rtrain,rtest in [('Fixed',train_fixed,test_fixed),('Adaptive',train_adapt,test_adapt)]:
  model=LogisticRegression(solver='liblinear',max_iter=1500)
  model.fit(features(pt,rtrain),yt);prob=model.predict_proba(features(pe,rtest))[:,1]
  results.append(dict(Dataset=dataset,Seed=seed,Method=method,N_test=len(yv),**evaluate(yv,prob)))
  for j,m in enumerate(MODS):
   vals=rtest[np.isfinite(pe[:,j]),j]
   if len(vals)>1:variation.append(dict(Dataset=dataset,Seed=seed,Method=method,Modality=m,N=len(vals),Within_modality_SD=float(vals.std()),Min=float(vals.min()),Max=float(vals.max())))
 return results,variation

def summarize(df):
 rows=[]
 for ds in sorted(df.Dataset.unique()):
  for metric in ['F1','MCC','ROC_AUC','Brier','ECE']:
   pivot=df[df.Dataset==ds].pivot(index='Seed',columns='Method',values=metric).dropna()
   a=pivot.Adaptive.to_numpy();b=pivot.Fixed.to_numpy();delta=(a-b)*(-1 if metric in ['Brier','ECE'] else 1)
   n=len(delta);margin=t.ppf(.975,n-1)*delta.std(ddof=1)/np.sqrt(n) if n>1 else np.nan
   rows.append(dict(Dataset=ds,Metric=metric,Fixed=b.mean(),Adaptive=a.mean(),Improvement=delta.mean(),CI_low=delta.mean()-margin,CI_high=delta.mean()+margin,N_seeds=n))
 return pd.DataFrame(rows)
def write_tex(summary,path):
 lines=[r'\begin{table*}[!t]',r'\centering',r'\caption{\textcolor{blue}{Fixed versus observation-adaptive reliability on DED and JMD. Positive improvement indicates better performance. JMD uses its original held-out partition; DED uses repeated outer splits.}}',r'\label{tab:sample_adaptive_reliability}',r'\begingroup\color{blue}\scriptsize',r'\begin{tabular}{llrrrr}',r'\toprule',r'Dataset & Metric & Fixed & Adaptive & Improvement & 95\% CI \\',r'\midrule']
 for _,r in summary.iterrows():lines.append(f"{r.Dataset} & {r.Metric} & {r.Fixed:.4f} & {r.Adaptive:.4f} & {r.Improvement:+.4f} & [{r.CI_low:+.4f},{r.CI_high:+.4f}] \\")
 lines +=[r'\bottomrule',r'\end{tabular}',r'\endgroup',r'\end{table*}'];path.write_text('\n'.join(lines),encoding='utf-8')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--ded-root',required=True);ap.add_argument('--jmd-root',required=True);ap.add_argument('--project-root',default=r'E:\other\QTrustAgentX');ap.add_argument('--output',default='reviewer12_results');ap.add_argument('--seeds',type=int,default=10);ap.add_argument('--audit-only',action='store_true');args=ap.parse_args()
 project=Path(args.project_root);dedroot=resolve(args.ded_root,project);jmdroot=resolve(args.jmd_root,project);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 print('RESOLVED DATASET PATHS\nDED:',dedroot,'\nJMD:',jmdroot,'\nOutput:',out.resolve())
 inv=[]
 for ds,root in [('DED',dedroot),('JMD',jmdroot)]:
  for p in root.rglob('*.csv'):
   try:inv.append(dict(Dataset=ds,File=str(p),Columns='|'.join(read(p).columns)))
   except Exception as e:inv.append(dict(Dataset=ds,File=str(p),Columns=f'ERROR {e}'))
 pd.DataFrame(inv).to_csv(out/'dataset_inventory.csv',index=False)
 if args.audit_only:print('Inventory written');return
 ded=load_ded(dedroot);jmd=load_jmd(jmdroot);rows=[];variations=[]
 for ds,data in [('DED',ded),('JMD',jmd)]:
  for seed in range(42,42+args.seeds):
   rr,vv=run(data,ds,seed);rows+=rr;variations+=vv
 result=pd.DataFrame(rows);var=pd.DataFrame(variations);summary=summarize(result)
 result.to_csv(out/'per_seed_metrics.csv',index=False);var.to_csv(out/'reliability_variation.csv',index=False);summary.to_csv(out/'paired_summary.csv',index=False);write_tex(summary,out/'reviewer12_table.tex')
 print('\n',summary.to_string(index=False));print('\nResults:',out.resolve())
 print('NOTE: JMD original split reused for every seed; repeated seeds change model refits, not independent test partitions.')
 print('NOTE: phase_pair_features is a classical quantum-inspired reconstruction, not the manuscript exact Qiskit circuit.')
if __name__=='__main__':main()
