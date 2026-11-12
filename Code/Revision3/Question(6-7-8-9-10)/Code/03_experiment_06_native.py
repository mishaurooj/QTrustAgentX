#!/usr/bin/env python3
"""Reviewer 6: measured train/test near-duplicate audit and provenance-backed subsets.

Uses frozen held-out predictions from experiment_05; does not retrain after
filtering. LLM authorship is derived from source-directory provenance, and SMS
raw labels are derived from the actual source CSV. 'smishing' is never silently
relabeled as 'phishing-only'. Missing annotations produce N/A, not invented data.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score


def norm(s):return ' '.join(str(s).casefold().split())


def hash_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def discover_zip(explicit):
    if explicit:return explicit
    choices=[Path('Dataset_Reorganized.zip'),Path('../Dataset_Reorganized.zip'),
             Path(__file__).resolve().parent/'Dataset_Reorganized.zip',
             Path(__file__).resolve().parent.parent/'Dataset_Reorganized.zip']
    return next((p for p in choices if p.exists()),None)


def read_source_zip(path):
    """Map normalized source text to verified metadata. Conflicts become unknown."""
    if path is None:return {}, {'source_zip':'unavailable','email_rows':0,'sms_rows':0}
    email={};sms={};ec=0;sc=0
    def insert(mapping,key,val):
        if not key or key=='nan':return
        if key in mapping and mapping[key]!=val:mapping[key]=None
        else:mapping.setdefault(key,val)
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            low=name.lower().replace('\\','/')
            if not low.endswith('.csv'):continue
            if '/03_email_human_llm/' in low:
                author='llm' if '/llm_' in low else 'human' if '/human_' in low else None
                if author is None:continue
                d=pd.read_csv(z.open(name),on_bad_lines='skip',encoding_errors='replace',low_memory=False)
                col='body' if 'body' in d.columns else 'text' if 'text' in d.columns else None
                if col is None:continue
                for text in d[col].dropna().astype(str):
                    insert(email,norm(text),(author,name));ec+=1
            elif low.endswith('/02_sms/sms_phishing_5971.csv'):
                d=pd.read_csv(z.open(name),encoding_errors='replace',low_memory=False)
                if not {'TEXT','LABEL'}.issubset(d.columns):continue
                for text,raw in zip(d.TEXT,d.LABEL):
                    if pd.isna(text) or pd.isna(raw):continue
                    raw=str(raw).strip().lower()
                    insert(sms,norm(text),(raw,name));sc+=1
    return {'email':email,'sms':sms},{'source_zip':str(path.resolve()),'sha256':hash_file(path),
        'email_rows':ec,'sms_rows':sc}


def load_provenance_csv(path):
    if not path:return None
    d=pd.read_csv(path,dtype={'source_id':str})
    if 'source_id' not in d.columns or d.source_id.duplicated().any():
        raise ValueError('Provenance CSV requires unique source_id')
    return d.set_index('source_id')


def annotate(d,source,extra):
    d=d.copy(); authors=[];raws=[];files=[]
    for row in d.itertuples(index=False):
        text=norm(row.text)
        if row.modality=='email':
            found=source.get('email',{}).get(text)
            authors.append(found[0] if found else None)
            raws.append(None);files.append(found[1] if found else None)
        elif row.modality=='sms':
            found=source.get('sms',{}).get(text)
            authors.append(None);raws.append(found[0] if found else None)
            files.append(found[1] if found else None)
        else:
            authors.append(None);raws.append(None);files.append(None)
    d['authoring_domain']=authors;d['sms_raw_label']=raws;d['provenance_source_file']=files
    if extra is not None:
        for col in ('authoring_domain','sms_raw_label'):
            if col in extra.columns:
                vals=d.source_id.map(extra[col])
                # Source-derived and external annotations that disagree are not silently reconciled.
                conflict=d[col].notna()&vals.notna()&(d[col].astype(str).str.lower()!=vals.astype(str).str.lower())
                if conflict.any():raise ValueError(f'Conflicting provenance for {col}: {int(conflict.sum())} rows')
                d[col]=d[col].where(d[col].notna(),vals)
    d['authoring_domain']=d.authoring_domain.astype('string').str.lower()
    d['sms_raw_label']=d.sms_raw_label.astype('string').str.lower()
    d.loc[~d.authoring_domain.isin(['human','llm']),'authoring_domain']=pd.NA
    return d


def max_similarity(train_text,test_text,threshold,chunk=256,max_features=150000):
    """Train-fitted TF-IDF char 3-5 grams; exact sparse cosine maximum."""
    v=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=1,
        max_features=max_features,dtype=np.float32,norm='l2')
    a=v.fit_transform(train_text)
    scores=np.zeros(len(test_text),dtype=float)
    for i in range(0,len(test_text),chunk):
        b=v.transform(test_text[i:i+chunk])
        prod=b@a.T
        scores[i:i+b.shape[0]]=np.asarray(prod.max(axis=1).toarray()).ravel()
    return scores,np.asarray(scores>=threshold)


def metric_f1(y,p):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=int)
    if len(y)==0 or len(np.unique(y))<2:return float('nan')
    return float(f1_score(y,p,zero_division=0))


def ci(a):
    v=np.asarray(a,dtype=float);v=v[np.isfinite(v)]
    if not len(v):return np.nan,np.nan,np.nan
    mean=float(v.mean())
    if len(v)<2:return mean,np.nan,np.nan
    h=float(student_t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v)))
    return mean,mean-h,mean+h


def measure(test,kind):
    if kind=='email_all':sel=test.modality.eq('email')
    elif kind=='email_llm':sel=test.modality.eq('email')&test.authoring_domain.eq('llm').fillna(False)
    elif kind=='email_human':sel=test.modality.eq('email')&test.authoring_domain.eq('human').fillna(False)
    elif kind=='sms_all':sel=test.modality.eq('sms')
    elif kind=='sms_smishing_vs_ham':sel=test.modality.eq('sms')&test.sms_raw_label.isin(['smishing','smish','ham']).fillna(False)
    elif kind=='sms_phishing_only_vs_ham':sel=test.modality.eq('sms')&test.sms_raw_label.isin(['phishing','ham']).fillna(False)
    else:raise ValueError(kind)
    s=test.loc[sel].copy()
    if kind=='sms_phishing_only_vs_ham' and not s.sms_raw_label.eq('phishing').any():s=s.iloc[:0]
    if kind=='sms_smishing_vs_ham' and not s.sms_raw_label.isin(['smishing','smish']).any():s=s.iloc[:0]
    kept=s.loc[~s.near_duplicate]
    return dict(group=kind,n_test=len(s),n_flagged=int(s.near_duplicate.sum()),
        flagged_share=float(s.near_duplicate.mean()) if len(s) else np.nan,
        n_retained=len(kept),positive_n=int(s.y_true.sum()),negative_n=int((s.y_true==0).sum()),
        f1_before=metric_f1(s.y_true,s.y_pred),f1_after=metric_f1(kept.y_true,kept.y_pred),
        recall_after=float(((kept.y_pred==1)&(kept.y_true==1)).sum()/max(1,(kept.y_true==1).sum()))
            if (kept.y_true==1).any() else np.nan)


def tex_table(summary,path):
    lines=[r'\begin{table}[t]',r'\centering\scriptsize',r'\begingroup\color{blue}',
           r'\caption{Measured train/test near-duplicate overlap using frozen held-out predictions; 0.90 character 3--5-gram TF--IDF cosine threshold. N/A denotes unavailable or single-class subsets.}',
           r'\label{tab:overlap_proposed}',r'\begin{tabular}{lccc}',r'\toprule',
           r'Subgroup & Flagged share (95\% CI) & F1 before & F1 after \\',r'\midrule']
    labels={'email_all':'Email (all)','email_llm':'Email (verified LLM)',
            'email_human':'Email (verified human)','sms_all':'SMS (all)',
            'sms_smishing_vs_ham':'SMS smishing vs ham','sms_phishing_only_vs_ham':'SMS phishing-only vs ham'}
    def fmt(v):return 'N/A' if not np.isfinite(v) else f'{v:.4f}'
    for _,r in summary.iterrows():
        rate=(f"{100*r['flagged_share_mean']:.2f}\\% [{100*r['flagged_share_ci_low']:.2f}, {100*r['flagged_share_ci_high']:.2f}]"
              if np.isfinite(r['flagged_share_ci_low']) else ('N/A' if not np.isfinite(r['flagged_share_mean']) else f"{100*r['flagged_share_mean']:.2f}\\%"))
        lines.append(f"{labels.get(r['group'],r['group'])} & {rate} & {fmt(r['f1_before_mean'])} & {fmt(r['f1_after_mean'])} " + r"\\")
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\endgroup',r'\end{table}',
        r'% Smishing-vs-ham is NOT synonymous with phishing-only SMS.',
        r'% The filtering analysis evaluates retained records; it does not retrain a detector.'])
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,default=Path('fresh_source_manifest.csv'))
    ap.add_argument('--experiment5-dir',type=Path,default=Path('Reviewer5_Phase_Reliability'))
    ap.add_argument('--output',type=Path,default=Path('Reviewer6_Near_Duplicate_Audit'))
    ap.add_argument('--source-zip',type=Path,default=None,help='Optional original source ZIP; not required with NATIVE_PROVENANCE.csv')
    ap.add_argument('--provenance',type=Path,default=None,help='Optional verified source_id,authoring_domain,sms_raw_label CSV')
    ap.add_argument('--condition',default='beta1.00_estimated')
    ap.add_argument('--threshold',type=float,default=0.90)
    ap.add_argument('--chunk',type=int,default=256)
    args=ap.parse_args()
    if not 0<args.threshold<=1:ap.error('threshold must be in (0,1]')
    args.output.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(args.manifest,dtype={'source_id':str},low_memory=False)
    for col in ('source_id','modality','text','label'):
        if col not in d:ap.error(f'Manifest missing {col}')
    d=d.copy()
    d['text']=d['text'].fillna('').astype(str)
    if d.source_id.duplicated().any():ap.error('Non-unique source_id in manifest')
    splits=pd.read_csv(args.experiment5_dir/'TABLE5_SPLIT_ASSIGNMENTS.csv',dtype={'source_id':str})
    pred=pd.read_csv(args.experiment5_dir/'TABLE5_TEST_PREDICTIONS.csv',dtype={'source_id':str})
    pred=pred.loc[pred.condition.eq(args.condition)].copy()
    if pred.empty:ap.error(f'Condition {args.condition} absent from frozen predictions')
    if pred.duplicated(['seed','source_id']).any():ap.error('Duplicate prediction keys')
    if splits.duplicated(['seed','source_id']).any():ap.error('Duplicate split keys')
    if args.provenance is None:
        candidate=args.manifest.parent/'NATIVE_PROVENANCE.csv'
        if candidate.exists():args.provenance=candidate
    if args.provenance is None:ap.error('Missing source provenance: supply --provenance NATIVE_PROVENANCE.csv')
    provenance,source_info=read_source_zip(args.source_zip)
    d=annotate(d,provenance,load_provenance_csv(args.provenance))
    d[['source_id','modality','label','authoring_domain','sms_raw_label','provenance_source_file']].to_csv(
        args.output/'TABLE9_PROVENANCE_MATCHES.csv',index=False)
    records=[];audits=[]
    kinds=['email_all','email_llm','email_human','sms_all','sms_smishing_vs_ham','sms_phishing_only_vs_ham']
    for seed in sorted(pred.seed.unique()):
        split=splits.loc[splits.seed.eq(seed),['source_id','split']]
        joined=split.merge(d,on='source_id',how='left',validate='one_to_one')
        if joined.label.isna().any():raise ValueError('Split IDs missing from manifest')
        train=joined[joined.split.eq('train')]
        test=joined[joined.split.eq('test')].merge(pred.loc[pred.seed.eq(seed),
            ['source_id','y_true','y_pred','y_score']],on='source_id',how='left',validate='one_to_one')
        if test.y_score.isna().any():raise ValueError('Missing held-out predictions')
        if not np.array_equal(test.label.to_numpy(dtype=int),test.y_true.to_numpy(dtype=int)):
            raise ValueError('Prediction labels differ from manifest labels')
        test['seed']=int(seed)
        test['max_train_cosine']=np.nan
        test['near_duplicate']=False
        for modality in ('email','sms'):
            a=train.loc[train.modality.eq(modality),'text'].fillna('').astype(str).tolist()
            mask=test.modality.eq(modality)
            b=test.loc[mask,'text'].fillna('').astype(str).tolist()
            if not a or not b:continue
            scores,flags=max_similarity(a,b,args.threshold,args.chunk)
            test.loc[mask,'max_train_cosine']=scores
            test.loc[mask,'near_duplicate']=flags
        for kind in kinds:
            rec=measure(test,kind);rec['seed']=int(seed);records.append(rec)
        audits.append(test.loc[test.modality.isin(['email','sms']),
            ['seed','source_id','modality','y_true','y_pred','y_score','authoring_domain',
             'sms_raw_label','max_train_cosine','near_duplicate']].copy())
        print('Completed similarity audit seed',seed,flush=True)
    audit_df=pd.concat(audits,ignore_index=True)
    audit_df.to_csv(args.output/'TABLE9_SIMILARITY_AUDIT.csv',index=False)
    raw=pd.DataFrame(records)
    raw.to_csv(args.output/'TABLE9_ALL_SEED_MEASUREMENTS.csv',index=False)
    rows=[]
    for kind in kinds:
        g=raw[raw.group.eq(kind)]
        row={'group':kind,'repeats':len(g),'total_test_evaluations':int(g.n_test.sum()),
             'total_flagged_evaluations':int(g.n_flagged.sum()),
             'mean_test_size':float(g.n_test.mean()),'mean_positive_n':float(g.positive_n.mean()),
             'mean_negative_n':float(g.negative_n.mean())}
        for metric in ('flagged_share','f1_before','f1_after','recall_after'):
            mean,lo,hi=ci(g[metric]);row.update({metric+'_mean':mean,metric+'_ci_low':lo,metric+'_ci_high':hi})
        rows.append(row)
    summary=pd.DataFrame(rows)
    summary.to_csv(args.output/'TABLE9_MEASURED_SUMMARY.csv',index=False)
    tex_table(summary,args.output/'TABLE9_REVISED.tex')
    report={'source_provenance':source_info,'manifest':str(args.manifest.resolve()),
        'manifest_sha256':hash_file(args.manifest),'threshold':args.threshold,
        'condition':args.condition,'seeds':[int(x) for x in sorted(pred.seed.unique())],
        'verified_llm_email_n':int((d.modality.eq('email')&d.authoring_domain.eq('llm').fillna(False)).sum()),
        'verified_human_email_n':int((d.modality.eq('email')&d.authoring_domain.eq('human').fillna(False)).sum()),
        'verified_sms_smishing_n':int((d.modality.eq('sms')&d.sms_raw_label.isin(['smishing','smish']).fillna(False)).sum()),
        'verified_sms_phishing_n':int((d.modality.eq('sms')&d.sms_raw_label.eq('phishing').fillna(False)).sum()),
        'similarity':'train-fitted char 3-5 TF-IDF cosine; maximum same-modality train similarity',
        'prediction_protocol':'frozen held-out experiment_05 predictions; no refitting after filtering',
        'statistical_unit':'outer resplit (repeated records across seeds are not independent)',
        'provenance_file':str(args.provenance.resolve()),'provenance_sha256':hash_file(args.provenance),
        'warning':'Unannotated subgroups and single-class F1 are N/A. Smishing is distinct from phishing-only. Never retain illustrative manuscript values.'}
    (args.output/'TABLE9_PROVENANCE_AUDIT.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\n',summary[['group','total_test_evaluations','flagged_share_mean','f1_after_mean']].to_string(index=False))
    print('Saved',args.output.resolve())

if __name__=='__main__':main()
