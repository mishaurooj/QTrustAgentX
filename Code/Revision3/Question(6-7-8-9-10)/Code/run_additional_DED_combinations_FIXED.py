#!/usr/bin/env python3
"""Additional DED *synthetically paired* modality benchmark and compact LaTeX updater.

DED source files are disjoint. This script first fits independent specialists on
source-level train splits, then combines held-out, same-label independent source
records into artificial joint observations. These are NOT real co-observed events.
Original JMD and DED rows are preserved in the generated LaTeX; only newly
measured synthetic DED combinations are added. No numerical results are invented.

pip install numpy pandas scipy scikit-learn torch pillow
python run_additional_DED_combinations.py --ded "E:\other\QTrustAgentX\Dataset_Reorganized" --original-tex TABLE_REVIEWER10_UPDATED_FULL.tex --output extra_ded --seeds 3 --qr-per-class 800 --gpu auto

To run all ten seeds, use --seeds 10. For a quick check: --seeds 1 --qr-per-class 100.
"""
import argparse, io, itertools, json, re, time, zipfile
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from PIL import Image
import torch

MODS=['url','email','sms','qr']
SHORT={'url':'U','email':'E','sms':'S','qr':'Q'}

class DirectoryDataset:
    """Expose an extracted dataset folder through the zip-like API used by parse_sources."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self._files = {p.relative_to(self.root).as_posix(): p
                       for p in self.root.rglob('*') if p.is_file()}
        print(f'Indexed {len(self._files)} extracted dataset files in {self.root}', flush=True)
    def namelist(self):
        return list(self._files)
    def read(self, name):
        return self._files[name].read_bytes()

@contextmanager
def open_dataset(path):
    path = Path(path)
    if path.is_dir():
        yield DirectoryDataset(path)
    elif path.is_file() and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            yield archive
    else:
        raise SystemExit(f'--ded must be an existing extracted directory or ZIP: {path}')

def load_csv(z,name):
    files=[f for f in z.namelist() if f == name or f.endswith('/'+name)]
    if len(files)!=1:raise RuntimeError(f'Expected one {name}; got {files[:3]}')
    raw=z.read(files[0]);return pd.read_csv(io.BytesIO(raw),on_bad_lines='skip',low_memory=False)

def parse_sources(z,qr_per_class,seed):
    sources={}
    email=[]
    for fn,label in [('human_legit_email_1000.csv',0),('human_phishing_email_1000.csv',1),('llm_legit_email_1000.csv',0),('llm_phishing_email_595.csv',1)]:
        df=load_csv(z,fn)
        col=next((c for c in ['text','email','body','message','content'] if c in df),None)
        if col is None:raise RuntimeError(f'Email text column absent: {fn}; columns={list(df)}')
        if 'label' in df:
            raw=pd.to_numeric(df.label,errors='coerce')
            n=int((raw.notna()&(raw!=label)).sum())
            if n:print(f'PROVENANCE WARNING: {fn} has {n} conflicting raw labels; folder label used')
        email.extend(((str(df.iloc[i].get('subject',''))+' '+str(v)).strip(),label,fn+':'+str(i)) for i,v in enumerate(df[col].fillna('')) if str(v).strip())
    sources['email']=email
    sms=load_csv(z,'sms_phishing_5971.csv')
    tc=next((c for c in ['TEXT','text','message','sms'] if c in sms),None)
    lc=next((c for c in ['LABEL','raw_label','label','category'] if c in sms),None)
    if tc is None or lc is None:raise RuntimeError('SMS text/label columns missing')
    labels=sms[lc].astype(str).str.strip().str.lower()
    # Strict smishing endpoint; ham and spam are negative.
    sources['sms']=[(str(sms.iloc[i][tc]),int(v=='smishing'),f'sms:{i}') for i,v in enumerate(labels) if v in ('ham','spam','smishing') and str(sms.iloc[i][tc]).strip()]
    url=load_csv(z,'url_phishing_11430_89features.csv')
    tc=next((c for c in ['url','URL','text'] if c in url),None)
    lc=next((c for c in ['label','status','STATUS','Label','class'] if c in url),None)
    if tc is None or lc is None:raise RuntimeError('URL text/label columns missing')
    labelmap={'phishing':1,'legitimate':0,'benign':0,'1':1,'0':0}
    sources['url']=[(str(row[tc]),labelmap[str(row[lc]).strip().lower()],f'url:{i}') for i,row in url.iterrows() if str(row[lc]).strip().lower() in labelmap and str(row[tc]).strip()]
    rng=np.random.default_rng(seed)
    qr=[]
    for folder,label in [('benign',0),('malicious',1)]:
        paths=[n for n in z.namelist() if (f'/04_qr/{folder}/' in ('/'+n)) and n.lower().endswith(('.png','.jpg','.jpeg'))]
        if not paths:raise RuntimeError(f'No QR files in {folder}')
        chosen=rng.choice(len(paths),size=min(qr_per_class,len(paths)),replace=False)
        for k in chosen:
            name=paths[int(k)]
            try:
                with Image.open(io.BytesIO(z.read(name))) as im:
                    arr=np.asarray(im.convert('L').resize((16,16)),dtype=np.float32).reshape(-1)/255.
                qr.append((arr,label,name))
            except Exception as exc:print('QR image skipped:',name,str(exc)[:100])
    sources['qr']=qr
    for m,items in sources.items():
        counts=np.bincount([x[1] for x in items],minlength=2)
        print('Source',m,'n=',len(items),'labels=',counts.tolist(),flush=True)
        if min(counts)<30:raise RuntimeError(f'Insufficient classes for {m}; check original source labels')
    return sources

def fit_sources(sources,seed):
    """Split original records before any pairing; fit source-only specialists."""
    pools={}; audits=[]
    for m,items in sources.items():
        y=np.array([x[1] for x in items],dtype=int)
        indices=np.arange(len(items))
        a,te=train_test_split(indices,test_size=.20,random_state=seed,stratify=y)
        tr,va=train_test_split(a,test_size=.20,random_state=seed+1000,stratify=y[a])
        if m=='qr':
            X=np.stack([x[0] for x in items]);model=LogisticRegression(max_iter=400,random_state=seed)
            model.fit(X[tr],y[tr]); raw=model.predict_proba(X)[:,1]
        else:
            vector=HashingVectorizer(n_features=2**15,alternate_sign=False,analyzer='char',ngram_range=(3,5))
            model=SGDClassifier(loss='log_loss',random_state=seed,alpha=1e-5,max_iter=1000)
            model.fit(vector.transform([items[i][0] for i in tr]),y[tr])
            raw=model.decision_function(vector.transform([x[0] for x in items]))
            raw=1/(1+np.exp(-np.clip(raw,-35,35)))
        # Calibration restricted to validation records.
        cal=LogisticRegression(max_iter=300,random_state=seed).fit(raw[va,None],y[va])
        probs=np.clip(cal.predict_proba(raw[:,None])[:,1],1e-4,1-1e-4)
        pools[m]={}
        for split,idx in [('train',tr),('val',va),('test',te)]:
            for label in (0,1):
                subset=idx[y[idx]==label]
                pools[m][split,label]=[(float(probs[i]),items[i][2]) for i in subset]
        audits.append({'modality':m,'train':len(tr),'val':len(va),'test':len(te),'label_definition':'smishing versus ham+spam' if m=='sms' else 'source-native'})
    return pools,audits

def paired(pools,combo,seed,n_per_class):
    rng=np.random.default_rng(seed);rows=[];provenance=[]
    for split in ('train','val','test'):
        for label in (0,1):
            selections={}
            for m in combo:
                pool=pools[m][split,label]
                if not pool:raise RuntimeError(f'Empty pool for {m}/{split}/{label}')
                # Sample without replacement where possible; no source can cross splits.
                selections[m]=rng.choice(len(pool),size=n_per_class,replace=len(pool)<n_per_class)
            for j in range(n_per_class):
                x=np.full(5,.5,dtype=np.float32)
                ids=[]
                for m in combo:
                    p,source_id=pools[m][split,label][selections[m][j]]
                    x[['email','sms','url','qr','image'].index(m)]=p
                    ids.append(source_id)
                rows.append((x,label,split))
                provenance.append({'combination':'+'.join(combo),'split':split,'label':label,'source_ids':'|'.join(ids),'synthetic_joint':True})
    return np.stack([r[0] for r in rows]),np.array([r[1] for r in rows]),np.array([r[2] for r in rows]),provenance

def features(X,device):
    """Same five-qubit circuit for both CPU and CUDA, 10 Z/ZZ observables."""
    outputs=[]
    for start in range(0,len(X),1024):
        x=torch.tensor(X[start:start+1024],device=device,dtype=torch.float32)
        n=len(x);idx=torch.arange(32,device=device)
        a=torch.pi*(x.clamp(.001,.999)-.5)
        st=torch.zeros((n,32),dtype=torch.complex64,device=device);st[:,0]=1+0j
        def gate(st,q,ang,kind):
            z=idx[((idx>>q)&1)==0];o=z|(1<<q)
            u=st[:,z];v=st[:,o]
            if kind=='ry':
                c=torch.cos(ang/2)[:,None];s=torch.sin(ang/2)[:,None]
                u1=c*u-s*v;v1=s*u+c*v
            else:
                u1=torch.exp(-.5j*ang)[:,None]*u;v1=torch.exp(.5j*ang)[:,None]*v
            out=st.clone();out[:,z]=u1;out[:,o]=v1;return out
        for q in range(5):st=gate(gate(st,q,a[:,q],'ry'),q,a[:,q],'rz')
        for c,t in [(0,1),(1,2),(2,3),(3,4),(4,0)]:
            z=idx[(((idx>>c)&1)==1)&(((idx>>t)&1)==0)];o=z|(1<<t)
            out=st.clone();out[:,z]=st[:,o];out[:,o]=st[:,z];st=out
        for q in range(5):st=gate(st,q,.5*a[:,(q+1)%5],'ry')
        p=st.abs().square()
        signs=torch.stack([1-2*((idx>>q)&1) for q in range(5)],dim=1).float()
        signs=torch.cat([signs,torch.stack([signs[:,q]*signs[:,(q+1)%5] for q in range(5)],dim=1)],dim=1)
        outputs.append((p@signs).cpu().numpy())
    if device=='cuda':torch.cuda.synchronize()
    return np.vstack(outputs)

def fit_eval(f,y,splits,seed):
    tr=splits=='train';va=splits=='val';te=splits=='test'
    scaler=StandardScaler().fit(f[tr]);v=scaler.transform(f)
    clf=LogisticRegression(max_iter=500,random_state=seed).fit(v[tr],y[tr])
    pv=clf.predict_proba(v[va])[:,1]
    th=max(np.linspace(.1,.9,33),key=lambda t:f1_score(y[va],pv>=t,zero_division=0))
    return float(f1_score(y[te],clf.predict_proba(v[te])[:,1]>=th,zero_division=0))

def parse_existing(path):
    content=Path(path).read_text(encoding='utf8')
    rows={}
    for line in content.splitlines():
        if not re.match(r'^\s*(DED|JMD):[UESQI+]+\s*&',line):continue
        cells=[c.strip().rstrip('\\').strip() for c in line.split('&')]
        for start in (0,7):
            if start+6<len(cells) and re.fullmatch(r'(DED|JMD):[UESQI+]+',cells[start]):
                name=cells[start];nums=cells[start+1:start+7]
                if len(nums)==6:rows[name]=[float(v) for v in nums]
    if not rows:raise RuntimeError('Could not read original 2-panel table')
    return rows

def render_table(rows,path):
    keys=sorted(rows,key=lambda x:(0 if x.startswith('DED:') else 1,x))
    half=(len(keys)+1)//2;left=keys[:half];right=keys[half:]
    def cell(key):
        if not key:return ['']*7
        return [key]+[f'{v:.3f}' for v in rows[key]]
    lines=[r'% Requires \usepackage{booktabs,graphicx,xcolor}',r'\begin{table*}[h]',r'\centering',
      r'\caption{\textcolor{blue}{Matched DED and JMD arbiter comparison. C=CPU exact; N=CPU no-pair; G=CUDA statevector. F1 and seconds are reported. Additional DED multimodal rows are synthetically paired independent source records, not real co-observed incidents. Times cover feature generation and arbiter fitting/inference; specialist training and preprocessing are excluded.}}',
      r'\label{tab:reviewer10_e2e}',r'\begingroup\color{blue}',r'\setlength{\tabcolsep}{2.1pt}',r'\renewcommand{\arraystretch}{0.95}',r'\scriptsize',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{@{}lrrrrrr@{\hspace{10pt}}lrrrrrr@{}}',r'\toprule',r'\multicolumn{7}{c}{\textbf{Panel A}} & \multicolumn{7}{c}{\textbf{Panel B}}\\',r'\cmidrule(lr){1-7}\cmidrule(l){8-14}',r'Combination & C F1 & C s & N F1 & N s & G F1 & G s & Combination & C F1 & C s & N F1 & N s & G F1 & G s \\',r'\midrule']
    for i,key in enumerate(left):
        vals=cell(key)+cell(right[i] if i<len(right) else '')
        lines.append(' & '.join(vals)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}}',r'\vspace{2pt}',r'{\raggedright\tiny U=URL, E=email, S=SMS, Q=QR, I=image. Original DED and JMD rows are preserved; newly measured DED combinations are synthetic joint examples assembled from independent DED records. Source-level train/validation/test splits precede pairing. CUDA is a simulator, not quantum hardware; no quantum speedup is claimed.\par}',r'\endgroup',r'\vspace{-0.5cm}',r'\end{table*}']
    path.write_text('\n'.join(lines)+'\n',encoding='utf8')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ded',default=r'E:\other\QTrustAgentX\Dataset_Reorganized',help='Extracted dataset folder or ZIP (defaults to E:\\other\\QTrustAgentX\\Dataset_Reorganized)')
    ap.add_argument('--original-tex',required=True)
    ap.add_argument('--output',default='extra_ded')
    ap.add_argument('--seeds',type=int,default=3)
    ap.add_argument('--start-seed',type=int,default=42)
    ap.add_argument('--qr-per-class',type=int,default=800)
    ap.add_argument('--samples-per-class',type=int,default=100)
    ap.add_argument('--gpu',choices=['auto','cpu','cuda'],default='auto')
    args=ap.parse_args()
    if args.gpu=='cuda' and not torch.cuda.is_available():ap.error('CUDA not available')
    gpu='cuda' if (args.gpu!='cpu' and torch.cuda.is_available()) else 'cpu'
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    rows=[];audit=[]
    # Re-read source files for each seed so QR selection and splits vary.
    for seed in range(args.start_seed,args.start_seed+args.seeds):
        with open_dataset(args.ded) as z:sources=parse_sources(z,args.qr_per_class,seed)
        pools,source_audit=fit_sources(sources,seed)
        audit.extend([dict(seed=seed,**v) for v in source_audit])
        combos=[c for n in (2,3,4) for c in itertools.combinations(MODS,n)]
        for combo in combos:
            X,y,splits,prov=paired(pools,combo,seed,args.samples_per_class)
            # Do not store individual source IDs in the main results table.
            timing={}
            t=time.perf_counter();f_cpu=features(X,'cpu');timing['CPU exact']=time.perf_counter()-t
            t=time.perf_counter();f_gpu=features(X,gpu);timing['CUDA statevector']=time.perf_counter()-t
            err=float(np.max(np.abs(f_cpu-f_gpu)))
            if err>2e-5:raise RuntimeError(f'Numerical mismatch: {err}')
            for method,f,feature_s in [('CPU exact',f_cpu,timing['CPU exact']),('CPU no-pair',f_cpu[:,:5],timing['CPU exact']),('CUDA statevector',f_gpu,timing['CUDA statevector'])]:
                t=time.perf_counter();score=fit_eval(f,y,splits,seed);fit_s=time.perf_counter()-t
                rows.append({'dataset':'DED_SYNTHETIC_JOINT','seed':seed,'combination':'+'.join(SHORT[m] for m in combo),'method':method,'f1':score,'seconds':feature_s+fit_s,'max_cpu_gpu_error':err,'synthetic_joint':True,'samples_per_class_per_split':args.samples_per_class,'device':gpu})
            print('Seed',seed,'DED','+'.join(combo),'completed',flush=True)
    pd.DataFrame(rows).to_csv(out/'DED_ADDITIONAL_COMBINATIONS_BY_SEED.csv',index=False)
    pd.DataFrame(audit).to_csv(out/'DED_SPECIALIST_SOURCE_SPLIT_AUDIT.csv',index=False)
    base=parse_existing(args.original_tex)
    frame=pd.DataFrame(rows)
    for combo,g in frame.groupby('combination'):
        vals=[]
        for method in ('CPU exact','CPU no-pair','CUDA statevector'):
            v=g[g.method==method];vals.extend([float(v.f1.mean()),float(v.seconds.mean())])
        base['DED:'+combo]=vals
    render_table(base,out/'TABLE_REVIEWER10_WITH_ADDITIONAL_DED.tex')
    (out/'METHOD_NOTE.txt').write_text('DED multimodal combinations in this extension are SYNTHETIC label-matched pairings of independently sourced records. They do NOT demonstrate real co-observed multimodal incidents. The original DED single-modality and JMD table values are preserved without recomputation. F1/time averages for new rows use the selected seeds. Timings exclude specialist training and raw preprocessing. QR uses 16x16 resized pixel logistic regression, not a decoded QR payload. The SMS endpoint is smishing versus ham+spam. Source splits precede pairing; repeated source reuse within a split is possible. CPU exact and CUDA use the same 5-qubit statevector simulator; no quantum advantage is claimed.\n')
    print('OUTPUT:',out/'TABLE_REVIEWER10_WITH_ADDITIONAL_DED.tex')

if __name__=='__main__':main()
