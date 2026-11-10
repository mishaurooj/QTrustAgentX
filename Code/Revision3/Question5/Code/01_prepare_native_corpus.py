#!/usr/bin/env python3
"""Build a provenance-audited, source-native four-modality corpus from Dataset_Reorganized.zip.
No proxy QR text, fabricated rows, or synthetic incident joining. Produces native URL
numeric features and native QR 68-dimensional image features for the reviewer-5 run.
"""
import argparse, hashlib, io, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

FILES = {'url':'url_phishing_11430_89features.csv', 'sms':'sms_phishing_5971.csv'}
EMAIL = [('human_legit_email_1000.csv',0,'human'),('human_phishing_email_1000.csv',1,'human'),
         ('llm_legit_email_1000.csv',0,'llm'),('llm_phishing_email_595.csv',1,'llm')]
def h(b): return hashlib.sha256(b).hexdigest()
def norm(x): return str(x).casefold().strip()
def read(z, suffix):
    matches=[n for n in z.namelist() if n.endswith('/'+suffix)]
    if len(matches)!=1: raise ValueError(f'Expected one source {suffix}, found {len(matches)}')
    return pd.read_csv(io.BytesIO(z.read(matches[0])),encoding='utf-8',encoding_errors='replace',on_bad_lines='skip',low_memory=False)
def qr_features(raw):
    with Image.open(io.BytesIO(raw)) as im:
        a=np.asarray(im.convert('L').resize((128,128)),dtype=np.float32)/255.
    blocks=a.reshape(8,16,8,16).mean(axis=(1,3)).ravel()
    return np.r_[a.mean(),a.std(),np.abs(np.diff(a,axis=0)).mean(),
                 np.abs(np.diff(a,axis=1)).mean(),blocks].astype(np.float32)
class FolderSource:
    """Minimal ZipFile-like interface for the actual Dataset_Reorganized directory."""
    def __init__(self, root):
        self.root=Path(root).resolve()
        if not self.root.is_dir():raise NotADirectoryError(self.root)
        self.names=[self.root.name+'/'+f.relative_to(self.root).as_posix() for f in self.root.rglob('*') if f.is_file()]
        if not self.names:raise ValueError(f'No source files in {self.root}')
    def namelist(self):return self.names
    def read(self,name):
        f=(self.root/Path(name).relative_to(self.root.name)).resolve()
        if not f.is_relative_to(self.root):raise ValueError('Path escapes dataset root')
        return f.read_bytes()

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-zip',type=Path,default=None)
    p.add_argument('--source-root',type=Path,default=None,help='E:/other/QTrustAgentX/Dataset_Reorganized')
    p.add_argument('--out',type=Path,default=Path('Reviewer5_Original_Native'))
    p.add_argument('--qr-per-class',type=int,default=10000)
    p.add_argument('--seed',type=int,default=42);args=p.parse_args()
    if bool(args.source_zip)==bool(args.source_root):p.error('Specify exactly one of --source-root or --source-zip')
    args.out.mkdir(parents=True,exist_ok=True)
    z=zipfile.ZipFile(args.source_zip) if args.source_zip else FolderSource(args.source_root)
    rng=np.random.default_rng(args.seed)
    rows=[];url_features=[];qr_features_all=[];audit={}
    u=read(z,FILES['url']); y=u.status.astype(str).str.casefold().eq('phishing').astype(int)
    numeric=[c for c in u.columns if c not in ('url','status')]
    x=u[numeric].apply(pd.to_numeric,errors='coerce').fillna(0).to_numpy(dtype=np.float32)
    for j,(text,label) in enumerate(zip(u.url.fillna('').astype(str),y)):
        rows.append(dict(source_id=f'url:{j}',modality='url',text=text,label=int(label),
                         group_id='url:'+h(norm(text).encode()),authoring_domain='',raw_label='',native_index=j))
    url_features=x; audit['url']={'records':len(u),'numeric_features':len(numeric)}
    print('URL',len(u),flush=True)
    s=read(z,FILES['sms']); ss=[]; seen=set();conflict=0
    for j,r in s.iterrows():
        text=str(r['TEXT']) if pd.notna(r['TEXT']) else ''
        key=norm(text)
        if not key or key in seen:continue
        seen.add(key);raw=norm(r['LABEL']); label=int(raw in ('spam','smishing','smish','phishing'))
        ss.append(dict(source_id=f'sms:{j}',modality='sms',text=text,label=label,
                       group_id='sms:'+h(key.encode()),authoring_domain='',raw_label=raw,native_index=-1))
    rows.extend(ss);audit['sms']={'records':len(ss),'raw_records':len(s)}
    print('SMS',len(ss),flush=True)
    seen_by_class=set(); email=[]; empty=0; conflicts=0
    for file,label,author in EMAIL:
        d=read(z,file); col='body' if 'body' in d else 'text'
        for j,r in d.iterrows():
            body=str(r[col]) if pd.notna(r[col]) else ''
            subject=str(r['subject']).strip() if 'subject' in d and pd.notna(r['subject']) else ''
            text=(' '.join([subject,body])).strip()
            key=norm(text)
            if not key: empty+=1;continue
            dedup_key=(label,key)
            if dedup_key in seen_by_class:continue
            seen_by_class.add(dedup_key)
            email.append(dict(source_id=f'email:{file}:{j}',modality='email',text=text,label=label,
                              group_id='email:'+h(key.encode()),authoring_domain=author,
                              raw_label='',native_index=-1))
    rows.extend(email)
    cross=(pd.DataFrame(email).groupby('group_id').label.nunique()>1).sum()
    audit['email']={'records':len(email),'empty_source_rows_dropped':empty,
                    'cross_label_duplicate_groups':int(cross),'source_files':[x[0] for x in EMAIL]}
    print('Email',len(email),'cross-label conflicts',cross,flush=True)
    # Deterministic QR source selection. Identity is based on file bytes, never filename as a feature.
    qrseen=set();qrows=[]
    for label,kind in [(0,'benign'),(1,'malicious')]:
        names=sorted(n for n in z.namelist() if f'/04_qr/{kind}/' in n and n.lower().endswith(('.png','.jpg','.jpeg','.webp')))
        if len(names)<args.qr_per_class:raise ValueError(f'{kind}: insufficient QR files')
        order=rng.permutation(len(names))
        selected=0
        for i in order:
            name=names[int(i)];raw=z.read(name);digest=h(raw)
            if digest in qrseen:continue
            try:feat=qr_features(raw)
            except Exception:continue
            qrseen.add(digest);idx=len(qr_features_all);qr_features_all.append(feat)
            qrows.append(dict(source_id='qr:'+digest,modality='qr',text='',label=label,
                              group_id='qr:'+digest,authoring_domain='',raw_label=kind,
                              native_index=idx,zip_member=name))
            selected+=1
            if selected==args.qr_per_class:break
        if selected!=args.qr_per_class:raise ValueError(f'Not enough distinct decodable {kind} QR images')
        print('QR',kind,selected,flush=True)
    rows.extend(qrows);audit['qr']={'records':len(qrows),'features':68,'per_class':args.qr_per_class}
    d=pd.DataFrame(rows)
    # Two labels may share identical text: group them together to prevent leakage.
    if d.source_id.duplicated().any():raise ValueError('Nonunique source IDs')
    d.to_csv(args.out/'NATIVE_SOURCE_MANIFEST.csv',index=False)
    d.assign(sms_raw_label=d.raw_label)[['source_id','authoring_domain','sms_raw_label']].to_csv(
        args.out/'NATIVE_PROVENANCE.csv',index=False)
    np.savez_compressed(args.out/'NATIVE_FEATURES.npz',url=np.asarray(url_features,dtype=np.float32),
                        qr=np.asarray(qr_features_all,dtype=np.float32))
    audit.update({'total_records':len(d),'by_modality':d.modality.value_counts().to_dict(),
                  'by_label':{str(k):int(v) for k,v in d.label.value_counts().items()},
                  'manuscript_claimed_total':39654,'matches_manuscript_total':len(d)==39654,
                  'manuscript_email_claimed':2393,'matches_manuscript_email':len(email)==2393,
                  'source_zip_sha256':h(args.source_zip.read_bytes()) if args.source_zip else None,
                  'source_root':str(args.source_root.resolve()) if args.source_root else None,
                  'note':'Rows follow parseable original ZIP evidence. No padding or invented messages. Source-specific preparation may differ from unpublished manuscript preprocessing.'})
    (args.out/'SOURCE_AUDIT.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('READY',len(d),'rows; outputs in',args.out.resolve(),flush=True)
if __name__=='__main__':main()
