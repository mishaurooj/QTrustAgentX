"""Build a reproducible *synthetic-composite* five-modality phishing benchmark.
All 31 nonempty subsets of {url,email,sms,qr,image} contain 1000 rows (500/class).
IMPORTANT: 
Screen-shot images are DERIVED renderings of email text, not an independent image corpus.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split
import argparse, collections, hashlib, io, itertools, json, re, zipfile
import numpy as np
import pandas as pd

MODS=['url','email','sms','qr','image']
URL_RE=re.compile(r'(?:https?://|www\.)[^\s<>"\']+|\b(?:[a-z0-9][a-z0-9.-]*\.(?:com|net|org|info|co|io|xyz|top|ru|uk|pk|edu|gov|biz))(?:/[^\s<>"\']*)?',re.I)

def read_csv(z,partial):
    name=next(n for n in z.namelist() if n.endswith(partial))
    return pd.read_csv(io.BytesIO(z.read(name)),encoding='utf-8',encoding_errors='replace',on_bad_lines='skip',low_memory=False),name

def text_hash(v):return hashlib.sha256(str(v).lower().strip().encode('utf8','replace')).hexdigest()

def dedup(df, key):
    # Normalize exact message identity BEFORE splitting to prevent train/test duplicates.
    d=df.copy();d['__normalized__']=d[key].astype(str).str.casefold().str.strip()
    ambiguous=d.groupby('__normalized__').label.nunique()
    conflict=set(ambiguous[ambiguous>1].index)
    if conflict:print('DROPPED conflicting-label texts',len(conflict))
    d=d[~d['__normalized__'].isin(conflict)].drop_duplicates('__normalized__')
    return d.drop(columns='__normalized__').reset_index(drop=True)

def assign_splits(df,seed=42):
    # Deterministic split by unique source records, label stratified.
    a,b=train_test_split(np.arange(len(df)),test_size=.30,stratify=df.label,random_state=seed)
    c,d=train_test_split(b,test_size=.50,stratify=df.iloc[b].label,random_state=seed)
    df=df.copy();df['split']='';df.loc[a,'split']='train';df.loc[c,'split']='val';df.loc[d,'split']='test';return df

def render_screenshot(txt,path):
    # Derived from source email. No label printed. Safe: not loading remote assets/URLs.
    im=Image.new('RGB',(480,240),'#fbfbfa'); dr=ImageDraw.Draw(im)
    dr.rounded_rectangle((10,10,469,229),radius=12,outline='#aaaaaa',width=2,fill='white')
    dr.rectangle((10,10,469,46),fill='#e8eef5')
    dr.text((26,22),'Message',fill='#1c2d45')
    msg=re.sub(r'\s+',' ',str(txt))[:480]
    import textwrap
    for j,line in enumerate(textwrap.wrap(msg,width=70)[:10]):dr.text((25,58+15*j),line,fill='#191919')
    im.save(path,quality=65)

def main():
    p=argparse.ArgumentParser();p.add_argument('--zip',default=r'E:\other\QTrustAgentX\Dataset_Reorganized.zip');p.add_argument('--out',default=r'E:\other\QTrustAgentX\Validation_2026');p.add_argument('--seed',type=int,default=42);args=p.parse_args()
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True);z=zipfile.ZipFile(args.zip)
    url,_=read_csv(z,'url_phishing_11430_89features.csv')
    u=url.copy();u['label']=u.status.astype(str).str.lower().eq('phishing').astype(int);u['source_id']=['url_'+str(i) for i in range(len(u))]
    u['text']=u.url.fillna('').astype(str);u=dedup(u,'text');u=assign_splits(u,args.seed)
    # Keep URL numeric metadata in source table for provenance, though comparison uses URL text uniformly.
    sms,_=read_csv(z,'sms_phishing_5971.csv');s=pd.DataFrame({'text':sms.TEXT.fillna('').astype(str),'raw_label':sms.LABEL.astype(str)})
    s['label']=s.raw_label.str.lower().isin(['smishing','spam']).astype(int);s['source_id']=['sms_'+str(i) for i in range(len(s))]
    s=dedup(s,'text');s=assign_splits(s,args.seed)
    em=[];audit=[]
    specs=[('human_legit_email_1000.csv',0,'human'),('human_phishing_email_1000.csv',1,'human'),('llm_legit_email_1000.csv',0,'llm'),('llm_phishing_email_595.csv',1,'llm')]
    for file,y,author in specs:
        if 'llm_phishing_' in file:
            name=next(n for n in z.namelist() if n.endswith(file))
            lines=z.read(name).decode('utf8','replace').splitlines()[1:]
            # Raw file is one observation per physical line, with unquoted internal commas.
            vals=[l.rsplit(',',1)[0].strip() for l in lines if l.strip() and l.rsplit(',',1)[-1].strip()=='1']
            cnt=len(vals);rows=pd.DataFrame({'text':vals})
            audit.append({'file':file,'expected_label':y,'source_rows':cnt,'parse':'split each physical row at final comma (unquoted CSV)'})
        else:
            raw,_=read_csv(z,file); col='body' if 'body' in raw else 'text'
            rows=pd.DataFrame({'text':raw[col].fillna('').astype(str)})
            audit.append({'file':file,'expected_label':y,'source_rows':len(rows),'parse':'folder label authoritative (llm_legit internal label=1)'})
        rows['label']=y;rows['author']=author;rows['source_id']=[file+':'+str(i) for i in range(len(rows))];em.append(rows)
    e=pd.concat(em,ignore_index=True);e=e[e.text.str.len()>5].copy();e=dedup(e,'text');e=assign_splits(e,args.seed)
    # image modality is a rendered copy of an email. Same split as underlying email source.
    img=e.groupby(['split','label'],group_keys=False).sample(n=60,random_state=args.seed)[['text','label','split','source_id']].rename(columns={'source_id':'origin_email_id'}).copy()
    img['source_id']='image:'+img.origin_email_id
    img_dir=root/'images'/'rendered_email';img_dir.mkdir(parents=True,exist_ok=True)
    # Render a source pool large enough for repeated combination sampling.
    for k,row in enumerate(img.itertuples()):
        img.at[row.Index if hasattr(row,'Index') else img.index[k],'image_path']='images/rendered_email/'+str(k).zfill(6)+'.jpg'
        render_screenshot(row.text,img_dir/(str(k).zfill(6)+'.jpg'))
    # QR physical images: 1000 per class sampled randomly from all 100000 per class.
    qrrows=[]; rng=np.random.default_rng(args.seed)
    for word,y in [('benign',0),('malicious',1)]:
        names=[n for n in z.namelist() if '/04_qr/'+word+'/' in n and n.lower().endswith(('.png','.jpg','.jpeg','.webp'))]
        pick=rng.choice(len(names),size=min(1200,len(names)),replace=False)
        qr_dir=root/'images'/'qr'/word;qr_dir.mkdir(parents=True,exist_ok=True)
        for i,idx in enumerate(pick):
            nm=names[int(idx)]; dest=qr_dir/Path(nm).name;(dest.write_bytes(z.read(nm)) if not dest.exists() else None)
            qrrows.append({'source_id':'qr:'+str(Path(nm).name),'text':'','qr_path':str(dest.relative_to(root)).replace('\\','/'),'label':y,'original_zip_path':nm})
    q=pd.DataFrame(qrrows);q=assign_splits(q,args.seed)
    # Source pools must never appear in multiple train/val/test partitions.
    sources={'url':u,'email':e,'sms':s,'qr':q,'image':img}
    for mod,df in sources.items():
        df.to_csv(root/(mod+'_sources.csv'),index=False)
        assert not df.source_id.duplicated().any()
        assert len(df.groupby('source_id').split.nunique().loc[lambda x:x>1])==0
    cache={(mod,sp,y):df[(df.split==sp)&(df.label==y)].reset_index(drop=True) for mod,df in sources.items() for sp in ['train','val','test'] for y in [0,1]}
    assert all(len(g)>0 for g in cache.values()),[(k,len(g)) for k,g in cache.items() if len(g)==0]
    # Per combination: train 350+350; validation 75+75; test 75+75 = 1000.
    # Class-matched independent reuse of sources is intentional synthetic compositing.
    records=[]
    for bits in itertools.product([0,1],repeat=5):
        if not any(bits): continue
        combo='+'.join(m for m,b in zip(MODS,bits) if b)
        for sp,n in [('train',350),('val',75),('test',75)]:
            for y in [0,1]:
                for j in range(n):
                    rec={'incident_id':f'{combo}|{sp}|{y}|{j}','combination':combo,'split':sp,'label':y,'construction':'synthetic_class_matched_composite','real_coobserved':False,'mask':'m'+''.join(map(str,bits))}
                    for m,b in zip(MODS,bits):
                        rec['mask_'+m]=b;rec[m+'_source_id']='';rec[m+'_text']='';rec[m+'_path']=''
                        if not b: continue
                        # If both email + image are present, image is a rendering of the SAME selected email.
                        if m=='email' and 'image' in combo:
                            ip=cache['image',sp,y];image_ref=ip.iloc[int(rng.integers(len(ip)))]
                            pp=cache['email',sp,y]; candidates=pp.index[pp.source_id==image_ref.origin_email_id].tolist()
                            row=pp.iloc[candidates[0]]
                        elif m=='image' and 'email' in combo:
                            pp=cache['image',sp,y]
                            candidates=pp.index[pp.origin_email_id==rec['email_source_id']].tolist()
                            row=pp.iloc[candidates[0]] if candidates else pp.iloc[int(rng.integers(len(pp)))]
                        else:
                            pp=cache[m,sp,y];row=pp.iloc[int(rng.integers(len(pp)))]
                        rec[m+'_source_id']=row.source_id
                        if m in ['url','email','sms']:rec[m+'_text']=row.text
                        if m=='qr':rec[m+'_path']=row.qr_path
                        if m=='image':rec[m+'_path']=row.image_path;rec['image_origin_email_id']=row.origin_email_id
                    records.append(rec)
    dataset=pd.DataFrame.from_records(records)
    dataset.to_csv(root/'multimodal_31_combinations_1000_each.csv',index=False)
    # Linked records from within same email or SMS message: independent of synthetic composition.
    linked=[]
    for m in ['email','sms']:
        for row in sources[m].itertuples(index=False):
            links=URL_RE.findall(row.text)
            for link in list(dict.fromkeys(links))[:3]:
                linked.append({'source_modality':m,'source_id':row.source_id,'split':row.split,'label':row.label,'message_text':row.text,'observed_url':link,'linkage':'extracted_from_same_message'})
    linked_df=pd.DataFrame(linked);linked_df.to_csv(root/'naturally_linked_message_url_pairs.csv',index=False)
    stat={'sources':{m:{'total':len(df),'by_split_and_label':{f'{sp}/{y}':len(cache[m,sp,y]) for sp in ['train','val','test'] for y in [0,1]}} for m,df in sources.items()},
          'combinations':dataset.groupby('combination').agg(n=('label','size'),benign=('label',lambda x:int((x==0).sum())),phishing=('label',lambda x:int((x==1).sum()))).to_dict('index'),
          'pair_gate_activations':int((dataset[[f'mask_{m}' for m in MODS]].sum(axis=1).ge(2)).sum()),
          'naturally_linked':linked_df.groupby(['source_modality','label']).size().to_dict().__str__(),
          'original_source_note':'All multimodal composite rows are synthetic; naturally linked URL pairs are a separate evaluation set.',
          'image_note':'Image modality contains derived email screenshot renderings, not an independent image corpus.',
          'email_label_audit':audit,'seed':args.seed}
    (root/'dataset_audit.json').write_text(json.dumps(stat,indent=2),encoding='utf8')
    print('CREATED',root,'ROWS',len(dataset),'COMBINATIONS',dataset.combination.nunique(),'PAIRED_ROWS',stat['pair_gate_activations']);print('NATURALLY_LINKED',stat['naturally_linked']);print('SOURCES',[(m,len(df)) for m,df in sources.items()])
if __name__=='__main__':main()
