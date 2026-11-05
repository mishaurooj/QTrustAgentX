"""QTrustAgentX multimodal benchmark and real circuit statevector execution.
Runs on CPU. Implements Ry/Rz/Rx and masked CNOT-Rz-CNOT (= RZZ), exact statevector.
"""
from pathlib import Path
import argparse, itertools, json, time, hashlib, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,brier_score_loss,matthews_corrcoef,confusion_matrix,roc_curve
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier

MODS=['url','email','sms','qr','image']

def report(y,p):
    y=np.asarray(y,int);p=np.asarray(p,float);yh=(p>=.5).astype(int)
    return {'n':len(y),'accuracy':float(accuracy_score(y,yh)),'precision':float(precision_score(y,yh,zero_division=0)),'recall':float(recall_score(y,yh,zero_division=0)),
            'f1':float(f1_score(y,yh,zero_division=0)),'mcc':float(matthews_corrcoef(y,yh)),'roc_auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else float('nan'),
            'brier':float(brier_score_loss(y,p)),'tn':int(confusion_matrix(y,yh,labels=[0,1])[0,0]),'fp':int(confusion_matrix(y,yh,labels=[0,1])[0,1]),
            'fn':int(confusion_matrix(y,yh,labels=[0,1])[1,0]),'tp':int(confusion_matrix(y,yh,labels=[0,1])[1,1])}

def img_features(p,root):
    # Pixel classifier consumes real rendered image / QR pixels, not raw label or underlying text.
    with Image.open(root/p) as im:
        im=im.convert('L').resize((24,24),resample=Image.Resampling.BILINEAR)
        a=np.asarray(im,dtype=np.float32)/255.
    return np.r_[a.flatten(), a.mean(axis=0)[::2],a.mean(axis=1)[::2],a.std(),a.mean()].astype('float32')

def oof_predict(df,mod,root,seed=42):
    n=len(df);ids=df.source_id.astype(str).tolist();split=df.split.to_numpy();y=df.label.astype(int).to_numpy()
    is_text=mod in ['url','email','sms']
    if mod=='url':
        # Original URL source includes its 87 numeric URL/webpage features. No labels as input.
        from sklearn.ensemble import HistGradientBoostingClassifier
        cols=[c for c in df.select_dtypes(include=[np.number]).columns if c!='label']
        X=df[cols].fillna(0).to_numpy(dtype=np.float32)
        def build():return HistGradientBoostingClassifier(max_iter=55, max_leaf_nodes=15, learning_rate=.13,random_state=seed)
    elif is_text:
        X=df.text.fillna('').astype(str).to_numpy()
        def build():
            return make_pipeline(TfidfVectorizer(analyzer='word',ngram_range=(1,1),min_df=2,max_features=9000,sublinear_tf=True), LogisticRegression(C=2.,max_iter=200,solver='liblinear'))
    else:
        field='qr_path' if mod=='qr' else 'image_path'
        X=np.stack([img_features(str(p),root) for p in df[field].astype(str).values],axis=0)
        def build():
            return make_pipeline(StandardScaler(),LogisticRegression(C=0.06,max_iter=120,solver='liblinear'))
    tr=np.where(split=='train')[0];ot=np.where(split!='train')[0]
    pred=np.full(n,np.nan,float)
    # 3-fold OOF specialist probabilities prevent fitting the meta arbiter on in-sample specialist scores.
    folds=StratifiedKFold(n_splits=3,shuffle=True,random_state=seed)
    for ii,jj in folds.split(tr,y[tr]):
        model=build();model.fit(X[tr[ii]],y[tr[ii]]);pred[tr[jj]]=model.predict_proba(X[tr[jj]])[:,1]
    m=build();m.fit(X[tr],y[tr]);pred[ot]=m.predict_proba(X[ot])[:,1]
    assert np.isfinite(pred).all()
    # Reliability learned only from validation data; AUC is a demonstrative estimate, not full paper methodology.
    v=split=='val';rho=float(np.clip(roc_auc_score(y[v],pred[v]),.05,1))
    ret=df[['source_id','split','label']].copy();ret['score']=np.clip(pred,1e-5,1-1e-5)
    return ret,rho,report(y[split=='test'],pred[split=='test'])

def statevector_features(scores, masks, reliability, with_pairs=True, return_amplitudes=False):
    """Exact CPU statevector. Data-dependent Ry,Rz,Rx + explicit Rzz equivalent of CX-Rz-CX.
    For original manuscript use four modalities; five modalities is an extension.
    Angles: theta=2asin(sqrt(score)), phi=pi*0.35*sqrt(rhoa*rhob)*(2*(1-|sa-sb|)-1).
    Local sequence: Ry,Rz, Rx, paired ZZ, Ry(1.1 theta),Rz. Returns Z,X,ZZ,XX observables.
    """
    scores=np.asarray(scores,np.float64);masks=np.asarray(masks,np.float64)
    nrow,n=scores.shape;rho=np.asarray(reliability,dtype=np.float64)[:n]
    amps=np.zeros((nrow,1<<n),dtype=np.complex128);amps[:,0]=1.0
    num=1<<n; ix=np.arange(num)
    pairs=[]
    for q in range(n):
        bit=1<<(n-1-q);i0=ix[(ix&bit)==0];i1=i0|bit;pairs.append((i0,i1,bit))
    theta=2*np.arcsin(np.sqrt(np.clip(scores,0,1)))*masks
    def gate(q,typ,ang):
        i0,i1,_=pairs[q];a=amps[:,i0].copy();b=amps[:,i1].copy();ang=np.asarray(ang)
        if typ=='RY':
            c=np.cos(ang/2)[:,None];s=np.sin(ang/2)[:,None];amps[:,i0]=c*a-s*b;amps[:,i1]=s*a+c*b
        elif typ=='RZ':
            amps[:,i0]=a*np.exp(-.5j*ang[:,None]);amps[:,i1]=b*np.exp(.5j*ang[:,None])
        elif typ=='RX':
            c=np.cos(ang/2)[:,None];s=-1j*np.sin(ang/2)[:,None];amps[:,i0]=c*a+s*b;amps[:,i1]=s*a+c*b
    for q in range(n):
        gate(q,'RY',theta[:,q]*np.sqrt(rho[q]));gate(q,'RZ',np.pi*(scores[:,q]-.5)*rho[q]*masks[:,q])
    for q in range(n):gate(q,'RX',.25*theta[:,q]*rho[q])
    if with_pairs:
        for a,b in itertools.combinations(range(n),2):
            phi=np.pi*.35*np.sqrt(rho[a]*rho[b])*(2*(1-np.abs(scores[:,a]-scores[:,b]))-1)*masks[:,a]*masks[:,b]
            # CX[a,b] RZ[b](phi) CX[a,b] is exp(-i*phi/2 Z_a Z_b).
            za=np.where((ix & pairs[a][2])==0,1,-1);zb=np.where((ix&pairs[b][2])==0,1,-1)
            amps*=np.exp(-.5j*phi[:,None]*(za*zb)[None,:])
    for q in range(n):
        gate(q,'RY',1.1*theta[:,q]*np.sqrt(rho[q]));gate(q,'RZ',np.pi*(scores[:,q]-.5)*rho[q]*masks[:,q])
    if return_amplitudes:return amps
    prob=abs(amps)**2
    z=np.column_stack([prob@np.where((ix&pairs[q][2])==0,1,-1) for q in range(n)])
    x=np.column_stack([np.sum(np.conj(amps)*amps[:,ix^pairs[q][2]],axis=1).real for q in range(n)])
    zz=np.column_stack([prob@(np.where((ix&pairs[a][2])==0,1,-1)*np.where((ix&pairs[b][2])==0,1,-1)) for a,b in itertools.combinations(range(n),2)])
    xx=np.column_stack([np.sum(np.conj(amps)*amps[:,ix^pairs[a][2]^pairs[b][2]],axis=1).real for a,b in itertools.combinations(range(n),2)])
    return np.column_stack([z,x,zz,xx]).astype('float32')

def get_quantum(scores,masks,rho,pair=True,batch=2500):
    return np.vstack([statevector_features(scores[i:i+batch],masks[i:i+batch],rho,with_pairs=pair) for i in range(0,len(scores),batch)])

def audit_statevector():
    from math import cos,sin
    sc=np.array([[.25,.7]]);mask=np.array([[1,1]]);rr=np.array([.95,.8])
    # Verify single qubit Ry probability identity and state norm over paired circuit.
    phi=2*np.arcsin(np.sqrt(.25));assert abs(np.sin(phi/2)**2-.25)<1e-12
    aa=statevector_features(sc,mask,rr,True,True);assert abs(np.sum(np.abs(aa)**2)-1)<1e-12
    off=statevector_features(sc,np.array([[1,0]]),rr,True,True)
    off2=statevector_features(sc,np.array([[1,0]]),rr,False,True)
    assert np.max(np.abs(off-off2))<1e-12,'Inactive pair must be exactly identity'
    # Direct check CX-RZ-CX = diagonal pair implementation for basis phases.
    for j in range(4):
        za=1 if not (j&2) else -1;zb=1 if not(j&1) else -1
        target=np.exp(-.5j*0.37*za*zb)
        state=np.eye(4,dtype=complex)[:,j]
        state=state[[0,1,3,2]];state*=np.array([np.exp(-.185j),np.exp(.185j),np.exp(-.185j),np.exp(.185j)]);state=state[[0,1,3,2]]
        assert abs(state[j]-target)<1e-12
    return True

def metrics_by_groups(df,y,p,name):
    rows=[]
    for combo,g in df.groupby('combination'):
        ind=g.index.to_numpy();r=report(y[ind],p[ind]);r.update(model=name,scope='exact_combination',combination=combo);rows.append(r)
    for k in range(1,6):
        ind=np.where(df[[f'mask_{m}' for m in MODS]].sum(axis=1).to_numpy()==k)[0];r=report(y[ind],p[ind]);r.update(model=name,scope='number_modalities',combination=str(k));rows.append(r)
    for a,b in [('email','url'),('sms','url'),('email','sms')]:
        ix=np.where((df['mask_'+a].to_numpy()==1)&(df['mask_'+b].to_numpy()==1))[0]
        r=report(y[ix],p[ix]);r.update(model=name,scope='pair_present',combination=a+'+'+b);rows.append(r)
    return rows

def plot_circuit(root,n=4):
    from matplotlib.patches import FancyBboxPatch,Circle
    fig,ax=plt.subplots(figsize=(14,2.2+n*.65))
    for q in range(n):
        y=n-q;ax.hlines(y,.2,12.8,color='#444',lw=1.8)
        for xx,txt in [(1,'RY(θ)'),(2.1,'RZ(β)'),(3.4,'RX'),(9.6,'RY(1.1θ)'),(11,'RZ(β)')]:
            ax.add_patch(FancyBboxPatch((xx-.43,y-.23),.86,.46,boxstyle='round,pad=.05',fc='#e6effa',ec='#294f83',lw=1));ax.text(xx,y,txt,fontsize=8,ha='center',va='center')
        ax.text(-.35,y,['URL','EMAIL','SMS','QR','IMAGE'][q],ha='right',va='center',fontsize=10)
    # Example pair gate between URL and email, precisely the CX RZ CX described in the manuscript.
    yy=[n,n-1];ax.plot([4.6,4.6],yy,color='#344e83');ax.plot(4.6,yy[0],'ko');ax.add_patch(Circle((4.6,yy[1]),.12,ec='#344e83',fc='white'));ax.plot([4.47,4.73],[yy[1],yy[1]],color='#344e83');ax.plot([5.95,5.95],yy,color='#344e83');ax.plot(5.95,yy[0],'ko');ax.add_patch(Circle((5.95,yy[1]),.12,ec='#344e83',fc='white'));ax.plot([5.82,6.08],[yy[1],yy[1]],color='#344e83')
    ax.add_patch(FancyBboxPatch((4.95,yy[1]-.23),.6,.46,boxstyle='round,pad=.05',fc='#ffe9cc',ec='#bb752f'));ax.text(5.25,yy[1],'RZ(φ)',ha='center',va='center',fontsize=8)
    ax.text(6.7,n-.45,'All active pairs: CX–RZ(φ)–CX\n(masked by observed modalities)',fontsize=10,ha='left',va='center')
    ax.set_xlim(-1,13);ax.set_ylim(.45,n+.85);ax.axis('off');fig.suptitle(f'Executable {n}-qubit RC-QTG circuit '+('(original 4 modalities)' if n==4 else '(5-modality extension)'),fontsize=13)
    fig.tight_layout();fig.savefig(root/f'circuit_{n}qubits.png',dpi=190,bbox_inches='tight');plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default=r'E:\other\QTrustAgentX\Validation_2026');p.add_argument('--seed',type=int,default=42);p.add_argument('--output',default=None);args=p.parse_args()
    root=Path(args.data);out=Path(args.output or root/'experiment');out.mkdir(parents=True,exist_ok=True);t0=time.perf_counter();assert audit_statevector()
    preds={};rhos={};special=[];lookups={}
    for m in MODS:
        df=pd.read_csv(root/(m+'_sources.csv'),low_memory=False)
        for c in ['text','qr_path','image_path']:
            if c in df:df[c]=df[c].fillna('').astype(str)
        score_path=out/f'specialist_scores_{m}.csv'
        if score_path.exists():
            pp=pd.read_csv(score_path)
            pp=pp.merge(df[['source_id','label','split']],on=['source_id','label','split'],how='inner')
            v=pp.split.eq('val');rho=float(np.clip(roc_auc_score(pp.loc[v,'label'],pp.loc[v,'score']),.05,1))
            ts=pp.split.eq('test');met=report(pp.loc[ts,'label'],pp.loc[ts,'score'])
            print('CACHED specialist',m,flush=True)
        else:
            pp,rho,met=oof_predict(df,m,root,args.seed)
            pp.to_csv(score_path,index=False)
        rhos[m]=rho;special.append({'modality':m,'val_auc_reliability':rho,**met})
        lookups[m]=pp.set_index('source_id')['score'].to_dict()
        pp.to_csv(out/f'specialist_scores_{m}.csv',index=False)
        print('SPECIALIST',m,'AUC(test)',round(met['roc_auc'],4),'rho(val)',round(rho,4),flush=True)
    pd.DataFrame(special).to_csv(out/'specialist_test_metrics.csv',index=False)
    df=pd.read_csv(root/'multimodal_31_combinations_1000_each.csv',low_memory=False)
    for m in MODS:df[m+'_source_id']=df[m+'_source_id'].fillna('').astype(str)
    masks=df[[f'mask_{m}' for m in MODS]].to_numpy(dtype=float)
    scores=np.zeros_like(masks,dtype=float)+.5
    for j,m in enumerate(MODS):
        v=df[m+'_source_id'].map(lookups[m]);active=masks[:,j]>0
        assert v[active].notna().all(),f'missing modality scores for {m}'
        scores[active,j]=v[active].to_numpy(dtype=float)
    rho=np.array([rhos[m] for m in MODS],float)
    base=np.column_stack([scores*masks,masks,np.sum(scores*masks,axis=1)/np.maximum(masks.sum(axis=1),1),masks.sum(axis=1)])
    base=np.asarray(base,np.float32)
    # Equal-information, high-capacity pair polynomial classical comparison.
    poly=np.column_stack([base]+[(scores[:,a]*scores[:,b]*masks[:,a]*masks[:,b]) for a,b in itertools.combinations(range(5),2)]).astype('float32')
    q0=get_quantum(scores,masks,rho,False);q1=get_quantum(scores,masks,rho,True)
    pair_delta=np.max(np.abs(q1-q0),axis=1)
    assert float(pair_delta[(masks.sum(axis=1)<2)].max())<1e-5
    assert float(pair_delta[(masks.sum(axis=1)>1)].mean())>1e-6
    np.savez_compressed(out/'circuit_feature_sample.npz',features_with_pairs=q1[:100],features_without_pairs=q0[:100],masks=masks[:100],scores=scores[:100],reliability=rho)
    cols=['quantum_z_'+m for m in MODS]+['quantum_x_'+m for m in MODS]
    cols+=['quantum_zz_'+MODS[a]+'_'+MODS[b] for a,b in itertools.combinations(range(5),2)]
    cols+=['quantum_xx_'+MODS[a]+'_'+MODS[b] for a,b in itertools.combinations(range(5),2)]
    assert len(cols)==q1.shape[1]==30
    pd.DataFrame(q1[:100],columns=cols).to_csv(out/'circuit_observables_first100.csv',index=False)
    features={'linear_fusion':base,'classical_pair_polynomial':poly,'classical_extra_trees':poly,
              'circuit_no_pairs':np.column_stack([q0,base]),'circuit_with_pairs':np.column_stack([q1,base])}
    tr=np.where(df.split=='train')[0];va=np.where(df.split=='val')[0];te=np.where(df.split=='test')[0]
    y=df.label.to_numpy(dtype=int);results=[];groups=[];prob_all={};trained={}
    for name,X in features.items():
        if name=='classical_extra_trees':model=ExtraTreesClassifier(n_estimators=90,min_samples_leaf=6,max_features=.9,n_jobs=-1,random_state=args.seed)
        else:model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=500,C=.3))
        t=time.perf_counter();model.fit(X[tr],y[tr]);tfit=time.perf_counter()-t
        ptest=model.predict_proba(X[te])[:,1];pv=model.predict_proba(X[va])[:,1]
        r=report(y[te],ptest);r.update(model=name,scope='synthetic_composite_test',fit_seconds=round(tfit,4),val_f1=f1_score(y[va],pv>=.5))
        results.append(r);prob_all[name]=ptest;trained[name]=model
        joblib.dump(model,out/f'{name}_model.joblib',compress=3)
        temp=df.iloc[te].reset_index(drop=True);groups.extend(metrics_by_groups(temp,y[te],ptest,name))
        print('MODEL',name,'test_f1',round(r['f1'],4),'auc',round(r['roc_auc'],4),'fit',round(tfit,2),flush=True)
    # Same paired model, remove paired circuit gates at test time (causal input ablation).
    mod=trained['circuit_with_pairs'];counter=mod.predict_proba(np.column_stack([q0[te],base[te]]))[:,1]
    rr=report(y[te],counter);rr.update(model='paired_model_gates_disabled_at_test',scope='counterfactual_gate_ablation');results.append(rr)
    testframe=df.iloc[te][['incident_id','combination','split','mask','label']].reset_index(drop=True)
    for name,pred in prob_all.items():testframe['p_'+name]=pred
    testframe['p_gate_disabled_counterfactual']=counter
    testframe['pair_gate_feature_change']=pair_delta[te]
    testframe.to_csv(out/'test_predictions.csv',index=False)
    try:
        import torch
        torch.save({'scores':torch.from_numpy(scores.astype('float32')),'masks':torch.from_numpy(masks.astype('float32')),'reliability':torch.from_numpy(rho.astype('float32')),'observables_with_pair_gates':torch.from_numpy(q1),'observables_without_pair_gates':torch.from_numpy(q0),'labels':torch.from_numpy(y),'split':df.split.tolist(),'note':'Executed CPU statevector features; these are tensors, not neural-network weights.'},out/'quantum_observables_and_scores.pt')
    except ImportError:pass
    pd.DataFrame(results).to_csv(out/'model_comparison_metrics.csv',index=False)
    pd.DataFrame(groups).to_csv(out/'per_combination_and_pair_metrics.csv',index=False)
    # Paired bootstrap intervals (row level). Source reuse makes these optimistic; record limitation.
    rng=np.random.default_rng(args.seed);aa=prob_all['circuit_with_pairs']; bb=prob_all['classical_pair_polynomial']
    ix=np.arange(len(te));boot=[]
    for i in range(300):
        r=rng.choice(ix,len(ix),replace=True)
        boot.append(float(f1_score(y[te][r],aa[r]>=.5)-f1_score(y[te][r],bb[r]>=.5)))
    report_extra={'seed':args.seed,'original_paper_qubits':4,'extension_qubits':5,'source_reliabilities_validation_auc':rhos,
       'statevector_backend':'NumPy CPU exact amplitudes (not hardware, not Qiskit Aer)','pair_gates_simulated':int(sum(int(c*(c-1)//2) for c in masks.sum(axis=1))),
       'active_pairs_rows':int((masks.sum(axis=1)>1).sum()),'synthetic_test_rows':len(te),
       'mean_abs_max_observable_change_gate_enabled':float(np.mean(pair_delta[masks.sum(axis=1)>1])),
       'paired_F1_delta_vs_classical_polynomial':float(f1_score(y[te],aa>=.5)-f1_score(y[te],bb>=.5)),
       'bootstrap_row_level_95pct_CI':np.quantile(boot,[.025,.975]).tolist(),
       'bootstrap_caution':'Repeated source records across composites violate iid rows; CI is optimistic; report source-grouped uncertainty in publication.',
       'elapsed_seconds':round(time.perf_counter()-t0,3)}
    (out/'experiment_audit.json').write_text(json.dumps(report_extra,indent=2))
    plot_circuit(out,4);plot_circuit(out,5)
    fig,ax=plt.subplots(figsize=(9,5))
    comp=pd.DataFrame(results).iloc[:5]
    xx=np.arange(len(comp));ax.bar(xx-.16,comp['f1'],width=.31,label='F1');ax.bar(xx+.16,comp['roc_auc'],width=.31,label='ROC-AUC')
    ax.set_xticks(xx,comp.model,rotation=25,ha='right');ax.set_ylim(0,1.05);ax.legend();fig.tight_layout();fig.savefig(out/'model_comparison.png',dpi=180);plt.close(fig)
    # Real incident-linked, same-message URL pair evaluation (email and SMS). No email-SMS linkage.
    linked=pd.read_csv(root/'naturally_linked_message_url_pairs.csv',low_memory=False)
    # Email addresses and sender domains are NOT observed hyperlinks.
    linked=linked[(linked.source_modality=='sms')|linked.observed_url.astype(str).str.lower().str.startswith(('http://','https://','www.'))].copy()
    linked=linked[linked.observed_url.astype(str).str.len()>5]
    linked=linked.drop_duplicates(['source_modality','source_id']).reset_index(drop=True)
    # independent URL text model transferred from URL source, classical specialist already fitted; train fresh on URL train only.
    url_df=pd.read_csv(root/'url_sources.csv');tr_url=url_df.split=='train'
    linkurl=make_pipeline(TfidfVectorizer(analyzer='char',ngram_range=(3,5),max_features=25000,min_df=2,sublinear_tf=True),LogisticRegression(C=2.,max_iter=500))
    linkurl.fit(url_df.loc[tr_url,'text'].astype(str),url_df.loc[tr_url,'label'].astype(int))
    linked['score_url']=linkurl.predict_proba(linked.observed_url.fillna('').astype(str))[:,1]
    linked['score_message']=[lookups[mod].get(sid,np.nan) for mod,sid in zip(linked.source_modality,linked.source_id)]
    assert linked.score_message.notna().all()
    linked.to_csv(out/'naturally_linked_scored.csv',index=False)
    linked_results=[];linked_predictions=[]
    for mod in ['email','sms']:
        sub=linked[linked.source_modality==mod].reset_index(drop=True)
        S=np.full((len(sub),4),.5);M=np.zeros_like(S);S[:,0]=sub.score_url;S[:,1 if mod=='email' else 2]=sub.score_message;M[:,0]=1;M[:,1 if mod=='email' else 2]=1
        R=np.array([rhos['url'],rhos['email'],rhos['sms'],rhos['qr']]);on=get_quantum(S,M,R,True);off=get_quantum(S,M,R,False)
        raw=np.column_stack([S*M,M]);features2={'additive_classical':raw,'pair_circuit_disabled':np.column_stack([off,raw]),'pair_circuit_enabled':np.column_stack([on,raw])}
        train=np.where(sub.split=='train')[0];test=np.where(sub.split=='test')[0]
        if len(test)<20 or len(np.unique(sub.label.iloc[test]))<2:
            linked_results.append({'modality':mod,'status':'insufficient_linked_test_samples'});continue
        for name,F in features2.items():
            clf=make_pipeline(StandardScaler(),LogisticRegression(C=.3,max_iter=500));clf.fit(F[train],sub.label.iloc[train]);pp=clf.predict_proba(F[test])[:,1]
            m=report(sub.label.iloc[test],pp);m.update(modality=mod,model=name,scope='naturally_linked_same_message',status='ok');linked_results.append(m)
            linked_predictions.append(pd.DataFrame({'modality':mod,'model':name,'label':sub.label.iloc[test].to_numpy(),'probability':pp,'source_id':sub.source_id.iloc[test].to_numpy()}))
            print('NATURAL PAIR',mod,name,'N',len(test),'F1',round(m['f1'],4),flush=True)
    pd.DataFrame(linked_results).to_csv(out/'naturally_linked_pair_results.csv',index=False)
    if linked_predictions:pd.concat(linked_predictions).to_csv(out/'naturally_linked_pair_test_predictions.csv',index=False)
    print('FINISHED',out,'elapsed seconds:',round(time.perf_counter()-t0,1),flush=True)
if __name__=='__main__':main()
