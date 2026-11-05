"""Executed 4-qubit circuit evaluation on all 15 synthetic combinations of original URL/email/SMS/QR channels.
Uses saved OOF specialist scores and source-level split from 02_train_evaluate.py. 4q gate family
is manuscript-referenced; validation-AUC reliability is supplementary and differs from primary Brier skill.
"""
from pathlib import Path
import argparse,itertools,importlib.util,json,time
import numpy as np,pandas as pd,joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,confusion_matrix

MODS=['url','email','sms','qr']
def modelmod():
    p=Path(__file__).with_name('02_train_evaluate.py');s=importlib.util.spec_from_file_location('qtrust_4q_eval',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default=str(Path(__file__).parent/'data'));ap.add_argument('--seed',type=int,default=42);a=ap.parse_args()
    d=Path(a.data);o=d/'experiment';raw=pd.read_csv(d/'multimodal_31_combinations_1000_each.csv',low_memory=False)
    raw=raw[raw.mask_image.eq(0)].reset_index(drop=True)
    assert raw.combination.nunique()==15 and len(raw)==15000
    m=raw[[f'mask_{t}' for t in MODS]].to_numpy(dtype=float)
    s=np.full_like(m,.5,dtype=float)
    for j,t in enumerate(MODS):
        lookup=pd.read_csv(o/f'specialist_scores_{t}.csv').set_index('source_id').score
        score=raw[f'{t}_source_id'].map(lookup);active=m[:,j]>0
        assert score[active].notna().all()
        s[active,j]=score[active].to_numpy(dtype=float)
    audit=json.loads((o/'experiment_audit.json').read_text())
    rho=np.array([audit['source_reliabilities_validation_auc'][t] for t in MODS],float)
    base=np.column_stack([s*m,m,np.sum(s*m,axis=1)/np.maximum(m.sum(axis=1),1),m.sum(axis=1)]).astype('float32')
    classical_pair=np.column_stack([base]+[s[:,i]*s[:,j]*m[:,i]*m[:,j] for i,j in itertools.combinations(range(4),2)]).astype('float32')
    ev=modelmod();q0=ev.get_quantum(s,m,rho,False);q1=ev.get_quantum(s,m,rho,True)
    features={'linear_fusion_4q':base,'classical_pair_polynomial_4q':classical_pair,
       'extra_trees_4q':classical_pair,'four_qubit_no_pair_gates':np.column_stack([q0,base]),
       'four_qubit_pair_gates':np.column_stack([q1,base])}
    y=raw.label.to_numpy(dtype=int);tr=np.where(raw.split=='train')[0];te=np.where(raw.split=='test')[0]
    rows=[];pred=raw.iloc[te][['incident_id','combination','label','split']].reset_index(drop=True)
    for name,X in features.items():
        if name=='extra_trees_4q':model=ExtraTreesClassifier(n_estimators=90,min_samples_leaf=6,max_features=.9,n_jobs=-1,random_state=a.seed)
        else:model=make_pipeline(StandardScaler(),LogisticRegression(C=.3,max_iter=500))
        model.fit(X[tr],y[tr]);p=model.predict_proba(X[te])[:,1];yp=p>=.5;c=confusion_matrix(y[te],yp,labels=[0,1]);r={'name':name,'n_test':len(te),'f1':f1_score(y[te],yp),'auc':roc_auc_score(y[te],p),
          'accuracy':accuracy_score(y[te],yp),'tn':int(c[0,0]),'fp':int(c[0,1]),'fn':int(c[1,0]),'tp':int(c[1,1])}
        rows.append(r);pred['p_'+name]=p
        joblib.dump(model,o/(name+'.joblib'),compress=3)
        print('4Q',name,'F1',round(r['f1'],6),'AUC',round(r['auc'],6),flush=True)
    pd.DataFrame(rows).to_csv(o/'ORIGINAL_FOUR_QUBIT_ABLATION.csv',index=False)
    pred.to_csv(o/'ORIGINAL_FOUR_QUBIT_TEST_PREDICTIONS.csv',index=False)
    print('FOUR QUBIT RUN COMPLETED',len(te),'held-out rows')
if __name__=='__main__':main()
