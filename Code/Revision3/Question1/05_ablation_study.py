"""Executed, reproducible held-out ablations for the QTrustAgentX =benchmark.
Uses the saved predictions and models from 02_train_evaluate.py. Does NOT retrain the specialists.
Outputs model, all-31-combination, cardinality, named-pair, leave-one-out and linked-message tables.
Leave-one-out rows are INPUT MASKING tests of a frozen model (distribution shift).
"""
from pathlib import Path
import argparse, itertools, importlib.util, json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix

MODS=['url','email','sms','qr','image']

def metrics(y,p):
    y=np.asarray(y,dtype=int);p=np.asarray(p,float);yh=(p>=0.5).astype(int)
    cm=confusion_matrix(y,yh,labels=[0,1]);return {'n':int(len(y)),'accuracy':float(accuracy_score(y,yh)),
        'f1':float(f1_score(y,yh,zero_division=0)),
        'auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else float('nan'),
        'tn':int(cm[0,0]),'fp':int(cm[0,1]),'fn':int(cm[1,0]),'tp':int(cm[1,1])}

def imp(path):
    spec=importlib.util.spec_from_file_location('qtrust_eval_ablation',path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
    return obj

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default=str(Path(__file__).parent/'data'));args=ap.parse_args()
    d=Path(args.data);o=d/'experiment';o.mkdir(parents=True,exist_ok=True)
    test=pd.read_csv(o/'test_predictions.csv');y=test.label.to_numpy(int)
    keys={'linear_fusion':'p_linear_fusion','classical_pair_polynomial':'p_classical_pair_polynomial',
          'classical_extra_trees':'p_classical_extra_trees','circuit_no_pairs':'p_circuit_no_pairs',
          'circuit_with_pairs':'p_circuit_with_pairs','paired_model_gates_disabled_at_test':'p_gate_disabled_counterfactual'}
    scored=[]
    for name,col in keys.items():
        scored.append({'ablation':name,**metrics(y,test[col])})
    pd.DataFrame(scored).to_csv(o/'ABLATION_01_overall_model_comparison.csv',index=False)
    # Full 31 combinations and all cardinalities with an explicitly common test subset.
    allcomb=[];strata=[]
    for combo,grp in test.groupby('combination',sort=True):
        ii=grp.index.to_numpy();r={'combination':combo,'active_modalities':1+combo.count('+'),'n_test':int(len(ii))}
        for name,col in keys.items():
            r['f1_'+name]=metrics(y[ii],test[col].to_numpy()[ii])['f1']
            r['auc_'+name]=metrics(y[ii],test[col].to_numpy()[ii])['auc']
        r['delta_pair_vs_no_pair']=r['f1_circuit_with_pairs']-r['f1_circuit_no_pairs']
        r['delta_pair_vs_classical_pair']=r['f1_circuit_with_pairs']-r['f1_classical_pair_polynomial']
        allcomb.append(r)
    combos=pd.DataFrame(allcomb);combos.to_csv(o/'ABLATION_02_all_31_combinations.csv',index=False)
    assert len(combos)==31 and (combos.n_test==150).all()
    for k in range(1,6):
        ii=test.index[test.combination.str.count(r'\+')+1==k].to_numpy()
        r={'modalities_active':k,'number_combinations':int(sum(combos.active_modalities==k)),'n_test':int(len(ii))}
        for name,col in keys.items():r['f1_'+name]=metrics(y[ii],test[col].to_numpy()[ii])['f1']
        r['delta_pair_vs_no_pair']=r['f1_circuit_with_pairs']-r['f1_circuit_no_pairs']
        strata.append(r)
    pd.DataFrame(strata).to_csv(o/'ABLATION_03_by_active_modality_count.csv',index=False)
    # Pair-present means both masks active in each source-composite, not a naturally linked incident.
    pairs=[]
    for a,b in [('email','url'),('sms','url'),('email','sms')]:
        ii=test.index[test.combination.apply(lambda x:a in x.split('+') and b in x.split('+'))].to_numpy()
        r={'pair':a+'+'+b,'scope':'synthetic_pair_present','n_test':int(len(ii))}
        for name,col in keys.items():r['f1_'+name]=metrics(y[ii],test[col].to_numpy()[ii])['f1']
        r['delta_pair_vs_no_pair']=r['f1_circuit_with_pairs']-r['f1_circuit_no_pairs']
        pairs.append(r)
    pd.DataFrame(pairs).to_csv(o/'ABLATION_04_required_named_pairs.csv',index=False)
    # Frozen trained model; input modality removed at inference, all its dependent circuit gates re-evaluated.
    orig=pd.read_csv(d/'multimodal_31_combinations_1000_each.csv',low_memory=False)
    orig=orig[orig.split=='test'].reset_index(drop=True)
    assert orig.incident_id.tolist()==test.incident_id.tolist(),'test order changed: abort'
    masks=orig[[f'mask_{m}' for m in MODS]].to_numpy(dtype=float)
    scores=np.full_like(masks,.5,dtype=float)
    for j,mod in enumerate(MODS):
        pp=pd.read_csv(o/f'specialist_scores_{mod}.csv').set_index('source_id')['score']
        ids=orig[f'{mod}_source_id'].astype(str).replace({'nan':''})
        active=masks[:,j]>0
        lookup=ids.map(pp)
        assert lookup[active].notna().all(),f'missing {mod} specialist scores'
        scores[active,j]=lookup[active].to_numpy(float)
    audit=json.loads((o/'experiment_audit.json').read_text());rho=np.array([audit['source_reliabilities_validation_auc'][m] for m in MODS])
    evalmod=imp(Path(__file__).parent/'02_train_evaluate.py')
    def base_features(s,m):
        return np.column_stack([s*m,m,np.sum(s*m,axis=1)/np.maximum(m.sum(axis=1),1),m.sum(axis=1)]).astype('float32')
    model=joblib.load(o/'circuit_with_pairs_model.joblib')
    b=base_features(scores,masks);q=evalmod.get_quantum(scores,masks,rho,True)
    original_scores=model.predict_proba(np.column_stack([q,b]))[:,1]
    assert np.allclose(original_scores,test.p_circuit_with_pairs.to_numpy(float),atol=1e-5,rtol=1e-4), 'Saved scores and rebuilt circuit differ.'
    occl=[]
    for j,mod in enumerate(MODS):
        ii=np.where(masks[:,j]==1)[0]
        altered_s=scores[ii].copy();altered_m=masks[ii].copy();altered_s[:,j]=.5;altered_m[:,j]=0
        bb=base_features(altered_s,altered_m)
        qq=evalmod.get_quantum(altered_s,altered_m,rho,True)
        pp=model.predict_proba(np.column_stack([qq,bb]))[:,1]
        before=metrics(y[ii],original_scores[ii]);after=metrics(y[ii],pp)
        occl.append({'removed_modality':mod,'scope':'frozen_model_masked_input','n_test':len(ii),
                      'f1_before':before['f1'],'f1_after':after['f1'],
                      'f1_drop_before_minus_after':before['f1']-after['f1'],
                      'auc_before':before['auc'],'auc_after':after['auc'],
                      'auc_drop_before_minus_after':before['auc']-after['auc'],
                      'changed_decisions':int(np.sum((original_scores[ii]>=.5)!=(pp>=.5)))})
    pd.DataFrame(occl).to_csv(o/'ABLATION_05_frozen_model_leave_one_modality_out.csv',index=False)
    linked=pd.read_csv(o/'naturally_linked_pair_results.csv')
    linked.to_csv(o/'ABLATION_06_naturally_linked_pair_evaluation.csv',index=False)
    # Inference-only observation checks on real held-out cases. Not independent events.
    all_text=[]
    all_text.append('# Supplementary executed ablation study\n')
    all_text.append('The 31,000 balanced records are **synthetic class-matched composites**; no independently collected email--SMS incidents are available. The comparison uses 4,650 held-out composite rows; source elements may be reused inside a split. The frozen-model leave-one-out experiment removes a source at inference without refitting and therefore includes distribution shift. No statistically reliable quantum advantage is demonstrated.\n')
    for t,frame in [('1. All-model held-out comparison',pd.DataFrame(scored)),('2. By modality cardinality',pd.DataFrame(strata)),('3. Required pair-presence comparison',pd.DataFrame(pairs)),('4. Frozen-model source knockout',pd.DataFrame(occl)),('5. Naturally linked same-message cases',linked)]:
        all_text.append('## '+t+'\n');all_text.append(frame.to_markdown(index=False,floatfmt='.6f')+'\n')
    four_path=o/'ORIGINAL_FOUR_QUBIT_ABLATION.csv'
    if four_path.exists():
        all_text.append('## Original four-qubit circuit, 15 synthetic combinations\n')
        all_text.append(pd.read_csv(four_path).to_markdown(index=False,floatfmt='.6f')+'\n')
    all_text.append('## Full 31-combination comparison\n');all_text.append(combos[['combination','n_test','f1_linear_fusion','f1_classical_pair_polynomial','f1_circuit_no_pairs','f1_circuit_with_pairs','delta_pair_vs_no_pair']].to_markdown(index=False,floatfmt='.6f')+'\n')
    all_text.append('Data provenance: QR images were sampled from the original uploaded archive; image screenshots were generated from source email content. Source-label conflicts and malformed email CSV recovery are documented in `data/dataset_audit.json`.\n')
    (o/'ABLATION_STUDY_FULL_TABLES.md').write_text('\n'.join(all_text),encoding='utf8')
    print('ABLATION COMPLETE; 31 combinations, 3 named pairs, 5 leave-one-out tests. Outputs:',o)

if __name__=='__main__':main()
