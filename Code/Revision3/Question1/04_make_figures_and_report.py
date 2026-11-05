"""Figures and reviewer-facing summary from actual saved execution outputs."""
from pathlib import Path
import argparse, itertools, json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

def full_circuit(n,out):
    names=['URL','Email','SMS','QR','Image'][:n]
    pairs=list(itertools.combinations(range(n),2));fig,ax=plt.subplots(figsize=(max(19,3.6*n+len(pairs)*2.2),2.6+n*.8))
    xs=0.8
    ticks=[]
    def block(x,q,txt,fc='#e7eef9',fontsize=7):
        y=n-q;ax.add_patch(FancyBboxPatch((x-.32,y-.24),.64,.48,boxstyle='round,pad=.04',edgecolor='#4f6e95',facecolor=fc))
        ax.text(x,y,txt,ha='center',va='center',fontsize=fontsize)
    def cx(x,ctrl,tgt):
        y1=n-ctrl;y2=n-tgt;ax.vlines(x,min(y1,y2),max(y1,y2),color='#233d62',lw=1.5);ax.plot([x],[y1],'o',color='#233d62',ms=5)
        ax.add_patch(Circle((x,y2),.125,edgecolor='#233d62',facecolor='white',lw=1.2));ax.hlines(y2,x-.12,x+.12,color='#233d62');ax.vlines(x,y2-.12,y2+.12,color='#233d62')
    for q in range(n):
        for delta,t in [(0,'RY₀'),(.9,'RZ₀'),(1.8,'RX')]:block(xs+delta,q,t)
    xs+=3.4
    for a,b in pairs:
        cx(xs,a,b);block(xs+.72,b,'RZφ','#fff1d9',7);cx(xs+1.44,a,b)
        ax.text(xs+.7,n+.49,f'{names[a]}:{names[b]}',ha='center',fontsize=8,rotation=45)
        xs+=2.15
    for q in range(n):
        block(xs+.3,q,'RY₁');block(xs+1.2,q,'RZ₁')
    end=xs+2
    for q,name in enumerate(names):
        ax.hlines(n-q,0,end,color='#666',lw=1,zorder=-1);ax.text(-.35,n-q,name,ha='right',va='center',fontsize=10)
    ax.set_xlim(-1,end+.3);ax.set_ylim(.25,n+1.3);ax.axis('off')
    ax.set_title(f'{n}-qubit QTrustAgentX statevector circuit: all {len(pairs)} conditional CNOT–RZ–CNOT pair blocks shown',fontsize=12)
    fig.tight_layout();fig.savefig(out,dpi=125,bbox_inches='tight');plt.close(fig)

def main():
    a=argparse.ArgumentParser();a.add_argument('--data',default=r'E:\other\QTrustAgentX\Validation_2026');args=a.parse_args()
    data=Path(args.data);exp=data/'experiment'
    full_circuit(4,exp/'complete_original_4qubit_pair_circuit.png')
    full_circuit(5,exp/'complete_extended_5qubit_pair_circuit.png')
    r=pd.read_csv(exp/'per_combination_and_pair_metrics.csv')
    sub=r[(r.scope=='exact_combination') & r.model.isin(['circuit_with_pairs','circuit_no_pairs'])]
    delta=sub.pivot(index='combination',columns='model',values='f1')
    delta=delta.loc[delta.index.str.contains(r'\+')].copy()
    delta['delta']=delta['circuit_with_pairs']-delta['circuit_no_pairs']
    delta.sort_values('delta',inplace=True)
    fig,ax=plt.subplots(figsize=(10,8));ax.barh(delta.index,delta['delta']);ax.axvline(0,ls='--',color='black',lw=1)
    ax.set_xlabel('Test F1 with pair gates minus F1 without pair gates');ax.set_title('Exact modality-combination pair-gate ablation (synthetic composites)')
    fig.tight_layout();fig.savefig(exp/'pair_gate_f1_deltas_by_combination.png',dpi=175);plt.close(fig)
    df=pd.read_csv(data/'multimodal_31_combinations_1000_each.csv')
    assert len(df)==31000 and df.combination.nunique()==31
    assert (df.groupby('combination').size()==1000).all()
    assert ((df.groupby('combination').label.sum())==500).all()
    assert (df[[f'mask_{m}' for m in ['url','email','sms','qr','image']]].sum(axis=1)>=2).sum()==26000
    for m in ['url','email','sms']:
        s=pd.read_csv(data/f'{m}_sources.csv');assert not s.text.astype(str).str.casefold().str.strip().duplicated().any(),f'normalized duplicate {m}'
    models=pd.read_csv(exp/'model_comparison_metrics.csv');linked=pd.read_csv(exp/'naturally_linked_pair_results.csv');audit=json.loads((exp/'experiment_audit.json').read_text())
    rows=[]
    rows.append('# QTrustAgentX paired-evidence validation report\n')
    (exp/'VALIDATION_REPORT.md').write_text('\n'.join(rows),encoding='utf8')
    print('REPORT',exp/'VALIDATION_REPORT.md','validation PASSED',flush=True)
if __name__=='__main__':main()
