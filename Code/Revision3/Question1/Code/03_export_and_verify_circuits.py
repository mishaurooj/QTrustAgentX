"""Export exact four-qubit ORIGINAL and five-qubit EXTENDED QTrustAgentX circuits.
"""
from pathlib import Path
import argparse,itertools,importlib.util,json,math
import numpy as np

def script_module(path):
    spec=importlib.util.spec_from_file_location('qtrust_experiment',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def qasm(scores,mask,rho,n):
    out=['OPENQASM 2.0;','include "qelib1.inc";',f'qreg q[{n}];','// Source-derived specialist scores; 4q is manuscript model, 5q is extension.']
    theta=2*np.arcsin(np.sqrt(np.clip(scores[:n],0,1)))*mask[:n]
    for q in range(n):
        if mask[q]==0:continue
        out.append(f'ry({theta[q]*math.sqrt(rho[q]):.14f}) q[{q}];')
        out.append(f'rz({math.pi*(scores[q]-.5)*rho[q]:.14f}) q[{q}];')
    for q in range(n):
        if mask[q]:out.append(f'rx({.25*theta[q]*rho[q]:.14f}) q[{q}];')
    for a,b in itertools.combinations(range(n),2):
        if not (mask[a] and mask[b]):continue
        phi=math.pi*.35*math.sqrt(rho[a]*rho[b])*(2*(1-abs(scores[a]-scores[b]))-1)
        out.extend([f'cx q[{a}],q[{b}];',f'rz({phi:.14f}) q[{b}];',f'cx q[{a}],q[{b}];'])
    for q in range(n):
        if not mask[q]:continue
        out.append(f'ry({1.1*theta[q]*math.sqrt(rho[q]):.14f}) q[{q}];')
        out.append(f'rz({math.pi*(scores[q]-.5)*rho[q]:.14f}) q[{q}];')
    return '\n'.join(out)+'\n'

def main():
    a=argparse.ArgumentParser();a.add_argument('--experiment',default=r'E:\other\QTrustAgentX\Validation_2026\experiment');args=a.parse_args()
    root=Path(args.experiment);p=root/'quantum_observables_and_scores.pt'
    import torch
    d=torch.load(p,map_location='cpu',weights_only=False)
    scores=d['scores'].numpy();masks=d['masks'].numpy();rho=d['reliability'].numpy()
    idx=int(np.flatnonzero(masks.sum(axis=1)==5)[0]);mod=script_module(Path(__file__).with_name('02_train_evaluate.py'))
    audit={'chosen_row_index':idx,'specialist_scores':scores[idx].tolist(),'reliability':rho.tolist(),'source_mask':masks[idx].tolist(),'qiskit_statevector_verified':False}
    for n,label in [(4,'original_manuscript_4qubit'),(5,'extended_five_modality_5qubit')]:
        txt=qasm(scores[idx],masks[idx],rho,n);dest=root/(label+'.qasm');dest.write_text(txt,encoding='utf8')
        audit[label+'_active_pair_gates']=int(sum(bool(masks[idx,a] and masks[idx,b]) for a,b in itertools.combinations(range(n),2)))
        print(label,'EXPORTED',dest,'CNOT count',txt.count('cx q['),flush=True)
    try:
        from qiskit import qasm2
        from qiskit.quantum_info import Statevector, Pauli
        for n,label in [(4,'original_manuscript_4qubit'),(5,'extended_five_modality_5qubit')]:
            qc=qasm2.load(str(root/(label+'.qasm')));state=Statevector.from_instruction(qc)
            expected=mod.statevector_features(scores[idx:idx+1,:n],masks[idx:idx+1,:n],rho[:n],True)[0]
            obs=[]
            def expect(axes):
                chars=['I']*n
                for q,axis in axes:chars[n-q-1]=axis
                return float(np.real(state.expectation_value(Pauli(''.join(chars)))))
            for axis in ['Z','X']:
                for q in range(n):obs.append(expect([(q,axis)]))
            for axis in ['Z','X']:
                for aa,bb in itertools.combinations(range(n),2):obs.append(expect([(aa,axis),(bb,axis)]))
            np.testing.assert_allclose(obs,expected,rtol=1e-6,atol=1e-6)
            audit[label+'_qiskit_verified']=True
            (root/(label+'_drawing.txt')).write_text(str(qc.draw(output='text')),encoding='utf8')
            print('QISKIT VERIFIED',label,'features',len(obs),flush=True)
        audit['qiskit_statevector_verified']=True
    except ImportError:
        print('Qiskit not installed here. QASM circuits exported; NumPy statevectors already executed.',flush=True)
        audit['qiskit_status']='Qiskit unavailable in this runtime; run on local Anaconda to cross-verify.'
    (root/'circuit_export_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
if __name__=='__main__':main()
