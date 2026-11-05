"""Create manuscript-ready full 31-row ablation appendix from executed CSV, no hardcoded metrics."""
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).parent
CSV=ROOT/'data'/'experiment'/'ABLATION_02_all_31_combinations.csv'
OUT=ROOT/'MANUSCRIPT_ABLATION_FULL_31_COMBINATIONS.tex'

def main():
    d=pd.read_csv(CSV)
    assert len(d)==31 and (d.n_test==150).all()
    s=[r'% Additional appendix table; requires booktabs and graphicx.',
       r'\begin{table*}[t]',r'\centering',
       r'\caption{Held-out F1 for every synthetic modality combination ($n=150$ test rows per combination). Pair and no-pair models are separately trained; $\Delta$F1 is pair-enabled minus no-pair circuit F1.}',
       r'\label{tab:all_31_combo_ablation}',r'\scriptsize',r'\setlength{\tabcolsep}{3.5pt}',r'\begin{tabular}{lrrrrr}',r'\toprule',
       r'Combination & Linear & Classical pair & No-pair circuit & Pair circuit & $\Delta$F1 \\',r'\midrule']
    for _,r in d.iterrows():
        combo=r.combination.replace('+',r'--')
        s.append(f"{combo} & {r.f1_linear_fusion:.4f} & {r.f1_classical_pair_polynomial:.4f} & {r.f1_circuit_no_pairs:.4f} & {r.f1_circuit_with_pairs:.4f} & {r.delta_pair_vs_no_pair:+.4f} \\\\")
    s += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
    OUT.write_text('\n'.join(s)+'\n',encoding='utf8')
    print('Created',OUT)

if __name__=='__main__':main()
