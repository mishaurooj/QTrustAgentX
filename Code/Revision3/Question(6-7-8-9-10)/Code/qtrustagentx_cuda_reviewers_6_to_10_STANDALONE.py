#!/usr/bin/env python3
"""CUDA-capable QTrustAgent-X reviewer 6-10 runner.

Runs the ORIGINAL uploaded reviewer 6-9 experiment unchanged on DED, then
a separate DED/JMD matched CUDA statevector versus CPU exact reconstruction
benchmark for reviewer 10. Does not invent target metrics or imply quantum hardware.
Includes the original Reviewer 6-9 script internally; no companion file required.
"""
import argparse, importlib.util, io, json, os, subprocess, sys, time, zipfile
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from scipy.stats import t as student_t
try:
 import torch
except ImportError:
 torch=None
MODS=['email','sms','url','qr','image']
def clock():
 if torch is not None and torch.cuda.is_available():torch.cuda.synchronize()
 return time.perf_counter()
def read_csv(path,filename):
 p=Path(path)
 if p.is_file():
  with zipfile.ZipFile(p) as z:
   matches=[n for n in z.namelist() if n.endswith('/'+filename) or n==filename]
   if len(matches)!=1:raise ValueError(f'Expected one {filename}, found {matches[:5]}')
   return pd.read_csv(io.BytesIO(z.read(matches[0])),low_memory=False)
 matches=list(p.rglob(filename))
 if len(matches)!=1:raise ValueError(f'Expected one {filename}, found {matches[:5]}')
 return pd.read_csv(matches[0],low_memory=False)
def jmd(path):
 d=read_csv(path,'multimodal_31_combinations_1000_each.csv')
 print('JMD columns:',d.columns.tolist(),flush=True)
 if not {'label','split','combination'}.issubset(d.columns):raise ValueError('JMD must have label/split/combination')
 d['label']=pd.to_numeric(d.label,errors='raise').astype(int)
 d['split']=d.split.astype(str).str.lower()
 for m in MODS:
  col=f'mask_{m}'
  if col not in d:raise ValueError(f'Missing {col}')
  d[col]=pd.to_numeric(d[col],errors='raise').astype(int)
 return d
def ded(path,seed):
 # Use original audited source parser, preserving source-label provenance.
 module=load_original_module()
 email,sms,audit=module.load_data(path)
 rows=[]
 for _,r in email.iterrows():
  rows.append(dict(label=int(r.label),combination='email',mask_email=1,
                   mask_sms=0,mask_url=0,mask_qr=0,mask_image=0,
                   group=r.group,source_id=r.source_id,email_text=r.text))
 for _,r in sms.iterrows():
  rows.append(dict(label=int(r.label==2),combination='sms',mask_email=0,
                   mask_sms=1,mask_url=0,mask_qr=0,mask_image=0,
                   group=r.group,source_id=r.source_id,sms_text=r.text))
 # SMS strictly smishing-positive, not legacy spam+smishing.
 # URL labels and other DED modalities need separately validated source mappings.
 d=pd.DataFrame(rows)
 groups=d.group.to_numpy()
 ids=np.arange(len(d))
 for offset in range(50):
  splitter=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed+offset)
  trva,te=next(splitter.split(ids,d.label,groups))
  splitter2=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed+1000+offset)
  a,b=next(splitter2.split(trva,d.label.iloc[trva],groups[trva]))
  tr,va=trva[a],trva[b]
  if all(len(set(d.label.iloc[x]))==2 for x in [tr,va,te]):break
 else:raise ValueError('Cannot make grouped DED split')
 d['split']='train';d.loc[va,'split']='val';d.loc[te,'split']='test'
 return d,audit

# Bundled original audited Reviewer 6-9 experiment; no companion .py needed.
import base64, zlib
_ORIGINAL_B85 = (
    'c-rlKYj@j5lIVB-ik?hn0NNz<@HnyQFguDQC)$-{uOvG&OX=W1Bq$*w0R{jm(OTW#e(TkZ#)FiUWbf{tb8}`a0_d)ORoAPls{j0_jd_-CTt@Lm'
    '7%$XpnO`UItJa@Y_eEC?lTj33_0&8cci-Tf)>mJB_3vltJj-{k!Z`nj+S}jjs5G3V$!I=|E+-*8ETZr>Ow~8tw<^4wg=sW}qM3?f_2GD5O_E_S'
    'Q6H%tHI62s*J_>Z{BXFh-t<(Kr_nH<ELE1w(_z>hCDR~^yLk%#h9mXUd>X{xb-zD6JW}v3&c@JA5RcStlwU(RuBGxY%QH`)zU`@L7-aJl>c?S_'
    'c1QEsBpL>JsFFpPPJ)@5P3D<8JvvpB;4+-3>mZ#bar7m~qa@Z~wl;f3lP4c`yE~tDJL=;J|J~o+@2D9JJ(z?s^Z@&mhLb3`j3!aOgdy$)d60#9'
    'Pwn?MlKeVMH|%-bkk9@}n50($G!DIWlT<al_r654w%0liq4#Q@g&B7{Q{yz5sys-q!dwM&nBNo@Pw^D0F#w}!gm)b^O-3-#ES!Wx9Arnqnnaf='
    'jkwdwuhTHQP9~$Ggo5f}oP|?d;@dE~x`r86$BCLHS+1^=Aryr!!+Pc+HcQ{4cmach*V7Q!6_(`3kDq!fnIS|#g56JhJAesiNjkhHBnmSqPo03J'
    '%5RhOro)S4sxbEKg+J_{oCRul9S(11NfhTW!v}Ge2a`z;a9DMxY8K6)1N2IDKdE>=h4(X9m>^TxFoFm8_a@4_6F`bs<=BbV2i=ae&;*RJAW5fT'
    'nzc^mF_qC+?|<We>%Z~e`biuTa%25&nk2dUYQJ}WPD_7&Zm<1UPyi<{0~~{&SF<^;UV>Y#MN8`Y<9QDF>-#F2&XN>XH;xmYr>v!4rB|~c&B6|#'
    'Z*~of*-=r_QNLuky3-)P?x-Zgt=du95`N^cD?92Ih7PL*&wy0eUv6=jOk~|N-Q;dEhowyU`wZYg6Z+dnc+~yid4@TTXZn>)I{a@o;`%iI9xmz>'
    'Fj|9|0&?PW$Ncg$SnVLSD`m5A7{LNboh%vW)8MXD0%pc^JI7|Td_IDi_VY3zoGxAN8fp!1<ogYwglFvuGYj%69PLhmEQ`ibnC^ZqR~v=H1WIR-'
    'T@lthMw&A^{k&H$J`Mv~-0%(%i{{+R;kT(cjz;6puo$qLzv%hbbte&0G@o~~R5(noqAZVwC*c+Brwrhg>&RZz#oYkNNN^SUu=|D)7Kces5KS&q'
    '7#u(NlOzKi87A>~PDKEr(&(<!QtSR3Z)s3a9Ps`&^WiZaCgGTfWx2UD8T!F|$dBzxpK*YGL5!sDKc>lic6vP@k0;?N5cRxVYesx&h?4+xQ0koK'
    'h>hv!6jrx7o!Me)W|3z0^z&}3m8Z)d4OF8PQ3@I$5nQ5P8vEl`csC4Zx!R{c&{?j_rTRb><B(p&v3_dpAMG3*`tJ`8_fO&7fIhp<HS#)t5?)2Q'
    '9|8%V_*<Kso8B;6I32}b;m2mXQz}2ZjxykcS8J+lnJSaXbY%nZtppy+R)D{L`?_2jAz!qBk@|-_KkOgkXzrcsVCwYLCKQ4lhd;OQ=aeDA<F<ap'
    '-w&;$4}1HE{?Y!~PapPZIvkDAK)W<XW0OB8=}4V^KEmI-z_R&w9V4BemVn8o=pG=^U4&a3SuDQ1Xr29ZvVZ#1hr>Pp$CHCSXf&ST)`I>HL)Y7U'
    '-BI3K_-E_QR=d?|jl!|Yt^;=jthvprn})E3V@-KI_`3c28!q5ohj*js3M3NOs_Y}Jb!So?5u;MKO&?IK0p1tJ?hLw~UOH_gXXBn3C35vk;E6XV'
    'T}42j-AORL90A`NdubSq+%5IpcWN7!QC&JtyH}c?Yi~Y-K87wqXqQIP(CuLmC$WpPsXXd5^s@P-n>y#&OPpZ^|9~9FC4q=$;TS{-x83$ox|<nz'
    'NGC_34Up(wzo@LDF|Z5|cwHRG8<2BC<v4cxOfARs*J@7vsjr+EFy7SB%>n{6v}z5{er-nQ{!7$8zjXhlcb(_6>|c5t8~=HGdwcG+&rf;SpH2?X'
    'KK$GMF^+#VnDfiQ`K@=}eR=Wn@BHiH<+(Qqx?e8tw>uB#mjsJQ!jqt#kM93wsvUniYN%w*6U@O78J9XGNtq`A1dS8G6%g^Fs|bZ3L|iqBfI_1v'
    'PE;<4)jS3nVFsJ#=O`W}x0#ek(j^Lq|8elqLx}|e;xzj{@bKLCU7*@yr0z*>p&!2VNGj@+^x)F>^5-w`O{5a<Kcd-t+%cve;f8tgqKtQxAc&!|'
    'M7GBoUX;-nlFk9C%JpG1Ze@djxD3Fs;o~HK4+79=KTVU=9XleH6O=K2=&5@H?ZK&xBo3w^rT3*9^f-xdXIXuteh*5`&tMBrK>S#Qex6uE_kzqv'
    '`Qi=$AOS$p)EY~AHAyaA=Y_-jr<j|Ia<5oA$K`{b0rLt+?tt}#*Z}070~nAs5)g0Qwim{u4C#{V5b<wcz<$<?jwHv2x3cINaOUW)ZIdgQDkSRF'
    'c8b1GTSXfij(Vx#D5}!$%kzc3!k~7!1fBN%R?|9u4kq(ZFW^2XrbARs5FhRt;6q0`cu#*GJhhAZ1&ZgiTRdxBabjQ~RB*6abcME#S!p7{ShTj%'
    '#d4|*0@kL&{^b&sSGQ>A&+6<tgaOY$yn)`L;;Xr^+CBY@B3TYfT?V`J*Li{i0^xHSpthYwqmYTB1??b5PNM<^+Eo|C{P`4!Jv0oe<mDW8gh)pO'
    '8wF%+2Zgrz3}kA+g$zW+%m(-#^-AcGBEftNY#QWAl#6L<yR++MhTOJ`qH1>*rd^;J$hjxMGMVS7kPJ}`w;+ZAX@gYi60|fF>~ic!0AwC|akz7O'
    '#-mg&NR^;-BF(yle*t|+6>b+D1Q9TZQdkkk3{BJ#J;E@32lDBJ6gub@bQ8vMLzK=b(GbEBM89>iP#N(UTA#4O6MpBAF$Kvr^lbtXh69Wd$4G^('
    'rPr8EFpy-?FF-cUM{Q<@-NLo*&;&SC9+XwLCJ#9s2Y1~hW{|O$QsXYXM*(io-7@g8oa=1Dmo2AV1B4rRLOYoIwbi?z?#8-$g_str5fTZgHT^im'
    'i+U$g-iIJ1AW43YkFIV76&6JBtUsW+AwiX<=c3&VX`sL3)-JsOnwu|Yp&P|{d&RI2$v}iKcvjhwJ&o#0?+IDjIc+i<i@R)x0!an5U=Zq@Qj17G'
    'dh2GwP~j-?en5J1@PTE~jtXNo>Gcg3>NkFflrifYtb9bdlm9?}yM)$gBz*_>4Zs9e;ZMUUkn{fg0D8M_#(v&8IX*c4v1g{MD+El6EI?8pu?{|l'
    'jTVmHDQhR=_J#M8$t7s-#WdBY=)Yllo;fulrBO#apf^FOOIZ3IaR=0qNHKtCqkF(hnt_p@f;a|W06AS^gb1VEUfD&i1~SAX8ZMolO!b2Z$dd$m'
    'jSaZW_I$MV$Y##cAxt6*kEPG_HJFd0oRmKP<d8wPR1W%#!U*>E9{8=J4N&krxTtiP_g>LWXCFyz$=Yc|RwRRf3M8+*BU^47+^SbymUgAe4QS-f'
    '12saBfC@25LCQ1cd1iAdnk<k5_`DN=Bw0$jCC~6;Me#U6rk@TA<RWmz>eAE<Wb79M`ifKrtJ+*<%!0*wCc~uLEHp|y@TrXO(x91NpkOedcvR;I'
    'W9Ob)edv)no;r{P%M4+5NGiB5+IlFc7a|>RLD-W{ozaLf-oT7dC;@pT1JM_RJst!vOd9?=2a0W$QycPGlo+xh-r_+>&3z#5NI))=(bB=G)4ya7'
    'iP}em-=1Mp7iD){_)&TBC}JxTwz0t)XROS;-IF*kGi3D>4Q7D6Im!T=C8a$Y-GS_5+V_ym<AhdK5K!}e0YMdI2o@jGfg|sLi_x(4V|C=^CBcr{'
    'Q1moZSfMDGu5R}w>htg~RxcFzs)z#UVX4n;^__ZC=TIoBM)5qX))p39bRQAJsQsy7!A<f+a4S2Mtp-+MBP5C5QmYFs7uB_-qfpeL4j?Tx-)W24'
    '6rtBhT8rg=2|7rr$UH)!XY9~M?rFri2Q+vnuM>XPTUFChKN<n~0}#-^X)+4%ND1j70mDuclJ{uw*6<~V4~3whCUP=c%sK4wwby}>PT|IeszNs%'
    'Gg$~<u*|y(LA>L)P{e7kg?okOTr;j7X@0=#EX?wV9fK;{ekc|cNhniRZr?Y{;4p>FUGt`GQv&n$iImeKl=r6Y12Gz*WBM%{?t~}JQjq%MARdPA'
    '=-@;bo)h6D61P<I8Q^OeOchQ7pkJQCT4|CO3b!grx%iGh%MPScqzP;m1FDfS=Qx3u13boKuv)2;w*IX45+aSGC>x@XWsXZTl_%;Nl}b4`_1FLp'
    'v<%2Ou>@QsI_Nb@8*5_OH1lDanmMI|Uk#^w{r1}}HY~In_~weTB~A)O3eG-{Z{p+@ZS8V^C6s_@uux#>T_X*KwKBFrpe|(Bw$%rmdnh8jq%eJ#'
    'QA&C-fT9<5+E62mLWB#7f42XRGmECR*3;fsz4hQ>yv-D5jMEE5>A|TG8hHIAGE-0($Ei@4Vrpm>RuF;;ZCSD06>0;2S3olq1;_OidcwF<+Gp^p'
    'go@ZDSx@c(lbI20dsp7q%A4K-Z2~Oh8u!StZrVIO3s!$`I%**(>6uF20=KVJ1&#)II)%Rvb}K9(GF|uC2cXmz(5?XZ&E&yS6Rz6rl1$<Tpr3$*'
    'euX_CfF4cY<<cGKm9<pphW2AaaSMV}n~ZWHoZfaF(lAZ3Ob^76AfNhE&~>-mOIdH=`7~|uJdS>yqfz*KwFR4JK6O_&#p8Q^Q(SLNTd+qfRb?t`'
    '-SnnW?BWLAUPr&l2ZlrxGzKya=XKAt>5PTxO)9jdNCh(VT}Va-j2V%Pxjo4@uhzAS8kGp;(|Lx(ZaA3(E4U7(9fb#b)R;cj^E4A-yRL9N&CW#A'
    '<yk?WkKB*qQF!N$(qz^@OXndVA?<(_-)Q+nTcug`*^q=vYey|R<crcFHy}TYzJz^myJHy=^t-k>{vbZq1&;R%`jW7pOh=EmsMz5_7EVT>E{4D~'
    '&|QH}D6w603L-O2syfauJ}leoCixJjX#-f0qOGf7Hm5;8yq0-mjMU~tdh+tbCyxtZ3NTk0pI-)P5MPC^=|REvf*k0_jQACo`DU}!s~E7!CvToU'
    'JN3LYRbHsA&CR!_>=l%#ox!^?uSVA2vX{X8>X;^S*k*mJJxjmC1%P>H05~9^*6tPCNnIQyEi+GuMa8b4(V+@HtnL<!9}mViUs>8bpxNc6?$@>S'
    'tLXJ|pnqTJUKVfKm2nbk0xVr%5Cl-mJY8NEW>4A5v@kHPnJ1)U*)GWhX!xxZ7(@{?jKZOYK@w0znorRUo}jAfRv#c%inR(XJ^^%%rW%hy)SB=K'
    'TXKn~6Fdblle`Wr9iuCi^yM8pa*-rGOwt)Te5G8e5YioeZVVNDabj@+31-k4?XTAN&|(3Uu)n2odMVo>3(_=LVucYJa{I=RJL3Yw2}o8}!I>^N'
    'o+LpIL}wFi)b_?+)?{s-ru!R^q_C1p$IrspVWMj?%D^#RFb?@g@CgbEU35GEA~oqv<1G@%hC}>AZaMUJ#((0p(d|L}X|TvF)xSctT;dDuspwoh'
    '=rQ4X#duse#*<t0PVf+s+Uu2xm`kIUbIIP0k>NE<oLhIMzV8=H)UH|5bHJ%FO!pZ6=2)ZZW`v8jgaXV(`U6}E5QfuW5l)2e!MM=G;%yo#H%U66'
    'tRQ-S;e|C09ax>Q1b23O(Cco&ep^>!t7jV@&`E0%nvqSCw)(4b;o&=IkW4br?ssa-+mxd|_|nHur9FzM7B5`}R@rYj8qgB>n>P&u6boLc;1Upi'
    'z;7?~>#mwrUX_^ay@g^LEi0~9dV0TvY2uHf1%2voR(16fdTLO4tTxm%xO0)T#xPssS*@Pnv8gh%s?4xTT&n^IuZwFZvZwYEcZsfBb30=owK}||'
    'bgio28xfg$7iUF{@$B(gz-?;E$B$2e-oCm{$vsZGWD9dkrtaH}b2j~{02X4P;jVE1`#mj$WclX;@Vi|mIQq*{qpLo5VD~=XPzvrsPplF<=y_}W'
    'l84txG^De$P@R7Gbh5jzKJOeJ?CqQ#d^lDgjt~E-+?X7s#UGmF2qqwMIt=~E%DiDvLN=Qhw26e(kq?%yy~?0P4%bM3%HCySJGd0NX67LtU|$A3'
    ';4$(QX()W9*F=(sfQX@0tV00WNC@rSE;|fr4@d;OLX84{0p%c=UFE%QlgkE8Ij(>|j9gMMGnwl}c~<yahheu(iN(O?6$?3=6*ht>vuM{iV+m*0'
    'Jn;z)^bJj>^nOvIZGx1Gs;<PXV97_JkHQH#YVZf+AneU%W7m88Er=1)LKjb_U*Lpf`N&;>R4_`$eR#J)?|w~Lja{_MbeB6pW9V{Y^7N9QG|orU'
    'XcDAAQdtP|$+p5$D*qf_&*Pi^_Umst0vh!p#C<tGG&;-~*6)s^oQ|B)A@WuCwK{v>J=lAX$60s4SF<P%(Tb;47_9P6#X6EeZx&J4EJa2ys_B@u'
    'xTqP3l!f~vL-xs{6TG4MeYIUX=5(o6r=b$c1w?ns!)2;NwQ8L*Gq<i+<?cVPi{m*6LRA3Wt!H+55#r%##i82`;$X6bAAJWG4WzU9Dh;MSsb+om'
    'RY$#UlS8~e0<>Fq>RcTV`d?j&hj>0_-I=Q2(S+e14u>8fjBg?u+<_R!+6DP4X7@r@XWE6e*VojOB^3b&qiJBT34ubF*I76~qSm}7I08d1X(_7p'
    '%1g0oC5%@fX~Bj53#eBGNW0GVSB%IoNW}(0FL+4>F4W&;GQBffwktPYmhIS~OGn~}?Hn*Q!c)+k!<^7T3r!qsrC(qfJV?b5oIqU93vlB}aCKFS'
    'ZF;AYc(Nq#HtH(E2QV3XW(7HVfPup1m4IOZ6l`&{UY;=hFEcn<3wsXwT#wid@G@9rP`}!hLoHnShDw8qr2^l9DAdr*#Tw_1sqXW0+P7HTv=}Uj'
    '@9;j)Q+J9cZ~owjW9by#^axGd?pBH5yWEE9`bsS1a+(1LH=!>PAi$zi)UJ{Ici2r?{M1XNUT@w3J2Gdy;WFN;<S=Dd<u>VfI#1hL@S(%V1^|m-'
    '9e_n3(g|~k<}QS5yC4(@NXKmhDJ)Sw*wWht<=Z9}hQGYGZ`b?YU^zXd{nxKIH#Zw7A-SkWGd~y(F*vdB%;N};0v_?}X(muGoZm~?x1TYlx0|cR'
    ')ZD$>O{Uhj6=F}ABt>AhZ?|=^q6py%^s@4rF@UQM1^RF`^dUeWZtG%=(1$Ca8#VSeN%4rvy@io;Z&9zHiG{aAc+0I+m62<T{=hOE;^hty!Vrl('
    'HWuWtI$7$hKA1Mu5d7Bjt8HVis@06rgLB5KC}PZ=nEw1FS=#8CPeJZbg3Paz5!<(g97JPz%T7bdRT^GdUZ@-z)^CUkLxSbwk!SxkV$*Ja%R{6}'
    'CJV&aNhHYSR1Bkqp|(tiF9Tv7CT_|VfMnwHIbyW$Tm}<72p_G6YInLh+RnwM0Pv#!O3z5&$)c-i0_!Z21_D5QtW9cJNT*_;o8sY}PZlhf=Ddws'
    '7Or(u3B4RrNS{N@=KgdtTfK_-XDOonoj#%%0e2sUWh9FIaFz_Ov;M11G+~W^DfYLv-*nJ<AVNu?{~EVUdI>6FRS7r=@f4n7%_z9Gi1;O&D=ZDh'
    'R_DoVhQdb@*UvcA4`}`%>A_Y2B+88gD5@60vh`O7^v>g85rB3BLX0d60Rbi*&w~k>*4<*=iD}?u5FQWbqrk#qu81$~+CJO4HH1GaGxvJq&|E9v'
    '$LM$29bq7P#__!@>YRSwtHd3PZ#zhJXpyI8gA9b&I7_S@%5;`L8)=)n-{(IfDldWtFw(mx1Ifdtz6c&Y70PnWh?a=SxUd&bho|>w9DAqXuXB`Q'
    'gNX}Ihb3wzGBxfcMr9Q4V(%n8{N%242EsynX);UZx$C{+N&=bTXn1O4gXDy#0xbcd%4pYbg9PZ>!=u<_Ghy`P?Tmuy&#sV!0G$JXdS=j~nxy?V'
    'VfSwxiH!G0;V@YCH@(g6!rH?o8>V;gI~W=3gA>JLG*aLSbi_|EjjvcnWisgji3#SD+=urT%9Bcf#G+7UH1%j#cz{O>;+9YR>9`Q8k`xYNuAHMn'
    'Bi<}br*qyDZlPo^Q8*XNB?gsy!!VkZzHb1jYF7`Cc>c(G1|kq{dyo_o9~8#W(o0R;6GsS8w?WlQ2iVPupCTG&8T-cbc*sHf9(&Krfj(d);Lp-W'
    '!si2DjPeXE-#~Pit3i6`M}8Zmql((Y1IY5w_P*hHMZ>1`lISdBO)R6WOvb{Xl`t(=MhfzEUA-{;mB_M28L9v49B7-nP~Aovv#f61wph`~gTyVL'
    'M=<>F*f7~uTVc22P%Jo1cuR#lmP@NH!03$*CpSH{SOD6B6m*TDDziEMJAoB<3s)ENb>V5GiavSKR6Jaobw2D!&65hp0q?)>8(vv=)Vxf?;6}3?'
    'tIxpO(4bW2Z`?QXh_^ae)Uw453lY{MfmXntgak5k_8~vJh>=005kww73PJ=R>WLiY7V>Bu&Qll)YNN}XHPYpRwbGK*qO!#ZCS(d4$u}Kkgccd;'
    '?@4kXp=D97_CN2R{Ii-H1oxpOyqulv92~R6qSdkdH>zXnl(4cUYP7MfEmp^xbuShqutzoZ$QBW(fUOwWtiSS6W8H`t6k7Q8I*>Mr6qC6w8?TLY'
    '1;rn6EjkAL594HhbzK;!^f+nHqP!%DSwy6Rrg5sR>=wX*DE~!FdNs7#AnL>1i|WBpzR)ZItjZ+a=~?<hvxEh#k9wBA&@5Fx>RHMt&jSXnloCI<'
    'rH1KMrEU0R&Z?`cD&65~Ru$;c$|~H+S{3N$sw%QcwPx`}Z^$otM}DbqNuxZ76kuX{pkWY8F2ghr8h19Q*qBkGm@-HKxQ)a=A71z06zYaamm}$#'
    '$mTf4NY8vu*prMga(w?E2d8HT$3IH;o}b<~CGfG74N(OCR96J*IIAIv*IPz<&pQrD#WBh0#wfXsk&I*LD-BhvBs#fAk<Y_Q@1yC{&OA@flA93o'
    'FTvd5;}HURpQJ4O4nwV>a<B?3$ean}fM8H}(XtC~tK+dsxX!9nL$m?AU0K5}4Wf(dg&ro@aRB_NmX#+rfQ6;ucvu;ae<jJ5i*Bg}HlTD%$U`pJ'
    'KCZAOvS}^g@foQ#Q`pw4?E#|_&ss#f#?}i$9bLtEP_JF(Kr2LVu8jNmyQTVV5Q8dGy|=m5HUaOnBhYe-j_<}pg$4PHeb7oOI`z20MUU#21v05#'
    '7GPE@1nQ;Qe)YEPNi`Q*Da-nD3{VsHBXp)~6zLJf292;L(8K!DG~T!rNR$oGMK)?*uvUSsu$>)^79)zNEU#Mmzns|zo~g4P6B#V@;3~f^oqV!v'
    'oAsSpUI|F%{*cgVt$E3%8j95A%EE;oBQhFAK&uIfE@G$Ps>IEo2727K9t~3ORihHE|64}kWP~kPT3w;iJwu@jDSX;srO<heiL@tvk7mXK=8G4g'
    '#(U3FmvFC$6-{uxw_c4RHw-I|<7ui@!xU5+E5rBTD~3Uc7|prF-0~<<aqOV<)ZY?|Us;|8eB)#uwF6}9go&FgyA(1-nxm+yr)Ggcy~R8kR|TG<'
    'R>c%XC(Hg26s)i;)&o#WoM&IUjIHgy?ul42_2ZJfIxV}h&RHTSKSfo2h$L>46k+d+vD+>;xVbQ%j|@#aw}eeqOZ{0{7r7UbCVE$(7P)ENu5vg*'
    'CmYmzVHoP~>K3G89Bo^@c%io2hB2@#O=;&`PIsnHn}LFCsSS10F3ioRho}GkvH7g%663eB{(!s)fg-*D3e;*27R+c1gOq&lAYM@D+ppx)Igcr>'
    '1yebx+a7tE%%?Cyl;|HC!EuH}1ce8TIr+0~Z*EFl%dO?)3MIWeq<Z)xn-4KT8lTbG1KzkP`Me^4Tk(0Jo`{ZiimQhi9ytYFh8V~GSB=JH!R^A('
    'N~{00^9gTj<)HyVA$vcIw85l&Hhb@|Rhzeh(=&6y7$F0#hM()PnOfXT^}Il+LS<Q2HVNqeJDUW;u%bzEo0a^W7-NqcMSI-wYvzg^w8?LblPqp3'
    'c-gLEm^0>JJ`m8-oYERnOjT`lQ9N@Vw9bJPeU3I0oi>$2Xta6(`#sZ4(eh%tO(Y3yNvGgcQ;0V2%1%lYMajMnFbdl>ylpfJ&Lu-cJOF7C(Jbo}'
    'sZD*IwqO8koBv;UiBPBxb`xsFk_s|b1J`8Jqtth>Rb0t|Cm-4umh%WDO`yRKIOv7J;z<Ypx>ynFQkZ+L@|7U0zC8+ODZ!T<d1Qh|DlYNm&;+y)'
    'U88)`F9q8G7!zw_SCbYZklQ)t;leU}*&hjX@$ovbKVpLyb`~T%t*y>!D=2V`Mk0loW>Y1YT25%I6V_Uh0-ilkPU_poSq83Dva?|{1t2O23KZ{T'
    'aM{OGgtAz{7Vue;NT@Clco1rVvyoQ^;FYK<v^WjImG~IdTmlgyS<8&1(J`H_zNfwx5PbGys&><iS3|)Wr{=$72xgidMX0(f9zkr3)vv1V<kHLQ'
    'np|dKy5JKy?a}lO!)_6<CM0^<O}8y2pl%?R=`4Bg>E01<kAv94TNXbg^7c9@Pb?@`%`;x)1M_8zU^~}I+cx&<45zk;gWAeN2mW7(@7mDTuXzPk'
    'S9uZC@S*5_Ss`FLe-pw+y}ge3plDEA8+TQbP4c?v7{7uB1F2g|#Hq_?i)nM=B2%Z*X4ut*%b7dHlS5ayC@=Tj9UhLzoDE>W<z!Kd4h8yH?iP1<'
    '<mm^qTOo!po$w7$Ea4A&ZqUS430-Mz2b9jGyut&W;su+o(Gzlzs*GUtdzZeWXmbwqR6RNcqcu}7;wc!>6ws8fpMs{$s+AdODza*V_z;4e0fNbG'
    'u*@7qA<ETn^E}~vI92a~>Zw~XdUL)}r>gadcqGkJeLdh!eKzeydIaK%BNnFY#?UWDS!+7<x$mmwIsLrXJv#gd+lqY95xh5_&CnE%Szg+_6lGRX'
    'TD&I#Xu@<dSo(#DuW@0j7V1A>ZT?G^2Lr&Ghqcb9r|~~~4K=t&7ExQfUAClaQ#<U+N0;xBs>Q-B&|*OG)B1;~6_UK8VpC}1ZY!p_Og>CqorYK%'
    't}*3Op}bnis0Nq}ii6|7K0i-U;0rLF$JZDqPYI$5ky(UvNtA?f^pz?E$EmlW1q4i|K+qn#-``sTxRuQ3&*~#+ZQ&3kvCzIqVVI&)OJb?EH)J*c'
    't9Kw#8QF-_<jZXoC4T@E7DUCXMXfciz^o};sgM(9>;$aWrDiUbs3pl3%U#f=NXqDo&EjU!Q|`@zSP1;tylA)KA3ITMMg6i8k^qvR;!AtUF*A=@'
    'g-WaN_y*I-M7XJN<g@h#AdxS_lk4Mz=nz33^8f){JOM)NlMIi>9FjlLXEpW<xbgw*eop}=74f`YYJ#XqAPogGUkYLON!{f7qUMaOEFqv2q1mYf'
    '@+mNj1y1Lj%mp)XW%(+X!zW}^bc|WVqDT*RvfLdNhgXy`Su+2juP918pc;`Bq4=g6bjH66D7`h>z7=2Bu&`6xnr%P{4Ud9f64a;!tgD-8vt_3U'
    'F+%^1j(YLJRDWnBXU_v_i}wWG2KCTdv5`yUQ+8=ZWi<hn{d)LiBLS(1ycFd?+5PaacJ+9e(t5AUeqQJaH5>F;cPb03WRIN<_*1X^ZK`!G+uQ>x'
    'TLZaUSSo+zD8n~l3^tfTC9VU@R{|3Y({CVR$}vQVR&MPca7^=<g-QSJtRYmCfQ?Z?G~P#AsE|wqlq3^ZXQkw*)7EdGdCwqr2%(9N!T4)bN>ixr'
    'Gf3Wx7kcp@%FAfk2mh;=^68VU9I=0h*|l`lmGd;>Uf~xw0l0<!Hx8eiJLO)mXOG^(rA8!}rO5Th&fpBa*57PwZC-E+oO}Ei?KR~=KlHSfS`VUs'
    'kKKN#Cqx|!@~Y$heW@#$b)Ig`!phC)jQQBclX-S+8Q;w18&9xpTZe&|fD0){NF=%xbiDL!o3D9k`3TB8>y!{J_6SWV{&g_jz+_MxIuVqW7G(KO'
    'VG^M*CEJFQK;_aYk;gX)d~|j7Hl8S9j(42{u4}T)V?rvxQ1%@HzK@q4fLc>X6?AYQ2aZLv<VTnSqgT^;J_pLbR76cOOzBa|VZynd=F@JT04Rx+'
    '<jluiOLk!!bVnV3IJ1)s6b9&!24MBU(=*2hYq)&BDHhInApVVF+@2M;wlX#tZF1TF!&tUAuN$2wSN89g4j5?Cn&QnY3+Rwa(s&*z`IH!`Z*`2+'
    '>q<xy`T9IYhD=n|<@hRF9s5*-Et%*sJUoDr)WXj6Zne+A$;G|u@PKAf1C6H5_9Vr!9hs!+`)ZrDJPXv@kxm<;{qWs86J{!I_hj9n&J~nJy8-}F'
    'sTL|??5WIzJS6TZjVmTg)Tuj%?9?-#;<gIKflCrf`J%I8AEMD+CCvvw9B|G>_A(&%AR%-Ywfs=VzW}=D+ZhICqzSXjh&9-a!s{Tsn@KG^nyJJk'
    'ms}LOn0_d^aycJJG@Vb~StT|aQ_AQ8K`*z$?$)O9=x_33kjwMLnL*qVTxp17YWVkSZ?kRrHc-h>htk=^vtJ$>ITCQ<gPJSBmZV}@5KUb*Qx@>@'
    '1m)FgHm0np$C=a&vT*BoSRP$G2XDl?-oVL!Bb>s`z9abpt)3S-SnzUEEAu3fK1$Cxzm(*09b4FENsP^o5-BUW`Ek1Dm-Y$faKcGcGCtu$-0~$0'
    '$ae)y)~=?T6;&wnmSyXmH!D4ty1*c|Dj=x+3K)L(7$_6>EAl6<1QE(kk=0e^rV#N=PSIL6B_L0Hs3@h}(p;4DlX33OYWgOw?q@(MiyTJlz{til'
    '8I;<_#|G4}TK9DB2Wu2GQMHa3OGPn~B}&UC=_pQ!=-t&ao2v%5<zI$?);O@~<^a=gT~OLNKNzzo(R54Rwkqj<hvk|&YZX?<k9PgyTSNA(G8k53'
    'FLbuM5*YohmUF5M0+Uc(aK^h*^^kMkNh(2<j_D}B*nXLDQ<-MAuSy~Do}z++`coIZ_Yn8qp$GPtFc!B$vfhfPGhJMS?H7r(;nOrBahF6&5M;}O'
    '=P&{IeCwLI!NN$Bqb^9onk@3hc?ZgV5{yt_71up9VvG4FqwrTC*AvX#My@nt&?!Qcnk7lA#MYpkI;HDUaeTkd14>L%mR2!6cE|GKV*R4Si5-1W'
    'M(VCKe^z{T2WCr^8$2_)vzc#@M*jj-II$baxa9g{{1uA3q`6ParFMy_8fy&i4pDpx(I+e<uS45^MOJxPa$zYla2VL+juMgxX1^ehj#7#6!X~|x'
    '5rbxh%`ZTi#dC<QBY2`kZbOAG;b7bJj_4WoqU1(o&$0Ggtt#ktI`3@Ng#oR{6mx6|0a*RFsP4lf*S_M&vFWQ+dzK+vGCiZY;Pka3x7_`<gYU5X'
    'rh4ujqV+E%k;p#tUUB4&IMY#b?$2l5;`CtS2Z%^dHkCsAl411OE<t%Xa0<qPbgamMWcdd*dInWy<!RSA*(aM-_n}(iv2s79XRMC+1^&^YB6iTc'
    'vSw!>wJr>pC3y?Lj~RQ{hs_HF>uI)^@c&|GaXxQOnhJYo<<NCW?ZbzSp|@J7vw4TVy2g+1aWL%o*9e511261dj1*K`*PbN$l^wD)M*>8Y%~K`D'
    't0)7Jgw)=K&nFE<e*e_du92lw{ArQ3{xML*DvR`%-V7y|rAREPw<LS7W$%U&KL*m;G6me?<zstcB`V{%R(xAe3lU3GPK!jhOXRT7c*mddo(A4e'
    '05^aOY>?KWQza*H!wps);@2P_TvDDe`_4l-xmvzL5}^~jH?YJkPkp+4C|d;bBM(JMF1dH2^s{`mk~v}<#P|f@MaO!@pi8f~k^lKDlCHY=pT;1O'
    'lg-N7qj0sFJ@;|H86u>Qa)qaO&O8lLjUy>LXA4CMs!hH~fN^~busrUXYfjOitUqlRR}9F+;CUpzLT<AFzRj;W&AWb|oA-P|mQWId8M-H^m$ho<'
    'doJHi_59kvtcd_d2A~|d?Iz#qj@8|^2B=$G1Mp9zxvd6BApXCo@gz}0uOG7MEQ2|5mwh8^0LR`g<tK*;?pwGd$y{W`8bQRNySV~{wM-o0Fd=Vz'
    '1kC0c{zYISSPT~voo|EiP^FoL+=GT8A(==BCZP8ZF|IS1`m@PA^Yvx{LP@Q6492ZrEC#1*`-{wn?MZM^8phWhvl)H^B)%Y?X262;#|ue+w2brw'
    '-cY##=?{^ODw2^!Wgdl7_yt(3gRXEY45*L5DEl1vvikoWvesiDXN=m#yj{}7nYZ@z5lwx9@J9Db%q}&buE=p6o6~e`{$UW$K2UyJI!*4$<vm!5'
    'Pkt-r1Uwoh3DFb)X$LP&ZP~}l`mR%nJ-?UO(UsRntvyYCih0Mrcn2+_^XU|%P}iJ94-*tr#RM!?v(dPINi!9<qhPJLD+*Y^XfvGn(b(Lb#IP|A'
    'B@Q^SzC+%K)zD@MNkKA-w*?PFmmt!*ReYe}1K#C;-`h3;(sz2{vU9Ugq+Kd9z}tYsz`AIV4r-v=m7BGAGAP**PfM>FKqDpO9ytW@#0xa1bf++8'
    'L%Tx0rZc)DgaHGsq`AU=OKUHc<6{Nt{&ZcbVdO(AR1W?=QWCpDMe&Cz3u~-NI#em4T?sLuoQWXJhOWjSDN6SVV|F9Dh1qp{JWX}ZeaCwD7k>8_'
    '``usp-CxZuGUI9PV*0!YkCgg<E4ShtZ|pctccR>h6@Tp9-}StEuKh56a0Duhp3+%R_II-`rkPho(#|SmtWpnPrjD$t0`MC`o?%?vJhxezXy`V6'
    'vwN-v_W;L!)49h!+ix!P@NUs}*{8hJZ(Izk>fo|UCxWXhtwUO5flCc6&!NIoXsrkB`pa=1y%I;T9y*1xAl@^HL}o)dXFER}?!WPO4iEjK{j-yU'
    '-BbVc)6vn+$v@?8L6(rM+qr>GQMlY-je{e!x*~+C2#=}+)!Ly@<w&P`m<+r$;TM?Trd@+peE#^U%|KTA`@`2#aPlKj$V`^`vU->W*hZJ(RTSS7'
    '9=v!M<QKfTh7^Ly;-)S{=x`47GfD3+fek%W8-s%5B*_7{LrmOq4w4;b{<uE}_(*ue*~9%S?{BmG;XDoF@Rsg}k4N|CbexaZrsl)zhx;vW>s1r<'
    'VL*}b_ovpG0=f5peLe}tx#jhbaRBDDA&JRvXK#P6SM}ve9lFV3SjumsDZ2{DLk;HpE*y1<-KQaF>X}E0-{oXOA{lFdizB0WREnF|f!CS2aW14A'
    'w3_2xYj82!=@hfcn@Ct}z=(F(IXW%smLqHN=%@&=u0ciCX3oLr;5(z^8GEGKF|NoE05qB62S66Wd^>sA1RC#6LG=1Ru;ydLw4RB)MNc=3gr6nQ'
    'vj?wf&8hoT3vW@!+kZm3BFTjBHdQi6JqMNa;r{RU50gpC|H1DEM(RA7kvn^>I$gh^zAi<PkU{US>WK3*{}ls5n4rP$(}JGC)B7zfx4R4f^3H@;'
    'A5ZY#;}dm$-q71=G@_0aGE)%<gBl^*ue;w%R66|We$OUt+g>)uOYDF~4v<b-ku?1R+J#pH0BgcslWuT)&?-^NCfNqRi)@3oErRw$0f$8~tc7wh'
    '!@8)(M-gE{)~7E^0-3}TZ`z#nVA^yE+aJMeE-HSs+DCK(*o}tO)C49q#U+*I-Cj4AHBt#sS`(k76Sj~5h!a!@K2^41af%<2Q&e)GB9Yf(T6G+F'
    '7yMu8Hh2`wu)VhRj{bUhfhOHkgr8CeakmsqzjCa-)%~!ry=}3gm%@vZJWr;)DRE;!4YP@B+B$p(nsdIOd*1F}0Z+L<&xUC<%Zd4<<m5pUU31_~'
    'oqfDrRg3aK9Qo#~k*4@S1YzT9Vt-6QC3*Kc84BsE1LRw*C`N@0{^B7FnmIY%;iVrDXShHQSI{vtWDN`rw?!Qy+0FQ_?(2X~^d&J2NC43kl%J_d'
    'qInx|wySg=dpIeLlLeE>W^H?Fk?9NFL7{3xi&Q83pASy=_s}Q^qPNJ3ctNIf9t&pB%C%qOi;kkpmoYb(GtbA}H%2tKq|CQ2T^kOq*s&Mu<V`9+'
    '5Mw-gAXHp;I<~uu?d0+->!t1aQgU^99+Z;nOU2uzQ*vr4d9pmO5pj>|*j`YL(y=V$Tab%)PCuRO?>#0Mi<Ho>p1nHk?cu^xn{%U-&s+6(dT!Ta'
    '|H}ufx-?M1#<GuDKG8zbqf>;Jg<HCKQLTcv!t^PiG?WGG^J@f)U!MkPB@j-Y(|)$B{d6%qo5Fgmkgvaaf^_|QopgP~R@zZ<oT=UL^Q231Q2Bl$'
    'l>qge1hcy2hk_|ty_gunrky6Sguhx+`8#u+nTcVmfC6RLeCv7Jpz6+veu`i!0SIUByLdwy`9KFlq~Nc*-Pa1e+hJj9y#qmwy}3AMgRXEQkFD7p'
    'p^?{?_<|Y9*h+d;2Z&x!Sg3Fp^VI3I8tq{Fa*a^SjIZWUCLA>g<&?w}CW-8jVOqg_HVLj5nq_SYA^U04@)ITH9SkMNVAfDF4)~fd^>zF;O1pBG'
    '6_IM?zI}cE7qxrPW&xQgA^+wzc%ooTraw0O8r8<Ko3I2-(k1@oYllHPDdiBs%bYmjURhA7iL3}NWgD&)?VNH}^zm2sTbr9NQm@35e74B|hMu4Q'
    '<*UjivBug{t;8m|>P^kq&P!}a4H}!`TGUu)RRK2tK#7Z@GbMoxb{5Uz11!dU>M?_nQEZO+Ep!BbIL)jIAfU7kjcZ;^)!S6pe4$m=9~=@WjXe2B'
    '$$)=*o(!ll6l63c!65(1rVJSbh7oCbkarQ?v~pFxV7?9)=oH<J{SM8@m<ma>?^C{+vio)ZwURPL39qSxn5O&aw8JX9&^pSagwCVt9B}yhUQ>ey'
    '&O#XFa}*itz>thcHa5*M+3dk8K672|h9U&JLCi({JDVkV*+ibGpm0I8972DcV-|(v39{&VG4$<!g&4ZEiL#8b9@tjTQT1TME@=FG=!)8IaYb#_'
    'ETZrB<MSa(XQLG*I~Bxj6vsgKX3_9Q=TFpdKw}?_yh=!A>FBswUe-m+Hd+##txJ15jrWN^M+~hvYnN5&EvInfUniPwZ9ZK#{a+_SJ;5kxMN`@V'
    'pLHQ1A-`%*y|?kILt=qgV<bw*5{PPdK-Hjf=7LC!o0=i(_*{ULHy9D$)!`6dFX$Z8&PxmdO(Q+L1|lkfqM~817hzUd3!g5?{7##n<SQ(;5WGA2'
    'u-o1Fv|EyEYIF1Pq*i8scfa(^tAoPNBZfdr>d_f1wLx&TMe4DWd)KxlcGg&@N)8=WDQktTu_SM4fop}3RhP2Hm`2-}0#mP;eV>oSUNsI*4((D;'
    '-?7exi)R*^DsNCqS`UuNq=6=n1+e19<FG<A2L4?@k*Ad)cq-VkK%&$|vrxyV(rC})d97OX{=2Mjup8zXw7j(`hb*_zD8DYzMvzE=?#NauOm)gw'
    'd7C7s<D`*(DNHhxPOYo-ienQ@NyiM_MBM*g`2|?qA1lCs&^erQG;<-f;B^YHPpDaJZvW@~6aO3kTmOxZ=AW~lKJ1-R`=+$rkJNFT)?d`l2sjsv'
    'ocA#uDAA3@<Y|J*sX*9U0?k%<2CVQx3SMLn>K?>dh!_7*g*P4f@s^<QZ&+UIfGEf<0F*3Fl1Sg%EW1@XGpZ@Fa`=?#sLSfL$)rGKVrFRV9i0B_'
    'hlArYwQo{TC=~QS08lRQ*EaiON0Z4MQ+bnkOwKkSAO9Sh`<z?CxcIK;G3D*TaVCJ`5n5*1XABj7!91~nbg@fU3CPwkfJGRiMc8Zo%yCo1*4h6A'
    'S-UXtilRXZLkBN=u9{`o{LyMQe{|9OQN48u4aRgB2+Jx44;CmmV7Vpsn8K$M1cdxH45h2Y7_B_7asnCZBg_r!1hZw%J3-N?)Z@T0yO{uU&jwIx'
    '(|fzAZbQs2!D^b<`iM4ow$+#uS}I{oHeymTmv0-hO{q;X9ks-=BU_%YD4;YEm9x%*LdeCNrFcNEbw&-!G3Y93NYuUVBwU0OTQ(A`D?;XA>aX#d'
    'nmOP{io?LLJ!%9EX=FqS&_$P)5bxvQ69LS9A*ppoVjBaApa>Og<yLAB(Lh}C(`J`!lchtjovz9;mZugL>LzK|n4ScLwX2P-s-~yvN-c}5cMQs)'
    'cOYt|R>3Z8^=zJX(UeM;xr2r}$3&SRAy0Wzk6TwDo@3_Fd7LFQ$Z)g)B0c;{=13E=EV!a+00g145xoQCQg+0eDWa!NM?y+=ImZd=#X-(TL2@Ng'
    '8crgjX|*&aEg3Q47>-U-9>ekj+X+*oP$<&P<6>&Ko0zJM$z4{^yH%&%6dL;2NqUiIndCP#+@ffQy@J2^Nbg%l8z~mjQ%9`EYHtLK&@^reUhh`P'
    'X`$+)z%){yUP;c^$_=(ak6LAy3TQRpyWo_SU%EzNg}0|6JyAX}yfp(avYn_R<B*NnH8rWK#M>!&T2YKSs<4kD_IE@c>5x2+V`Dj95@u>Uum}>k'
    'U2rGi&3vM<0U8vxZr-kCc?GIt=rAy!op>qa3p&SNyL#DL;~<*MS&uVBkC1x;BvoiVtjTqV?Eyi!46!LqmZ&TsJaD|0or_7y-ZF_QA6ky{sBn^o'
    '!H9kSKx#~qTwe=7ey94vd!2IAasK@*g?4wKsr(<%`9%~0EvHS1m(8hfI_ldFN)tG{Z+nKpO0q0ei5YUM{|Elp(jo!wcUZ4*)Km93m<RI(qXBxX'
    'k&|nTY#jBkM|-fnCupUQGU5TTA<akz=;&8(bDWL(Nvl%5Qn+*t(+i^&`U?pU)S%v0L9LgI^w|vS3Z<hN4Y4f00>mct@0eSrt2y0UG-rs)z@6<O'
    'qrcwi{yvDM!)(z}>%I7j;WbrjKH%&Wl|F(ne6qj$;RFgC9333)oE)6}vjpJu<Ke-XzjJze@Z<5({_)wVwCd{M8RGRC?+*4PPnS@0ype8M6NuW@'
    '6gG8xh4Bk&v3b$o`*5^#a18Yh_jl3A$aUr8$CD4A_m6ju;pfh$y@NCFmkbuI#lA^6&o9)E4YFgVE)f;esW>AF^HHZi?QFmPM!A$lQ^e3rxR(xb'
    '0Z9XSo^f3RIe|IL_cmYoo(&P@TRpNoSQ@JAQSVM`X+s0NO%^2AB+1C(p<plUxX7ZJq0#IZL$G!&Y4<fA0ip;j7t^ufJc(-}bH|oBO=~pR`*Qrr'
    'gJ({%XpwbZ1f91~{d@o#)~7QzW#tmh_Cl3i`FnUQ+}~trtG`vqTm5aXo`|e#g>-Z|`HM*zVWLQxPsw>?re;N=HJw8OUO{W^m)0kCKW_Bf<8H%-'
    '?xDH{<PT^qYvZVadulg{$Efw7@}mXZUg@ml?d!iN>qLY7ZSlfJ+wiBg95MaX<X>j`S%3~M8TSd4`7!BMuq5+|oSed{kGjB)=mJus?bB>`*$Jnr'
    '%oD6bqIR$^f-8`Rnbja_!BTapWtfhPy$XO@E~zOqSwP$V^%V9Z@34=bdg=^s@Tw|Ta;qwO<B<A^-dbn`lS1!o8s93e@t~_AOE+%9LV-A!MI=a}'
    '>d|y%q&obtyL0G&+&TLR*vrS0{qE_fAC3-yoP9V}yFcyk{u@#RNe_ZOrJT761&@-ltF?CPG>kc19NwxxpYvV^xUR*aj~r%i@Q%z_3oy#Cm;~P0'
    '=iCIZ!1b0~)rn4S`g%!qAYQUVv^jNzt#%h#7d@po(uUdA5Rh_0_&<e_fOtcl%Z3w_f_Y&>O4v41+J#E@O)K(al`VnMgI*C0J(P1SE6&OO&fd{J'
    '@Q>3^hrk!|yS!|~;T(#q#G)zP=_d|Om|+a&3ehX{cG9ak;@(GmN?o&wlDO~tqh#p&o$6h9Q_@3x+WUP2=p?w^D=PmKPG;|QQ8{{<c|f}TKpJ$N'
    'ZWqNFjJC$e(~p$vUhbcyCpUHWd*=y?jOVt9w#t?agnia`euf>K+)_s7U+~6e5roj#D^AcNKKUnMl3oRPTd8(EZRny-h}zW2)k*reEBm7JHnphx'
    '1$_HI4nD4KpGB76*8YhQDK>;BDd>?>m5gX7A3mIIoQSw<%yd#|&5VTMDAOGR5?l0VYjbra*qu35ZLImX?Ugl)KtZWUzPLqI@9mma4QK6AAaG;b'
    'AaJY%a5bAdn8+}pu;BsU3(}FRK=3bozbHWQ_m$9C*=z(Tmyy7)WIV2eT(`<tfC98KmMxU_#3hO2_BzyL9}U((uaFQv*2gYi7|}Riuhz`>V${Wp'
    'L>lY8T|LcoZ4Ff7JKPJJ@<WnfwlaMm^o|@qX^2IjVFOH&ij0EVMH%`JZlmGW<lnSWKYT>NdgoAXvIg$l)i+B6rKsO(O^5fo1iB+W+QT*Lxq9o7'
    '?_ddA*$-1sDgG`JoRzV5GFj89blC-oq*x3m%h6XShq+#h>W3&xg3vod6P(yj3L7?`5M;*YAwn=qUxuK8NR9k&=F#$FzcSa{X3yyZkie<$)D{wF'
    '6jj{K+6`vvJIsw&oJ0b=kgp#*Ve_04LW$}7wut>B5R04<1by8-3jlp-pRHK>!e$W;vdRU*sPU1|g~zZ|54q`v)5P5*l6_?X?@KgW<#$o9=`=TT'
    'QH#*P?s?%I2wb%*0!pLCa;PbU!Odf%F6MccO`3FOBLP+OjHaEdg6JT!LrHizKVQxMIbN6s%xt%~8;U335oU=><@5%)Eyl^BJTLc7(-ip0jh%M_'
    'rF9?K2((D~$?1)KS2UZYi+g#a%s2E~=lano3w)}$qKYaal%kQw2}Gt~j7#crp093=yfe@X1_ffiM4qaDT$0+MuguezW$BLC-2*gr4c}xwS&Wcp'
    'g73{ip4wV})suZQ*PnE2O=E1-Qj7sT;6Z@;<doy;(W1v_)KmBT<KaC1KsW(!V4dM{kJ^XF@}*S|l|Xuz6*Fqd%xM3<81_TEA#%QyERwyi^nId8'
    '^L<nhUH%|+O}^g0cwv_(x2hyhh6(11xHmm>MzWWC3P+1;^2>DOqh%0YaR_`-n4~vfl3Fm0GH)=ji6p|FMLq4hMKTsU=R+m6Mp51zBh2QY&sDQ%'
    '>TLe2Z_jMoZI`d;F;~Wrm(qMHMyz#r##r~wpm%`2&_*t7ug{cq^zgI_#*trpLi)sYd*sh=<*e^-Q-&k4@RFXYkDb_%Q|k0I^R43}-|7`Y)^-~^'
    '?-OQqMrPTRK)-YmuiXqI8lL0(%uU%;fo&WG6EY)=)b8nLaq7r!BG@CNPGQENS|dsjVFQ+{=V;GrT;X?Cp*mThi;Yuv1<`1)BPc|A%3_+s`x>vL'
    'kX!SJ&J<)3GnbfMY){J#2$X5;<Iy3{xbH_(9I@}a`YUMfzAj<w*v@PTs7iM<FEdXHbZEbZ{yZPThuO$Gi>5%egXzqJS8;Odwt-+Ltee{zlZ*qf'
    'A4rlWv2AHge0Z>I4C`#VWorYz@kaCMEOUj-L<E+wv~GkbXe&qE^qewP8%SghTkZ9pVQ~({$|`jfSQznAgM_$ml9a50VT24_@fKDEdY##Vj5Jit'
    'i28Zw<oMwDM{4-RDgd-o`_6f+R988TZ8IXKOQ1<{t-ZaIf2xyD$7=V((Z|F6v;Cee<%(2Zg<0fdU^DF)-E!}nH*0nsfYyr);igB9NHA^uit~`)'
    'bm%R`)s}{cwPAeb3O5_;$@Eizh9A_JOQsy38}{4nKZvcY;U!HwS5)s+w4`*`Wx15Rq{~YAnoQFreQl;qGEkB2S(mY5gk2Mx(uEp7Hc+`j`Fn)Q'
    'HPkH+tNivsV++$k?Y#Br*x&td{Qls_PbcJA&NM(HuP$zcdlJZqD4=wbxuyV~#4^dKQH3^&Zcxy2=cK!wPw=#Ss&8{<p{p=qDfW=l_IJr=y?5~b'
    '{r<`R@fn$6qpLYHyH#-_GCzRn+Bi4@$@UX89*|8~o87Et^Mt%^(a=04H)SnWWm@<(V{WFC@c9W;l9D)vY~bfh8fyEZ@<P3|Hi)HVWn)>)v4cl`'
    'K4*vW&rWuZPhqae!M(VN@~W}$-oeg~#~)754t7t=<+Gr%yfBdBsIb5qiY0IE(((SziNE*h<Ke;X&e?vYrFcb4<^WZARJl^tT&d}J)u~-AOLj9-'
    'SE}}oc1^{F67H<n2BegeMxKp3Fsr}_f{O;#S1TH5&Oh2$mhX$n<F$gsH{VK5Ysj}XS=Th-ENJf=LtsoEdV@^*ZrS=Jo}~zsQCmk&;Mi)B5v%J#'
    'r%ct+)gx82V8#9^M)ibkw01cyZKA%Q@wiOX;-K=Jt+8lSmbGZ;_bn{0Q_#^T=l$p1FYw6X6ssCPYgL1ws2Dt_B9Ee|oHbNt%US<7QNY%-w=#Ts'
    'Q_gNW8!_Y3VM2zCROlyf_>ANvY<zV-RB31Q<aVm0_t{OosmD20++Ca%%ZB}I43c_M?NTx?><5RnHGSv!w`u_~H&b?&sXMTwis~FSvzxXhbhI0$'
    'ZD3`tD`DmUrQwAZfa&@s?u!<85v^{>;3O%m9;mS7+c3qmi)N{#U7FN2-Zf2A`%G#EAKyM1*P0mcv+?b1Q_6=YVa(Gy*~szhy=<8JdcT{xy3#xA'
    '35fm{@%XJI6e(*X&mLx@jPeJLv$=tWF*e3|NaX2<7(KAEJnZ5)mK^w@lWSRl7o8&Nf|59O#5WWzO)(mt&;@rsy`+Fiua4+w(?&cm5F5|&Q3%Jp'
    'd4ExI*n)<n;SKtUka_S176sVK9ffc-pKCZKR)k~zF@z^1zgq6_x1ji<lSYQlvUGC}drRWj-m)j?)~2@-_6|k{#N^EaqnO9yOUi)>$z+n;Qqlwr'
    'Y#`4)?I>X8qSp*M1=tuUDzPve@AGi~t&aFH*EP?lL?pPkTbu2AdI8L+CNWxqgM=XWQ1zh_E3{6DGjJ^=JEgM`U>H9fe%c2;^5}<y;}wBNCiF<U'
    'v{IsDa}F8`%PCzcF;6KTk+Pv|Qhpf8ud>AtMedeuJBYA=TwG6EO0bMAX!8k8ywiJ<&53SFrrMk9qaVqb5wb&#Y4-7q&W@y~As0JiKh{@$75RwQ'
    '1JUlb7`$DzMVDYbo*LRI!VV<8GFpNQHz%}8OQ?eX`t7${NO{!Z&JX*Cj@*`8;>X3pR5<6+sZX{1qn(c*4~~Dd7{W?AX1j!Q-^xzmwq5Z&-urO8'
    'Z+R_?-!LdA3kp<nh}3)H(1i(`S6C=_cRug$ob}YM?b2>1;u2pm^~-#8r4zlFW_M3NXYWdcz=;z~xP`_*aTrWPbv4KP9CP3cUPZTPwSb!Xbd!Yd'
    '6EXFDvatF>QQ0r7^?w0;KzRB'
)

def original_script_path():
    dest = Path(__file__).with_name("_qtrustagentx_bundled_reviewers_6_9.py")
    payload = zlib.decompress(base64.b85decode(_ORIGINAL_B85))
    if not dest.exists() or dest.read_bytes() != payload:
        dest.write_bytes(payload)
    return dest

def load_original_module():
 path=original_script_path()
 spec=importlib.util.spec_from_file_location('reviewer_original',path)
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 return m
def learned_specialist_scores(d,seed):
 # Training-only specialist fitting, with source-validation calibration when possible.
 # This baseline uses textual evidence. Image/QR file pixels are not decoded.
 mask=d[[f'mask_{m}' for m in MODS]].to_numpy(int)
 score=np.full((len(d),len(MODS)),.5,np.float32)
 audit=[]
 for j,m in enumerate(MODS):
  col=m+'_text'
  if col not in d:continue
  texts=d[col].fillna('').astype(str)
  tr=np.where((mask[:,j]>0)&d.split.eq('train').to_numpy())[0]
  va=np.where((mask[:,j]>0)&d.split.eq('val').to_numpy())[0]
  ix=np.where(mask[:,j]>0)[0]
  if len(tr)<30 or len(np.unique(d.label.iloc[tr]))<2:continue
  tic=clock()
  vector=HashingVectorizer(n_features=2**16,analyzer='char',ngram_range=(3,5),alternate_sign=False)
  model=SGDClassifier(loss='log_loss',alpha=1e-5,max_iter=1000,random_state=seed)
  model.fit(vector.transform(texts.iloc[tr]),d.label.iloc[tr])
  z=model.decision_function(vector.transform(texts.iloc[ix]))
  if len(va)>20 and len(set(d.label.iloc[va]))==2:
   zv=model.decision_function(vector.transform(texts.iloc[va]))
   cal=LogisticRegression(max_iter=400).fit(zv.reshape(-1,1),d.label.iloc[va])
   score[ix,j]=cal.predict_proba(z.reshape(-1,1))[:,1]
  else:score[ix,j]=1/(1+np.exp(-np.clip(z,-35,35)))
  audit.append(dict(specialist=m,train_n=len(tr),val_n=len(va),time_seconds=clock()-tic))
 return score,audit
def scores_from_file(d,path):
 p=pd.read_csv(path)
 needed=['source_id']+[f'p_{m}' for m in MODS]
 if any(c not in p for c in needed):raise ValueError('Scores CSV requires '+str(needed))
 if 'source_id' not in d:raise ValueError('Dataset must have stable source_id to join specialist scores')
 if p.source_id.duplicated().any():raise ValueError('Duplicate score IDs')
 merged=d[['source_id']].merge(p[needed],on='source_id',how='left',validate='one_to_one')
 if merged[[f'p_{m}' for m in MODS]].isna().any().any():raise ValueError('Missing specialist scores')
 mask=d[[f'mask_{m}' for m in MODS]].to_numpy()
 score=merged[[f'p_{m}' for m in MODS]].to_numpy(np.float32)
 if np.any((score<0)|(score>1)):raise ValueError('Scores outside [0,1]')
 return np.where(mask>0,score,.5)
def torch_statevector_features(X,device,batch=4096):
 # Exact statevector evolution of fixed five-qubit RY/RZ/CX/RY circuit.
 # This is a simulator on CUDA, NOT a physical quantum processor.
 if torch is None:raise RuntimeError('pip install torch with CUDA support')
 outputs=[]
 idx=torch.arange(32,device=device)
 sign_z=torch.stack([1-2*((idx>>q)&1) for q in range(5)],dim=1).float()
 sign_zz=torch.stack([sign_z[:,q]*sign_z[:,(q+1)%5] for q in range(5)],dim=1)
 signs=torch.cat([sign_z,sign_zz],dim=1)
 for start in range(0,len(X),batch):
  x=torch.tensor(X[start:start+batch],device=device,dtype=torch.float32)
  a=torch.pi*(x.clamp(.001,.999)-.5)
  n=len(x)
  state=torch.zeros((n,32),dtype=torch.complex64,device=device)
  state[:,0]=1+0j
  def gate(st,q,ang,kind):
   # reshape by qubit's stride, apply 2x2 rotation to all paired amplitudes.
   ids0=idx[((idx>>q)&1)==0];ids1=ids0|(1<<q)
   u=st[:,ids0];v=st[:,ids1]
   if kind=='ry':
    c=torch.cos(ang/2)[:,None];s=torch.sin(ang/2)[:,None]
    out0=c*u-s*v;out1=s*u+c*v
   else:
    out0=torch.exp(-.5j*ang)[:,None]*u
    out1=torch.exp(.5j*ang)[:,None]*v
   result=st.clone();result[:,ids0]=out0;result[:,ids1]=out1
   return result
  for q in range(5):
   state=gate(state,q,a[:,q],'ry')
   state=gate(state,q,a[:,q],'rz')
  for c,t in [(0,1),(1,2),(2,3),(3,4),(4,0)]:
   swap0=idx[((idx>>c)&1).bool() & ~((idx>>t)&1).bool()]
   swap1=swap0|(1<<t)
   new=state.clone();new[:,swap0]=state[:,swap1];new[:,swap1]=state[:,swap0];state=new
  for q in range(5):
   state=gate(state,q,.5*a[:,(q+1)%5],'ry')
  p=state.abs().square()
  outputs.append((p@signs).cpu().numpy())
 return np.vstack(outputs)
def cpu_features(X):
 # Reference is same torch statevector implementation on CPU, float32.
 return torch_statevector_features(X,'cpu',batch=1024)
def ece(y,p,bins=10):
 edges=np.linspace(0,1,bins+1);out=0.
 for lo,hi in zip(edges[:-1],edges[1:]):
  ix=(p>=lo)&(p<(hi if hi<1 else 1+1e-12))
  if ix.any():out+=ix.mean()*abs(y[ix].mean()-p[ix].mean())
 return float(out)
def fit_eval(features,d,seed):
 y=d.label.to_numpy(int);split=d.split.to_numpy()
 tr=split=='train';va=split=='val';te=split=='test'
 if min(tr.sum(),va.sum(),te.sum())<10 or len(set(y[tr]))<2 or len(set(y[te]))<2:return None
 tic=clock()
 scaler=StandardScaler().fit(features[tr]);X=scaler.transform(features)
 clf=LogisticRegression(max_iter=700,random_state=seed).fit(X[tr],y[tr])
 pva=clf.predict_proba(X[va])[:,1]
 threshold=max(np.linspace(.1,.9,33),key=lambda t:f1_score(y[va],pva>=t,zero_division=0))
 p=clf.predict_proba(X[te])[:,1];runtime=clock()-tic
 pred=p>=threshold
 return dict(f1=f1_score(y[te],pred,zero_division=0),
             mcc=matthews_corrcoef(y[te],pred),
             roc_auc=roc_auc_score(y[te],p),pr_auc=average_precision_score(y[te],p),
             brier=brier_score_loss(y[te],p),ece=ece(y[te],p),
             test_n=int(te.sum()),threshold=float(threshold),fit_infer_seconds=runtime)
def tex_table(df,out):
 lines=[r'% Requires \usepackage{booktabs,xcolor,graphicx}',
 r'\begin{table*}[!t]\centering',
 r'\caption{\textcolor{blue}{Measured DED (email/SMS) and JMD matched simulation versus classical reconstruction.',
 r'Reported runtime covers feature generation and arbiter fitting/inference; separate specialist',
 r'training times are exported separately. CUDA simulation is not quantum hardware.}}',
 r'\label{tab:reviewer10_e2e}\begingroup\color{blue}\scriptsize',
 r'\resizebox{\textwidth}{!}{%',
 r'\begin{tabular}{lllrrrrrr}\toprule',
 r'Dataset & Combination & Method & F1 & MCC & ROC-AUC & PR-AUC & Brier & Runtime (s)\\',
 r'\midrule']
 for keys,g in df.groupby(['dataset','combination','method']):
  ds,co,method=keys
  nums=[g[k].mean() for k in ['f1','mcc','roc_auc','pr_auc','brier','total_seconds']]
  vals=' & '.join(f'{x:.4f}' for x in nums)
  co_tex = co.replace("_", "\\_")
  lines.append(f'{ds} & {co_tex} & {method} & {vals} '+r'\\')
 lines += [r'\bottomrule\end{tabular}}',r'\endgroup\end{table*}']
 (out/'TABLE_REVIEWER10_BLUE.tex').write_text('\n'.join(lines),encoding='utf8')
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--ded',required=True)
 ap.add_argument('--jmd',required=True)
 ap.add_argument('--output',default='Reviewer_6_to_10_CUDA')
 ap.add_argument('--seeds',type=int,default=10)
 ap.add_argument('--start-seed',type=int,default=42)
 ap.add_argument('--gpu',choices=['auto','cuda','cpu'],default='auto')
 ap.add_argument('--run-original-6-9',action='store_true')
 ap.add_argument('--specialist-scores',default=None,help='Optional verified source_id,p_email,p_sms,p_url,p_qr,p_image CSV')
 ap.add_argument('--max-combinations',type=int,default=0,help='Smoke test only')
 args=ap.parse_args()
 if torch is None:raise SystemExit('Install PyTorch: pip install torch')
 if args.gpu=='cuda' and not torch.cuda.is_available():raise SystemExit('CUDA unavailable; install CUDA-enabled PyTorch')
 device='cuda' if args.gpu!='cpu' and torch.cuda.is_available() else 'cpu'
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 if args.run_original_6_9:
  original=original_script_path()
  cmd=[sys.executable,str(original),'--source',args.ded,'--output',str(out/'DED_Reviewer_6_9'),
       '--seeds',str(args.seeds),'--start-seed',str(args.start_seed),'--gpu',device]
  subprocess.run(cmd,check=True)
 allrows=[];audit=[]
 for seed in range(args.start_seed,args.start_seed+args.seeds):
  for ds in ['DED','JMD']:
   target=out/f'{ds}_{seed}_TIMED.csv'
   if target.exists():
    allrows.extend(pd.read_csv(target).to_dict('records'));continue
   if ds=='DED':
    d,provenance=ded(args.ded,seed)
    (out/'DED_SOURCE_AUDIT.json').write_text(json.dumps(provenance,indent=2))
   else:d=jmd(args.jmd)
   rows=[]
   specialist_start=clock()
   if args.specialist_scores:
    all_X=scores_from_file(d,args.specialist_scores); source_times=[]
   else:
    all_X,source_times=learned_specialist_scores(d,seed)
   specialist_sec=clock()-specialist_start
   combos=sorted(d.combination.unique())
   if args.max_combinations:combos=combos[:args.max_combinations]
   for combo in combos:
    sub=d[d.combination==combo].copy()
    if len(sub)<40:continue
    X=all_X[sub.index.to_numpy()]
    input_type=('externally supplied specialist scores' if args.specialist_scores else 'train-only fitted textual specialists; image/QR pixels NOT decoded')
    # A proxy contains no text/content information and cannot validate accuracy.
    t=clock();cpu=cpu_features(X);cpu_sec=clock()-t
    t=clock();sim=torch_statevector_features(X,device);sim_sec=clock()-t
    err=float(np.max(np.abs(cpu-sim)))
    if err>2e-5:raise RuntimeError(f'CPU/CUDA statevectors differ: {err}')
    audit.append(dict(dataset=ds,seed=seed,combination=combo,max_abs_error=err,
                      input_type=input_type,cpu_feature_seconds=cpu_sec,
                      simulation_feature_seconds=sim_sec,device=device))
    for method,feat,ft in [('CPU exact',cpu,cpu_sec),
                            (f'{device.upper()} statevector',sim,sim_sec),
                            ('CPU no-pair',cpu[:,:5],cpu_sec)]:
     result=fit_eval(feat,sub,seed)
     if result is None:continue
     result.update(dataset=ds,seed=seed,combination=combo,method=method,
                   input_type=input_type,feature_seconds=ft,
                   total_seconds=ft+result['fit_infer_seconds'],
                   specialist_training_seconds= specialist_sec,
                   amortized_end_to_end_seconds=ft+result['fit_infer_seconds']+specialist_sec/max(1,len(combos)))
     rows.append(result)
   pd.DataFrame(rows).to_csv(target,index=False)
   allrows+=rows
   print('Finished',ds,seed,len(rows),'rows',flush=True)
 df=pd.DataFrame(allrows)
 if len(df):
  df.to_csv(out/'ALL_METHODS_SEEDS_TIMED.csv',index=False)
  pd.DataFrame(audit).to_csv(out/'NUMERICAL_AND_RUNTIME_AUDIT.csv',index=False)
  tex_table(df,out)
  df.groupby(['dataset','method'])[['f1','mcc','roc_auc','brier','total_seconds']].mean().to_csv(out/'SUMMARY.csv')
  print(df.groupby(['dataset','method'])[['f1','mcc','total_seconds']].mean())
 print('IMPORTANT: DED benchmark covers email and SMS only; JMD includes 31 combinations.')
 print('IMPORTANT: text-specialist scores are learned from train only; QR/image pixels are not modeled.')
 print('IMPORTANT: total_seconds is feature+arbiter; amortized_end_to_end_seconds includes specialist training.')
 print('Results:',out.resolve())
if __name__=='__main__':main()
