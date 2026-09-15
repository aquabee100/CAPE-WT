import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import detect_final as D
import normalise as N
import paths as P
import protocol as PR
CASES = [('MF07', '2018-07-10', '2018-10-23', 'MF07 (~ WT6): gearbox fault, maintenance 30 May 2019', '2018-10-19'), ('MF12', '2021-09-07', '2021-10-07', 'MF12 (= WT8): gearbox failure 23 Feb 2022', None)]
if __name__ == '__main__':
    Ft, Tt = (P.load('filter_traj'), P.load('causal_traj')['VA'])
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for i, (t, warn, conf, lab, paper) in enumerate(CASES):
        ev = pd.Timestamp(PR.CASES[t]['event'])
        Zf = N.rolling_z(Ft, scale_end=PR.SCALE_END[t])
        Zt = N.rolling_z(Tt, scale_end=PR.SCALE_END[t])
        for j, (Z, nm) in enumerate([(Zf, 'filter differential pressure'), (Zt, 'grey-box thermal index')]):
            a = ax[i, j]
            sl = slice(pd.Timestamp(warn) - pd.Timedelta('420D'), ev + pd.Timedelta('120D'))
            for h in D.HEALTHY:
                a.plot(Z[h].loc[sl], color='0.8', lw=0.9)
            a.plot(Z[t].loc[sl], color='crimson', lw=2.2, label=t)
            a.axvline(ev, color='k', ls='--', lw=1.5, label='failure/maintenance')
            a.axvline(pd.Timestamp(warn), color='darkorange', lw=1.8, label=f'warning {warn}')
            a.axvline(pd.Timestamp(conf), color='g', ls='-.', lw=1.5, label=f'confirmation {conf}')
            if paper and j == 0:
                a.axvline(pd.Timestamp(paper), color='b', ls=':', lw=1.8, label='Milani et al.')
            a.axhline(0, color='k', lw=0.4)
            a.grid(alpha=0.3)
            a.legend(fontsize=7, loc='upper left')
            a.set_ylabel('z')
            a.set_title(nm, fontsize=9, loc='left')
        ax[i, 0].text(0.01, 1.18, lab, transform=ax[i, 0].transAxes, fontsize=10, weight='bold')
    plt.tight_layout()
    plt.savefig('figures/fig_twostage.png', dpi=150)
    print('written: figures/fig_twostage.png')
