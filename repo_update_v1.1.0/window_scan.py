"""Where is friction measurable inside the model's validity?  Scan q at
fixed (m1, SNR, window placement R); report the lowest q at which the
friction window covers the band and the largest q at which
rho_test < rho_valid(|delta| = 0.1) and (|delta| = 1)."""
import numpy as np
from dm_dephasing_full import Binary
import physical_model as pm
from validity import rho_valid

def scan(m1, snr, R, qgrid=np.arange(6, 61, 1.0)):
    q_win, q01, q1 = None, None, None
    for q in qgrid:
        m2 = m1 / q; b = Binary(m1, m2)
        fmin, fmax = b.lisa_band()
        if fmax <= 1.5 * fmin:
            continue
        r = pm.analyse_physical("q", m1, m2, snr, fdfplus_over_fmax=R, verbose=False)
        win = b.f_df_minus(r['y']) < fmin and b.nu < 0.16
        if not win:
            continue
        q_win = q if q_win is None else q_win
        v = rho_valid(m1, m2, fdfplus_over_fmax=R, n=8000)
        if r['rho0_test'] < v:      q01 = q
        if r['rho0_test'] < 10 * v: q1 = q
    return q_win, q01, q1

if __name__ == "__main__":
    print("   m1     SNR    R   | window opens at q | measurable (|d|<0.1) up to q | (|d|<1) up to q")
    for m1, snr, R in [(1e5,30,10),(1e5,10,10),(1e5,100,10),(1e5,30,3),(1e5,30,100),
                       (1e4,30,10),(1e6,30,10)]:
        qw, q01, q1 = scan(m1, snr, R)
        print(f" {m1:6.0e}  {snr:4.0f}  {R:4.0f}  | {str(qw):>16s}  | {str(q01):>27s}  | {str(q1):>14s}")
