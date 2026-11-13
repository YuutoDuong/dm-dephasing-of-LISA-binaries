"""validity.py -- where does the linearized dephasing model stop applying?

The model assumes the environment changes the frequency drift by a small
fraction delta. With dPsi/df = 2*pi*t(f), d^2Psi/df^2 = 2*pi/fdot, so
    delta(f) = -Psi_env''(f) / Psi_gw''(f).
delta is linear in rho0, so rho_valid(eps) = eps / max_f |delta(rho0=1)|.
"""
import numpy as np
from dm_dephasing_full import Binary, psi_halo, psi_df, psi_vacuum
import physical_model as pm


def _f2_second_deriv(psi, lnf):
    d1 = np.gradient(psi, lnf)
    return np.gradient(d1, lnf) - d1          # f^2 * d^2psi/df^2


def rho_valid(m1, m2, eps=0.1, fdfplus_over_fmax=10.0, n=20000):
    b = Binary(m1, m2)
    fmin, fmax = b.lisa_band()
    f = np.logspace(np.log10(fmin), np.log10(fmax), n)
    lnf = np.log(f)
    fp = fdfplus_over_fmax * fmax
    y = b.y_for_fdf_plus(fp)
    T = psi_halo(f, b, 1.0) + pm.psi_acc_full(f, b, 1.0, y) + psi_df(f, b, 1.0, fp)
    delta1 = np.abs(_f2_second_deriv(T, lnf) / _f2_second_deriv(psi_vacuum(f, b), lnf))
    return eps / delta1[20:-20].max()          # drop np.gradient edge points


if __name__ == "__main__":
    print("   q     rho_det    rho_valid(0.1)  rho_test   test/valid   window")
    for m1, m2, snr in [(1e4, 10., 64.), (1e5, 10., 22.),
                        (1e5, 3e3, 30.), (1e5, 1e3, 30.), (1e5, 1e2, 30.), (1e5, 10., 30.)]:
        r = pm.analyse_physical("x", m1, m2, snr, verbose=False)
        rv = rho_valid(m1, m2)
        b = Binary(m1, m2); fmin, _ = b.lisa_band()
        win = b.f_df_minus(r['y']) < fmin and b.nu < 0.16
        print(f" {m1/m2:6.0f}  {r['rho0_detect']:.2e}   {rv:.2e}      {r['rho0_test']:.2e}  {r['rho0_test']/rv:.1e}    {win}")
