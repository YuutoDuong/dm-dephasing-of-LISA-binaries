"""
physical_model.py -- consistency test of the Boudon et al. (2024) cloud model.

In the model every dephasing channel is fixed by the same cloud parameters
(bulk density rho0, density ratio y = rho_a/rho0) and the masses. So the
model predicts the RATIO between channels. This module asks whether LISA
can check that prediction for dynamical friction:

  Psi_DM = rho0 * [ T_rest(f) + lam * T_df(f) ],   model predicts lam = 1

  T_rest = halo gravity + accretion on BOTH holes (each in whichever regime,
           high-f or low-f, it occupies at each frequency; Eq. 4.19 in full)
  T_df   = dynamical friction on the lighter hole (Eq. 4.20)

Outputs per system:
  rho0_detect : 1-sigma threshold for detecting the full model signal
  rho0_test   : density at which sigma(lam) = 1, i.e. the friction term
                is measured at its predicted size at 1 sigma
  ratio       : rho0_test / rho0_detect  (how far above detection you must
                be before the friction channel is seen at all)

Requires dm_dephasing_full.py (or the colab file) to be imported first.
"""
import numpy as np
from dm_dephasing_full import (Binary, x_of_f, psi_halo, psi_df, _dm_prefac,
                               FSTAR, make_fgrid, fisher, invert_fisher,
                               binary_param_derivs)


def psi_acc_full(f, b, rho0_gcm3, y):
    """Full Eq. (4.19): each hole uses the high-f branch above its threshold
    f_acc,i and the low-f branch (density rho_a = y*rho0, with the bracket
    kept) below it. The two branches join continuously at f_acc,i."""
    x = x_of_f(f, b)
    out = np.zeros_like(f, dtype=float)
    for mi, i in ((b.m1, 1), (b.m2, 2)):
        fa = b.f_acc(i, y)
        k = 3.0 + 2.0 * mi ** 2 / (b.m * b.mu)
        hi = f > fa
        out[hi] += -(25.0 * np.pi / 38912.0) * _dm_prefac(b, rho0_gcm3) \
            * mi ** 3 / (b.mu ** 2 * b.m) * k * x[hi] ** (-16.0 / 3.0)
        lo = ~hi
        r = f[lo] / fa
        bracket = 1.0 - r ** (13.0 / 3.0) + (13.0 / 19.0) * r ** (16.0 / 3.0)
        out[lo] += -(75.0 * np.pi * FSTAR / 26624.0) * b.nu ** 0.4 \
            * _dm_prefac(b, y * rho0_gcm3) * k * x[lo] ** (-13.0 / 3.0) * bracket
    return out


def analyse_physical(name, m1, m2, snr, fdfplus_over_fmax=10.0, verbose=True):
    b = Binary(m1, m2)
    fmin, fmax = b.lisa_band()
    f = make_fgrid(fmin, fmax)
    f_plus = fdfplus_over_fmax * fmax
    y = b.y_for_fdf_plus(f_plus)

    T_halo = psi_halo(f, b, 1.0)
    T_acc = psi_acc_full(f, b, 1.0, y)
    T_df = psi_df(f, b, 1.0, f_plus)
    T_rest = T_halo + T_acc
    base = list(binary_param_derivs(b, f))

    # (1) detecting the full model signal: one DM parameter, rho0
    _, s5, _, _ = invert_fisher(fisher(base + [T_rest + T_df], f, snr))
    rho0_detect = s5[4]

    # (2) consistency test: rho0 and lam both free, evaluated at rho0 = 1.
    #     dPsi/drho0 = T_rest + T_df, dPsi/dlam = rho0 * T_df.
    #     sigma(lam) scales as 1/rho0, so rho0_test = sigma(lam) at rho0 = 1.
    _, s6, c6, cond = invert_fisher(fisher(base + [T_rest + T_df, T_df], f, snr))
    rho0_test = s6[5]

    res = dict(name=name, y=y, f_acc1=b.f_acc(1, y), fmin=fmin, fmax=fmax,
               df_over_rest_fmin=abs(T_df[0] / T_rest[0]),
               df_over_rest_fmax=abs(T_df[-1] / T_rest[-1]),
               halo_over_acc_fmin=abs(T_halo[0] / T_acc[0]),
               rho0_detect=rho0_detect, rho0_test=rho0_test,
               ratio=rho0_test / rho0_detect, r_rho0_lam=c6[4, 5], cond=cond)
    if verbose:
        print(f"\n--- {name}  (y = {y:.3e}, f_acc,1 = {res['f_acc1']:.3e} Hz, "
              f"band [{fmin:.2e}, {fmax:.2e}] Hz)")
        print(f"  |T_df / T_rest|     at fmin {res['df_over_rest_fmin']:.2e}, "
              f"at fmax {res['df_over_rest_fmax']:.2e}")
        print(f"  |T_halo / T_acc|    at fmin {res['halo_over_acc_fmin']:.2e}")
        print(f"  rho0 to detect the environment : {rho0_detect:.2e} g/cm^3")
        print(f"  rho0 to see friction (sig_lam=1): {rho0_test:.2e} g/cm^3")
        print(f"  ratio test/detect               : {res['ratio']:.2e}")
        print(f"  corr(rho0, lam) = {res['r_rho0_lam']:+.6f}   cond = {cond:.1e}")
    return res


if __name__ == "__main__":
    for s in [("LISA-IMRI", 1e4, 10.0, 64.0), ("LISA-EMRI", 1e5, 10.0, 22.0)]:
        analyse_physical(*s)
