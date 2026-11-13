"""
mcmc_check.py -- does the Fisher result survive without linearization?

For signals h = A f^{-7/6} exp(i Psi) with fixed amplitude, the noise-free
log-likelihood is exactly
    ln L(theta) = -SNR^2 * [1 - <cos(Psi(theta) - Psi(theta_0))>_w],
    <g>_w = int w g df / int w df,   w = f^{-7/3} / S_n(f).
Expanding cos to second order gives back the Fisher matrix of the paper,
so any difference between the chain and Fisher measures the non-linearity
the Fisher approximation throws away.

Parameters: theta = (t_c, Phi_c, ln m1, ln m2, rho_A, rho_B), injected at
rho_A = rho_B = 0 as in the Fisher analysis. Flat priors, negative densities
allowed (same as Fisher). Sampling uses emcee in coordinates whitened by the
Fisher covariance, so the near-perfect correlation does not stall the walkers.
"""
import numpy as np
import emcee
from dm_dephasing_full import (Binary, Sn_LISA, Msun, psi_vacuum, psi_acc_high,
                               psi_df, analyse_system, make_fgrid, fisher,
                               invert_fisher, binary_param_derivs)


def run(name, m1, m2, snr, n_grid=4000, nwalkers=32, nsteps=6000, burn=1500, seed=1):
    r = analyse_system(name, m1, m2, snr, verbose=False)
    b0 = Binary(m1, m2)
    fmin, fmax = b0.lisa_band()
    f = make_fgrid(fmin, fmax, n_grid)
    w = f ** (-7.0 / 3.0) / Sn_LISA(f)
    # trapezoid weights so <.>_w matches the Fisher quadrature
    dw = np.zeros_like(f)
    dw[1:] += 0.5 * np.diff(f)
    dw[:-1] += 0.5 * np.diff(f)
    W = w * dw / np.sum(w * dw)
    f_plus = r['f_plus']
    inc = r['acc_include'] if r['acc_include'] else (2,)

    def psi(theta):
        tc, phic, lm1, lm2, rA, rB = theta
        b = Binary(np.exp(lm1) / Msun, np.exp(lm2) / Msun)
        return (2 * np.pi * f * tc - phic + psi_vacuum(f, b)
                + psi_acc_high(f, b, rA, include=inc) + psi_df(f, b, rB, f_plus))

    theta0 = np.array([0.0, 0.0, np.log(b0.m1), np.log(b0.m2), 0.0, 0.0])
    psi0 = psi(theta0)

    def lnlike(theta):
        if theta[3] > theta[2]:          # keep the m1 >= m2 labelling
            return -np.inf
        return -snr ** 2 * (1.0 - np.sum(W * np.cos(psi(theta) - psi0)))

    # Fisher on the same grid, for whitening and for the comparison
    base = list(binary_param_derivs(b0, f))
    tA = psi_acc_high(f, b0, 1.0, include=inc)
    tB = psi_df(f, b0, 1.0, f_plus)
    Sigma, sig, corr, _ = invert_fisher(fisher(base + [tA, tB], f, snr))
    # whitening: theta = theta0 + L z with Sigma = L L^T (eigen-form, robust)
    lam, V = np.linalg.eigh(corr)
    L = np.diag(sig) @ V @ np.diag(np.sqrt(np.clip(lam, 0, None)))

    def lnprob_z(z):
        return lnlike(theta0 + L @ z)

    rng = np.random.default_rng(seed)
    p0 = 0.1 * rng.standard_normal((nwalkers, 6))
    sampler = emcee.EnsembleSampler(nwalkers, 6, lnprob_z)
    sampler.run_mcmc(p0, nsteps, progress=False)
    z = sampler.get_chain(discard=burn, flat=True)
    th = theta0 + z @ L.T

    sig_mc = th.std(axis=0)
    r_mc = np.corrcoef(th[:, 4], th[:, 5])[0, 1]
    try:
        tau = sampler.get_autocorr_time(discard=burn, quiet=True)
    except Exception:
        tau = np.full(6, np.nan)
    out = dict(name=name, sig_fisher=sig, sig_mcmc=sig_mc,
               r_fisher=corr[4, 5], r_mcmc=r_mc,
               z_std=z.std(axis=0), acc=np.mean(sampler.acceptance_fraction),
               tau_max=np.nanmax(tau), n_eff=len(z) / np.nanmax(tau))
    labels = ["t_c", "Phi_c", "ln m1", "ln m2", "rho_A", "rho_B"]
    print(f"\n=== {name}  (grid {n_grid}, {nwalkers}x{nsteps} steps, burn {burn})")
    print(f"  acceptance {out['acc']:.2f}, max autocorr time {out['tau_max']:.0f}, "
          f"effective samples ~{out['n_eff']:.0f}")
    print("  param    sigma_Fisher   sigma_MCMC   ratio")
    for i, lab in enumerate(labels):
        print(f"  {lab:6s}   {sig[i]:.3e}    {sig_mc[i]:.3e}   {sig_mc[i]/sig[i]:.3f}")
    print(f"  r_AB: Fisher {corr[4,5]:+.6f}   MCMC {r_mc:+.6f}")
    print(f"  whitened-coordinate std (1 if Fisher exact): {np.round(out['z_std'], 3)}")
    return out


if __name__ == "__main__":
    run("LISA-IMRI", 1e4, 10.0, 64.0)
    run("LISA-EMRI", 1e5, 10.0, 22.0)
