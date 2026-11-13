"""boudon_table4.py -- reproduce the LISA rows of Boudon et al. (2024), Table IV.

Large y: both holes accrete supersonically across the band; the signal
measures rho0. Small y: the primary is subsonic across the band; the signal
measures rho_a = y * rho0. Within each regime the threshold does not depend
on y, so any y deep inside the regime gives the same answer.
"""
from dm_dephasing_full import (Binary, make_fgrid, fisher, invert_fisher,
                               binary_param_derivs, psi_halo)
import physical_model as pm

PUBLISHED = {  # (system): (rho0* large y, rho_a* small y), g/cm^3
    "MBBH": (8e-13, 5e-9), "IMRI": (3e-20, 2e-8), "EMRI": (1e-22, 1e-8)}
SYSTEMS = {"MBBH": (1e6, 5e5, 3e4), "IMRI": (1e4, 10.0, 64.0),
           "EMRI": (1e5, 10.0, 22.0)}


def threshold(m1, m2, snr, y):
    b = Binary(m1, m2)
    fmin, fmax = b.lisa_band()
    f = make_fgrid(fmin, fmax)
    T = psi_halo(f, b, 1.0) + pm.psi_acc_full(f, b, 1.0, y)
    _, s, _, _ = invert_fisher(fisher(list(binary_param_derivs(b, f)) + [T], f, snr))
    return s[4], b, fmin, fmax


def compare(verbose=True):
    rows = []
    for name, (m1, m2, snr) in SYSTEMS.items():
        b = Binary(m1, m2)
        fmin, fmax = b.lisa_band()
        # large y: push both accretion thresholds below fmin
        y_hi = 1e16
        assert b.f_acc(1, y_hi) < fmin and b.f_acc(2, y_hi) < fmin
        r0, *_ = threshold(m1, m2, snr, y_hi)
        # small y: keep the primary's threshold above fmax (y >= 1)
        y_lo = 10.0
        assert b.f_acc(1, y_lo) > fmax
        rl, *_ = threshold(m1, m2, snr, y_lo)
        ra = y_lo * rl
        rows += [(name, "large y", "rho0*", r0, PUBLISHED[name][0]),
                 (name, "small y", "rho_a*", ra, PUBLISHED[name][1])]
    if verbose:
        print("system  regime   quantity   this work   Boudon+24   ratio")
        for n, reg, q, ours, pub in rows:
            print(f"{n:6s}  {reg:7s}  {q:7s}   {ours:.2e}   {pub:.0e}    {ours/pub:.2f}")
    return rows


if __name__ == "__main__":
    compare()
