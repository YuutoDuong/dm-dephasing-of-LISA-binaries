"""make_fig6.py -- Figure 6: can friction be measured where the model holds?

(a) For three binaries: the detection threshold rho_det, the density at
    which the linearized model stops applying (|delta| = 0.1), and the
    density at which friction is measured at its predicted size (rho_test).
(b) rho_test / rho_valid versus mass ratio at m1 = 1e5 Msun, SNR = 30.
    Below 1, friction is measurable inside the model's validity.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dm_dephasing_full import Binary
import physical_model as pm
from validity import rho_valid

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
C_DET, C_VALID, C_TEST = "#1baf7a", "#2a78d6", "#eb6834"   # validated set
INK, MUTED = "#0b0b0b", "#52514e"


def tex(x):
    m, e = f"{x:.0e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


# ---------------- data
rows = [("IMRI, $q=10^3$\n$\\varrho=64$", 1e4, 10., 64.),
        ("EMRI, $q=10^4$\n$\\varrho=22$", 1e5, 10., 22.),
        ("$q=15$, $m_1=10^5\\,M_\\odot$\n$\\varrho=30$", 1e5, 1e5 / 15, 30.)]
bench = []
for name, m1, m2, snr in rows:
    r = pm.analyse_physical(name, m1, m2, snr, verbose=False)
    bench.append((name, r["rho0_detect"], rho_valid(m1, m2), r["rho0_test"]))

qs, ratio, ok = [], [], []
for q in np.concatenate([np.arange(9, 40, 1.0), np.logspace(np.log10(40), 5, 16)]):
    m2 = 1e5 / q
    b = Binary(1e5, m2)
    r = pm.analyse_physical("q", 1e5, m2, 30., verbose=False)
    qs.append(q)
    ratio.append(r["rho0_test"] / rho_valid(1e5, m2))
    ok.append(b.f_df_minus(r["y"]) < b.lisa_band()[0] and b.nu < 0.16)
qs, ratio, ok = map(np.array, (qs, ratio, ok))
q_edge = qs[ok].min()

# ---------------- figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2),
                               gridspec_kw={"width_ratios": [1.15, 1]})

# (a) density ladder
for k, (name, det, val, tst) in enumerate(bench):
    yk = 2 - k
    ax1.plot([det, val], [yk, yk], color=C_VALID, lw=6, alpha=0.25, solid_capstyle="butt")
    if tst > val:
        ax1.plot([val, tst], [yk, yk], color=MUTED, lw=1.5, ls=(0, (3, 3)))
    ax1.plot(det, yk, "o", ms=9, color=C_DET, mec="white", mew=1.5, zorder=3)
    ax1.plot(val, yk, "s", ms=9, color=C_VALID, mec="white", mew=1.5, zorder=3)
    ax1.plot(tst, yk, "D", ms=7.5, color=C_TEST, mec="white", mew=1.5, zorder=4)
    close = tst < val * 10
    ax1.annotate(f"detect\n{tex(det)}", (det, yk), xytext=(0, -14),
                 textcoords="offset points", ha="center", va="top", fontsize=8.5, color=INK)
    ax1.annotate(f"model breaks\n{tex(val)}", (val, yk), xytext=(0, 14 if close else -14),
                 textcoords="offset points", ha="center", va="bottom" if close else "top",
                 fontsize=8.5, color=INK)
    ax1.annotate(f"friction seen\n{tex(tst)}", (tst, yk),
                 xytext=(14, -14) if close else (0, -14), textcoords="offset points",
                 ha="left" if close else "center", va="top", fontsize=8.5, color=INK)
    if tst > val:
        ax1.annotate(f"friction needs {tex(tst / val)} more\nthan the model allows",
                     (np.sqrt(val * tst), yk), xytext=(0, 8), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8.5, color=MUTED)
    else:
        ax1.annotate("friction measurable\ninside model validity", (val, yk),
                     xytext=(30, 0), textcoords="offset points", ha="left", va="center",
                     fontsize=8.5, color=MUTED)
ax1.set_xscale("log")
ax1.set_xlim(1e-24, 1e-1)
ax1.set_ylim(-0.75, 2.55)
ax1.set_yticks([2, 1, 0])
ax1.set_yticklabels([b[0] for b in bench], fontsize=9.5)
ax1.grid(axis="y", visible=False)
ax1.set_xlabel(r"bulk density $\rho_0$  [g cm$^{-3}$]")
ax1.set_title("(a) Detection, model validity and friction", fontsize=11)

# (b) one curve, one axis
ax2.axvspan(8.5, q_edge - 0.5, color="0.85", alpha=0.7, lw=0)
ax2.fill_between([q_edge - 0.5, 1.1e5], 1e-2, 1, color=C_TEST, alpha=0.12, lw=0)
ax2.text(1.2e3, 0.3, "friction measurable inside model validity",
         ha="center", va="center", fontsize=8.5, color=INK)
ax2.axhline(1, color=C_VALID, lw=2)
ax2.text(1e5, 1.4, r"model validity limit, $|\delta|=0.1$", ha="right", va="bottom",
         fontsize=8.5, color=INK)
ax2.axhline(10, color=C_VALID, lw=1.2, ls=(0, (4, 3)))
ax2.text(1e5, 14, r"$|\delta|=1$", ha="right", va="bottom", fontsize=8.5, color=INK)
ax2.plot(qs[ok], ratio[ok], color=C_TEST, lw=2)
ax2.plot(qs[ok][::3], ratio[ok][::3], "D", color=C_TEST, ms=5, mec="white")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlim(8.5, 1.1e5)
ax2.set_ylim(1e-2, 1e15)
ax2.set_xlabel(r"mass ratio $q=m_1/m_2$   ($m_1=10^5\,M_\odot$, $\varrho=30$)")
ax2.set_ylabel(r"$\rho_{\rm test}\,/\,\rho_{\rm valid}$")
ax2.set_title("(b) Where friction can be measured", fontsize=11)

fig.tight_layout()
fig.savefig("fig6_friction_visibility.png")
meas = ok & (ratio < 1)
print("window opens at q =", q_edge, "| measurable (ratio<1) for q =", qs[meas].min(), "-", qs[meas].max())
for name, det, val, tst in bench:
    print(name.replace("\n", " "), f"| det {det:.2e} valid {val:.2e} test {tst:.2e}")
