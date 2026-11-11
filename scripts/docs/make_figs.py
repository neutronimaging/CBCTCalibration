"""Figures for the fitting explainer, computed from the real calibration data."""
import os
import sys
from types import SimpleNamespace

import numpy as np
from scipy import ndimage as ndi

# usage: python docs/make_figs.py [data folder with xcal_/xob_/xdc_ FITS files]
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = sys.argv[1] if len(sys.argv) > 1 else "/Users/Shared/Data/P20250297_1_SoilLehmann/02_rawdata/00_setup/03_xcal"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig")
sys.path.insert(0, CODE)
import xcal_geometry as xg  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

os.makedirs(OUT, exist_ok=True)
R, PIXEL, STEP = 25.0, 0.139, 5
cache = os.path.join(OUT, "..", "cache.npz")

# ------------------------------------------------------------- data ---------
dc, gain, bad = xg.load_references(DATA, "xdc_*.fits", "xob_*.fits")
fill_idx = ndi.distance_transform_edt(bad, return_distances=False, return_indices=True)
files = sorted(f for f in os.listdir(DATA) if f.startswith("xcal_") and f.endswith(".fits"))
files = [os.path.join(DATA, f) for f in files]
if os.path.exists(cache):
    obs = np.load(cache)["obs"]
else:
    frames, angs = [], []
    for i in range(0, len(files), STEP):
        n = xg.normalize(xg.read_image(files[i]), dc, gain, fill_idx)
        frames.append(xg.detect_balls(n, 56.1))
        angs.append(xg.read_angle(files[i], "sp2_ry"))
    spacing = np.median(np.concatenate([np.diff(np.sort(f[:, 1])) for f in frames]))
    tracks = [t for t in xg.track_balls(frames, 0.4 * spacing) if len(t) >= 0.5 * len(frames)]
    tracks.sort(key=lambda t: np.median([d[1] for d in t.values()]))
    obs = np.array([(k, angs[fi], d[0], d[1], fi, d[2]) for k, t in enumerate(tracks) for fi, d in t.items()])
    np.savez(cache, obs=obs)

sense = -1
model = xg.Model(obs, R, PIXEL, sense)
p_init = xg.initial_guess(obs, R, PIXEL, sense)
p0 = xg.fit(model, p_init).p
opt = SimpleNamespace(nboot=200, radius_tol=0.01, pixel_tol=0.0)
full = xg.calibrate(obs, model, p0, np.ones(xg.N_GEO, bool), opt)
red = xg.calibrate(obs, model, full["p"], np.array([1, 1, 1, 1, 0, 0, 1], bool),
                   SimpleNamespace(nboot=0, radius_tol=0.0, pixel_tol=0.0))
good = full["good"]
pf = full["p"]
geo = full["geo"]
_, zk, phik = model.unpack(pf)
nb = model.nb
print("full", np.round(geo, 3), "reduced", np.round(red["geo"], 3))
print("init", np.round(p_init[:7], 3))

# profiles: fix one tilt, refit the rest
def profile(idx, values):
    rms, other = [], []
    for val in values:
        p = pf.copy()
        p[idx] = val
        free = np.ones(p.size, bool)
        free[idx] = False
        r = xg.fit(model, p, good, free)
        rms.append(np.sqrt(np.mean(r.fun ** 2)))
        other.append(r.p[:xg.N_GEO])
    return np.array(rms), np.array(other)

eta_vals = geo[4] + np.linspace(-0.9, 0.9, 19)
th_vals = geo[5] + np.linspace(-0.3, 0.3, 13)
eta_rms, eta_geo = profile(4, eta_vals)
th_rms, th_geo = profile(5, th_vals)

# stats for the page
rr = full["rr"]
fr = obs[:, 4].astype(int)
frame_mean = np.array([rr[(fr == f) & good, 0].mean() for f in range(fr.max() + 1)])
frame_ang = np.array([obs[fr == f, 1][0] for f in range(fr.max() + 1)])
summary = dict(full=full, red=red)
np.savez(os.path.join(OUT, "..", "numbers.npz"), geo=geo, red=red["geo"],
         sig=full["sig_tot"], sig_cov=full["sig_cov"], sig_boot=full["sig_boot"],
         sig_red=red["sig_cov"], init=p_init[:7], cor=full["cor"],
         rms=np.sqrt(np.mean(rr[good] ** 2, axis=0)), nobs=good.sum(), nrej=(~good).sum(), nb=nb,
         frame_rms=np.std(frame_mean), resid_rms_after=np.std(rr[good, 0] - frame_mean[fr[good]]))

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", muted="#52514e", grid="#e4e3df",
                  s=["#2a78d6", "#eb6834", "#1baf7a"], neutral="#a3a29b"),
    "dark": dict(surface="#1a1a19", text="#f0efec", muted="#c3c2b7", grid="#34342f",
                 s=["#3987e5", "#d95926", "#199e70"], neutral="#6d6c66"),
}


def setup(t):
    plt.rcParams.update({
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "text.color": t["text"], "axes.labelcolor": t["muted"], "xtick.color": t["muted"],
        "ytick.color": t["muted"], "axes.edgecolor": t["grid"], "axes.grid": True, "grid.color": t["grid"],
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlecolor": t["text"], "axes.titlelocation": "left",
        "legend.frameon": False, "legend.labelcolor": t["text"], "lines.linewidth": 2,
    })


def save(fig, name, theme):
    fig.savefig(os.path.join(OUT, f"{name}-{theme}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


frame46 = xg.normalize(xg.read_image(files[45]), dc, gain, fill_idx)
raw46 = xg.read_image(files[45])
size = int(1.6 * 56.1) | 1
bg46 = ndi.uniform_filter(ndi.grey_closing(frame46, size=(size, size)), 9)
att46 = np.clip(-np.log(np.clip(frame46 / bg46, 1e-3, None)), 0, None)
det46 = xg.detect_balls(frame46, 56.1)

dense = np.deg2rad(np.linspace(0, 360, 721))
show = [1, int(np.argmin(np.abs(np.array([np.median(obs[obs[:, 0] == k, 3]) for k in range(nb)]) - red["geo"][3]))), nb - 1]

for name, t in THEMES.items():
    setup(t)
    s = t["s"]

    # 1 preprocessing
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.9))
    ax[0].imshow(raw46, cmap="gray")
    ax[0].set_title("raw projection $I$")
    ax[1].imshow(frame46, cmap="gray", vmin=0, vmax=1.2)
    ax[1].set_title("normalized $T$")
    ax[2].imshow(bg46, cmap="gray", vmin=0, vmax=1.2)
    ax[2].set_title("background $B$ (grey closing)")
    z0, z1, c0, c1 = 520, 760, 120, 360
    ax[3].imshow(att46[z0:z1, c0:c1], cmap="magma", extent=(c0, c1, z1, z0))
    sel = (det46[:, 1] > z0) & (det46[:, 1] < z1)
    ax[3].plot(det46[sel, 0], det46[sel, 1], "+", color="#ffffff", ms=12, mew=1.6)
    ax[3].set_title("attenuation $a$ + centroids")
    for a in ax:
        a.grid(False)
        a.tick_params(labelsize=8)
    ax[1].add_patch(plt.Rectangle((c0, z0), c1 - c0, z1 - z0, fill=False, ec=s[1], lw=1.5))
    fig.tight_layout()
    save(fig, "preproc", name)

    # 2 sinusoids
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True)
    for j, k in enumerate(show):
        m = (obs[:, 0] == k) & good
        u, v = xg.project(geo, zk, phik, np.full(dense.size, k), dense, R, PIXEL, sense)
        ax[0].plot(np.rad2deg(dense), u, color=s[j], lw=1.6)
        ax[0].plot(obs[m, 1], obs[m, 2], "o", ms=4.5, mfc=s[j], mec=t["surface"], mew=1)
        vm = obs[m, 3].mean()
        ax[1].plot(np.rad2deg(dense), v - vm, color=s[j], lw=1.6, label=f"ball {k}  (v ≈ {vm:.0f} px)")
        ax[1].plot(obs[m, 1], obs[m, 3] - vm, "o", ms=4.5, mfc=s[j], mec=t["surface"], mew=1)
    ax[0].set_ylabel("u  [px]")
    ax[0].set_title("Horizontal position: a sinusoid of amplitude R·M/p (the three balls overlap)")
    ax[1].set_ylabel("v − mean(v)  [px]")
    ax[1].set_title("Vertical position: magnification changes with depth; sign flips across the central plane")
    ax[1].set_xlabel("rotation angle θ  [deg]")
    ax[1].set_xticks(range(0, 361, 45))
    ax[1].legend(loc="lower center", fontsize=9, ncol=3)
    fig.tight_layout()
    save(fig, "sinus", name)

    # 3 initial guess regression
    ang = np.deg2rad(obs[:, 1])
    vmean, vq = [], []
    for k in range(nb):
        m = obs[:, 0] == k
        A = np.stack([np.ones(m.sum()), np.cos(ang[m]), np.sin(ang[m])], 1)
        cu = np.linalg.lstsq(A, obs[m, 2], rcond=None)[0]
        cv = np.linalg.lstsq(A, obs[m, 3], rcond=None)[0]
        al = np.arctan2(cu[2], cu[1])
        vmean.append(cv[0])
        vq.append(-cv[1] * np.sin(al) + cv[2] * np.cos(al))
    vmean, vq = np.array(vmean), np.array(vq)
    sl, ic = np.polyfit(vmean, vq, 1)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    xx = np.array([vmean.min() - 50, vmean.max() + 50])
    ax.axhline(0, color=t["neutral"], lw=1)
    ax.plot(xx, sl * xx + ic, color=s[0], lw=2, label=f"fit: slope = R/SOD = {abs(sl):.4f}")
    ax.plot(vmean, vq, "o", ms=7, mfc=s[0], mec=t["surface"], mew=1.5, label="one point per ball")
    ax.axvline(-ic / sl, color=s[1], lw=1.5, ls="--")
    ax.annotate(f"zero crossing: ppy₀ = {-ic / sl:.0f} px", (-ic / sl, 0), xytext=(12, -34),
                textcoords="offset points", color=t["text"], fontsize=9)
    ax.set_xlabel("mean row of the ball trajectory  v̄ₖ  [px]")
    ax.set_ylabel("depth term  qₖ  [px]")
    ax.set_title("Start values: depth modulation is proportional to the distance from the central plane")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    save(fig, "init", name)

    # 4 detector view
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.4), gridspec_kw=dict(width_ratios=[1.25, 1]))
    for k in range(nb):
        u, v = xg.project(geo, zk, phik, np.full(dense.size, k), dense, R, PIXEL, sense)
        ax[0].plot(u, v, color=t["neutral"], lw=1)
    ax[0].plot(obs[good, 2], obs[good, 3], ".", ms=1.5, color=s[0])
    ax[0].axhline(geo[3], color=s[1], lw=1.5, ls="--")
    ax[0].axhline(red["geo"][3], color=s[2], lw=1.5, ls="--")
    ax[0].plot(geo[2], geo[3], "X", ms=11, color=s[1], mec=t["surface"], label=f"ppx, ppy full model ({geo[2]:.0f}, {geo[3]:.0f})")
    ax[0].plot(red["geo"][2], red["geo"][3], "P", ms=11, color=s[2], mec=t["surface"],
               label=f"reduced model ({red['geo'][2]:.0f}, {red['geo'][3]:.0f})")
    ax[0].axvline(full["cor"], color=t["muted"], lw=1, ls=":")
    ax[0].set_xlim(0, 1792)
    ax[0].set_ylim(2176, 0)
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("u  [px]")
    ax[0].set_ylabel("v  [px]")
    ax[0].set_title("Fitted trajectories on the detector")
    ax[0].legend(loc="lower center", fontsize=8.5, bbox_to_anchor=(0.5, -0.32))
    # signed ellipse half height: positive when the near side (larger magnification) is below
    hb = []
    for k in range(nb):
        u, v = xg.project(geo, zk, phik, np.full(dense.size, k), dense, R, PIXEL, sense)
        a = sense * dense + phik[k]
        vn = v[np.argmin(np.sin(a))]  # ball closest to the source
        vf = v[np.argmax(np.sin(a))]  # ball farthest from the source
        hb.append(((vf - vn) / 2, v.mean()))
    hb = np.array(hb)
    ax[1].axhline(0, color=t["neutral"], lw=1)
    ax[1].plot(hb[:, 1], hb[:, 0], "o-", color=s[0], ms=5, lw=1.5, mec=t["surface"])
    ax[1].axvline(red["geo"][3], color=s[2], lw=1.5, ls="--")
    ax[1].axvline(geo[3], color=s[1], lw=1.5, ls="--")
    ax[1].annotate("trajectory is a line\n(plane contains the source)", (red["geo"][3], 0), xytext=(10, 40),
                   textcoords="offset points", fontsize=8.5, color=t["text"])
    ax[1].set_xlabel("trajectory centre row  [px]")
    ax[1].set_ylabel("signed half-height of ellipse  [px]")
    ax[1].set_title("Ellipse opening vs. row")
    fig.tight_layout()
    save(fig, "detector", name)

    # 5 profiles
    fig, ax = plt.subplots(2, 2, figsize=(10, 5.8), sharex="col")
    ax[0, 0].plot(eta_vals, eta_rms, "o-", color=s[0], ms=4)
    ax[0, 0].axvline(geo[4], color=t["neutral"], lw=1)
    ax[0, 0].set_ylabel("fit rms  [px]")
    ax[0, 0].set_title("Fix η, refit everything else")
    ax[1, 0].plot(eta_vals, eta_geo[:, 3], "o-", color=s[1], ms=4)
    ax[1, 0].axvline(geo[4], color=t["neutral"], lw=1)
    ax[1, 0].set_ylabel("fitted ppy  [px]")
    ax[1, 0].set_xlabel("η  [deg]")
    ax[0, 1].plot(th_vals, th_rms, "o-", color=s[0], ms=4)
    ax[0, 1].axvline(geo[5], color=t["neutral"], lw=1)
    ax[0, 1].set_title("Fix θ, refit everything else")
    ax[1, 1].plot(th_vals, th_geo[:, 2], "o-", color=s[1], ms=4)
    ax[1, 1].axvline(geo[5], color=t["neutral"], lw=1)
    ax[1, 1].set_ylabel("fitted ppx  [px]")
    ax[1, 1].set_xlabel("θ  [deg]")
    fig.tight_layout()
    save(fig, "profile", name)

    # 6 residuals
    fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for j, lab in enumerate(["u", "v"]):
        ax[j].plot(obs[good, 1], rr[good, j], "o", ms=2.5, color=t["neutral"], mec="none", label="single ball")
        fm = np.array([rr[(fr == f) & good, j].mean() for f in range(fr.max() + 1)])
        ax[j].plot(frame_ang, fm, "-", color=s[0], lw=1.8, label="mean of all balls in the projection")
        ax[j].set_ylabel(f"residual {lab}  [px]")
        ax[j].set_ylim(-1.3, 1.3)
    ax[0].set_title("Residuals: balls in one projection move together")
    ax[0].legend(loc="upper right", fontsize=8.5, ncol=2)
    ax[1].set_xlabel("rotation angle θ  [deg]")
    ax[1].set_xticks(range(0, 361, 45))
    fig.tight_layout()
    save(fig, "resid", name)

    # 7 bootstrap
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.2))
    for j, k in enumerate([0, 1, 2, 3]):
        b = full["boot"][:, k]
        ax[j].hist(b, bins=22, color=s[0], ec=t["surface"], lw=1, density=True, label="bootstrap")
        xs = np.linspace(b.min(), b.max(), 200)
        sg = full["sig_cov"][k]
        ax[j].plot(xs, np.exp(-0.5 * ((xs - geo[k]) / sg) ** 2) / (sg * np.sqrt(2 * np.pi)), color=s[1], lw=2,
                   label="covariance σ")
        ax[j].set_title(f"{xg.GEO_NAMES[k]}  [{xg.GEO_UNITS[k]}]")
        ax[j].set_yticks([])
        ax[j].tick_params(labelsize=8)
    ax[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    save(fig, "boot", name)

print("done")
