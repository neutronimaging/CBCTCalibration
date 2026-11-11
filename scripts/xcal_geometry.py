# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "astropy", "matplotlib"]
# ///
"""
Cone-beam geometry calibration from a ball phantom.

Reads every N-th projection (xcal_*.fits), normalizes with open-beam (xob_*)
and dark-current (xdc_*) references, detects the steel balls, tracks them over
the rotation and fits a full cone-beam geometry model by least squares:

    SOD    source -> rotation axis distance               [mm]
    SDD    source -> detector distance (along central ray) [mm]
    ppx,ppy piercing point of the central ray             [pixel, 0-based col,row]
    eta    detector tilt about the detector u-axis        [deg]
    theta  detector tilt about the rotation axis (yaw)    [deg]
    psi    detector in-plane rotation (axis tilt in image)[deg]

plus nuisance parameters (height z_k and phase phi_k of every ball).
All balls are assumed to lie on circles of radius R around the rotation axis.

Uncertainties are reported as
  * statistical, from the Jacobian covariance of the fit,
  * statistical, from a bootstrap over projections (robust against correlated
    residuals),
  * systematic, from the tolerance of the ball circle radius R and of the
    pixel size (SOD scales with R, SDD scales with the pixel size).

Run:  pixi run calibrate --step 5      (or: uv run xcal_geometry.py --step 5)
"""

import argparse
import glob
import json
import os
import re
import warnings

import numpy as np
from astropy.io import fits
from scipy import ndimage as ndi
from scipy.optimize import least_squares, linear_sum_assignment

warnings.filterwarnings("ignore", module="astropy")

GEO_NAMES = ["SOD", "SDD", "ppx", "ppy", "eta", "theta", "psi"]
GEO_UNITS = ["mm", "mm", "px", "px", "deg", "deg", "deg"]
N_GEO = len(GEO_NAMES)


# ---------------------------------------------------------------- I/O --------
def read_image(fname):
    return fits.getdata(fname).astype(np.float32)


def read_angle(fname, key):
    """Read the rotation angle from the header.

    `key` is matched against the full HIERARCH keyword or its last path component
    (e.g. 'sp2_ry' matches 'sample_position_2/sp2_ry'). The header as a whole does
    not verify (NUL bytes in MODESTR), so only the angle card is fixed; its value
    is a string like "b'0.972 deg'".
    """
    header = fits.getheader(fname)
    names = [k for k in header.keys() if k == key or k.split("/")[-1] == key]
    if not names:
        return None
    card = header.cards[names[0]]
    card.verify("silentfix")
    m = re.search(r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", str(card.value))
    return float(m.group(0)) if m else None


def load_references(path, dc_pattern, ob_pattern):
    dc_files = sorted(glob.glob(os.path.join(path, dc_pattern)))
    ob_files = sorted(glob.glob(os.path.join(path, ob_pattern)))
    if not dc_files or not ob_files:
        raise FileNotFoundError("Reference images not found")
    dc = np.median([read_image(f) for f in dc_files], axis=0)
    ob = np.median([read_image(f) for f in ob_files], axis=0)
    gain = ob - dc
    bad = (gain < 0.5 * np.median(gain)) | (ob >= 65000) | (dc >= 65000)
    print(f"References: {len(dc_files)} dark, {len(ob_files)} open beam, {bad.sum()} bad pixels")
    return dc, gain, bad


# ---------------------------------------------------------- detection --------
def normalize(proj, dc, gain, fill_idx):
    n = (proj - dc) / np.maximum(gain, 1.0)
    return n[tuple(fill_idx)]  # bad pixels replaced by their nearest good neighbour


def sphere_profile(q, r, grid):
    """Radial transmission profile of a blurred sphere shadow.

    q = (r0, mu, gamma, sigma, b): radius [px], attenuation, exponent of the
    chord length (gamma < 1 mimics beam hardening), Gaussian blur [px] and
    background level. The blur is applied along r, valid for sigma << r0.
    """
    r0, mu, gam, sig, b = q
    chord = 2 * np.sqrt(np.clip(r0**2 - grid**2, 0, None))
    t = ndi.gaussian_filter1d(b * np.exp(-mu * chord**gam), max(sig, 1e-3) / (grid[1] - grid[0]),
                              mode="nearest")
    return np.interp(r, grid, t)


def edge_diameter(t, u, v, r_guess, bin_px=0.25):
    """Shadow diameter [px] of a ball from a blurred-sphere fit to its radial profile.

    `t` is the background-corrected transmission around the ball, (u, v) the
    centre in the coordinates of `t`. The edge of a sphere shadow is not
    symmetric (chord ~ sqrt(r0^2 - r^2)), so blur shifts both its 50 % level and
    its steepest point inwards; fitting the blurred profile removes that bias.
    """
    rr, cc = np.mgrid[0:t.shape[0], 0:t.shape[1]]
    r = np.hypot(cc - u, rr - v).ravel()
    sel = (r > 0.6 * r_guess) & (r < 1.6 * r_guess)
    edges = np.arange(0.6 * r_guess, 1.6 * r_guess + bin_px, bin_px)
    idx = np.digitize(r[sel], edges) - 1
    cnt = np.bincount(idx, minlength=edges.size)[:edges.size - 1]
    prof = np.bincount(idx, t.ravel()[sel], minlength=edges.size)[:edges.size - 1]
    ok = cnt > 0
    rc, prof, cnt = (edges[:-1] + bin_px / 2)[ok], prof[ok] / cnt[ok], cnt[ok]
    grid = np.arange(0, 1.7 * r_guess, 0.05)
    b0 = np.median(prof[rc > 1.3 * r_guess])
    mu0 = -np.log(max(prof.min(), 1e-3) / b0) / (1.6 * r_guess)
    res = least_squares(lambda q: (sphere_profile(q, rc, grid) - prof) * np.sqrt(cnt),
                        [r_guess, mu0, 1.0, 1.0, b0],
                        bounds=([0.5 * r_guess, 0, 0.3, 0.05, 0], [1.5 * r_guess, np.inf, 1.5, 10, 2]))
    return 2 * res.x[0]


def detect_balls(n, ball_px, att_threshold=0.5, border=3):
    """Return (u, v, diameter) of all fully visible balls in a normalized projection.

    The smooth background (sample container etc.) is removed by a grey closing
    larger than the ball shadow; the ball centre is the attenuation-weighted
    centroid, the diameter comes from the steepest point of the radial edge
    profile (`edge_diameter`).
    """
    size = int(1.6 * ball_px) | 1
    bg = ndi.uniform_filter(ndi.grey_closing(n, size=(size, size)), 9)
    att = np.clip(-np.log(np.clip(n / np.maximum(bg, 1e-6), 1e-3, None)), 0, None)

    lab, nlab = ndi.label(att > att_threshold)
    if nlab == 0:
        return np.zeros((0, 3))
    objs = ndi.find_objects(lab)
    areas = ndi.sum_labels(np.ones_like(att), lab, np.arange(1, nlab + 1))
    a_nom = np.pi / 4 * ball_px**2
    out = []
    H, W = n.shape
    for k, sl in enumerate(objs):
        if not (0.4 * a_nom < areas[k] < 2.0 * a_nom):
            continue
        r0, r1, c0, c1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        if r0 < border or c0 < border or r1 > H - border or c1 > W - border:
            continue  # truncated by the detector edge
        h, w = r1 - r0, c1 - c0
        if max(h, w) / max(min(h, w), 1) > 1.3:
            continue  # not round: overlapping or truncated
        pad = max(4, int(0.3 * max(h, w)))
        R0, R1, C0, C1 = max(r0 - pad, 0), min(r1 + pad, H), max(c0 - pad, 0), min(c1 + pad, W)
        mask = ndi.binary_dilation(lab[R0:R1, C0:C1] == k + 1, iterations=3)
        a = att[R0:R1, C0:C1] * mask
        rr, cc = np.mgrid[R0:R1, C0:C1]
        s = a.sum()
        u, v = (a * cc).sum() / s, (a * rr).sum() / s
        d = edge_diameter(np.exp(-att[R0:R1, C0:C1]), u - C0, v - R0, 0.25 * (h + w))
        out.append((u, v, d))
    return np.array(out).reshape(-1, 3)


def track_balls(frames, max_dv):
    """Link detections of consecutive frames by vertical position."""
    tracks = []  # each: dict frame_idx -> (u, v, d)
    active = []  # (track_id, last_u, last_v)
    for fi, det in enumerate(frames):
        new_active = []
        used = set()
        if active and len(det):
            cost = np.abs(np.array([a[2] for a in active])[:, None] - det[None, :, 1]) \
                + 0.1 * np.abs(np.array([a[1] for a in active])[:, None] - det[None, :, 0])
            ri, ci = linear_sum_assignment(cost)
            for r, c in zip(ri, ci):
                if abs(active[r][2] - det[c, 1]) < max_dv:
                    tid = active[r][0]
                    tracks[tid][fi] = det[c]
                    new_active.append((tid, det[c, 0], det[c, 1]))
                    used.add(c)
        for c in range(len(det)):
            if c not in used:
                # a ball may re-enter the field of view; reconnect to a lost track
                tid = None
                for t_id, tr in enumerate(tracks):
                    if t_id in [a[0] for a in new_active]:
                        continue
                    last = tr[max(tr)]
                    if abs(last[1] - det[c, 1]) < max_dv and abs(last[0] - det[c, 0]) < 3 * max_dv:
                        tid = t_id
                        break
                if tid is None:
                    tracks.append({})
                    tid = len(tracks) - 1
                tracks[tid][fi] = det[c]
                new_active.append((tid, det[c, 0], det[c, 1]))
        active = new_active
    return tracks


# -------------------------------------------------------------- model --------
def rot(axis, a):
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def project(geo, z, phi, ball_idx, ang, R, pixel, sense, return_depth=False):
    """Project ball centres onto the detector.

    World frame: rotation axis = z (up), source at (0, -SOD, 0), beam along +y.
    Detector: u (columns) along +x, v (rows) along -z, normal +y before tilts.
    R is the circle radius, common to all balls or one value per ball.
    With return_depth, also return the depth (X - S) . n of each ball along the
    detector normal, so that SDD / depth is its local magnification.
    """
    SOD, SDD, ppx, ppy, eta, theta, psi = geo
    Rd = rot("z", np.deg2rad(theta)) @ rot("x", np.deg2rad(eta)) @ rot("y", np.deg2rad(psi))
    eu, ev, nn = Rd @ np.array([1.0, 0, 0]), Rd @ np.array([0, 0, -1.0]), Rd @ np.array([0, 1.0, 0])
    S = np.array([0.0, -SOD, 0.0])
    a = sense * ang + phi[ball_idx]
    Rb = np.asarray(R)[ball_idx] if np.ndim(R) else R
    X = np.stack([Rb * np.cos(a), Rb * np.sin(a), z[ball_idx]], axis=1)
    d = X - S
    depth = d @ nn
    Y = d * (SDD / depth)[:, None] - SDD * nn  # point on the detector relative to the piercing point
    u, v = ppx + (Y @ eu) / pixel, ppy + (Y @ ev) / pixel
    return (u, v, depth) if return_depth else (u, v)


class Model:
    """Parameter vector: geometry (7), ball heights z_k, phases phi_k and, with
    per_ball_radius, coefficients w of radius offsets dR = Q w.

    The columns of Q are orthonormal to a constant, so the offsets have zero mean
    and the mean radius R sets the scale. Without radius_trend they are also
    orthonormal to the ball height: a linear trend of the radius with height
    (a ball column leaning radially) cannot be told apart from the detector
    tilt eta, so it is only allowed when eta is fixed.
    """

    def __init__(self, obs, R, pixel, sense, per_ball_radius=False, radius_trend=False):
        self.ball = obs[:, 0].astype(int)
        self.ang = np.deg2rad(obs[:, 1])
        self.uv = obs[:, 2:4]
        self.nb = self.ball.max() + 1
        self.R, self.pixel, self.sense = R, pixel, sense
        self.radius_trend = radius_trend
        nfix = 1 if radius_trend else 2
        self.per_ball_radius = per_ball_radius and self.nb > nfix
        if self.per_ball_radius:
            self.vbar = np.array([self.uv[self.ball == k, 1].mean() for k in range(self.nb)])
            A = np.column_stack([np.ones(self.nb), self.vbar - self.vbar.mean()])[:, :nfix]
            q, _ = np.linalg.qr(np.column_stack([A, np.eye(self.nb)]))
            self.Q = q[:, nfix:self.nb]

    @property
    def n_params(self):
        return N_GEO + 2 * self.nb + (self.Q.shape[1] if self.per_ball_radius else 0)

    def variant(self, **kw):
        """Same observations, other radius options or scale."""
        opts = dict(R=self.R, pixel=self.pixel, sense=self.sense, per_ball_radius=self.per_ball_radius,
                    radius_trend=self.radius_trend)
        opts.update(kw)
        obs = np.column_stack([self.ball, np.rad2deg(self.ang), self.uv])
        return Model(obs, **opts)

    def convert(self, p, other):
        """Map parameter vector p of this model to `other` (radius offsets projected)."""
        q = other.start(p[:N_GEO + 2 * self.nb])
        if other.per_ball_radius:
            q[N_GEO + 2 * self.nb:] = other.Q.T @ (self.radii(p) - self.R)
        return q

    def radius_lean_deg(self, p, z):
        """Radial lean of the ball column implied by a linear radius trend with height."""
        return np.rad2deg(np.arctan(np.polyfit(z, self.radii(p), 1)[0]))

    def start(self, p):
        """Extend a vector without radius offsets (e.g. from initial_guess) to this model."""
        return np.concatenate([p, np.zeros(self.n_params - p.size)])

    def unpack(self, p):
        return p[:N_GEO], p[N_GEO:N_GEO + self.nb], p[N_GEO + self.nb:N_GEO + 2 * self.nb]

    def radii(self, p):
        if not self.per_ball_radius:
            return np.full(self.nb, float(self.R))
        return self.R + self.Q @ p[N_GEO + 2 * self.nb:]

    def predict(self, p, return_depth=False):
        geo, z, phi = self.unpack(p)
        u, v, depth = project(geo, z, phi, self.ball, self.ang, self.radii(p), self.pixel, self.sense,
                              return_depth=True)
        return (np.stack([u, v], 1), depth) if return_depth else np.stack([u, v], 1)

    def residuals(self, p, sel=None):
        r = self.predict(p) - self.uv
        if sel is not None:
            r = r[sel]
        return r.ravel()


def initial_guess(obs, R, pixel, sense):
    """Closed-form start values from per-ball sinusoid fits."""
    ball, ang = obs[:, 0].astype(int), np.deg2rad(obs[:, 1])
    nb = ball.max() + 1
    amp, cu, alpha, vmean, vquad = [], [], [], [], []
    for k in range(nb):
        s = ball == k
        A = np.stack([np.ones(s.sum()), np.cos(ang[s]), np.sin(ang[s])], 1)
        cu_k = np.linalg.lstsq(A, obs[s, 2], rcond=None)[0]
        cv_k = np.linalg.lstsq(A, obs[s, 3], rcond=None)[0]
        al = np.arctan2(cu_k[2], cu_k[1])
        amp.append(np.hypot(cu_k[1], cu_k[2]))
        cu.append(cu_k[0])
        alpha.append(al)
        vmean.append(cv_k[0])
        # v-modulation in quadrature with u (depth term)
        vquad.append(-cv_k[1] * np.sin(al) + cv_k[2] * np.cos(al))
    amp, vmean, vquad = map(np.array, (amp, vmean, vquad))
    M = np.median(amp) * pixel / R
    slope, icpt = np.polyfit(vmean, vquad, 1)
    ppy = -icpt / slope
    SOD = R / abs(slope)
    SDD = M * SOD
    ppx = np.median(cu)
    z = -(vmean - ppy) * pixel / M
    phi = -sense * np.array(alpha)
    return np.concatenate([[SOD, SDD, ppx, ppy, 0, 0, 0], z, phi])


def fit(model, p0, sel=None, free=None):
    """Least-squares fit; parameters with free == False stay at their p0 value."""
    free = np.ones(p0.size, bool) if free is None else free

    def fun(q):
        p = p0.copy()
        p[free] = q
        return model.residuals(p, sel)

    res = least_squares(fun, p0[free], x_scale="jac", method="trf",
                        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=2000)
    p = p0.copy()
    p[free] = res.x
    res.p = p
    return res


def calibrate(obs, model, p_start, free_geo, args):
    """Fit with outlier rejection and estimate the parameter uncertainties."""
    free = np.ones(p_start.size, bool)
    free[:N_GEO] = free_geo
    p0 = p_start.copy()
    p0[:N_GEO][~free_geo] = 0.0  # fixed tilts are zero
    res = fit(model, p0, None, free)

    # outlier rejection (5 sigma on the point distance), then refit
    for _ in range(3):
        rr = model.residuals(res.p).reshape(-1, 2)
        dist = np.hypot(rr[:, 0], rr[:, 1])
        sigma = np.median(dist) / 1.1774  # Rayleigh median -> sigma per axis
        good = dist < 5 * sigma
        res = fit(model, res.p, good, free)
    p_best = res.p
    rr = model.residuals(p_best).reshape(-1, 2)

    # covariance of the free parameters, scaled by the residual variance
    s2 = 2 * res.cost / (res.fun.size - free.sum())
    cov_f = np.linalg.pinv(res.jac.T @ res.jac) * s2
    cov = np.zeros((p_best.size, p_best.size))
    cov[np.ix_(free, free)] = cov_f
    sig_cov = np.sqrt(np.diag(cov))[:N_GEO]
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov[:N_GEO, :N_GEO] / np.outer(sig_cov, sig_cov)

    # bootstrap over projections: resample whole frames, since residuals of
    # balls in the same frame are correlated (angle jitter, stage wobble)
    frame_of_obs = obs[:, 4].astype(int)
    nfr = frame_of_obs.max() + 1
    boot = []
    rng = np.random.default_rng(0)
    for b in range(args.nboot):
        w = np.bincount(rng.integers(0, nfr, nfr), minlength=nfr)[frame_of_obs]
        idx = np.repeat(np.arange(len(obs)), w)
        idx = idx[good[idx]]
        boot.append(fit(model, p_best, idx, free).p[:N_GEO])
        print(f"  bootstrap {b + 1}/{args.nboot}", end="\r")
    if args.nboot:
        print()
    boot = np.array(boot).reshape(-1, N_GEO)
    sig_boot = boot.std(axis=0, ddof=1) if len(boot) > 1 else np.full(N_GEO, np.nan)

    # systematics: refit with perturbed radius and pixel size
    def refit_with(Rx, px):
        return fit(model.variant(R=Rx, pixel=px), p_best, good, free).p[:N_GEO]

    sig_sys = np.zeros(N_GEO)
    R, pixel = model.R, model.pixel
    if args.radius_tol > 0:
        sig_sys = np.hypot(sig_sys, (refit_with(R + args.radius_tol, pixel)
                                     - refit_with(R - args.radius_tol, pixel)) / 2)
    if args.pixel_tol > 0:
        sig_sys = np.hypot(sig_sys, (refit_with(R, pixel + args.pixel_tol)
                                     - refit_with(R, pixel - args.pixel_tol)) / 2)

    sig_tot = np.hypot(np.fmax(sig_cov, np.nan_to_num(sig_boot)), sig_sys)
    sig_tot[~free_geo] = 0

    # projection of the rotation axis: intersection of the axis image with row ppy
    geo = p_best[:N_GEO]
    ua, va = project(geo, np.array([-50.0, 50.0]), np.zeros(2), np.arange(2), np.zeros(2),
                     0.0, pixel, model.sense)
    cor = ua[0] + (geo[3] - va[0]) * (ua[1] - ua[0]) / (va[1] - va[0])
    axis_tilt = np.rad2deg(np.arctan2(ua[1] - ua[0], -(va[1] - va[0])))
    return dict(p=p_best, geo=geo, rr=rr, good=good, sig_cov=sig_cov, sig_boot=sig_boot,
                sig_sys=sig_sys, sig_tot=sig_tot, corr=corr, boot=boot, free=free_geo,
                cor=cor, axis_tilt=axis_tilt)


def ball_scale(model, p, diam_px, good, ball_diameter):
    """Scale factor lambda = measured / predicted ball shadow diameter.

    A small sphere of diameter D at depth L casts a shadow of D * SDD / (p * L).
    Scaling R, SOD and z by 1/lambda (SDD and p fixed) leaves the ball positions
    unchanged and makes the predicted sizes match. Returns lambda, its standard
    error from the scatter between balls, and the per-ball ratios.
    """
    _, depth = model.predict(p, return_depth=True)
    ratio = diam_px * model.pixel * depth / (ball_diameter * p[1])
    per_ball = np.array([np.median(ratio[(model.ball == k) & good]) for k in range(model.nb)])
    lam = per_ball.mean()
    err = per_ball.std(ddof=1) / np.sqrt(model.nb) if model.nb > 1 else np.nan
    return lam, err, per_ball


# --------------------------------------------------------------- main --------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--proj", default="xcal_*.fits", help="projection file pattern")
    ap.add_argument("--dc", default="xdc_*.fits", help="dark current file pattern")
    ap.add_argument("--ob", default="xob_*.fits", help="open beam file pattern")
    ap.add_argument("--step", "-N", type=int, default=5, help="use every N-th projection")
    ap.add_argument("--pixel-size", type=float, default=0.139, help="detector pixel pitch [mm]")
    ap.add_argument("--pixel-tol", type=float, default=0.0, help="1-sigma tolerance of pixel pitch [mm]")
    ap.add_argument("--ball-diameter", type=float, default=2.0, help="[mm]")
    ap.add_argument("--radius", type=float, default=25.0, help="ball circle radius around the axis [mm]")
    ap.add_argument("--radius-tol", type=float, default=0.01, help="1-sigma tolerance of the radius [mm]")
    ap.add_argument("--ball-diameter-tol", type=float, default=0.005,
                    help="1-sigma tolerance of the ball diameter [mm], used with --scale ball")
    ap.add_argument("--radii", choices=["per-ball", "common"], default="per-ball",
                    help="fit one circle radius per ball (mean fixed to R) or one common radius")
    ap.add_argument("--scale", choices=["radius", "ball"], default="radius",
                    help="absolute scale from the nominal circle radius or from the ball diameter")
    ap.add_argument("--angle-key", default="sp2_ry", help="header key with rotation angle in degrees")
    ap.add_argument("--arc", type=float, default=360.0, help="scan arc if no angle in header [deg]")
    ap.add_argument("--min-coverage", type=float, default=0.5,
                    help="keep balls seen in at least this fraction of the projections")
    ap.add_argument("--nboot", type=int, default=200, help="bootstrap repetitions (0 = off)")
    ap.add_argument("--outdir", default=None, help="output directory (default <path>/xcal_results)")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    outdir = args.outdir or os.path.join(args.path, "xcal_results")
    os.makedirs(outdir, exist_ok=True)
    pixel, R = args.pixel_size, args.radius

    # --- references and projections
    dc, gain, bad = load_references(args.path, args.dc, args.ob)
    fill_idx = ndi.distance_transform_edt(bad, return_distances=False, return_indices=True)

    files = sorted(glob.glob(os.path.join(args.path, args.proj)))
    nfiles = len(files)
    angles = [read_angle(f, args.angle_key) for f in files]
    if any(a is None for a in angles):
        print(f"Angle key '{args.angle_key}' not found in all headers, assuming uniform {args.arc} deg arc")
        angles = list(np.linspace(0, args.arc, nfiles))
    sel_idx = list(range(0, nfiles, args.step))
    print(f"{nfiles} projections, using {len(sel_idx)} (every {args.step}.)")

    ball_px_guess = None
    frames, frame_ang, frame_name = [], [], []
    for i in sel_idx:
        n = normalize(read_image(files[i]), dc, gain, fill_idx)
        if ball_px_guess is None:
            # first estimate of the ball shadow size from a dark-blob threshold
            lab, nl = ndi.label(n < 0.5 * np.median(n))
            areas = ndi.sum_labels(np.ones_like(n), lab, np.arange(1, nl + 1))
            areas = areas[(areas > 50) & (areas < 0.01 * n.size)]
            ball_px_guess = 2 * np.sqrt(np.median(areas) / np.pi)
            print(f"Ball shadow diameter ~ {ball_px_guess:.1f} px")
        det = detect_balls(n, ball_px_guess)
        frames.append(det)
        frame_ang.append(angles[i])
        frame_name.append(os.path.basename(files[i]))
        print(f"  {frame_name[-1]}  angle {angles[i]:8.3f}  balls {len(det)}", end="\r")
    print()

    # --- track
    spacing = np.median(np.concatenate([np.diff(np.sort(f[:, 1])) for f in frames if len(f) > 1]))
    tracks = track_balls(frames, max_dv=0.4 * spacing)
    keep = [t for t in tracks if len(t) >= args.min_coverage * len(frames)]
    keep.sort(key=lambda t: np.median([d[1] for d in t.values()]))
    print(f"{len(tracks)} tracks, {len(keep)} balls kept (coverage >= {args.min_coverage:.0%})")

    obs = []  # ball, angle, u, v, frame, diameter
    for k, t in enumerate(keep):
        for fi, d in t.items():
            obs.append((k, frame_ang[fi], d[0], d[1], fi, d[2]))
    obs = np.array(obs)

    # --- fit (try both rotation senses, keep the better one)
    best = None
    for sense in (+1, -1):
        m = Model(obs, R, pixel, sense)
        r = fit(m, initial_guess(obs, R, pixel, sense))
        print(f"rotation sense {sense:+d}: rms {np.sqrt(np.mean(r.fun**2)):.4f} px")
        if best is None or r.cost < best[1].cost:
            best = (m, r, sense)
    _, res0, sense = best
    per_ball = args.radii == "per-ball"
    model = Model(obs, R, pixel, sense, per_ball)
    p_start = fit(model, model.start(res0.p)).p

    # --- absolute scale from the ball size: R, SOD and z scale together, SDD does not
    D, all_obs = args.ball_diameter, np.ones(len(obs), bool)
    lam, lam_err, lam_ball = ball_scale(model, p_start, obs[:, 5], all_obs, D)
    R_ball = R / lam
    R_ball_err = R_ball * np.hypot(lam_err / lam, args.ball_diameter_tol / D)
    print(f"Ball size implies a circle radius of {R_ball:.3f} +/- {R_ball_err:.3f} mm (nominal {R} mm)")
    R_nominal, radius_tol_nominal = R, args.radius_tol
    if args.scale == "ball":
        R, args.radius_tol = R_ball, R_ball_err
        p_start = p_start.copy()
        p_start[0] /= lam                                   # SOD
        p_start[N_GEO:N_GEO + model.nb] /= lam              # heights
        p_start[N_GEO + 2 * model.nb:] /= lam               # radius offsets
        model = Model(obs, R, pixel, sense, per_ball)
        p_start = fit(model, p_start).p

    print("Full model (all detector tilts free)")
    full = calibrate(obs, model, p_start, np.ones(N_GEO, bool), args)
    print("Reduced model (eta = theta = 0: detector perpendicular to the source-axis plane)")
    # with eta fixed, a radial lean of the ball column (radius trend with height) is identifiable
    red_model = model.variant(radius_trend=True) if per_ball else model
    reduced = calibrate(obs, red_model, model.convert(full["p"], red_model),
                        np.array([1, 1, 1, 1, 0, 0, 1], bool), args)

    p_best, rr, good = full["p"], full["rr"], full["good"]
    geo = full["geo"]
    SOD, SDD, ppx, ppy = geo[:4]
    mag = SDD / SOD
    radii = model.radii(p_best)
    lean_red = red_model.radius_lean_deg(reduced["p"], red_model.unpack(reduced["p"])[1]) if per_ball else 0.0
    lam_fit, lam_fit_err, _ = ball_scale(model, p_best, obs[:, 5], good, D)

    def table(c):
        t = [f"{'param':6s} {'value':>12s} {'sig_cov':>10s} {'sig_boot':>10s} {'sig_sys':>10s} "
             f"{'sig_total':>10s}  unit", "-" * 72]
        for k in range(N_GEO):
            if c["free"][k]:
                t.append(f"{GEO_NAMES[k]:6s} {c['geo'][k]:12.4f} {c['sig_cov'][k]:10.4f} "
                         f"{c['sig_boot'][k]:10.4f} {c['sig_sys'][k]:10.4f} {c['sig_tot'][k]:10.4f}  "
                         f"{GEO_UNITS[k]}")
            else:
                t.append(f"{GEO_NAMES[k]:6s} {c['geo'][k]:12.4f} {'(fixed)':>10s}{'':33s}  {GEO_UNITS[k]}")
        g = c["good"]
        t += ["-" * 72,
              f"fit rms: u {np.sqrt(np.mean(c['rr'][g, 0]**2)):.4f} px, v {np.sqrt(np.mean(c['rr'][g, 1]**2)):.4f} px"
              f",  {g.sum()} points used, {(~g).sum()} rejected",
              f"centre of rotation (axis image at row ppy): {c['cor']:.3f} px   (ppx - COR = "
              f"{c['geo'][2] - c['cor']:.3f} px)",
              f"rotation axis tilt in the image          : {c['axis_tilt']:.4f} deg"]
        return t

    # --- report
    lines = ["Cone-beam geometry calibration", "=" * 72,
             f"data            : {args.path}",
             f"projections     : {len(frames)} of {nfiles} (step {args.step})",
             f"balls tracked   : {model.nb}",
             f"pixel size      : {pixel} +/- {args.pixel_tol} mm,  ball diameter {D} mm",
             f"circle radius R : {R:.4f} +/- {args.radius_tol:.4f} mm  "
             f"(scale from {'ball diameter' if args.scale == 'ball' else 'nominal radius'}, "
             f"radii {args.radii})",
             f"rotation sense  : {sense:+d}",
             "",
             "Scale check (independent of the nominal radius):",
             f"  ball diameter {D} mm implies R = {R_ball:.3f} +/- {R_ball_err:.3f} mm"
             f"  (nominal {R_nominal} +/- {radius_tol_nominal} mm)",
             f"  measured/predicted shadow size with the final R: {lam_fit:.4f} +/- {lam_fit_err:.4f}"
             f"  (per ball {np.min(lam_ball):.3f} .. {np.max(lam_ball):.3f} before rescaling)",
             *([f"  per-ball radii (full model): {radii.min():.4f} .. {radii.max():.4f} mm, "
                f"spread (std) {radii.std(ddof=1) * 1000:.1f} um, no trend with height",
                f"  reduced model: radial lean of the ball column {lean_red:.3f} deg "
                f"(trades off against the detector tilt eta of the full model)"] if per_ball else []),
             "",
             "FULL MODEL (detector tilts eta, theta, psi free)",
             *table(full),
             "",
             "REDUCED MODEL (eta = theta = 0, i.e. ppx = COR; geometry assumed by most",
             "reconstruction tools - use this if your reconstructor has no detector tilt)",
             *table(reduced),
             "",
             "sig_cov  : 1 sigma from the fit covariance (residual-scaled)",
             "sig_boot : 1 sigma from bootstrap over projections",
             "sig_sys  : systematic from radius/pixel-size tolerance",
             "sig_total: sqrt(max(sig_cov, sig_boot)^2 + sig_sys^2)",
             "",
             "Derived (full model):",
             f"  magnification SDD/SOD     : {mag:.5f}",
             f"  ODD = SDD - SOD           : {SDD - SOD:.3f} mm",
             f"  effective pixel size      : {pixel / mag * 1000:.3f} um",
             "",
             "Correlation matrix of the geometry parameters (full model):",
             "        " + " ".join(f"{n:>7s}" for n in GEO_NAMES)]
    for k in range(N_GEO):
        lines.append(f"{GEO_NAMES[k]:7s} " + " ".join(f"{c:7.3f}" for c in full["corr"][k]))
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(outdir, "geometry.txt"), "w") as f:
        f.write(report + "\n")

    settings = {"data": args.path, "projections_used": len(frames), "projections_total": nfiles,
                "step": args.step, "rotation_sense": sense, "pixel_size_mm": pixel,
                "pixel_size_tol_mm": args.pixel_tol, "radius_mm": R, "radius_tol_mm": args.radius_tol,
                "radius_nominal_mm": R_nominal, "radii": args.radii, "scale": args.scale,
                "ball_diameter_mm": D, "radius_from_ball_size_mm": R_ball,
                "radius_from_ball_size_err_mm": R_ball_err, "nboot": args.nboot}
    for name, c in (("full", full), ("reduced", reduced)):
        save_model_json(os.path.join(outdir, f"geometry_{name}.json"), name, c,
                        model if name == "full" else red_model, settings)
    np.savetxt(os.path.join(outdir, "ball_positions.csv"),
               np.column_stack([obs[:, :4], obs[:, 5], rr, good]), delimiter=",", fmt="%.4f",
               header="ball,angle_deg,u_px,v_px,diameter_px,res_u,res_v,used")

    if not args.no_plots:
        make_plots(outdir, obs, rr, good, model, p_best, full["boot"], files, sel_idx, dc, gain, fill_idx)
    print(f"\nResults written to {outdir}")


MODEL_DESCRIPTIONS = {
    "full": "all detector tilts (eta, theta, psi) free; ball radii without trend in height",
    "reduced": "eta = theta = 0: detector perpendicular to the source-axis plane, ppx = COR; "
               "ball radii may vary linearly with height (radially leaning ball column)",
}


def model_record(name, c, model, settings):
    """JSON-serializable result of one calibrate() run."""
    _, z, phi = model.unpack(c["p"])
    g = c["good"]
    return {
        "model": name,
        "description": MODEL_DESCRIPTIONS.get(name, ""),
        "parameters": {n: {"value": float(c["geo"][k]), "unit": GEO_UNITS[k], "free": bool(c["free"][k]),
                           "sigma_cov": float(c["sig_cov"][k]), "sigma_boot": float(c["sig_boot"][k]),
                           "sigma_sys": float(c["sig_sys"][k]), "sigma_total": float(c["sig_tot"][k])}
                       for k, n in enumerate(GEO_NAMES)},
        "cor_px": float(c["cor"]),
        "axis_tilt_deg": float(c["axis_tilt"]),
        "magnification": float(c["geo"][1] / c["geo"][0]),
        "rms_px": {"u": float(np.sqrt(np.mean(c["rr"][g, 0] ** 2))),
                   "v": float(np.sqrt(np.mean(c["rr"][g, 1] ** 2)))},
        "points_used": int(g.sum()),
        "points_rejected": int((~g).sum()),
        "correlation": {"names": GEO_NAMES,
                        "matrix": np.nan_to_num(c["corr"]).round(6).tolist()},
        "ball_heights_mm": z.tolist(),
        "ball_radii_mm": model.radii(c["p"]).tolist(),
        "ball_phases_deg": np.rad2deg(phi).tolist(),
        "settings": settings,
    }


def save_model_json(fname, name, c, model, settings):
    with open(fname, "w") as f:
        json.dump(model_record(name, c, model, settings), f, indent=2)


def make_plots(outdir, obs, rr, good, model, p, boot, files, sel_idx, dc, gain, fill_idx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    geo = p[:N_GEO]
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    n = normalize(read_image(files[sel_idx[0]]), dc, gain, fill_idx)
    ax[0].imshow(n, cmap="gray", vmin=0, vmax=1.2)
    dense = np.linspace(0, 2 * np.pi, 361)
    for k in range(model.nb):
        _, z, phi = model.unpack(p)
        u, v = project(geo, z, phi, np.full(dense.size, k), dense, model.radii(p), model.pixel, model.sense)
        ax[0].plot(u, v, "-", lw=0.8)
    ax[0].scatter(obs[:, 2], obs[:, 3], s=1, c="r")
    ax[0].plot(geo[2], geo[3], "c+", ms=20, mew=2, label="piercing point")
    ax[0].legend()
    ax[0].set_title("Detected balls and fitted trajectories")
    sc = ax[1].scatter(obs[good, 1], obs[good, 0] + 0 * rr[good, 0], c=np.hypot(*rr[good].T), s=6, cmap="viridis")
    plt.colorbar(sc, ax=ax[1], label="|residual| [px]")
    ax[1].set_xlabel("angle [deg]")
    ax[1].set_ylabel("ball index")
    ax[1].set_title("Residual magnitude")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "trajectories.png"), dpi=100)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for j, lab in enumerate(["u", "v"]):
        ax[j].scatter(obs[good, 1], rr[good, j], s=2, c=obs[good, 0], cmap="tab20")
        ax[j].scatter(obs[~good, 1], rr[~good, j], s=8, c="r", marker="x")
        ax[j].set_ylabel(f"residual {lab} [px]")
        ax[j].set_ylim(np.array([-1, 1]) * 6 * np.std(rr[good, j]))
    ax[1].set_xlabel("angle [deg]")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "residuals.png"), dpi=100)

    if len(boot) > 2:
        fig, ax = plt.subplots(1, N_GEO, figsize=(3 * N_GEO, 3))
        for k in range(N_GEO):
            ax[k].hist(boot[:, k], bins=25)
            ax[k].axvline(geo[k], color="r")
            ax[k].set_title(f"{GEO_NAMES[k]} [{GEO_UNITS[k]}]")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "bootstrap.png"), dpi=100)
    plt.close("all")


if __name__ == "__main__":
    main()
