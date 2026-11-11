"""Unit tests for xcal_geometry.py.

Most tests use synthetic data generated with the forward model, so the true
geometry is known. The tests at the end use the calibration data in the parent
directory and are skipped when it is not present.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from scipy import ndimage as ndi

import xcal_geometry as xg

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R, PIXEL = 25.0, 0.139

# geometry close to the real set-up: SOD, SDD, ppx, ppy, eta, theta, psi
GEO_TILTED = np.array([350.0, 1300.0, 900.0, 1300.0, 0.6, 0.1, 0.08])
GEO_IDEAL = np.array([350.0, 1300.0, 900.0, 1300.0, 0.0, 0.0, 0.08])


def make_obs(geo, sense=-1, nballs=15, step=5.0, noise=0.0, seed=0, radii=R, ball_diameter=2.0):
    """Synthetic observations [ball, angle, u, v, frame, diameter] from the forward model.

    The diameter column is the exact shadow size of a ball of `ball_diameter` mm.
    """
    rng = np.random.default_rng(seed)
    z = np.linspace(-35, 35, nballs)
    phi = 0.3 + rng.normal(0, 0.01, nballs)
    ang = np.arange(0, 360, step)
    ball = np.repeat(np.arange(nballs), ang.size)
    frame = np.tile(np.arange(ang.size), nballs)
    a = np.tile(ang, nballs)
    u, v, depth = xg.project(geo, z, phi, ball, np.deg2rad(a), radii, PIXEL, sense, return_depth=True)
    u = u + rng.normal(0, noise, u.size)
    v = v + rng.normal(0, noise, v.size)
    diam = ball_diameter * geo[1] / (PIXEL * depth)
    return np.column_stack([ball, a, u, v, frame, diam])


def options(**kw):
    return SimpleNamespace(**{"nboot": 0, "radius_tol": 0.0, "pixel_tol": 0.0, **kw})


# ---------------------------------------------------------------- I/O --------
def write_raw_fits(fname, cards, data):
    """Write a FITS file byte by byte, so that non-standard cards survive."""
    head = b"".join(c.ljust(80)[:80] for c in cards + [b"END"])
    head += b" " * (-len(head) % 2880)
    body = data.astype(">i2").tobytes()
    body += b"\0" * (-len(body) % 2880)
    with open(fname, "wb") as f:
        f.write(head + body)


def detector_like_fits(fname, angle_card):
    cards = [b"SIMPLE  =                    T", b"BITPIX  =                   16",
             b"NAXIS   =                    2", b"NAXIS1  =                    4",
             b"NAXIS2  =                    3", b"BZERO   =                32768",
             b"BSCALE  =                    1",
             b"MODESTR = '1x1 0.5pF VG1" + b"\0" * 60,  # NUL bytes as written by the detector
             b"HIERARCH sample_position_2/sp2_ry_brake = b'open'",
             angle_card]
    data = (np.arange(12).reshape(3, 4) + 1000 - 32768)
    write_raw_fits(fname, cards, data)


def test_read_angle_from_bytes_card(tmp_path):
    f = tmp_path / "a.fits"
    detector_like_fits(f, b"HIERARCH sample_position_2/sp2_ry = b'12.500 deg'")
    assert xg.read_angle(str(f), "sp2_ry") == pytest.approx(12.5)
    assert xg.read_angle(str(f), "sample_position_2/sp2_ry") == pytest.approx(12.5)


def test_read_angle_negative_and_exponent(tmp_path):
    f = tmp_path / "a.fits"
    detector_like_fits(f, b"HIERARCH sample_position_2/sp2_ry = b'-1.5e1 deg'")
    assert xg.read_angle(str(f), "sp2_ry") == pytest.approx(-15.0)


def test_read_angle_missing_key(tmp_path):
    f = tmp_path / "a.fits"
    detector_like_fits(f, b"HIERARCH other/key = b'1.0 deg'")
    assert xg.read_angle(str(f), "sp2_ry") is None


def test_read_image_applies_bzero(tmp_path):
    f = tmp_path / "a.fits"
    detector_like_fits(f, b"HIERARCH sample_position_2/sp2_ry = b'0 deg'")
    img = xg.read_image(str(f))
    assert img.dtype == np.float32
    np.testing.assert_array_equal(img, np.arange(12).reshape(3, 4) + 1000)


def test_load_references_median_and_bad_pixels(tmp_path):
    rng = np.random.default_rng(1)
    for i in range(5):
        dc = 100 + rng.normal(0, 1, (20, 30))
        ob = 10000 + rng.normal(0, 10, (20, 30))
        ob[5, 7] = 120  # dead pixel: no gain
        if i == 0:
            dc[0, 0] = 60000  # outlier in a single frame, removed by the median
        fits.writeto(tmp_path / f"xdc_{i:05d}.fits", dc.astype(np.uint16))
        fits.writeto(tmp_path / f"xob_{i:05d}.fits", ob.astype(np.uint16))
    dc, gain, bad = xg.load_references(str(tmp_path), "xdc_*.fits", "xob_*.fits")
    assert dc[0, 0] < 200
    assert np.median(gain) == pytest.approx(9900, rel=0.01)
    assert bad[5, 7] and bad.sum() == 1


def test_load_references_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        xg.load_references(str(tmp_path), "xdc_*.fits", "xob_*.fits")


def test_normalize_fills_bad_pixels():
    dc = np.full((10, 10), 100.0)
    gain = np.full((10, 10), 1000.0)
    bad = np.zeros((10, 10), bool)
    bad[4:6, 4:6] = True
    fill_idx = ndi.distance_transform_edt(bad, return_distances=False, return_indices=True)
    proj = np.full((10, 10), 600.0)
    proj[bad] = 0
    n = xg.normalize(proj, dc, gain, fill_idx)
    np.testing.assert_allclose(n, 0.5)


# ---------------------------------------------------------- detection --------
def ball_image(centres, d_px=50, shape=(400, 300), mu=0.08):
    """Transmission image of spheres in front of a sloped background."""
    rr, cc = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
    att = np.zeros(shape)
    r = d_px / 2
    for u, v in centres:
        # average the chord length over 4x4 sub-pixels for a smooth edge
        for dy in np.linspace(-0.375, 0.375, 4):
            for dx in np.linspace(-0.375, 0.375, 4):
                q = r**2 - (cc + dx - u) ** 2 - (rr + dy - v) ** 2
                att += 2 * np.sqrt(np.clip(q, 0, None)) / 16
    bg = 0.7 + 0.2 * cc / shape[1]
    return bg * np.exp(-mu * att)


def test_detect_balls_subpixel_centres():
    centres = [(100.3, 80.7), (180.6, 200.2), (120.25, 320.9)]
    det = xg.detect_balls(ball_image(centres), 50)
    assert len(det) == 3
    det = det[np.argsort(det[:, 1])]
    np.testing.assert_allclose(det[:, :2], centres, atol=0.05)
    np.testing.assert_allclose(det[:, 2], 50, rtol=0.01)  # edge-based shadow diameter


def test_detect_balls_rejects_truncated():
    det = xg.detect_balls(ball_image([(150, 200), (10, 100), (150, 395)]), 50)
    assert len(det) == 1
    np.testing.assert_allclose(det[0, :2], (150, 200), atol=0.05)


def test_detect_balls_empty():
    assert xg.detect_balls(np.full((100, 100), 0.8), 50).shape == (0, 3)


# ----------------------------------------------------------- tracking --------
def test_track_balls_reconnects_lost_ball():
    nfr, vs = 12, np.array([100.0, 200.0, 300.0])
    frames = []
    for f in range(nfr):
        det = np.column_stack([500 + 10 * f + np.zeros(3), vs + 2 * f, np.full(3, 50)])
        if 4 <= f <= 6:
            det = det[[0, 2]]  # middle ball not detected
        frames.append(det)
    tracks = xg.track_balls(frames, max_dv=40)
    assert len(tracks) == 3
    lengths = sorted(len(t) for t in tracks)
    assert lengths == [nfr - 3, nfr, nfr]
    for t in tracks:  # every track follows one ball
        v0 = np.array([d[1] - 2 * f for f, d in t.items()])
        assert np.ptp(v0) < 1e-9


def test_track_balls_new_ball_entering():
    frames = [np.array([[500, 100, 50], [500, 200, 50]]),
              np.array([[500, 101, 50], [500, 201, 50], [500, 301, 50]])]
    tracks = xg.track_balls(frames, max_dv=40)
    assert sorted(len(t) for t in tracks) == [1, 2, 2]


# -------------------------------------------------------------- model --------
def test_project_untilted_magnification():
    geo = np.array([350.0, 1300.0, 900.0, 1000.0, 0, 0, 0])
    M = 1300 / 350
    # ball at angle 0: on the x axis, at the same depth as the axis
    u, v = xg.project(geo, np.array([10.0]), np.zeros(1), np.zeros(1, int), np.zeros(1), R, PIXEL, 1)
    assert u[0] == pytest.approx(900 + R * M / PIXEL)
    assert v[0] == pytest.approx(1000 - 10 * M / PIXEL)
    # ball at -90 deg: on the axis line, closer to the source -> larger magnification
    u, v = xg.project(geo, np.array([10.0]), np.zeros(1), np.zeros(1, int),
                      np.array([-np.pi / 2]), R, PIXEL, 1)
    assert u[0] == pytest.approx(900)
    assert v[0] == pytest.approx(1000 - 10 * 1300 / (350 - R) / PIXEL)


def test_project_rotation_sense():
    def proj(ang, sense):
        return np.array(xg.project(GEO_TILTED, np.array([20.0]), np.zeros(1), np.zeros(1, int),
                                   np.array([ang]), R, PIXEL, sense)).ravel()

    # reversing the sense is the same as negating the angle
    np.testing.assert_allclose(proj(0.3, -1), proj(-0.3, 1))
    # +/- angle puts the ball at opposite depth: different magnification of z
    assert abs(proj(0.3, 1)[1] - proj(-0.3, 1)[1]) > 1


@pytest.mark.parametrize("geo", [GEO_IDEAL, GEO_TILTED,
                                 np.array([400.0, 1100.0, 850.0, 1100.0, -2.0, 1.5, -0.5])])
def test_project_central_ray_hits_piercing_point(geo):
    SOD, SDD = geo[:2]
    eta, theta, psi = np.deg2rad(geo[4:])
    n = xg.rot("z", theta) @ xg.rot("x", eta) @ xg.rot("y", psi) @ np.array([0, 1.0, 0])
    t = np.array([100.0, 300.0, 500.0])
    X = np.array([0, -SOD, 0]) + t[:, None] * n
    r = np.hypot(X[:, 0], X[:, 1])
    a = np.arctan2(X[:, 1], X[:, 0])
    for k in range(3):  # radius differs per point, so project one at a time
        u, v = xg.project(geo, X[k:k + 1, 2], np.zeros(1), np.zeros(1, int), a[k:k + 1], r[k], PIXEL, 1)
        assert u[0] == pytest.approx(geo[2], abs=1e-6)
        assert v[0] == pytest.approx(geo[3], abs=1e-6)


def test_rot_is_orthonormal():
    for ax in "xyz":
        m = xg.rot(ax, 0.7)
        np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(m) == pytest.approx(1)


def test_model_residuals_zero_at_truth():
    obs = make_obs(GEO_TILTED)
    nb = int(obs[:, 0].max()) + 1
    rng = np.random.default_rng(0)
    z = np.linspace(-35, 35, nb)
    phi = 0.3 + rng.normal(0, 0.01, nb)
    m = xg.Model(obs, R, PIXEL, -1)
    p = np.concatenate([GEO_TILTED, z, phi])
    np.testing.assert_allclose(m.residuals(p), 0, atol=1e-9)
    sel = np.zeros(len(obs), bool)
    sel[:10] = True
    assert m.residuals(p, sel).size == 20


# ---------------------------------------------------------------- fit --------
def test_initial_guess_untilted():
    obs = make_obs(GEO_IDEAL, noise=0.1)
    p0 = xg.initial_guess(obs, R, PIXEL, -1)
    assert p0[0] == pytest.approx(350, rel=0.05)
    assert p0[1] == pytest.approx(1300, rel=0.05)
    assert p0[2] == pytest.approx(900, abs=2)
    assert p0[3] == pytest.approx(1300, abs=10)


@pytest.mark.parametrize("sense", [1, -1])
def test_fit_noise_free_recovers_geometry(sense):
    obs = make_obs(GEO_TILTED, sense=sense)
    m = xg.Model(obs, R, PIXEL, sense)
    res = xg.fit(m, xg.initial_guess(obs, R, PIXEL, sense))
    err = np.abs(res.p[:xg.N_GEO] - GEO_TILTED)
    assert np.all(err < [1e-3, 1e-3, 0.02, 0.02, 1e-4, 1e-4, 1e-5]), err
    assert np.sqrt(np.mean(res.fun**2)) < 1e-6


def test_wrong_rotation_sense_fits_worse():
    obs = make_obs(GEO_TILTED, sense=-1, noise=0.1)
    cost = {s: xg.fit(xg.Model(obs, R, PIXEL, s), xg.initial_guess(obs, R, PIXEL, s)).cost for s in (1, -1)}
    assert cost[-1] < 0.01 * cost[1]


def test_fit_fixed_parameters_stay_fixed():
    obs = make_obs(GEO_IDEAL)
    m = xg.Model(obs, R, PIXEL, -1)
    p0 = xg.initial_guess(obs, R, PIXEL, -1)
    free = np.ones(p0.size, bool)
    free[4:6] = False
    res = xg.fit(m, p0, free=free)
    assert res.p[4] == p0[4] and res.p[5] == p0[5]
    np.testing.assert_allclose(res.p[:4], GEO_IDEAL[:4], atol=0.02)


def calibrated(geo, free_geo, noise=0.1, seed=0, outliers=(), **kw):
    obs = make_obs(geo, noise=noise, seed=seed)
    obs[list(outliers), 2] += 20.0
    m = xg.Model(obs, R, PIXEL, -1)
    p0 = xg.fit(m, xg.initial_guess(obs, R, PIXEL, -1)).p
    return obs, xg.calibrate(obs, m, p0, free_geo, options(**kw))


def test_calibrate_full_model_within_uncertainty():
    _, c = calibrated(GEO_TILTED, np.ones(xg.N_GEO, bool), nboot=30)
    err = np.abs(c["geo"] - GEO_TILTED)
    assert np.all(err < 4 * c["sig_tot"]), (err, c["sig_tot"])
    # bootstrap and covariance agree for independent noise
    ratio = c["sig_boot"] / c["sig_cov"]
    assert np.all((ratio > 0.5) & (ratio < 2)), ratio
    # COR is the image of the axis: shifted from ppx by the yaw theta
    shift = GEO_TILTED[1] * np.tan(np.deg2rad(GEO_TILTED[5])) / PIXEL
    assert c["geo"][2] - c["cor"] == pytest.approx(-shift, rel=0.05)
    assert c["axis_tilt"] == pytest.approx(-GEO_TILTED[6], abs=0.01)


def test_calibrate_reduced_model():
    free = np.array([1, 1, 1, 1, 0, 0, 1], bool)
    _, c = calibrated(GEO_IDEAL, free)
    assert c["geo"][4] == 0 and c["geo"][5] == 0
    assert c["sig_tot"][4] == 0 and c["sig_tot"][5] == 0
    err = np.abs(c["geo"] - GEO_IDEAL)[free]
    assert np.all(err < 4 * c["sig_tot"][free])
    assert c["cor"] == pytest.approx(c["geo"][2], abs=1e-6)


def test_calibrate_rejects_outliers():
    bad = [5, 100, 400]
    obs, c = calibrated(GEO_TILTED, np.ones(xg.N_GEO, bool), outliers=bad)
    assert not c["good"][bad].any()
    assert c["good"].sum() >= len(obs) - 10
    assert np.all(np.abs(c["geo"][:2] - GEO_TILTED[:2]) < [0.5, 2])


def test_calibrate_systematics_scale():
    _, c = calibrated(GEO_TILTED, np.ones(xg.N_GEO, bool), radius_tol=0.05, pixel_tol=0.001)
    # SOD scales with R, SDD with the pixel size; the other parameter is unaffected
    assert c["sig_sys"][0] == pytest.approx(c["geo"][0] * 0.05 / R, rel=0.05)
    assert c["sig_sys"][1] == pytest.approx(c["geo"][1] * 0.001 / PIXEL, rel=0.05)


@pytest.mark.parametrize("name, free", [("full", np.ones(xg.N_GEO, bool)),
                                        ("reduced", np.array([1, 1, 1, 1, 0, 0, 1], bool))])
def test_save_model_json(tmp_path, name, free):
    import json
    obs, c = calibrated(GEO_IDEAL, free, nboot=3)
    model = xg.Model(obs, R, PIXEL, -1)
    fname = tmp_path / f"geometry_{name}.json"
    xg.save_model_json(str(fname), name, c, model, {"step": 5})
    rec = json.loads(fname.read_text())
    assert rec["model"] == name and rec["settings"] == {"step": 5}
    assert list(rec["parameters"]) == xg.GEO_NAMES
    for k, n in enumerate(xg.GEO_NAMES):
        assert rec["parameters"][n]["value"] == pytest.approx(c["geo"][k])
        assert rec["parameters"][n]["free"] == bool(free[k])
        assert rec["parameters"][n]["sigma_total"] == pytest.approx(c["sig_tot"][k])
    assert rec["cor_px"] == pytest.approx(c["cor"])
    assert len(rec["ball_heights_mm"]) == len(rec["ball_phases_deg"]) == model.nb
    assert np.array(rec["correlation"]["matrix"]).shape == (xg.N_GEO, xg.N_GEO)
    assert rec["points_used"] + rec["points_rejected"] == len(obs)


@pytest.mark.parametrize("d_px", [30, 50, 90])
def test_edge_diameter_with_blur(d_px):
    img = ndi.gaussian_filter(ball_image([(150, 200)], d_px=d_px), 1.5)
    det = xg.detect_balls(img, d_px)
    assert len(det) == 1
    assert det[0, 2] == pytest.approx(d_px, rel=0.01)


def test_project_per_ball_radius_matches_scalar():
    z, phi = np.array([5.0, -5.0]), np.array([0.2, 0.4])
    ang = np.array([0.3, 1.1])
    for k, r in enumerate([20.0, 30.0]):
        u1, v1 = xg.project(GEO_TILTED, z, phi, np.array([k, k]), ang, np.array([20.0, 30.0]), PIXEL, -1)
        u2, v2 = xg.project(GEO_TILTED, z, phi, np.array([k, k]), ang, r, PIXEL, -1)
        np.testing.assert_allclose([u1, v1], [u2, v2])


def per_ball_radii(nballs=15, spread=0.05, seed=3):
    """Random radii with mean R and no linear trend with height, as in the model."""
    dr = np.random.default_rng(seed).normal(0, spread, nballs)
    A = np.column_stack([np.ones(nballs), np.arange(nballs)])
    dr -= A @ np.linalg.lstsq(A, dr, rcond=None)[0]
    return R + dr


def test_model_radii_mean_is_fixed():
    obs = make_obs(GEO_IDEAL)
    m = xg.Model(obs, R, PIXEL, -1, per_ball_radius=True)
    p = m.start(np.zeros(xg.N_GEO + 2 * m.nb))
    assert p.size == m.n_params == xg.N_GEO + 3 * m.nb - 2
    p[xg.N_GEO + 2 * m.nb:] = np.linspace(-0.1, 0.2, m.nb - 2)
    dr = m.radii(p) - R
    assert dr.mean() == pytest.approx(0, abs=1e-12)
    assert np.polyfit(np.arange(m.nb), dr, 1)[0] == pytest.approx(0, abs=1e-12)  # no trend with height
    assert np.ptp(dr) > 0.05
    np.testing.assert_allclose(xg.Model(obs, R, PIXEL, -1).radii(p), R)


def test_fit_recovers_per_ball_radii():
    radii = per_ball_radii()
    obs = make_obs(GEO_TILTED, radii=radii, noise=0.05)
    common = xg.Model(obs, R, PIXEL, -1)
    r_common = xg.fit(common, xg.initial_guess(obs, R, PIXEL, -1))
    m = xg.Model(obs, R, PIXEL, -1, per_ball_radius=True)
    r = xg.fit(m, m.start(r_common.p))
    assert np.sqrt(np.mean(r.fun ** 2)) < 0.06 < np.sqrt(np.mean(r_common.fun ** 2))
    np.testing.assert_allclose(m.radii(r.p), radii, atol=0.003)
    err = np.abs(r.p[:xg.N_GEO] - GEO_TILTED)
    assert np.all(err < [0.2, 1.0, 3, 3, 0.03, 0.03, 0.005]), err


def test_ball_scale_recovers_true_radius():
    # data from a phantom with R = 8 mm, fitted with a wrong nominal radius of 5 mm
    geo = GEO_IDEAL.copy()
    geo[0] = 112.0
    obs = make_obs(geo, ball_diameter=1.0)
    obs[:, 2:4] = make_obs(geo, ball_diameter=1.0, radii=8.0)[:, 2:4]
    obs[:, 5] = make_obs(geo, ball_diameter=1.0, radii=8.0)[:, 5]
    m = xg.Model(obs, 5.0, PIXEL, -1)
    p = xg.fit(m, xg.initial_guess(obs, 5.0, PIXEL, -1)).p
    assert p[0] == pytest.approx(112.0 * 5 / 8, rel=1e-3)  # SOD scales with the assumed R
    assert p[1] == pytest.approx(geo[1], rel=1e-3)         # SDD does not
    lam, err, per_ball = xg.ball_scale(m, p, obs[:, 5], np.ones(len(obs), bool), 1.0)
    assert 5.0 / lam == pytest.approx(8.0, rel=1e-3)
    assert err < 1e-3 and per_ball.size == m.nb


def test_reduced_model_fits_leaning_column():
    # ideal detector, ball column leaning radially by 0.3 deg: the radius grows with height
    nb = 15
    z = np.linspace(-35, 35, nb)
    radii = R + np.tan(np.deg2rad(0.3)) * z + per_ball_radii(nb, 0.02) - R
    obs = make_obs(GEO_IDEAL, radii=radii, noise=0.05)
    full = xg.Model(obs, R, PIXEL, -1, per_ball_radius=True)
    red = full.variant(radius_trend=True)
    assert red.n_params == full.n_params + 1
    p0 = xg.fit(full, full.start(xg.initial_guess(obs, R, PIXEL, -1))).p
    q0 = full.convert(p0, red)
    q0[4:6] = 0
    free = np.ones(q0.size, bool)
    free[4:6] = False
    r = xg.fit(red, q0, free=free)
    assert np.sqrt(np.mean(r.fun ** 2)) < 0.06
    err = np.abs(r.p[:4] - GEO_IDEAL[:4])
    assert np.all(err < [0.1, 0.5, 0.05, 0.5]), err
    # heights run opposite to the rows, the lean is recovered in magnitude
    assert abs(red.radius_lean_deg(r.p, red.unpack(r.p)[1])) == pytest.approx(0.3, abs=0.01)


# ------------------------------------------------------ real data (opt.) -----
real = pytest.mark.skipif(not os.path.exists(os.path.join(DATA_DIR, "xcal_00002.fits")),
                          reason="calibration data not available")


@real
def test_real_angles():
    for name, ang in [("xcal_00001.fits", 0.0), ("xcal_00002.fits", 0.972), ("xcal_00091.fits", 90.0)]:
        assert xg.read_angle(os.path.join(DATA_DIR, name), "sp2_ry") == pytest.approx(ang)


@real
def test_real_projection_detects_all_balls():
    dc, gain, bad = xg.load_references(DATA_DIR, "xdc_*.fits", "xob_*.fits")
    fill_idx = ndi.distance_transform_edt(bad, return_distances=False, return_indices=True)
    n = xg.normalize(xg.read_image(os.path.join(DATA_DIR, "xcal_00091.fits")), dc, gain, fill_idx)
    det = xg.detect_balls(n, 56)
    assert 19 <= len(det) <= 22
    spacing = np.diff(np.sort(det[:, 1]))
    assert np.ptp(spacing) < 0.2 * np.median(spacing)  # one ball per position, none missing in the column
