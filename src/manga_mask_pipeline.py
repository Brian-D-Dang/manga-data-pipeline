"""MaNGA LOGCUBE -> metallicity + ionization-parameter + spaxel-mask ETL.

Given a MaNGA LOGCUBE FITS, this script

1. **Extract**   -- sums flux inside narrow windows around the strong optical
   emission lines needed by the Kewley & Dopita (2002) strong-line diagnostics,
   using the same MASK-aware slicing pattern as
   ``testing_notebooks/showcase_cube.ipynb``.
2. **Transform** --
   a. estimates ``12 + log(O/H)`` per spaxel with the K&D02 ``[N II]/[O II]``
      N2O2 calibration (eq. 5),
   b. inverts the K&D02 cubic ``R = k0 + k1 x + k2 x^2 + k3 x^3`` (Table 2)
      for ``x = log10(q [cm/s])`` on both the ``[O III] 5007 / [O II] 3726,29``
      and ``[S III] 9069+9532 / [S II] 6717+31`` diagnostics, then converts to
      the dimensionless ionization parameter ``log U = log q - log c``,
   c. builds a per-spaxel selection mask keeping spaxels with
      ``12 + log(O/H) >= solar_oh`` and
      ``log U ([O III]/[O II]) < logu_limit``.
3. **Load**      -- writes a single output FITS whose extensions are the
   metallicity, both log U maps, and the spaxel-selection mask.

Reference:
    Kewley & Dopita 2002, ApJS 142, 35 (arXiv:astro-ph/0206495).

Usage:
    python manga_mask_pipeline.py \
        --cube data/manga-11019-12701-LOGCUBE.fits \
        --out  data/manga-11019-12701-MASK.fits
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from astropy.io import fits

log = logging.getLogger("manga_mask")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOG_C_CGS = np.log10(2.99792458e10)  # log10(speed of light [cm/s])

DEFAULT_SOLAR_OH = 8.93   # K&D02 Table 1 / eq. 5 -- log(Z/Zsun) = 0 on the AG89 scale
DEFAULT_LOGU_LIMIT = -2.75

# K&D02 metallicity grid corresponding to each column of Table 2
KD02_OH_GRID = np.array([7.6, 7.9, 8.2, 8.6, 8.9, 9.1, 9.2, 9.4])
Z_SOLAR_COLUMN = 4  # index of the Z = 1.0 Zsun column in KD02_OH_GRID

# K&D02 Table 2 bottom block -- log10([S III] 9069+9532 / [S II] 6717+31)
KD02_COEF_SIII_SII = np.array([
    # k0            k1           k2            k3
    [ 30.0116,  -14.8970,     2.30577,  -0.112314 ],  # Z=0.05 Zsun
    [ 16.8569,   -9.62876,    1.59938,  -0.0804552],  # Z=0.1
    [ 32.2358,  -15.2438,     2.27251,  -0.106913 ],  # Z=0.2
    [ -3.06247,  -0.864092,   0.328467, -0.0196089],  # Z=0.5
    [ -2.94394,  -0.546041,   0.239226, -0.0136716],  # Z=1.0
    [-38.1338,   13.0914,    -1.51014,   0.0605926],  # Z=1.5
    [-21.0240,    6.55748,   -0.683584,  0.0258690],  # Z=2.0
    [ -6.61131,   1.36836,   -0.0717560, 0.00225792], # Z=3.0
])

# K&D02 Table 2 top block -- log10([O III] 5007 / [O II] 3726,29)
KD02_COEF_OIII_OII = np.array([
    [-36.9772,   10.2838,   -0.957421,    0.0328614 ],  # Z=0.05 Zsun
    [-74.2814,   24.6206,   -2.79194,     0.110773  ],  # Z=0.1
    [-36.7948,   10.0581,   -0.914212,    0.0300472 ],  # Z=0.2
    [-81.1880,   27.5082,   -3.19126,     0.128252  ],  # Z=0.5
    [-52.6367,   16.0880,   -1.67443,     0.0608004 ],  # Z=1.0
    [-86.8674,   28.0455,   -3.01747,     0.108311  ],  # Z=1.5
    [-24.4044,    2.51913,   0.452486,   -0.0491711 ],  # Z=2.0
    [ 49.4728,  -27.4711,    4.50304,    -0.232228  ],  # Z=3.0
])

# Emission-line integration windows in Angstroms (observed frame, same as
# showcase_cube.ipynb). Small +/- ~2 A windows to bracket each line.
LINE_WINDOWS: Mapping[str, tuple[float, float]] = {
    "oii":    (3726.0, 3729.0),  # [O II] 3726, 3729 doublet
    "nii":    (6582.0, 6586.0),  # [N II] 6584
    "oiii":   (5005.0, 5009.0),  # [O III] 5007
    "sii":    (6714.0, 6734.0),  # [S II] 6717, 6731 doublet
    "siii_a": (9066.0, 9072.0),  # [S III] 9069
    "siii_b": (9529.0, 9535.0),  # [S III] 9532
}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class LineMap:
    """Wavelength-summed flux for a single emission-line window."""

    flux: np.ndarray            # (ny, nx) float
    bad_spaxel: np.ndarray      # (ny, nx) bool -- any voxel in window had MASK != 0
    n_channels: int


@dataclass
class DiagnosticResult:
    """One K&D02 line-ratio diagnostic solved per spaxel."""

    ratio_map: np.ndarray       # log10 line ratio, (ny, nx)
    logq_map: np.ndarray        # log10(q [cm/s]), (ny, nx), NaN where unsolved
    logu_map: np.ndarray        # log10 U (dimensionless), (ny, nx), NaN where unsolved


@dataclass
class PipelineOutput:
    oh: np.ndarray              # 12 + log(O/H), (ny, nx)
    oh_ok: np.ndarray           # (ny, nx) bool, True where N2O2 is trustworthy
    oiii_oii: DiagnosticResult
    siii_sii: DiagnosticResult
    include_spaxel: np.ndarray  # (ny, nx) bool -- final selection mask


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------

def load_cube(cube_path: Path):
    """Read FLUX, MASK, WAVE (and FLUX header) from a MaNGA LOGCUBE."""
    log.info("loading cube: %s", cube_path)
    with fits.open(cube_path) as hdul:
        flux = np.asarray(hdul["FLUX"].data)
        mask = np.asarray(hdul["MASK"].data)
        wave = np.asarray(hdul["WAVE"].data)
        flux_header = hdul["FLUX"].header.copy()
    log.info("FLUX shape=%s  WAVE=[%.2f, %.2f] A", flux.shape, wave[0], wave[-1])
    return flux, mask, wave, flux_header


def line_map(flux: np.ndarray, mask: np.ndarray, wave: np.ndarray,
             wave_lo: float, wave_hi: float) -> LineMap:
    """Sum ``flux`` between ``wave_lo`` and ``wave_hi``, zeroing MASK != 0 voxels."""
    sel = (wave >= wave_lo) & (wave <= wave_hi)
    if not sel.any():
        raise ValueError(f"no channels in window {wave_lo}-{wave_hi} A")
    chunk = np.where(mask[sel] == 0, flux[sel], 0.0)
    return LineMap(
        flux=chunk.sum(axis=0),
        bad_spaxel=np.any(mask[sel] != 0, axis=0),
        n_channels=int(sel.sum()),
    )


def compute_line_maps(flux: np.ndarray, mask: np.ndarray, wave: np.ndarray) -> dict[str, LineMap]:
    maps: dict[str, LineMap] = {}
    for name, (lo, hi) in LINE_WINDOWS.items():
        m = line_map(flux, mask, wave, lo, hi)
        maps[name] = m
        log.info("line %-6s window %.1f-%.1f A: %d channels, %d masked spaxels",
                 name, lo, hi, m.n_channels, int(m.bad_spaxel.sum()))
    return maps


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

def _safe_log10_ratio(num: np.ndarray, denom: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log10(num / denom)
    return np.where(np.isfinite(r), r, np.nan)


def compute_metallicity(nii: LineMap, oii: LineMap) -> tuple[np.ndarray, np.ndarray]:
    """K&D02 N2O2 metallicity from eq. 5. Returns ``(OH, oh_ok)``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.log10(nii.flux / oii.flux)
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    with np.errstate(invalid="ignore"):
        OH = np.log10(1.54020 + 1.26602 * R + 0.167977 * R ** 2) + 8.93
    OH = np.nan_to_num(OH, nan=0.0, posinf=0.0, neginf=0.0)
    # eq. 5 is only calibrated for 12+log(O/H) >= 8.6; keep a small buffer at 8.4
    oh_ok = (OH >= 8.4) & ~(nii.bad_spaxel | oii.bad_spaxel)
    log.info("N2O2 metallicity: %d/%d spaxels usable (OH>=8.4 and unmasked)",
             int(oh_ok.sum()), oh_ok.size)
    return OH, oh_ok


def pick_z_column(OH: np.ndarray, oh_ok: np.ndarray) -> np.ndarray:
    """Per-spaxel index into the K&D02 metallicity grid; defaults to solar."""
    iZ = np.argmin(np.abs(OH[..., None] - KD02_OH_GRID), axis=-1)
    return np.where(oh_ok, iZ, Z_SOLAR_COLUMN)


def _evaluate_cubic(coef_table: np.ndarray, logq_grid: np.ndarray) -> np.ndarray:
    """Evaluate the K&D02 cubics ``R(logq)`` for every Z column."""
    return (coef_table[:, 0][:, None]
            + coef_table[:, 1][:, None] * logq_grid
            + coef_table[:, 2][:, None] * logq_grid ** 2
            + coef_table[:, 3][:, None] * logq_grid ** 3)


def invert_kd02(R_map: np.ndarray, spaxel_ok: np.ndarray, coef_table: np.ndarray,
                logq_grid: np.ndarray, z_index: np.ndarray) -> np.ndarray:
    """Invert ``R = k0 + k1 x + k2 x^2 + k3 x^3`` for ``x = log10(q)``, per spaxel.

    For each Z column the cubic is tabulated on ``logq_grid`` and inverted with
    ``np.interp``. Non-monotonic Z columns (e.g. Z=0.05 for [S III]/[S II]) are
    skipped. Spaxels whose measured ``R`` falls outside the model curve range
    are left as NaN.
    """
    R_curve = _evaluate_cubic(coef_table, logq_grid)
    out = np.full(R_map.shape, np.nan)
    for iz in range(coef_table.shape[0]):
        sel = (z_index == iz) & np.isfinite(R_map) & spaxel_ok
        if not sel.any():
            continue
        curve = R_curve[iz]
        diffs = np.diff(curve)
        if not (np.all(diffs > 0) or np.all(diffs < 0)):
            log.debug("skipping non-monotonic K&D02 column iz=%d", iz)
            continue
        if curve[0] > curve[-1]:
            xp, fp = curve[::-1], logq_grid[::-1]
        else:
            xp, fp = curve, logq_grid
        r = R_map[sel]
        inside = (r >= xp.min()) & (r <= xp.max())
        vals = np.full(r.shape, np.nan)
        vals[inside] = np.interp(r[inside], xp, fp)
        out[sel] = vals
    return out


def compute_diagnostic(numerator: np.ndarray, denominator: np.ndarray,
                       bad_num: np.ndarray, bad_denom: np.ndarray,
                       coef_table: np.ndarray, z_index: np.ndarray,
                       logq_grid: np.ndarray, label: str) -> DiagnosticResult:
    ratio = _safe_log10_ratio(numerator, denominator)
    spaxel_ok = ~(bad_num | bad_denom)
    logq = invert_kd02(ratio, spaxel_ok, coef_table, logq_grid, z_index)
    logu = logq - LOG_C_CGS
    n_ok = int(np.isfinite(logq).sum())
    median_logu = float(np.nanmedian(logu)) if n_ok else float("nan")
    log.info("%-18s log U valid spaxels=%d, median log U=%.2f", label, n_ok, median_logu)
    return DiagnosticResult(ratio_map=ratio, logq_map=logq, logu_map=logu)


def build_include_mask(OH: np.ndarray, oh_ok: np.ndarray, logu_map: np.ndarray,
                       solar_oh: float, logu_limit: float) -> np.ndarray:
    """Keep spaxels with metallicity >= solar AND log U below the given threshold."""
    logu_valid = np.isfinite(logu_map)
    mask = (
        oh_ok
        & np.isfinite(OH)
        & logu_valid
        & (OH >= solar_oh)
        & (logu_map < logu_limit)
    )
    log.info(
        "mask: OH>=%.2f -> %d, log U<%.2f -> %d, BOTH -> %d (of %d spaxels)",
        solar_oh, int((oh_ok & (OH >= solar_oh)).sum()),
        logu_limit, int((logu_valid & (logu_map < logu_limit)).sum()),
        int(mask.sum()), mask.size,
    )
    return mask


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def write_output(out_path: Path, cube_path: Path, result: PipelineOutput,
                 solar_oh: float, logu_limit: float) -> None:
    """Write metallicity map, both log U maps, and the spaxel mask as a FITS."""
    log.info("writing outputs -> %s", out_path)
    primary = fits.PrimaryHDU()
    primary.header["ORIGIN"]   = ("manga_mask_pipeline", "producing script")
    primary.header["CUBEIN"]   = (str(cube_path), "input LOGCUBE")
    primary.header["SOLAR_OH"] = (solar_oh, "12+log(O/H) threshold treated as solar")
    primary.header["LOGULIM"]  = (logu_limit, "log U threshold applied to [OIII]/[OII]")

    hdus = [
        primary,
        fits.ImageHDU(result.oh.astype(np.float32),                     name="OH_N2O2"),
        fits.ImageHDU(result.oh_ok.astype(np.uint8),                    name="OH_OK"),
        fits.ImageHDU(result.oiii_oii.ratio_map.astype(np.float32),     name="R_OIII_OII"),
        fits.ImageHDU(result.oiii_oii.logu_map.astype(np.float32),      name="LOGU_OIII_OII"),
        fits.ImageHDU(result.siii_sii.ratio_map.astype(np.float32),     name="R_SIII_SII"),
        fits.ImageHDU(result.siii_sii.logu_map.astype(np.float32),      name="LOGU_SIII_SII"),
        fits.ImageHDU(result.include_spaxel.astype(np.uint8),           name="INCLUDE_SPAXEL"),
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(out_path, overwrite=True)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_pipeline(cube_path: Path, out_path: Path | None,
                 solar_oh: float = DEFAULT_SOLAR_OH,
                 logu_limit: float = DEFAULT_LOGU_LIMIT) -> PipelineOutput:
    """Full extract-transform-load pipeline. Returns the in-memory result."""
    flux, mask, wave, _ = load_cube(cube_path)
    maps = compute_line_maps(flux, mask, wave)

    OH, oh_ok = compute_metallicity(maps["nii"], maps["oii"])
    z_index = pick_z_column(OH, oh_ok)

    # K&D02 grid runs q in [5e6, 3e8] cm/s -> log q in ~[6.7, 8.48]
    logq_grid = np.linspace(6.7, 8.5, 2001)

    oiii_oii = compute_diagnostic(
        maps["oiii"].flux, maps["oii"].flux,
        maps["oiii"].bad_spaxel, maps["oii"].bad_spaxel,
        KD02_COEF_OIII_OII, z_index, logq_grid,
        label="[O III]/[O II]",
    )
    siii_sii = compute_diagnostic(
        maps["siii_a"].flux + maps["siii_b"].flux,
        maps["sii"].flux,
        maps["siii_a"].bad_spaxel | maps["siii_b"].bad_spaxel,
        maps["sii"].bad_spaxel,
        KD02_COEF_SIII_SII, z_index, logq_grid,
        label="[S III]/[S II]",
    )

    include = build_include_mask(OH, oh_ok, oiii_oii.logu_map, solar_oh, logu_limit)
    result = PipelineOutput(
        oh=OH, oh_ok=oh_ok,
        oiii_oii=oiii_oii, siii_sii=siii_sii,
        include_spaxel=include,
    )
    if out_path is not None:
        write_output(out_path, cube_path, result, solar_oh, logu_limit)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cube", required=True, type=Path,
                   help="input MaNGA LOGCUBE .fits")
    p.add_argument("--out",  required=True, type=Path,
                   help="output FITS path (parent dirs will be created)")
    p.add_argument("--solar-oh", type=float, default=DEFAULT_SOLAR_OH,
                   help="12+log(O/H) treated as solar (default: %(default)s)")
    p.add_argument("--logu-limit", type=float, default=DEFAULT_LOGU_LIMIT,
                   help="upper bound on log U from [O III]/[O II] "
                        "(default: %(default)s)")
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    run_pipeline(args.cube, args.out, args.solar_oh, args.logu_limit)


if __name__ == "__main__":
    main()
