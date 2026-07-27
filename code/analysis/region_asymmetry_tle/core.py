"""Core logic: bilateral region pairs from GAM mni_micro (Glasser, 4S156 cortex+subcortex, HCP1065), asymmetry = (ipsi - contra)/(|ipsi| + |contra|)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config as cfg


def _asymmetry_value(ipsi_z: float, contra_z: float) -> float:
    """(ipsi - contra) / (|ipsi| + |contra|). NaN if denominator is 0."""
    denom = abs(ipsi_z) + abs(contra_z)
    if denom == 0:
        return float("nan")
    return (ipsi_z - contra_z) / denom


def get_glasser_bilateral_pairs(gam_glasser_dir: Path) -> List[Tuple[str, str, str]]:
    """Return list of (left_region, right_region, region_base) for Glasser. region_base = suffix after Left_/Right_."""
    gam_glasser_dir = Path(gam_glasser_dir)
    if not gam_glasser_dir.is_dir():
        return []
    pairs: List[Tuple[str, str, str]] = []
    for d in gam_glasser_dir.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if name.startswith("Left_"):
            suffix = name[5:]  # after "Left_"
            right_name = "Right_" + suffix
            right_path = gam_glasser_dir / right_name
            if right_path.is_dir():
                pairs.append((name, right_name, suffix))
    return sorted(pairs, key=lambda x: x[2])


def get_4s156_subcortical_bilateral_pairs(
    gam_4s156_dir: Path,
    atlas_tsv_4s: Path,
) -> List[Tuple[str, str, str]]:
    """Return list of (left_region, right_region, region_base) for 4S156 subcortical only. region_base = suffix after LH_/RH_ or LH-/RH-."""
    gam_4s156_dir = Path(gam_4s156_dir)
    if not gam_4s156_dir.is_dir():
        return []
    subcortical_labels = set(cfg.get_4s_subcortical_regions(atlas_tsv_4s))
    existing = set(p.name for p in gam_4s156_dir.iterdir() if p.is_dir())
    subcortical_in_gam = subcortical_labels & existing
    pairs: List[Tuple[str, str, str]] = []
    for label in sorted(subcortical_in_gam):
        if label.startswith("LH_"):
            suffix = label[3:]
            right_name = "RH_" + suffix
            if right_name in subcortical_in_gam:
                pairs.append((label, right_name, suffix))
        elif label.startswith("LH-"):
            suffix = label[3:]
            right_name = "RH-" + suffix
            if right_name in subcortical_in_gam:
                pairs.append((label, right_name, suffix))
    return pairs


def get_4s156_cortical_bilateral_pairs(
    gam_4s156_dir: Path,
    atlas_tsv_4s: Path,
) -> List[Tuple[str, str, str]]:
    """Return list of (left_region, right_region, region_base) for 4S156 cortical only (homotopic LH/RH pairs). region_base = suffix after LH_/RH_."""
    gam_4s156_dir = Path(gam_4s156_dir)
    if not gam_4s156_dir.is_dir():
        return []
    cortical_labels = set(cfg.get_4s_cortical_regions(atlas_tsv_4s))
    existing = set(p.name for p in gam_4s156_dir.iterdir() if p.is_dir())
    cortical_in_gam = cortical_labels & existing
    pairs: List[Tuple[str, str, str]] = []
    for label in sorted(cortical_in_gam):
        if label.startswith("LH_"):
            suffix = label[3:]
            right_name = "RH_" + suffix
            if right_name in cortical_in_gam:
                pairs.append((label, right_name, suffix))
        elif label.startswith("LH-"):
            suffix = label[3:]
            right_name = "RH-" + suffix
            if right_name in cortical_in_gam:
                pairs.append((label, right_name, suffix))
    return pairs


def get_hcp1065_bilateral_pairs(gam_hcp1065_dir: Path) -> List[Tuple[str, str, str]]:
    """Return list of (left_region, right_region, region_base) for HCP1065 WM tracts. Pairs have _L/_R (e.g. AF_L/AF_R, IFOF_L_core/IFOF_R_core)."""
    gam_hcp1065_dir = Path(gam_hcp1065_dir)
    if not gam_hcp1065_dir.is_dir():
        return []
    existing = set(p.name for p in gam_hcp1065_dir.iterdir() if p.is_dir())
    pairs: List[Tuple[str, str, str]] = []
    seen_bases: set = set()
    for name in sorted(existing):
        if "_L" in name:
            right_name = name.replace("_L", "_R", 1)
            if right_name in existing:
                base = name.replace("_L", "")
                if base not in seen_bases:
                    seen_bases.add(base)
                    pairs.append((name, right_name, base))
    return sorted(pairs, key=lambda x: x[2])


def get_scalars_for_atlas(gam_dir: Path, region: str, stat: str = "mean") -> List[str]:
    """Discover scalar names from GAM dir: files {region}_{scalar}_stat-{stat}_gam.csv, exclude config EXCLUDED_SCALARS."""
    gam_dir = Path(gam_dir)
    region_dir = gam_dir / region
    if not region_dir.is_dir():
        return []
    prefix = f"{region}_"
    suffix = f"_stat-{stat}_gam.csv"
    scalars: List[str] = []
    for f in region_dir.iterdir():
        if not f.is_file() or f.suffix != ".csv":
            continue
        name = f.name
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        scalar = name[len(prefix) : -len(suffix)]
        if scalar not in cfg.EXCLUDED_SCALARS:
            scalars.append(scalar)
    return sorted(set(scalars))


def _load_gam_z(gam_dir: Path, region: str, scalar: str, stat: str) -> Optional[pd.DataFrame]:
    """Load GAM CSV for (region, scalar, stat). Returns DataFrame with 'sub' and '{scalar}_z'."""
    path = gam_dir / region / f"{region}_{scalar}_stat-{stat}_gam.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    z_col = f"{scalar}_z"
    if z_col not in df.columns:
        return None
    return df[["sub", z_col]].copy()


class RegionAsymmetryTLE:
    """
    Compute region-level asymmetry from GAM mni_micro z-scores (Glasser, 4S156 cortex+subcortex, HCP1065).
    One row per (sub, region, scalar) or (sub, region, scalar, stat).
    Asymmetry = (ipsi - contra) / (|ipsi| + |contra|).
    """

    def __init__(
        self,
        gam_glasser_dir: Path,
        gam_4s156_dir: Path,
        atlas_tsv_4s: Path,
        inclusion_path: Path,
        gam_hcp1065_dir: Optional[Path] = None,
        normative_dir: Optional[Path] = None,
    ):
        self.gam_glasser_dir = Path(gam_glasser_dir)
        self.gam_4s156_dir = Path(gam_4s156_dir)
        self.atlas_tsv_4s = Path(atlas_tsv_4s)
        self.inclusion_path = Path(inclusion_path)
        self.gam_hcp1065_dir = Path(gam_hcp1065_dir) if gam_hcp1065_dir else None
        self.normative_dir = Path(normative_dir) if normative_dir is not None else None
        self._gam_cache: Dict[Tuple[str, str, str, str], pd.DataFrame] = {}
        self._invcov_cache: Dict[Tuple[str, str, bool], Optional[Tuple[np.ndarray, List[str]]]] = {}

    def _load_gam(self, atlas: str, region: str, scalar: str, stat: str) -> Optional[pd.DataFrame]:
        if atlas == "Glasser":
            gam_dir = self.gam_glasser_dir
        elif atlas == "4S156":
            gam_dir = self.gam_4s156_dir
        elif atlas == "HCP1065" and self.gam_hcp1065_dir:
            gam_dir = self.gam_hcp1065_dir
        else:
            return None
        key = (atlas, region, scalar, stat)
        if key not in self._gam_cache:
            result = _load_gam_z(gam_dir, region, scalar, stat)
            self._gam_cache[key] = result if result is not None else pd.DataFrame()
        df = self._gam_cache[key]
        return df if not df.empty else None

    def _load_invcov(
        self,
        atlas: str,
        region: str,
        use_mcd: bool,
    ) -> Optional[Tuple[np.ndarray, List[str]]]:
        """
        Load inverse covariance from region_asymmetry_tle_normative: normative_dir/atlas/region/invcov.csv or invcov_mincovdet.csv.
        Returns (invcov matrix, list of scalar names in order) or None.
        """
        if self.normative_dir is None or not self.normative_dir.is_dir():
            return None
        key = (atlas, region, use_mcd)
        if key in self._invcov_cache:
            return self._invcov_cache[key]
        fname = "invcov_mincovdet.csv" if use_mcd else "invcov.csv"
        path = self.normative_dir / atlas / region / fname
        if not path.exists():
            self._invcov_cache[key] = None
            return None
        try:
            df = pd.read_csv(path, index_col=0)
        except Exception:
            self._invcov_cache[key] = None
            return None
        scalars = df.columns.tolist()
        if list(df.index) != scalars:
            df = df.reindex(index=scalars, columns=scalars)
        arr = df.values.astype(float)
        if np.any(np.isnan(arr)):
            self._invcov_cache[key] = None
            return None
        self._invcov_cache[key] = (arr, scalars)
        return arr, scalars

    @staticmethod
    def _mahalanobis_from_zero(z_vec: np.ndarray, invcov: np.ndarray) -> float:
        """Mahalanobis distance of z_vec from 0: sqrt(z' invcov z)."""
        q = float(z_vec.ravel() @ invcov @ z_vec.ravel())
        return float(np.sqrt(max(0.0, q)))

    def get_eligible_subjects(self) -> List[str]:
        """Eligible = in TLE inclusion (temporal) AND present in at least one GAM (Glasser sample region)."""
        tle_subs, _ = cfg.load_tle_inclusion(self.inclusion_path)
        sample_region = "Left_1"
        sample_path = self.gam_glasser_dir / sample_region
        if not sample_path.is_dir():
            return sorted(set(tle_subs))
        # Discover one scalar to read subjects
        scalars = get_scalars_for_atlas(self.gam_glasser_dir, sample_region, "mean")
        if not scalars:
            return sorted(set(tle_subs))
        df = self._load_gam("Glasser", sample_region, scalars[0], "mean")
        if df is None:
            return sorted(set(tle_subs))
        gam_subs = set(df["sub"].astype(str).unique())
        return sorted(set(tle_subs) & gam_subs)

    def compute_asymmetry(
        self,
        subjects: Optional[List[str]] = None,
        subject_ipsi_hemi: Optional[Dict[str, str]] = None,
        stats: Optional[List[str]] = None,
        show_progress: bool = False,
    ) -> pd.DataFrame:
        """
        Build asymmetry table: sub, region, scalar, ipsi_mean_z, contra_mean_z, asymmetry, hemi_ipsi.
        If stats = ["mean", "standard_deviation"], adds column "stat" and one row per (sub, region, scalar, stat).
        If show_progress=True, show a tqdm bar over ROI pairs.
        """
        if subjects is None:
            subjects = self.get_eligible_subjects()
        if subject_ipsi_hemi is None:
            _, subject_ipsi_hemi = cfg.load_tle_inclusion(self.inclusion_path)
            subject_ipsi_hemi = {s: subject_ipsi_hemi.get(s, "") for s in subjects}
        if stats is None:
            stats = ["mean", "standard_deviation"]

        rows: List[dict] = []

        # Count total ROI pairs for progress bar
        glasser_pairs = get_glasser_bilateral_pairs(self.gam_glasser_dir)
        s4_subcortex_pairs = get_4s156_subcortical_bilateral_pairs(self.gam_4s156_dir, self.atlas_tsv_4s)
        s4_cortex_pairs = get_4s156_cortical_bilateral_pairs(self.gam_4s156_dir, self.atlas_tsv_4s)
        hcp_pairs = get_hcp1065_bilateral_pairs(self.gam_hcp1065_dir) if (self.gam_hcp1065_dir and self.gam_hcp1065_dir.is_dir()) else []
        total_rois = len(glasser_pairs) + len(s4_subcortex_pairs) + len(s4_cortex_pairs) + len(hcp_pairs)
        pbar = tqdm(total=total_rois, desc="ROIs", unit="roi", leave=False, disable=not show_progress)

        def _advance():
            if show_progress:
                pbar.update(1)

        try:
            # Glasser
            for left_r, right_r, region_base in glasser_pairs:
                for stat in stats:
                    scalars = get_scalars_for_atlas(self.gam_glasser_dir, left_r, stat)
                    for scalar in scalars:
                        z_col = f"{scalar}_z"
                        left_df = self._load_gam("Glasser", left_r, scalar, stat)
                        right_df = self._load_gam("Glasser", right_r, scalar, stat)
                        if left_df is None or right_df is None:
                            continue
                        for sub in subjects:
                            hemi = subject_ipsi_hemi.get(sub)
                            if hemi not in ("L", "R"):
                                continue
                            left_row = left_df[left_df["sub"].astype(str) == sub]
                            right_row = right_df[right_df["sub"].astype(str) == sub]
                            if left_row.empty or right_row.empty:
                                continue
                            left_z = float(left_row[z_col].iloc[0])
                            right_z = float(right_row[z_col].iloc[0])
                            ipsi_z = left_z if hemi == "L" else right_z
                            contra_z = right_z if hemi == "L" else left_z
                            asym = _asymmetry_value(ipsi_z, contra_z)
                            row = {
                                "sub": sub,
                                "region": region_base,
                                "scalar": scalar,
                                "ipsi_mean_z": ipsi_z,
                                "contra_mean_z": contra_z,
                                "asymmetry": asym,
                                "hemi_ipsi": hemi,
                            }
                            if len(stats) > 1:
                                row["stat"] = stat
                            rows.append(row)
                _advance()

            # 4S156 subcortical
            for left_r, right_r, region_base in s4_subcortex_pairs:
                for stat in stats:
                    scalars = get_scalars_for_atlas(self.gam_4s156_dir, left_r, stat)
                    for scalar in scalars:
                        z_col = f"{scalar}_z"
                        left_df = self._load_gam("4S156", left_r, scalar, stat)
                        right_df = self._load_gam("4S156", right_r, scalar, stat)
                        if left_df is None or right_df is None:
                            continue
                        for sub in subjects:
                            hemi = subject_ipsi_hemi.get(sub)
                            if hemi not in ("L", "R"):
                                continue
                            left_row = left_df[left_df["sub"].astype(str) == sub]
                            right_row = right_df[right_df["sub"].astype(str) == sub]
                            if left_row.empty or right_row.empty:
                                continue
                            left_z = float(left_row[z_col].iloc[0])
                            right_z = float(right_row[z_col].iloc[0])
                            ipsi_z = left_z if hemi == "L" else right_z
                            contra_z = right_z if hemi == "L" else left_z
                            asym = _asymmetry_value(ipsi_z, contra_z)
                            row = {
                                "sub": sub,
                                "region": region_base,
                                "scalar": scalar,
                                "ipsi_mean_z": ipsi_z,
                                "contra_mean_z": contra_z,
                                "asymmetry": asym,
                                "hemi_ipsi": hemi,
                            }
                            if len(stats) > 1:
                                row["stat"] = stat
                            rows.append(row)
                _advance()

            # 4S156 cortical (homotopic LH/RH pairs)
            for left_r, right_r, region_base in s4_cortex_pairs:
                for stat in stats:
                    scalars = get_scalars_for_atlas(self.gam_4s156_dir, left_r, stat)
                    for scalar in scalars:
                        z_col = f"{scalar}_z"
                        left_df = self._load_gam("4S156", left_r, scalar, stat)
                        right_df = self._load_gam("4S156", right_r, scalar, stat)
                        if left_df is None or right_df is None:
                            continue
                        for sub in subjects:
                            hemi = subject_ipsi_hemi.get(sub)
                            if hemi not in ("L", "R"):
                                continue
                            left_row = left_df[left_df["sub"].astype(str) == sub]
                            right_row = right_df[right_df["sub"].astype(str) == sub]
                            if left_row.empty or right_row.empty:
                                continue
                            left_z = float(left_row[z_col].iloc[0])
                            right_z = float(right_row[z_col].iloc[0])
                            ipsi_z = left_z if hemi == "L" else right_z
                            contra_z = right_z if hemi == "L" else left_z
                            asym = _asymmetry_value(ipsi_z, contra_z)
                            row = {
                                "sub": sub,
                                "region": region_base,
                                "scalar": scalar,
                                "ipsi_mean_z": ipsi_z,
                                "contra_mean_z": contra_z,
                                "asymmetry": asym,
                                "hemi_ipsi": hemi,
                            }
                            if len(stats) > 1:
                                row["stat"] = stat
                            rows.append(row)
                _advance()

            # HCP1065 WM (tract segments: _L / _R pairs)
            for left_r, right_r, region_base in hcp_pairs:
                for stat in stats:
                    scalars = get_scalars_for_atlas(self.gam_hcp1065_dir, left_r, stat)
                    for scalar in scalars:
                        z_col = f"{scalar}_z"
                        left_df = self._load_gam("HCP1065", left_r, scalar, stat)
                        right_df = self._load_gam("HCP1065", right_r, scalar, stat)
                        if left_df is None or right_df is None:
                            continue
                        for sub in subjects:
                            hemi = subject_ipsi_hemi.get(sub)
                            if hemi not in ("L", "R"):
                                continue
                            left_row = left_df[left_df["sub"].astype(str) == sub]
                            right_row = right_df[right_df["sub"].astype(str) == sub]
                            if left_row.empty or right_row.empty:
                                continue
                            left_z = float(left_row[z_col].iloc[0])
                            right_z = float(right_row[z_col].iloc[0])
                            ipsi_z = left_z if hemi == "L" else right_z
                            contra_z = right_z if hemi == "L" else left_z
                            asym = _asymmetry_value(ipsi_z, contra_z)
                            row = {
                                "sub": sub,
                                "region": region_base,
                                "scalar": scalar,
                                "ipsi_mean_z": ipsi_z,
                                "contra_mean_z": contra_z,
                                "asymmetry": asym,
                                "hemi_ipsi": hemi,
                            }
                            if len(stats) > 1:
                                row["stat"] = stat
                            rows.append(row)
                _advance()
        finally:
            if show_progress:
                pbar.close()

        df = pd.DataFrame(rows)
        # Ensure stat column order: put stat after scalar if present
        if "stat" in df.columns:
            cols = ["sub", "region", "scalar", "stat", "ipsi_mean_z", "contra_mean_z", "asymmetry", "hemi_ipsi"]
        else:
            cols = ["sub", "region", "scalar", "ipsi_mean_z", "contra_mean_z", "asymmetry", "hemi_ipsi"]
        return df[[c for c in cols if c in df.columns]]

    def compute_mahal_asymmetry(
        self,
        subjects: Optional[List[str]] = None,
        subject_ipsi_hemi: Optional[Dict[str, str]] = None,
        stat: str = "mean",
        show_progress: bool = False,
    ) -> pd.DataFrame:
        """
        Mahalanobis distance asymmetry per region using normative invcov (raw and MCD).
        One row per (sub, atlas, region): ipsi/contra Mahalanobis distance from 0 (patient z-vector vs normative),
        and asymmetry = (ipsi_mahal - contra_mahal) / (|ipsi_mahal| + |contra_mahal|).
        Uses normative_dir/atlas/region/invcov.csv and invcov_mincovdet.csv. GAM z-scores loaded for given stat (default mean).
        """
        if subjects is None:
            subjects = self.get_eligible_subjects()
        if subject_ipsi_hemi is None:
            _, subject_ipsi_hemi = cfg.load_tle_inclusion(self.inclusion_path)
            subject_ipsi_hemi = {s: subject_ipsi_hemi.get(s, "") for s in subjects}
        if self.normative_dir is None or not self.normative_dir.is_dir():
            return pd.DataFrame()

        rows: List[dict] = []
        glasser_pairs = get_glasser_bilateral_pairs(self.gam_glasser_dir)
        s4_subcortex_pairs = get_4s156_subcortical_bilateral_pairs(self.gam_4s156_dir, self.atlas_tsv_4s)
        s4_cortex_pairs = get_4s156_cortical_bilateral_pairs(self.gam_4s156_dir, self.atlas_tsv_4s)
        hcp_pairs = get_hcp1065_bilateral_pairs(self.gam_hcp1065_dir) if (self.gam_hcp1065_dir and self.gam_hcp1065_dir.is_dir()) else []
        all_pairs: List[Tuple[str, List[Tuple[str, str, str]]]] = [
            ("Glasser", glasser_pairs),
            ("4S156", s4_subcortex_pairs + s4_cortex_pairs),
            ("HCP1065", hcp_pairs),
        ]
        total_rois = sum(len(p) for _, p in all_pairs)
        pbar = tqdm(total=total_rois, desc="Mahal ROIs", unit="roi", leave=False, disable=not show_progress)

        def _advance():
            if show_progress:
                pbar.update(1)

        try:
            for atlas, pairs in all_pairs:
                if not pairs:
                    continue
                for left_r, right_r, region_base in pairs:
                    inv_left_raw = self._load_invcov(atlas, left_r, use_mcd=False)
                    inv_left_mcd = self._load_invcov(atlas, left_r, use_mcd=True)
                    inv_right_raw = self._load_invcov(atlas, right_r, use_mcd=False)
                    inv_right_mcd = self._load_invcov(atlas, right_r, use_mcd=True)
                    if inv_left_raw is None or inv_right_raw is None:
                        _advance()
                        continue
                    _, scalars = inv_left_raw
                    for sub in subjects:
                        hemi = subject_ipsi_hemi.get(sub)
                        if hemi not in ("L", "R"):
                            continue
                        left_z_vec = []
                        right_z_vec = []
                        for scalar in scalars:
                            left_df = self._load_gam(atlas, left_r, scalar, stat)
                            right_df = self._load_gam(atlas, right_r, scalar, stat)
                            if left_df is None or right_df is None:
                                break
                            left_row = left_df[left_df["sub"].astype(str) == sub]
                            right_row = right_df[right_df["sub"].astype(str) == sub]
                            if left_row.empty or right_row.empty:
                                break
                            z_col = f"{scalar}_z"
                            if z_col not in left_row.columns or z_col not in right_row.columns:
                                break
                            left_z_vec.append(float(left_row[z_col].iloc[0]))
                            right_z_vec.append(float(right_row[z_col].iloc[0]))
                        if len(left_z_vec) != len(scalars) or len(right_z_vec) != len(scalars):
                            continue
                        ipsi_vec = np.array(left_z_vec if hemi == "L" else right_z_vec, dtype=float)
                        contra_vec = np.array(right_z_vec if hemi == "L" else left_z_vec, dtype=float)
                        invcov_ipsi_raw = inv_left_raw[0] if hemi == "L" else inv_right_raw[0]
                        invcov_contra_raw = inv_right_raw[0] if hemi == "L" else inv_left_raw[0]
                        ipsi_mahal_raw = self._mahalanobis_from_zero(ipsi_vec, invcov_ipsi_raw)
                        contra_mahal_raw = self._mahalanobis_from_zero(contra_vec, invcov_contra_raw)
                        asym_raw = _asymmetry_value(ipsi_mahal_raw, contra_mahal_raw)
                        row = {
                            "sub": sub,
                            "atlas": atlas,
                            "region": region_base,
                            "hemi_ipsi": hemi,
                            "ipsi_mahal_raw": ipsi_mahal_raw,
                            "contra_mahal_raw": contra_mahal_raw,
                            "asymmetry_mahal_raw": asym_raw,
                        }
                        if inv_left_mcd is not None and inv_right_mcd is not None:
                            invcov_ipsi_mcd = inv_left_mcd[0] if hemi == "L" else inv_right_mcd[0]
                            invcov_contra_mcd = inv_right_mcd[0] if hemi == "L" else inv_left_mcd[0]
                            row["ipsi_mahal_mcd"] = self._mahalanobis_from_zero(ipsi_vec, invcov_ipsi_mcd)
                            row["contra_mahal_mcd"] = self._mahalanobis_from_zero(contra_vec, invcov_contra_mcd)
                            row["asymmetry_mahal_mcd"] = _asymmetry_value(row["ipsi_mahal_mcd"], row["contra_mahal_mcd"])
                        else:
                            row["ipsi_mahal_mcd"] = np.nan
                            row["contra_mahal_mcd"] = np.nan
                            row["asymmetry_mahal_mcd"] = np.nan
                        rows.append(row)
                    _advance()
        finally:
            if show_progress:
                pbar.close()

        return pd.DataFrame(rows)
