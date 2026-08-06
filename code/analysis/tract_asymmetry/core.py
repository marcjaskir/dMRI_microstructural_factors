"""Core scalar asymmetry only: one row per (sub, tract_base, segment, scalar). No intervention, no factors."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from . import config as cfg

# GAM outputs are now {tract}_{scalar}_stat-mean_gam.csv or _stat-standard_deviation_gam.csv
GAM_STAT_SUFFIXES = ("_stat-mean_gam", "_stat-standard_deviation_gam")
DEFAULT_GAM_STAT = "mean"  # which stat to load for asymmetry (mean vs standard_deviation)


def _parse_segment(s: str) -> str:
    """Return segment label as used in GAM."""
    if pd.isna(s) or str(s).strip().upper() in ("NA", ""):
        return ""
    return str(s).strip()


def _tract_base(tract_label: str) -> str:
    """Strip _L or _R suffix for tract base (e.g. ILF_L -> ILF)."""
    if tract_label.endswith("_L"):
        return tract_label[:-2]
    if tract_label.endswith("_R"):
        return tract_label[:-2]
    return tract_label


class TractAsymmetry:
    """
    Compute ipsi vs contra mean z and normalized asymmetry for scalars only.
    One row per (sub, tract_base, segment, scalar). No intervention, no factor z-scores.
    """

    def __init__(
        self,
        base_dir: Path,
        metadata_path: Path,
        gam_dir: Path,
        inclusion_path: Optional[Path] = None,
        gam_stat: str = DEFAULT_GAM_STAT,
        normative_dir: Optional[Path] = None,
    ):
        self.base_dir = Path(base_dir)
        self.metadata_path = Path(metadata_path)
        self.gam_dir = Path(gam_dir)
        self.inclusion_path = Path(inclusion_path) if inclusion_path is not None else (
            self.base_dir / "results" / "inclusion" / "penn_epilepsy_included_basic_metadata_tle.csv"
        )
        self.gam_stat = gam_stat  # "mean" or "standard_deviation" for GAM file suffix
        self.normative_dir = Path(normative_dir) if normative_dir is not None else (
            self.base_dir / "derivatives" / "analysis" / "tract_asymmetry_normative"
        )
        self._meta: Optional[pd.DataFrame] = None
        self._bilateral_pairs: Optional[Dict[str, str]] = None
        self._scalars: Optional[List[str]] = None
        self._gam_cache: Dict[tuple, pd.DataFrame] = {}
        self._invcov_cache: Dict[tuple, Optional[Tuple[np.ndarray, List[str]]]] = {}  # (tract, seg_or_node, level, mcd) -> (invcov, scalars) or None

    def load_metadata(self) -> tuple:
        """Load HCP1065 metadata and build bilateral pairs. Returns (meta, bilateral_pairs)."""
        meta = pd.read_csv(self.metadata_path)
        meta = meta[meta["hemi"].isin(["left", "right"])]
        meta = meta[meta["profilable"].astype(str).str.upper() == "TRUE"]
        meta = meta.dropna(subset=["end1", "end2"])
        left_tracts = meta[meta["hemi"] == "left"]["label"].tolist()
        right_tracts = set(meta[meta["hemi"] == "right"]["label"].tolist())
        gam_tracts = set(p.name for p in self.gam_dir.iterdir() if p.is_dir())
        bilateral_pairs = {}
        for lt in left_tracts:
            rt = lt.replace("_L", "_R")
            if rt in right_tracts and lt in gam_tracts and rt in gam_tracts:
                bilateral_pairs[lt] = rt
        self._meta = meta
        self._bilateral_pairs = bilateral_pairs
        return meta, bilateral_pairs

    def get_all_bilateral_tract_segment_pairs(self) -> List[Tuple[str, str]]:
        """Return list of (left_tract_label, segment) for all bilateral left-tract segments."""
        if self._meta is None:
            self.load_metadata()
        out: List[Tuple[str, str]] = []
        for tract in self._bilateral_pairs:
            row = self._meta[self._meta["label"] == tract]
            if row.empty:
                continue
            row = row.iloc[0]
            segments = ["core"]
            e1, e2 = _parse_segment(row.get("end1", "")), _parse_segment(row.get("end2", ""))
            if e1:
                segments.append(e1)
            if e2 and e2 not in segments:
                segments.append(e2)
            for seg in segments:
                if self.segment_to_nodes(tract, seg) is not None:
                    out.append((tract, seg))
        return out

    def segment_to_nodes(self, tract_label: str, segment: str) -> Optional[List[int]]:
        """Map (tract, segment) to list of node indices."""
        if self._meta is None:
            self.load_metadata()
        if segment == "core":
            return cfg.CORE_NODES
        row = self._meta[self._meta["label"] == tract_label]
        if row.empty:
            return None
        row = row.iloc[0]
        e1, e2 = _parse_segment(row.get("end1", "")), _parse_segment(row.get("end2", ""))
        if segment == e1:
            return cfg.END1_NODES
        if segment == e2:
            return cfg.END2_NODES
        return None

    @property
    def scalars(self) -> List[str]:
        """List of scalar names from GAM. Supports _stat-mean_gam and _stat-standard_deviation_gam stems."""
        if self._scalars is not None:
            return self._scalars
        sample_tract = "ILF_L"
        all_scalars = []
        prefix = f"{sample_tract}_"
        for p in (self.gam_dir / sample_tract).iterdir():
            if not p.is_file() or p.suffix != ".csv":
                continue
            stem = p.stem
            if not stem.startswith(prefix):
                continue
            scalar = None
            for suffix in GAM_STAT_SUFFIXES:
                if stem.endswith(suffix):
                    scalar = stem[len(prefix) : -len(suffix)]
                    break
            if scalar is None:
                continue
            all_scalars.append(scalar)
        self._scalars = sorted(set(all_scalars))
        return self._scalars

    def get_subjects_with_gam(self) -> List[str]:
        """Subjects present in GAM (sample tract). Uses current gam_stat convention."""
        sample_tract = "ILF_L"
        stat_suffix = f"_stat-{self.gam_stat}_gam" if self.gam_stat in ("mean", "standard_deviation") else "_gam"
        sample_gam = self.gam_dir / sample_tract / f"{sample_tract}_dki_ad{stat_suffix}.csv"
        if not sample_gam.exists():
            return []
        return sorted(pd.read_csv(sample_gam)["sub"].astype(str).unique().tolist())

    def get_eligible_subjects(self) -> List[str]:
        """Eligible = in TLE inclusion list (from CSV, temporal lobe only if lobe col present) AND has GAM data."""
        tle_subjects, _ = cfg.load_tle_inclusion(self.inclusion_path)
        gam_subs = set(self.get_subjects_with_gam())
        return sorted(set(tle_subjects) & gam_subs)

    def _load_gam(self, tract: str, scalar: str) -> Optional[pd.DataFrame]:
        """Load GAM CSV for (tract, scalar); cache by (tract, scalar). Uses gam_stat suffix (_stat-mean_gam or _stat-standard_deviation_gam)."""
        key = (tract, scalar)
        if key not in self._gam_cache:
            stat_suffix = f"_stat-{self.gam_stat}_gam" if self.gam_stat in ("mean", "standard_deviation") else "_gam"
            path = self.gam_dir / tract / f"{tract}_{scalar}{stat_suffix}.csv"
            if not path.exists():
                return None
            self._gam_cache[key] = pd.read_csv(path)
        return self._gam_cache[key]

    @staticmethod
    def _asymmetry_value(ipsi_mean_z: float, contra_mean_z: float) -> float:
        """Normalized asymmetry: (ipsi - contra) / (|ipsi| + |contra|). NaN if denominator is 0."""
        denom = abs(ipsi_mean_z) + abs(contra_mean_z)
        if denom == 0:
            return float("nan")
        return (ipsi_mean_z - contra_mean_z) / denom

    def _load_invcov(
        self,
        tract: str,
        segment_or_node: str,
        level: str,
        use_mcd: bool,
    ) -> Optional[Tuple[np.ndarray, List[str]]]:
        """
        Load inverse covariance from tract_asymmetry_normative outputs.
        level: 'segment' or 'node'. For node, segment_or_node is node index as str (e.g. '1') or int-like.
        Returns (invcov matrix, list of scalar names in order) or None.
        """
        key = (tract, str(segment_or_node), level, use_mcd)
        if key in self._invcov_cache:
            return self._invcov_cache[key]
        if level == "segment":
            subdir = self.normative_dir / "segment_level" / tract / str(segment_or_node)
        else:
            k = int(segment_or_node)
            subdir = self.normative_dir / "node_level" / tract / f"node_{k:03d}"
        fname = "invcov_mincovdet.csv" if use_mcd else "invcov.csv"
        path = subdir / fname
        if not path.exists():
            self._invcov_cache[key] = None
            return None
        try:
            df = pd.read_csv(path, index_col=0)
        except Exception:
            self._invcov_cache[key] = None
            return None
        scalars = df.columns.tolist()
        if df.index.tolist() != scalars:
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

    def compute_asymmetry(
        self,
        subjects: Optional[List[str]] = None,
        subject_ipsi_hemi: Optional[Dict[str, str]] = None,
        return_missing: bool = False,
    ) -> Any:
        """
        Build scalar asymmetry table: one row per (sub, tract_base, segment, scalar).
        subject_ipsi_hemi: sub -> 'L' or 'R' (required; from TLE laterality).
        If return_missing=True, returns (df, missing_info) where missing_info[sub] has
        'missing_segments' and 'missing_scalars' (sets of expected but absent).
        """
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = lambda x, **kw: x  # noqa: E731
        if self._meta is None:
            self.load_metadata()
        if subjects is None:
            subjects = self.get_eligible_subjects()
        if subject_ipsi_hemi is None:
            subject_ipsi_hemi = {}
        pairs = self.get_all_bilateral_tract_segment_pairs()
        expected_pairs = set((_tract_base(t), seg) for (t, seg) in pairs)
        expected_scalars = set(self.scalars)
        rows = []
        missing_info: Dict[str, Dict[str, set]] = {}
        for sub in subjects:
            hemi = subject_ipsi_hemi.get(sub)
            if hemi not in ("L", "R"):
                continue
            pairs_with_data: set = set()
            scalars_with_data: set = set()
            for (tract, segment) in tqdm(pairs, desc=f"Scalar asym {sub}", leave=False):
                if tract not in self._bilateral_pairs:
                    continue
                contra_tract = self._bilateral_pairs[tract]
                node_list = self.segment_to_nodes(tract, segment)
                if node_list is None:
                    continue
                z_cols = [f"node{k}_z" for k in node_list]
                tract_base = _tract_base(tract)
                for scalar in self.scalars:
                    left_gam = self._load_gam(tract, scalar)
                    right_gam = self._load_gam(contra_tract, scalar)
                    if left_gam is None or right_gam is None:
                        continue
                    left_row = left_gam[left_gam["sub"].astype(str) == sub]
                    right_row = right_gam[right_gam["sub"].astype(str) == sub]
                    if left_row.empty or right_row.empty:
                        continue
                    left_mean_z = float(left_row[z_cols].iloc[0].mean())
                    right_mean_z = float(right_row[z_cols].iloc[0].mean())
                    ipsi_mean_z = left_mean_z if hemi == "L" else right_mean_z
                    contra_mean_z = right_mean_z if hemi == "L" else left_mean_z
                    asymmetry = self._asymmetry_value(ipsi_mean_z, contra_mean_z)
                    rows.append({
                        "sub": sub, "tract": tract_base, "segment": segment, "scalar": scalar,
                        "ipsi_mean_z": ipsi_mean_z, "contra_mean_z": contra_mean_z,
                        "asymmetry": asymmetry, "hemi_ipsi": hemi,
                    })
                    pairs_with_data.add((tract_base, segment))
                    scalars_with_data.add(scalar)
            if return_missing:
                missing_info[sub] = {
                    "missing_segments": expected_pairs - pairs_with_data,
                    "missing_scalars": expected_scalars - scalars_with_data,
                }
        df = pd.DataFrame(rows)
        if return_missing:
            return df, missing_info
        return df

    def get_all_bilateral_tract_nodes(self) -> List[Tuple[str, int]]:
        """Return list of (left_tract_label, node) for nodes 1..100 for all bilateral tracts."""
        if self._meta is None:
            self.load_metadata()
        out: List[Tuple[str, int]] = []
        for tract in self._bilateral_pairs:
            for node in range(1, 101):
                out.append((tract, node))
        return out

    def compute_asymmetry_node_level(
        self,
        subjects: Optional[List[str]] = None,
        subject_ipsi_hemi: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Build scalar asymmetry table at node level: one row per (sub, tract_base, node, scalar).
        Same structure as segment-level but with node (1..100) instead of segment.
        """
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = lambda x, **kw: x  # noqa: E731
        if self._meta is None:
            self.load_metadata()
        if subjects is None:
            subjects = self.get_eligible_subjects()
        if subject_ipsi_hemi is None:
            subject_ipsi_hemi = {}
        pairs = self.get_all_bilateral_tract_nodes()
        rows = []
        for sub in subjects:
            hemi = subject_ipsi_hemi.get(sub)
            if hemi not in ("L", "R"):
                continue
            for (tract, node) in tqdm(pairs, desc=f"Node asym {sub}", leave=False):
                if tract not in self._bilateral_pairs:
                    continue
                contra_tract = self._bilateral_pairs[tract]
                z_col = f"node{node}_z"
                tract_base = _tract_base(tract)
                for scalar in self.scalars:
                    left_gam = self._load_gam(tract, scalar)
                    right_gam = self._load_gam(contra_tract, scalar)
                    if left_gam is None or right_gam is None or z_col not in left_gam.columns or z_col not in right_gam.columns:
                        continue
                    left_row = left_gam[left_gam["sub"].astype(str) == sub]
                    right_row = right_gam[right_gam["sub"].astype(str) == sub]
                    if left_row.empty or right_row.empty:
                        continue
                    left_z = float(left_row[z_col].iloc[0])
                    right_z = float(right_row[z_col].iloc[0])
                    ipsi_z = left_z if hemi == "L" else right_z
                    contra_z = right_z if hemi == "L" else left_z
                    asymmetry = self._asymmetry_value(ipsi_z, contra_z)
                    rows.append({
                        "sub": sub, "tract": tract_base, "node": node, "scalar": scalar,
                        "ipsi_z": ipsi_z, "contra_z": contra_z,
                        "asymmetry": asymmetry, "hemi_ipsi": hemi,
                    })
        return pd.DataFrame(rows)

    def _mahal_asymmetry_value(self, ipsi_mahal: float, contra_mahal: float) -> float:
        """Same normalized asymmetry formula for Mahalanobis distances."""
        return self._asymmetry_value(ipsi_mahal, contra_mahal)

    def compute_mahal_asymmetry_segment(
        self,
        subjects: Optional[List[str]] = None,
        subject_ipsi_hemi: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        One row per (sub, tract_base, segment): ipsi/contra Mahalanobis distance from 0
        using normative invcov (raw and MCD), and asymmetry from those distances.
        """
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = lambda x, **kw: x  # noqa: E731
        if self._meta is None:
            self.load_metadata()
        if subjects is None:
            subjects = self.get_eligible_subjects()
        if subject_ipsi_hemi is None:
            subject_ipsi_hemi = {}
        pairs = self.get_all_bilateral_tract_segment_pairs()
        rows = []
        for sub in subjects:
            hemi = subject_ipsi_hemi.get(sub)
            if hemi not in ("L", "R"):
                continue
            for (tract, segment) in tqdm(pairs, desc=f"Mahal seg {sub}", leave=False):
                contra_tract = self._bilateral_pairs.get(tract)
                if not contra_tract:
                    continue
                node_list = self.segment_to_nodes(tract, segment)
                if node_list is None:
                    continue
                z_cols = [f"node{k}_z" for k in node_list]
                tract_base = _tract_base(tract)
                # Build ipsi and contra z vectors (mean over segment nodes per scalar) in invcov scalar order
                for use_mcd in (False, True):
                    inv_ipsi = self._load_invcov(tract, segment, "segment", use_mcd)
                    inv_contra = self._load_invcov(contra_tract, segment, "segment", use_mcd)
                    if inv_ipsi is None or inv_contra is None:
                        continue
                    invcov_ipsi, scalars = inv_ipsi
                    invcov_contra, _ = inv_contra
                    ipsi_z_vec = []
                    contra_z_vec = []
                    for s in scalars:
                        left_gam = self._load_gam(tract, s)
                        right_gam = self._load_gam(contra_tract, s)
                        if left_gam is None or right_gam is None:
                            break
                        left_row = left_gam[left_gam["sub"].astype(str) == sub]
                        right_row = right_gam[right_gam["sub"].astype(str) == sub]
                        if left_row.empty or right_row.empty:
                            break
                        missing = [c for c in z_cols if c not in left_gam.columns or c not in right_gam.columns]
                        if missing:
                            break
                        ipsi_z_vec.append(float(left_row[z_cols].iloc[0].mean()) if hemi == "L" else float(right_row[z_cols].iloc[0].mean()))
                        contra_z_vec.append(float(right_row[z_cols].iloc[0].mean()) if hemi == "L" else float(left_row[z_cols].iloc[0].mean()))
                    if len(ipsi_z_vec) != len(scalars) or len(contra_z_vec) != len(scalars):
                        continue
                    ipsi_vec = np.array(ipsi_z_vec, dtype=float)
                    contra_vec = np.array(contra_z_vec, dtype=float)
                    ipsi_mahal = self._mahalanobis_from_zero(ipsi_vec, invcov_ipsi)
                    contra_mahal = self._mahalanobis_from_zero(contra_vec, invcov_contra)
                    asym = self._mahal_asymmetry_value(ipsi_mahal, contra_mahal)
                    suffix = "_mcd" if use_mcd else "_raw"
                    row_key = (sub, tract_base, segment)
                    existing = next((r for r in rows if (r["sub"], r["tract"], r["segment"]) == row_key), None)
                    if existing is None:
                        existing = {
                            "sub": sub, "tract": tract_base, "segment": segment, "hemi_ipsi": hemi,
                            "ipsi_mahal_raw": np.nan, "contra_mahal_raw": np.nan, "asymmetry_mahal_raw": np.nan,
                            "ipsi_mahal_mcd": np.nan, "contra_mahal_mcd": np.nan, "asymmetry_mahal_mcd": np.nan,
                        }
                        rows.append(existing)
                    existing[f"ipsi_mahal{suffix}"] = ipsi_mahal
                    existing[f"contra_mahal{suffix}"] = contra_mahal
                    existing[f"asymmetry_mahal{suffix}"] = asym
        return pd.DataFrame(rows)

    def compute_mahal_asymmetry_node(
        self,
        subjects: Optional[List[str]] = None,
        subject_ipsi_hemi: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        One row per (sub, tract_base, node): ipsi/contra Mahalanobis distance from 0
        using normative invcov (raw and MCD), and asymmetry.
        """
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = lambda x, **kw: x  # noqa: E731
        if self._meta is None:
            self.load_metadata()
        if subjects is None:
            subjects = self.get_eligible_subjects()
        if subject_ipsi_hemi is None:
            subject_ipsi_hemi = {}
        pairs = self.get_all_bilateral_tract_nodes()
        rows = []
        for sub in subjects:
            hemi = subject_ipsi_hemi.get(sub)
            if hemi not in ("L", "R"):
                continue
            for (tract, node) in tqdm(pairs, desc=f"Mahal node {sub}", leave=False):
                contra_tract = self._bilateral_pairs.get(tract)
                if not contra_tract:
                    continue
                z_col = f"node{node}_z"
                tract_base = _tract_base(tract)
                for use_mcd in (False, True):
                    inv_ipsi = self._load_invcov(tract, str(node), "node", use_mcd)
                    inv_contra = self._load_invcov(contra_tract, str(node), "node", use_mcd)
                    if inv_ipsi is None or inv_contra is None:
                        continue
                    invcov_ipsi, scalars = inv_ipsi
                    invcov_contra, _ = inv_contra
                    ipsi_z_vec = []
                    contra_z_vec = []
                    for s in scalars:
                        left_gam = self._load_gam(tract, s)
                        right_gam = self._load_gam(contra_tract, s)
                        if left_gam is None or right_gam is None or z_col not in left_gam.columns or z_col not in right_gam.columns:
                            break
                        left_row = left_gam[left_gam["sub"].astype(str) == sub]
                        right_row = right_gam[right_gam["sub"].astype(str) == sub]
                        if left_row.empty or right_row.empty:
                            break
                        lz = float(left_row[z_col].iloc[0])
                        rz = float(right_row[z_col].iloc[0])
                        ipsi_z_vec.append(lz if hemi == "L" else rz)
                        contra_z_vec.append(rz if hemi == "L" else lz)
                    if len(ipsi_z_vec) != len(scalars) or len(contra_z_vec) != len(scalars):
                        continue
                    ipsi_vec = np.array(ipsi_z_vec, dtype=float)
                    contra_vec = np.array(contra_z_vec, dtype=float)
                    ipsi_mahal = self._mahalanobis_from_zero(ipsi_vec, invcov_ipsi)
                    contra_mahal = self._mahalanobis_from_zero(contra_vec, invcov_contra)
                    asym = self._mahal_asymmetry_value(ipsi_mahal, contra_mahal)
                    suffix = "_mcd" if use_mcd else "_raw"
                    row_key = (sub, tract_base, node)
                    existing = next((r for r in rows if (r["sub"], r["tract"], r["node"]) == row_key), None)
                    if existing is None:
                        existing = {
                            "sub": sub, "tract": tract_base, "node": node, "hemi_ipsi": hemi,
                            "ipsi_mahal_raw": np.nan, "contra_mahal_raw": np.nan, "asymmetry_mahal_raw": np.nan,
                            "ipsi_mahal_mcd": np.nan, "contra_mahal_mcd": np.nan, "asymmetry_mahal_mcd": np.nan,
                        }
                        rows.append(existing)
                    existing[f"ipsi_mahal{suffix}"] = ipsi_mahal
                    existing[f"contra_mahal{suffix}"] = contra_mahal
                    existing[f"asymmetry_mahal{suffix}"] = asym
        return pd.DataFrame(rows)
