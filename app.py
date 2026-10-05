
import re
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------

st.set_page_config(
    page_title="Polymer Series Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp {
            background: #F7F9FC;
        }

        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }

        section[data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid #E6EAF0;
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 1rem;
        }

        .app-hero {
            background: linear-gradient(135deg, #0E223D 0%, #163A63 100%);
            border-radius: 18px;
            padding: 26px 30px;
            margin-bottom: 18px;
            box-shadow: 0 10px 28px rgba(24, 46, 78, 0.10);
        }

        .app-hero h1 {
            color: white;
            font-size: 2rem;
            line-height: 1.15;
            margin: 0;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        .app-hero p {
            color: rgba(255,255,255,0.82);
            margin: 8px 0 0 0;
            font-size: 0.98rem;
        }

        .eyebrow {
            color: #A8C4E6;
            text-transform: uppercase;
            font-size: 0.73rem;
            font-weight: 700;
            letter-spacing: 0.13em;
            margin-bottom: 7px;
        }

        .metric-card {
            background: #FFFFFF;
            border: 1px solid #E6EAF0;
            border-radius: 14px;
            padding: 16px 18px;
            min-height: 98px;
            box-shadow: 0 3px 12px rgba(18, 38, 63, 0.035);
        }

        .metric-label {
            color: #687587;
            font-size: 0.78rem;
            font-weight: 650;
            text-transform: uppercase;
            letter-spacing: 0.055em;
            margin-bottom: 6px;
        }

        .metric-value {
            color: #10243E;
            font-size: 1.45rem;
            font-weight: 720;
            line-height: 1.1;
        }

        .metric-subtext {
            color: #8A96A6;
            font-size: 0.78rem;
            margin-top: 5px;
        }

        .section-card {
            background: #FFFFFF;
            border: 1px solid #E6EAF0;
            border-radius: 14px;
            padding: 18px 20px;
            margin-bottom: 14px;
            box-shadow: 0 3px 12px rgba(18, 38, 63, 0.03);
        }

        .status-pill {
            display: inline-block;
            background: #EAF4EE;
            color: #2B6843;
            border-radius: 999px;
            padding: 6px 10px;
            font-size: 0.76rem;
            font-weight: 650;
        }

        .muted-pill {
            display: inline-block;
            background: #EEF2F7;
            color: #536274;
            border-radius: 999px;
            padding: 6px 10px;
            font-size: 0.76rem;
            font-weight: 650;
            margin-right: 6px;
            margin-bottom: 5px;
        }

        .small-note {
            color: #738195;
            font-size: 0.84rem;
            line-height: 1.45;
        }

        h2, h3 {
            color: #10243E;
            letter-spacing: -0.012em;
        }

        div[data-testid="stButton"] > button {
            border-radius: 10px;
            font-weight: 650;
            min-height: 42px;
        }

        div[data-testid="stDownloadButton"] > button {
            border-radius: 10px;
            font-weight: 650;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #E6EAF0;
            border-radius: 12px;
            overflow: hidden;
        }

        div[data-testid="stFileUploader"] {
            background: white;
            border-radius: 12px;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: #FFFFFF;
            padding: 5px;
            border: 1px solid #E6EAF0;
            border-radius: 12px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 9px;
            padding-left: 18px;
            padding-right: 18px;
        }

        .stTabs [aria-selected="true"] {
            background: #EFF4FA;
        }

        hr {
            border-color: #E8ECF2;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-hero">
        <div class="eyebrow">LCMS polymer screening</div>
        <h1>Polymer Series Explorer</h1>
        <p>Map homologous polymer series across an LCMS feature table, connect companion adduct forms, and inspect every assignment interactively.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

MONOISOTOPIC_MASS = {
    "H": 1.00782503223,
    "C": 12.00000000000,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "F": 18.99840316273,
    "Na": 22.9897692820,
    "Mg": 23.985041697,
    "Al": 26.98153853,
    "Si": 27.97692653465,
    "P": 30.97376199842,
    "S": 31.9720711744,
    "Cl": 34.968852682,
    "K": 38.9637064864,
    "Ca": 39.962590863,
    "Fe": 55.93493633,
    "Br": 78.9183376,
    "I": 126.904468,
}

PROTON_MASS = 1.007276466621
ELECTRON_MASS = 0.000548579909

ADDUCT_LIBRARY = {
    "[M+H]+": {
        "shift": PROTON_MASS,
        "charge": 1,
        "polarity": "positive",
        "description": "Protonated",
    },
    "[M+Na]+": {
        "shift": MONOISOTOPIC_MASS["Na"] - ELECTRON_MASS,
        "charge": 1,
        "polarity": "positive",
        "description": "Sodiated",
    },
    "[M+K]+": {
        "shift": MONOISOTOPIC_MASS["K"] - ELECTRON_MASS,
        "charge": 1,
        "polarity": "positive",
        "description": "Potassiated",
    },
    "[M+NH4]+": {
        "shift": (
            MONOISOTOPIC_MASS["N"]
            + 4 * MONOISOTOPIC_MASS["H"]
            - ELECTRON_MASS
        ),
        "charge": 1,
        "polarity": "positive",
        "description": "Ammoniated",
    },
    "[M-H]-": {
        "shift": -PROTON_MASS,
        "charge": -1,
        "polarity": "negative",
        "description": "Deprotonated",
    },
}

REPEAT_LIBRARY = {
    "PEG / EO": {
        "full_name": "Polyethylene glycol / ethylene oxide",
        "formula": "C2H4O",
        "notes": "Ethylene oxide repeat",
    },
    "PPG / PO": {
        "full_name": "Polypropylene glycol / propylene oxide",
        "formula": "C3H6O",
        "notes": "Propylene oxide repeat",
    },
    "PE": {
        "full_name": "Polyethylene",
        "formula": "C2H4",
        "notes": "Ethylene repeat",
    },
    "PP": {
        "full_name": "Polypropylene",
        "formula": "C3H6",
        "notes": "Propylene repeat",
    },
    "PS": {
        "full_name": "Polystyrene",
        "formula": "C8H8",
        "notes": "Styrene repeat",
    },
    "PMMA": {
        "full_name": "Poly(methyl methacrylate)",
        "formula": "C5H8O2",
        "notes": "Methyl methacrylate repeat",
    },
    "PLA": {
        "full_name": "Poly(lactic acid)",
        "formula": "C3H4O2",
        "notes": "Lactyl repeat",
    },
    "Nylon 6": {
        "full_name": "Nylon 6",
        "formula": "C6H11NO",
        "notes": "Caprolactam derived repeat",
    },
    "PDMS": {
        "full_name": "Polydimethylsiloxane",
        "formula": "C2H6OSi",
        "notes": "Dimethylsiloxane repeat",
    },
    "PET": {
        "full_name": "Polyethylene terephthalate",
        "formula": "C10H8O4",
        "notes": "Ethylene terephthalate repeat",
    },
}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def formula_mass(formula: str) -> float:
    if not formula:
        raise ValueError("Formula is empty.")

    formula = formula.strip()
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    rebuilt = "".join(element + count for element, count in tokens)

    if rebuilt != formula:
        raise ValueError("Use a simple formula such as C2H4O, C8H8, or C2H6OSi.")

    total = 0.0
    for element, count_text in tokens:
        if element not in MONOISOTOPIC_MASS:
            raise ValueError(f"Element {element} is not in the built in mass table.")
        count = int(count_text) if count_text else 1
        total += MONOISOTOPIC_MASS[element] * count

    return total


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "mz": ["mz", "m/z", "mass_to_charge", "mass to charge"],
        "rt": ["rt", "retention_time", "retention time", "retentiontime"],
        "intensity": ["intensity", "height", "abundance", "area", "peak_area", "peak area"],
    }

    lowered = {str(c).strip().lower(): c for c in df.columns}
    rename = {}

    for canonical, names in aliases.items():
        for name in names:
            if name in lowered:
                rename[lowered[name]] = canonical
                break

    out = df.rename(columns=rename).copy()

    if "mz" not in out.columns:
        raise ValueError("A column named m/z or mz is required.")

    out["mz"] = pd.to_numeric(out["mz"], errors="coerce")
    out["rt"] = (
        pd.to_numeric(out["rt"], errors="coerce")
        if "rt" in out.columns
        else np.nan
    )
    out["intensity"] = (
        pd.to_numeric(out["intensity"], errors="coerce")
        if "intensity" in out.columns
        else 1.0
    )

    out = out.dropna(subset=["mz"]).reset_index(drop=True)
    out["feature_id"] = np.arange(1, len(out) + 1)

    return out[["feature_id", "mz", "rt", "intensity"]]


def find_series(
    features: pd.DataFrame,
    repeat_mass: float,
    charge: int,
    ppm_tolerance: float,
    min_members: int,
    max_repeat_gap: int,
    rt_tolerance,
):
    df = features.sort_values("mz").reset_index(drop=True).copy()
    spacing = repeat_mass / abs(charge)
    mz = df["mz"].to_numpy()
    n = len(df)

    graph = defaultdict(set)

    for i in range(n):
        for j in range(i + 1, n):
            delta = mz[j] - mz[i]

            max_delta = spacing * max_repeat_gap * (1 + ppm_tolerance * 1e-6)
            if delta > max_delta:
                break

            repeats = int(round(delta / spacing))
            if repeats < 1 or repeats > max_repeat_gap:
                continue

            target = repeats * spacing
            ppm_error_value = (delta - target) / target * 1e6

            if abs(ppm_error_value) <= ppm_tolerance:
                if rt_tolerance is not None:
                    rt_i = df.loc[i, "rt"]
                    rt_j = df.loc[j, "rt"]

                    if pd.notna(rt_i) and pd.notna(rt_j):
                        if abs(rt_j - rt_i) > rt_tolerance:
                            continue

                graph[i].add(j)
                graph[j].add(i)

    visited = set()
    components = []

    for start in range(n):
        if start in visited or start not in graph:
            continue

        queue = deque([start])
        visited.add(start)
        comp = []

        while queue:
            node = queue.popleft()
            comp.append(node)

            for nxt in graph[node]:
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)

        if len(comp) >= min_members:
            components.append(sorted(comp, key=lambda x: mz[x]))

    components.sort(key=len, reverse=True)

    rows = []

    for local_series_number, comp in enumerate(components, start=1):
        base_idx = comp[0]
        base_mz = mz[base_idx]

        for idx in comp:
            repeat_number = int(round((mz[idx] - base_mz) / spacing))
            predicted = base_mz + repeat_number * spacing

            ppm_from_series = (
                0.0
                if repeat_number == 0
                else (mz[idx] - predicted) / predicted * 1e6
            )

            row = df.loc[idx].to_dict()
            row.update({
                "local_series_number": local_series_number,
                "repeat_index": repeat_number,
                "charge": charge,
                "repeat_mass": repeat_mass,
                "spacing_mz": spacing,
                "series_ppm_error": ppm_from_series,
            })
            rows.append(row)

    return pd.DataFrame(rows)


def add_kendrick(df: pd.DataFrame, repeat_mass: float):
    out = df.copy()
    nominal = round(repeat_mass)

    out["kendrick_mass"] = out["mz"] * nominal / repeat_mass
    out["kendrick_nominal_mass"] = np.round(out["kendrick_mass"])
    out["kmd"] = out["kendrick_nominal_mass"] - out["kendrick_mass"]

    return out


def run_multi_repeat_search(
    features,
    search_units,
    charge,
    ppm_tolerance,
    min_members,
    max_repeat_gap,
    rt_tolerance,
):
    all_results = []

    for search_unit in search_units:
        result = find_series(
            features=features,
            repeat_mass=search_unit["mass"],
            charge=charge,
            ppm_tolerance=ppm_tolerance,
            min_members=min_members,
            max_repeat_gap=max_repeat_gap,
            rt_tolerance=rt_tolerance,
        )

        if result.empty:
            continue

        result = add_kendrick(result, search_unit["mass"])
        result["repeat_unit"] = search_unit["label"]
        result["repeat_formula"] = search_unit["formula"]
        result["repeat_description"] = search_unit["description"]

        result["series_id"] = result["local_series_number"].apply(
            lambda x: f'{search_unit["label"]} | Series {int(x)}'
        )

        all_results.append(result)

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


def build_summary(results: pd.DataFrame):
    if results.empty:
        return pd.DataFrame()

    summary = (
        results.groupby(
            ["repeat_unit", "repeat_formula", "repeat_mass", "series_id"],
            as_index=False
        )
        .agg(
            members=("feature_id", "count"),
            mz_min=("mz", "min"),
            mz_max=("mz", "max"),
            rt_min=("rt", "min"),
            rt_max=("rt", "max"),
            total_intensity=("intensity", "sum"),
            max_abs_ppm=(
                "series_ppm_error",
                lambda x: np.nanmax(np.abs(x))
            ),
        )
    )

    return summary.sort_values(
        ["repeat_unit", "members", "mz_min"],
        ascending=[True, False, True]
    ).reset_index(drop=True)


def ppm_error(observed, expected):
    return (observed - expected) / expected * 1e6


def find_adduct_series_links(
    results: pd.DataFrame,
    selected_adducts,
    ppm_tolerance: float,
    rt_tolerance: float,
    min_matches: int = 3,
):
    """
    Compare complete homologous series with the same repeat unit.

    Each detected homologous series is treated as a single ion/adduct form.
    Companion series are linked only when multiple members show the expected
    exact adduct mass offset and coelution.

    Returns
    -------
    links : DataFrame
        One row per supported series-to-series adduct relationship.
    member_matches : DataFrame
        Feature-level evidence used to draw visual connectors between linked
        adduct series.
    """
    if results.empty or len(selected_adducts) < 2:
        return pd.DataFrame(), pd.DataFrame()

    link_rows = []
    member_rows = []
    seen = set()

    for repeat_unit in results["repeat_unit"].unique():
        ru = results[results["repeat_unit"] == repeat_unit]
        series_ids = list(ru["series_id"].unique())

        for i in range(len(series_ids)):
            a_id = series_ids[i]
            a = ru[ru["series_id"] == a_id].copy()

            for j in range(i + 1, len(series_ids)):
                b_id = series_ids[j]
                b = ru[ru["series_id"] == b_id].copy()

                for adduct_a in selected_adducts:
                    for adduct_b in selected_adducts:
                        if adduct_a == adduct_b:
                            continue

                        info_a = ADDUCT_LIBRARY[adduct_a]
                        info_b = ADDUCT_LIBRARY[adduct_b]

                        # Positive and negative ion forms are never merged into
                        # one adduct relationship.
                        if info_a["polarity"] != info_b["polarity"]:
                            continue

                        expected_delta = info_b["shift"] - info_a["shift"]

                        # Evaluate each pair once in the increasing-mass direction.
                        if expected_delta <= 0:
                            continue

                        matches = []

                        for _, row_a in a.iterrows():
                            expected_b_mz = row_a["mz"] + expected_delta

                            candidates = b.copy()
                            candidates["abs_ppm"] = np.abs(
                                (candidates["mz"] - expected_b_mz)
                                / expected_b_mz
                                * 1e6
                            )

                            candidates = candidates[
                                candidates["abs_ppm"] <= ppm_tolerance
                            ]

                            if candidates.empty:
                                continue

                            if pd.notna(row_a["rt"]) and candidates["rt"].notna().any():
                                candidates = candidates[
                                    np.abs(candidates["rt"] - row_a["rt"])
                                    <= rt_tolerance
                                ]

                            if candidates.empty:
                                continue

                            best = candidates.sort_values("abs_ppm").iloc[0]

                            matches.append({
                                "repeat_unit": repeat_unit,
                                "series_a": a_id,
                                "candidate_adduct_a": adduct_a,
                                "series_b": b_id,
                                "candidate_adduct_b": adduct_b,
                                "feature_a": int(row_a["feature_id"]),
                                "feature_b": int(best["feature_id"]),
                                "mz_a": float(row_a["mz"]),
                                "mz_b": float(best["mz"]),
                                "rt_a": float(row_a["rt"]) if pd.notna(row_a["rt"]) else np.nan,
                                "rt_b": float(best["rt"]) if pd.notna(best["rt"]) else np.nan,
                                "repeat_index_a": int(row_a["repeat_index"]),
                                "repeat_index_b": int(best["repeat_index"]),
                                "delta_mz": float(best["mz"] - row_a["mz"]),
                                "ppm_error": float(
                                    ppm_error(
                                        best["mz"] - row_a["mz"],
                                        expected_delta
                                    )
                                ),
                                "rt_difference": (
                                    float(abs(best["rt"] - row_a["rt"]))
                                    if pd.notna(best["rt"]) and pd.notna(row_a["rt"])
                                    else np.nan
                                ),
                            })

                        unique_pairs = {
                            (m["feature_a"], m["feature_b"]) for m in matches
                        }

                        if len(unique_pairs) >= min_matches:
                            key = (
                                repeat_unit,
                                a_id,
                                adduct_a,
                                b_id,
                                adduct_b,
                            )

                            if key in seen:
                                continue

                            seen.add(key)

                            link_rows.append({
                                "repeat_unit": repeat_unit,
                                "series_a": a_id,
                                "candidate_adduct_a": adduct_a,
                                "series_b": b_id,
                                "candidate_adduct_b": adduct_b,
                                "matched_members": len(unique_pairs),
                                "expected_delta_mz": expected_delta,
                                "median_observed_delta_mz": np.median(
                                    [m["delta_mz"] for m in matches]
                                ),
                                "median_abs_ppm": np.median(
                                    [abs(m["ppm_error"]) for m in matches]
                                ),
                                "median_rt_difference": np.nanmedian(
                                    [m["rt_difference"] for m in matches]
                                ),
                            })

                            member_rows.extend(matches)

    return pd.DataFrame(link_rows), pd.DataFrame(member_rows)

def build_series_adduct_map(links: pd.DataFrame):
    """
    Return one series-level candidate adduct label.

    A homologous series is intentionally assigned one adduct identity. If the
    evidence is inconsistent, the series is marked Ambiguous rather than
    assigning different adducts to individual members.
    """
    candidates = defaultdict(set)

    if links.empty:
        return {}

    for _, row in links.iterrows():
        candidates[row["series_a"]].add(row["candidate_adduct_a"])
        candidates[row["series_b"]].add(row["candidate_adduct_b"])

    output = {}
    for series_id, labels in candidates.items():
        if len(labels) == 1:
            output[series_id] = next(iter(labels))
        else:
            output[series_id] = "Ambiguous: " + " / ".join(sorted(labels))

    return output


def build_family_groups(series_ids, links: pd.DataFrame):
    """
    Union series that are supported as companion adduct forms.

    Each line remains its own homologous series/adduct, while linked lines are
    assigned the same underlying polymer family group.
    """
    parent = {sid: sid for sid in series_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    if not links.empty:
        for _, row in links.iterrows():
            if row["series_a"] in parent and row["series_b"] in parent:
                union(row["series_a"], row["series_b"])

    roots = []
    for sid in series_ids:
        root = find(sid)
        if root not in roots:
            roots.append(root)

    root_to_group = {
        root: f"Family {i + 1}"
        for i, root in enumerate(roots)
    }

    return {
        sid: root_to_group[find(sid)]
        for sid in series_ids
    }


def annotate_summary_with_adducts(summary, links):
    out = summary.copy()
    out["adduct_evidence"] = "No companion adduct series identified"

    if links.empty:
        return out

    evidence_map = defaultdict(list)

    for _, row in links.iterrows():
        evidence_map[row["series_a"]].append(
            f'{row["candidate_adduct_a"]} linked to '
            f'{row["series_b"]} ({row["candidate_adduct_b"]})'
        )
        evidence_map[row["series_b"]].append(
            f'{row["candidate_adduct_b"]} linked to '
            f'{row["series_a"]} ({row["candidate_adduct_a"]})'
        )

    for idx, row in out.iterrows():
        if row["series_id"] in evidence_map:
            out.loc[idx, "adduct_evidence"] = "; ".join(
                evidence_map[row["series_id"]]
            )

    return out


def linked_series_ids(selected_series, links: pd.DataFrame):
    linked = set()

    if not selected_series or links.empty:
        return linked

    for _, row in links.iterrows():
        if row["series_a"] == selected_series:
            linked.add(row["series_b"])
        elif row["series_b"] == selected_series:
            linked.add(row["series_a"])

    return linked


def linked_relationship_table(selected_series, links: pd.DataFrame):
    if not selected_series or links.empty:
        return pd.DataFrame()

    rows = []

    for _, row in links.iterrows():
        if row["series_a"] == selected_series:
            rows.append({
                "Selected adduct": row["candidate_adduct_a"],
                "Linked series": row["series_b"],
                "Linked adduct": row["candidate_adduct_b"],
                "Matched members": row["matched_members"],
                "Median |ppm|": row["median_abs_ppm"],
                "Median RT difference": row["median_rt_difference"],
            })
        elif row["series_b"] == selected_series:
            rows.append({
                "Selected adduct": row["candidate_adduct_b"],
                "Linked series": row["series_a"],
                "Linked adduct": row["candidate_adduct_a"],
                "Matched members": row["matched_members"],
                "Median |ppm|": row["median_abs_ppm"],
                "Median RT difference": row["median_rt_difference"],
            })

    return pd.DataFrame(rows)


def selected_series_from_plot_event(event, trace_series_map):
    if event is None:
        return None

    try:
        points = event.selection.points
    except Exception:
        try:
            points = event["selection"]["points"]
        except Exception:
            points = []

    if not points:
        return None

    point = points[0]
    curve_number = point.get("curve_number")

    if curve_number is None:
        curve_number = point.get("curveNumber")

    if curve_number is None:
        return None

    if 0 <= int(curve_number) < len(trace_series_map):
        return trace_series_map[int(curve_number)]

    return None


def make_all_series_figure(
    results,
    summary,
    selected_series,
    adduct_links,
    adduct_member_matches,
):
    """
    Build one interactive plot containing every detected homologous series.

    One connected trace = one homologous series = one candidate adduct form.
    Adduct-linked companion series share a family color. When a series is
    selected, its linked adduct forms are emphasized and member-level adduct
    connectors are drawn.
    """
    use_rt = results["rt"].notna().all()

    family_groups = summary.set_index("series_id")["family_group"].to_dict()
    adduct_labels = summary.set_index("series_id")["candidate_adduct"].to_dict()

    families = list(dict.fromkeys(summary["family_group"].tolist()))
    palette = (
        px.colors.qualitative.Safe
        + px.colors.qualitative.Bold
        + px.colors.qualitative.Set2
        + px.colors.qualitative.Dark24
    )
    family_colors = {
        family: palette[i % len(palette)]
        for i, family in enumerate(families)
    }

    dash_map = {
        "[M+H]+": "solid",
        "[M+Na]+": "dash",
        "[M+K]+": "dot",
        "[M+NH4]+": "dashdot",
        "[M-H]-": "longdash",
    }

    linked_ids = linked_series_ids(selected_series, adduct_links)

    fig = go.Figure()
    trace_series_map = []

    # Series traces come first so curve_number maps cleanly to series_id.
    for series_id in summary["series_id"]:
        sdf = (
            results[results["series_id"] == series_id]
            .sort_values("repeat_index")
            .copy()
        )

        family = family_groups.get(series_id, series_id)
        adduct = adduct_labels.get(series_id, "Unassigned")
        color = family_colors[family]

        is_selected = series_id == selected_series
        is_linked = series_id in linked_ids

        if is_selected:
            width = 5
            marker_size = 10
            opacity = 1.0
        elif is_linked:
            width = 3.5
            marker_size = 8
            opacity = 0.95
        else:
            width = 1.8
            marker_size = 6
            opacity = 0.40 if selected_series else 0.82

        dash = dash_map.get(adduct, "solid")

        if use_rt:
            x = sdf["rt"]
            x_title = "Retention time"
            x_hover = sdf["rt"]
        else:
            x = sdf["repeat_index"]
            x_title = "Repeat index"
            x_hover = sdf["repeat_index"]

        customdata = np.column_stack([
            [series_id] * len(sdf),
            [family] * len(sdf),
            [adduct] * len(sdf),
            sdf["repeat_unit"].astype(str),
            sdf["repeat_formula"].astype(str),
            sdf["feature_id"].astype(int),
            sdf["repeat_index"].astype(int),
            x_hover,
        ])

        fig.add_trace(
            go.Scatter(
                x=x,
                y=sdf["mz"],
                mode="lines+markers",
                name=f"{family} · {adduct}",
                customdata=customdata,
                line=dict(
                    color=color,
                    width=width,
                    dash=dash,
                ),
                marker=dict(
                    color=color,
                    size=marker_size,
                    line=dict(width=1, color="#FFFFFF"),
                ),
                opacity=opacity,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Family: %{customdata[1]}<br>"
                    "Adduct: %{customdata[2]}<br>"
                    "Repeat unit: %{customdata[3]} · %{customdata[4]}<br>"
                    "Feature: %{customdata[5]}<br>"
                    "Repeat index: %{customdata[6]}<br>"
                    + (
                        "RT: %{x:.3f}<br>"
                        if use_rt
                        else "Repeat index: %{x}<br>"
                    )
                    + "m/z: %{y:.5f}<extra></extra>"
                ),
                showlegend=False,
                selected=dict(
                    marker=dict(size=max(marker_size + 2, 10)),
                ),
                unselected=dict(
                    marker=dict(opacity=opacity),
                ),
            )
        )
        trace_series_map.append(series_id)

    # Draw dotted connectors between the selected series and its supported
    # companion adduct series. These are visual links, not merged homologs.
    if (
        selected_series
        and not adduct_member_matches.empty
        and use_rt
    ):
        connector_rows = adduct_member_matches[
            (adduct_member_matches["series_a"] == selected_series)
            | (adduct_member_matches["series_b"] == selected_series)
        ]

        for _, row in connector_rows.iterrows():
            family = family_groups.get(selected_series, selected_series)
            color = family_colors[family]

            fig.add_trace(
                go.Scatter(
                    x=[row["rt_a"], row["rt_b"]],
                    y=[row["mz_a"], row["mz_b"]],
                    mode="lines",
                    line=dict(
                        color=color,
                        width=1.1,
                        dash="dot",
                    ),
                    opacity=0.30,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    fig.update_layout(
        template="plotly_white",
        height=650,
        margin=dict(l=25, r=20, t=30, b=25),
        xaxis_title=x_title,
        yaxis_title="m/z",
        hovermode="closest",
        clickmode="event+select",
        dragmode="select",
        selectionrevision="series-selection",
        uirevision="series-overview",
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#EEF1F5",
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#EEF1F5",
        zeroline=False,
    )

    return fig, trace_series_map, use_rt

def metric_card(label, value, subtext=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-subtext">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------

with st.sidebar:
    st.markdown("### Analysis setup")
    st.caption("Configure the screen, then run or rerun without reloading your data.")

    with st.expander("1. Repeat units", expanded=True):
        st.caption("Select every repeat unit to search in the same run.")

        selected_builtin = []

        for short_name, info in REPEAT_LIBRARY.items():
            checked = st.checkbox(
                f'{short_name}  ·  {info["formula"]}',
                value=short_name in ["PEG / EO", "PPG / PO"],
                key=f"repeat_{short_name}",
                help=info["full_name"],
            )
            if checked:
                selected_builtin.append(short_name)

        st.markdown("---")

        use_custom = st.checkbox("Add custom repeat unit", value=False)

        custom_name = st.text_input(
            "Custom name",
            value="Custom",
            disabled=not use_custom,
        )

        custom_formula = st.text_input(
            "Custom formula",
            value="C2H4O",
            disabled=not use_custom,
        )

        custom_error = None
        custom_mass = None

        if use_custom:
            try:
                custom_mass = formula_mass(custom_formula)
                st.caption(f"Exact mass: {custom_mass:.6f} Da")
            except Exception as exc:
                custom_error = str(exc)
                st.error(custom_error)

    with st.expander("2. Adducts", expanded=True):
        st.caption("All common adducts are included by default.")

        selected_adducts = []

        for adduct_name, info in ADDUCT_LIBRARY.items():
            checked = st.checkbox(
                f'{adduct_name}  ·  {info["description"]}',
                value=True,
                key=f"adduct_{adduct_name}",
            )
            if checked:
                selected_adducts.append(adduct_name)

    with st.expander("3. Search settings", expanded=False):
        charge = st.selectbox(
            "Homologous series charge state",
            [1, 2, 3, 4],
            index=0,
        )

        ppm_tolerance = st.number_input(
            "Repeat spacing tolerance (ppm)",
            min_value=0.1,
            max_value=100.0,
            value=5.0,
            step=0.5,
        )

        min_members = st.number_input(
            "Minimum members in a series",
            min_value=2,
            max_value=50,
            value=4,
            step=1,
        )

        max_repeat_gap = st.number_input(
            "Maximum skipped repeat units",
            min_value=1,
            max_value=50,
            value=8,
            step=1,
        )

        use_rt = st.checkbox(
            "Apply RT proximity filter to series search",
            value=False,
        )

        rt_tolerance_value = st.number_input(
            "Max series RT difference",
            min_value=0.0,
            value=3.0,
            step=0.1,
            disabled=not use_rt,
        )

        rt_tolerance = rt_tolerance_value if use_rt else None

        st.markdown("##### Adduct matching")

        adduct_ppm_tolerance = st.number_input(
            "Adduct matching tolerance (ppm)",
            min_value=0.1,
            max_value=100.0,
            value=10.0,
            step=0.5,
        )

        adduct_rt_tolerance = st.number_input(
            "Adduct RT matching tolerance",
            min_value=0.0,
            value=0.20,
            step=0.05,
        )

    with st.expander("Reference libraries", expanded=False):
        st.markdown("**Repeat units**")
        for short_name, info in REPEAT_LIBRARY.items():
            st.caption(
                f'{short_name}: {info["formula"]} · '
                f'{formula_mass(info["formula"]):.6f} Da'
            )

        st.markdown("**Adducts**")
        for adduct_name, info in ADDUCT_LIBRARY.items():
            st.caption(
                f'{adduct_name}: {info["description"]} · '
                f'{info["polarity"]}'
            )


# ------------------------------------------------------------
# Data input
# ------------------------------------------------------------

upload_col, info_col = st.columns([1.5, 1])

with upload_col:
    st.markdown("### Data")
    uploaded = st.file_uploader(
        "Upload LCMS feature table",
        type=["csv"],
        key="feature_file",
        help="CSV with at minimum an m/z or mz column. RT and intensity are recommended.",
    )

with info_col:
    st.markdown("### Expected input")
    st.markdown(
        """
        <div class="section-card">
            <div class="small-note">
                <b>Required:</b> m/z<br>
                <b>Recommended:</b> RT, intensity<br><br>
                Common column aliases such as <i>retention_time</i>, <i>area</i>,
                <i>height</i>, and <i>abundance</i> are recognized automatically.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if uploaded is None:
    st.info(
        "Upload a CSV feature table to begin. The included example file contains "
        "multiple polymer families and adduct forms."
    )
    st.stop()

try:
    raw = pd.read_csv(uploaded)
    features = normalize_columns(raw)
except Exception as exc:
    st.error(f"Could not read the file: {exc}")
    st.stop()


# ------------------------------------------------------------
# Search configuration summary
# ------------------------------------------------------------

search_units = []

for short_name in selected_builtin:
    info = REPEAT_LIBRARY[short_name]
    search_units.append({
        "label": short_name,
        "formula": info["formula"],
        "mass": formula_mass(info["formula"]),
        "description": info["full_name"],
    })

if use_custom and custom_error is None and custom_mass is not None:
    custom_label = custom_name.strip() if custom_name.strip() else "Custom"
    search_units.append({
        "label": custom_label,
        "formula": custom_formula.strip(),
        "mass": custom_mass,
        "description": "User defined repeat unit",
    })


st.markdown("### Search summary")

summary_cols = st.columns(4)
with summary_cols[0]:
    metric_card("Features loaded", f"{len(features):,}", "Input feature table")
with summary_cols[1]:
    metric_card(
        "m/z range",
        f"{features.mz.min():.1f}–{features.mz.max():.1f}",
        "Observed features",
    )
with summary_cols[2]:
    metric_card(
        "Repeat units",
        len(search_units),
        "Selected for screening",
    )
with summary_cols[3]:
    metric_card(
        "Adducts",
        len(selected_adducts),
        "Selected for evaluation",
    )

st.write("")

if search_units:
    repeat_badges = " ".join(
        f'<span class="muted-pill">{unit["label"]} · {unit["formula"]}</span>'
        for unit in search_units
    )
else:
    repeat_badges = '<span class="muted-pill">No repeat units selected</span>'

if selected_adducts:
    adduct_badges = " ".join(
        f'<span class="muted-pill">{name}</span>'
        for name in selected_adducts
    )
else:
    adduct_badges = '<span class="muted-pill">No adducts selected</span>'

st.markdown(
    f"""
    <div class="section-card">
        <div class="metric-label">Active repeat units</div>
        <div>{repeat_badges}</div>
        <div style="height:10px;"></div>
        <div class="metric-label">Active adducts</div>
        <div>{adduct_badges}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

button_label = (
    "Rerun analysis"
    if "search_results" in st.session_state
    else "Run analysis"
)

run_search = st.button(
    button_label,
    type="primary",
    use_container_width=True,
    disabled=(
        len(search_units) == 0
        or len(selected_adducts) == 0
        or custom_error is not None
    ),
)

if run_search:
    with st.spinner(
        f"Screening {len(search_units)} repeat unit"
        f'{"s" if len(search_units) != 1 else ""} and evaluating '
        f'{len(selected_adducts)} adduct{"s" if len(selected_adducts) != 1 else ""}...'
    ):
        results = run_multi_repeat_search(
            features=features,
            search_units=search_units,
            charge=charge,
            ppm_tolerance=ppm_tolerance,
            min_members=int(min_members),
            max_repeat_gap=int(max_repeat_gap),
            rt_tolerance=rt_tolerance,
        )

        adduct_links, adduct_member_matches = find_adduct_series_links(
            results=results,
            selected_adducts=selected_adducts,
            ppm_tolerance=adduct_ppm_tolerance,
            rt_tolerance=adduct_rt_tolerance,
            min_matches=max(2, min(3, int(min_members))),
        )

    st.session_state["search_results"] = results
    st.session_state["adduct_links"] = adduct_links
    st.session_state["adduct_member_matches"] = adduct_member_matches
    st.session_state["last_search_units"] = search_units
    st.session_state["last_selected_adducts"] = selected_adducts
    st.session_state["selected_series_id"] = None
    st.session_state["search_run_counter"] = (
        st.session_state.get("search_run_counter", 0) + 1
    )

if "search_results" not in st.session_state:
    st.stop()

results = st.session_state["search_results"]
adduct_links = st.session_state.get("adduct_links", pd.DataFrame())
adduct_member_matches = st.session_state.get(
    "adduct_member_matches",
    pd.DataFrame(),
)
last_search_units = st.session_state.get("last_search_units", [])
last_selected_adducts = st.session_state.get("last_selected_adducts", [])
run_number = st.session_state.get("search_run_counter", 1)

if results.empty:
    st.warning("No candidate homologous series met the current criteria.")
    st.stop()

summary = build_summary(results)
summary = annotate_summary_with_adducts(summary, adduct_links)
series_ids = summary["series_id"].tolist()

series_adduct_map = build_series_adduct_map(adduct_links)
family_map = build_family_groups(series_ids, adduct_links)

summary["candidate_adduct"] = summary["series_id"].map(
    lambda sid: series_adduct_map.get(sid, "Unassigned")
)
summary["family_group"] = summary["series_id"].map(family_map)

if (
    st.session_state.get("selected_series_id") not in series_ids
):
    st.session_state["selected_series_id"] = series_ids[0]

st.markdown(
    f'<span class="status-pill">Analysis complete · Run #{run_number}</span>',
    unsafe_allow_html=True,
)
st.write("")


# ------------------------------------------------------------
# Main result tabs
# ------------------------------------------------------------

tab_overview, tab_series, tab_adducts, tab_export = st.tabs(
    ["Overview", "Interactive Series Map", "Adduct Relationships", "Results & Export"]
)

with tab_overview:
    st.markdown("### Analysis overview")

    result_cols = st.columns(4)
    with result_cols[0]:
        metric_card(
            "Repeat units searched",
            len(last_search_units),
            "Last completed run",
        )
    with result_cols[1]:
        metric_card(
            "Candidate series",
            len(series_ids),
            "Across all repeat units",
        )
    with result_cols[2]:
        metric_card(
            "Assigned feature hits",
            f"{len(results):,}",
            "Assignments may overlap",
        )
    with result_cols[3]:
        metric_card(
            "Adduct relationships",
            0 if adduct_links.empty else len(adduct_links),
            "Evidence supported links",
        )

    st.write("")
    st.markdown("#### Detection by repeat unit")

    repeat_summary = (
        summary.groupby(["repeat_unit", "repeat_formula"], as_index=False)
        .agg(
            candidate_series=("series_id", "nunique"),
            assigned_feature_hits=("members", "sum"),
        )
    )

    st.dataframe(
        repeat_summary,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### All candidate series")
    overview_cols = [
        "family_group",
        "repeat_unit",
        "repeat_formula",
        "series_id",
        "candidate_adduct",
        "members",
        "mz_min",
        "mz_max",
        "rt_min",
        "rt_max",
        "max_abs_ppm",
        "adduct_evidence",
    ]
    st.dataframe(
        summary[overview_cols],
        use_container_width=True,
        hide_index=True,
    )


with tab_series:
    st.markdown("### Interactive series map")
    st.caption(
        "Every connected line is one candidate homologous series and one adduct form. "
        "Companion adducts remain separate lines, but are grouped as the same polymer family."
    )

    selected_series = st.session_state["selected_series_id"]
    selected_summary = summary[
        summary["series_id"] == selected_series
    ].iloc[0]

    left_panel, right_panel = st.columns([2.35, 1], gap="large")

    with left_panel:
        st.markdown("#### All detected homologous series")

        st.markdown(
            """
            <div class="section-card">
                <div class="small-note">
                    Click a marker on any line to select that series. The complete line
                    represents the software's current homologous-series assignment.
                    Lines with the same family color are supported companion adduct forms;
                    dotted cross-links appear for matched adduct members of the selected family.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        fig_all, trace_series_map, using_rt = make_all_series_figure(
            results=results,
            summary=summary,
            selected_series=selected_series,
            adduct_links=adduct_links,
            adduct_member_matches=adduct_member_matches,
        )

        plot_event = st.plotly_chart(
            fig_all,
            use_container_width=True,
            key=f"all_series_map_{run_number}",
            on_select="rerun",
            selection_mode="points",
            config={
                "displaylogo": False,
                "scrollZoom": True,
                "modeBarButtonsToRemove": [
                    "lasso2d",
                ],
            },
        )

        clicked_series = selected_series_from_plot_event(
            plot_event,
            trace_series_map,
        )

        if (
            clicked_series
            and clicked_series != st.session_state["selected_series_id"]
        ):
            st.session_state["selected_series_id"] = clicked_series
            st.rerun()

        if using_rt:
            st.caption(
                "X axis = retention time. The connected points are ordered by repeat index. "
                "Adduct companions should follow the same chromatographic trend at shifted m/z."
            )
        else:
            st.caption(
                "Retention time was incomplete, so the overview uses repeat index on the X axis."
            )

    with right_panel:
        selected_series = st.session_state["selected_series_id"]

        # Direct-selection fallback is useful for dense plots.
        direct_series = st.selectbox(
            "Selected series",
            series_ids,
            index=series_ids.index(selected_series),
            key="direct_series_picker",
            help="You can also select a series directly if the plot is crowded.",
        )

        if direct_series != selected_series:
            st.session_state["selected_series_id"] = direct_series
            st.rerun()

        selected_series = st.session_state["selected_series_id"]
        selected = (
            results[results["series_id"] == selected_series]
            .sort_values("repeat_index")
            .copy()
        )
        selected_summary = summary[
            summary["series_id"] == selected_series
        ].iloc[0]

        family_id = selected_summary["family_group"]
        candidate_adduct = selected_summary["candidate_adduct"]

        st.markdown(
            f"""
            <div class="section-card">
                <div class="eyebrow">Selected homologous series</div>
                <div style="font-size:1.2rem;font-weight:750;color:#10243E;">
                    {selected_series}
                </div>
                <div style="height:10px;"></div>
                <span class="status-pill">{family_id}</span>
                <span class="muted-pill">{candidate_adduct}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        detail_cols = st.columns(2)
        with detail_cols[0]:
            metric_card(
                "Repeat unit",
                selected_summary["repeat_unit"],
                selected_summary["repeat_formula"],
            )
        with detail_cols[1]:
            metric_card(
                "Members",
                int(selected_summary["members"]),
                f'Max |ppm| {selected_summary["max_abs_ppm"]:.2f}',
            )

        st.write("")

        st.markdown("#### Linked adduct forms")

        selected_links = linked_relationship_table(
            selected_series,
            adduct_links,
        )

        if selected_links.empty:
            st.caption(
                "No supported companion adduct series were detected for this line."
            )
        else:
            st.dataframe(
                selected_links,
                use_container_width=True,
                hide_index=True,
            )

        family_series_ids = summary[
            summary["family_group"] == family_id
        ]["series_id"].tolist()

        family_results = results[
            results["series_id"].isin(family_series_ids)
        ].copy()

        st.markdown("#### Series members")
        member_cols = [
            "feature_id",
            "mz",
            "rt",
            "repeat_index",
            "intensity",
            "series_ppm_error",
        ]
        st.dataframe(
            selected[member_cols],
            use_container_width=True,
            hide_index=True,
            height=280,
        )

        st.markdown("#### Download")

        st.download_button(
            "Download selected series",
            data=selected.to_csv(index=False).encode("utf-8"),
            file_name=(
                selected_series.replace(" ", "_")
                .replace("/", "-")
                .replace("|", "")
                + ".csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

        st.download_button(
            "Download linked adduct family",
            data=family_results.to_csv(index=False).encode("utf-8"),
            file_name=(
                family_id.replace(" ", "_").lower()
                + "_linked_adducts.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

        st.download_button(
            "Download all series",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="polymer_series_all_assignments.csv",
            mime="text/csv",
            use_container_width=True,
        )


with tab_adducts:
    st.markdown("### Adduct relationships")

    st.markdown(
        """
        <div class="section-card">
            <div class="small-note">
                Companion series are linked only when multiple homologous members show
                the expected exact adduct mass offset and similar retention times.
                This keeps adduct assignments evidence based instead of forcing a label
                onto every detected series.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not adduct_links.empty:
        st.dataframe(
            adduct_links,
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("Matched feature pairs used as adduct evidence"):
            st.dataframe(
                adduct_member_matches,
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info(
            "No companion adduct series met the current evidence criteria. "
            "A homologous series may still be valid when only one ion form is observed."
        )

    st.markdown("#### Adducts evaluated in last run")
    adduct_table = pd.DataFrame([
        {
            "Adduct": name,
            "Description": ADDUCT_LIBRARY[name]["description"],
            "Polarity": ADDUCT_LIBRARY[name]["polarity"],
            "m/z shift (z=1)": ADDUCT_LIBRARY[name]["shift"],
        }
        for name in last_selected_adducts
    ])

    st.dataframe(
        adduct_table,
        use_container_width=True,
        hide_index=True,
    )


with tab_export:
    st.markdown("### Results & export")

    st.markdown(
        """
        <div class="section-card">
            <div class="small-note">
                Export the complete feature assignments for downstream review.
                Adduct relationships are exported separately so the evidence layer
                remains distinct from the core homologous series assignments.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Complete candidate series table")
    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
    )

    current_selected_series = st.session_state.get(
        "selected_series_id",
        series_ids[0],
    )
    export_selected = results[
        results["series_id"] == current_selected_series
    ].copy()

    export_family_id = summary.loc[
        summary["series_id"] == current_selected_series,
        "family_group",
    ].iloc[0]

    export_family_series = summary[
        summary["family_group"] == export_family_id
    ]["series_id"].tolist()

    export_family = results[
        results["series_id"].isin(export_family_series)
    ].copy()

    export_row1 = st.columns(2)

    with export_row1[0]:
        st.download_button(
            "Download all assignments",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="polymer_series_all_assignments.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with export_row1[1]:
        st.download_button(
            "Download series summary",
            data=summary.to_csv(index=False).encode("utf-8"),
            file_name="polymer_series_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )

    export_row2 = st.columns(2)

    with export_row2[0]:
        st.download_button(
            "Download currently selected series",
            data=export_selected.to_csv(index=False).encode("utf-8"),
            file_name="polymer_series_selected.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with export_row2[1]:
        st.download_button(
            "Download selected linked adduct family",
            data=export_family.to_csv(index=False).encode("utf-8"),
            file_name="polymer_series_selected_family.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if not adduct_links.empty:
        st.download_button(
            "Download adduct relationships",
            data=adduct_links.to_csv(index=False).encode("utf-8"),
            file_name="polymer_series_adduct_relationships.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.caption(
        "Change any repeat unit, adduct, or search setting in the sidebar and "
        "click Rerun analysis. The uploaded feature table stays loaded."
    )
