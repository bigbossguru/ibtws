"""
SPX GEX Analysis
================
Calculates Net GEX per strike and Zero Gamma Level from an options chain CSV.

Zero Gamma Level method: Black-Scholes repricing sweep — for each candidate
spot price, all options are repriced with their own IV and total dealer GEX
is recomputed. ZGL is the root where total GEX = 0.

Expected CSV columns:
    strike, right (C/P), gamma, open_interest, underlying_price, iv,
    expiry (YYYYMMDD), timestamp (unix)
"""

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import norm
from scipy.optimize import brentq


# ── Config ────────────────────────────────────────────────────────────────────

FIGSIZE = (12, 13)
COLOR_POS = "#2a78d6"
COLOR_NEG = "#e34948"
COLOR_ZGL = "#e34948"
COLOR_SPOT = "#eda100"
COLOR_CUMGEX = "#4a3aa7"
FILL_POS = "#dbeafe"
FILL_NEG = "#fde8e8"
GRID_COLOR = "#e1e0d9"
BAR_WIDTH = 4.0

RISK_FREE_RATE = 0.045  # annualised, adjust as needed
SWEEP_POINTS = 500  # resolution of GEX(spot) sweep


# ── Black-Scholes ─────────────────────────────────────────────────────────────


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Analytical BS gamma (identical for calls and puts)."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


# ── Core calculations ─────────────────────────────────────────────────────────


def load_chain(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"strike", "right", "gamma", "open_interest", "underlying_price", "iv", "expiry", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df


def derive_params(df: pd.DataFrame) -> tuple[float, float, float]:
    """Return (spot, T_years, r)."""
    spot = df["underlying_price"].iloc[0]
    expiry = datetime.strptime(str(df["expiry"].iloc[0]), "%Y%m%d")
    now = datetime.fromtimestamp(df["timestamp"].iloc[0])
    T = max((expiry - now).total_seconds() / (365.25 * 24 * 3600), 1e-6)
    return spot, T, RISK_FREE_RATE


def calc_static_gex(df: pd.DataFrame, spot: float) -> pd.DataFrame:
    """Per-strike net GEX using pre-computed gammas (snapshot at current spot)."""
    df = df.copy()
    df["gex"] = df["gamma"] * df["open_interest"] * spot**2 * 0.01
    df.loc[df["right"] == "P", "gex"] *= -1
    net = (
        df.groupby("strike")["gex"]
        .sum()
        .reset_index()
        .rename(columns={"gex": "net_gex"})
        .sort_values("strike")
        .reset_index(drop=True)
    )
    net["cumgex"] = net["net_gex"].cumsum()
    return net


def total_gex_at(S: float, df: pd.DataFrame, T: float, r: float) -> float:
    """
    Reprice ALL options at hypothetical spot S using each option's own IV,
    then return total dealer net GEX.
    """
    total = 0.0
    for _, row in df.iterrows():
        g = bs_gamma(S, row["strike"], T, r, row["iv"])
        gex = g * row["open_interest"] * S**2 * 0.01
        total += gex if row["right"] == "C" else -gex
    return total


def find_zero_gamma(df: pd.DataFrame, spot: float, T: float, r: float) -> dict:
    """
    Sweep spot prices and find roots of total_gex_at() via Brent's method.
    Primary ZGL = the neg→pos crossing closest to current spot.
    """
    strikes = sorted(df["strike"].unique())
    S_min = strikes[0] - 100
    S_max = strikes[-1] + 100
    sweep = np.linspace(S_min, S_max, SWEEP_POINTS)
    gex_curve = np.array([total_gex_at(s, df, T, r) for s in sweep])

    crossings = []
    for i in range(1, len(sweep)):
        if gex_curve[i - 1] * gex_curve[i] <= 0:
            try:
                zgl = brentq(lambda s: total_gex_at(s, df, T, r), sweep[i - 1], sweep[i], xtol=0.1)
                direction = "neg→pos" if gex_curve[i - 1] < 0 else "pos→neg"
                crossings.append({"level": zgl, "direction": direction})
            except Exception:
                pass

    # Primary = neg→pos crossing nearest to spot
    neg_pos = [c for c in crossings if c["direction"] == "neg→pos"]
    primary = min(neg_pos, key=lambda c: abs(c["level"] - spot)) if neg_pos else (crossings[0] if crossings else None)

    regime = None
    pts_from_zgl = None
    if primary:
        pts_from_zgl = spot - primary["level"]
        regime = "POSITIVE" if pts_from_zgl > 0 else "NEGATIVE"

    return {
        "all_crossings": crossings,
        "primary": primary,
        "regime": regime,
        "pts_from_zgl": pts_from_zgl,
        "sweep": sweep,
        "gex_curve": gex_curve,
    }


def summary_stats(df: pd.DataFrame, net: pd.DataFrame, spot: float, zgl_info: dict) -> None:
    total_call = df[df["right"] == "C"]["gex"].sum()
    total_put = df[df["right"] == "P"]["gex"].sum()
    call_wall = net.loc[net["net_gex"].idxmax()]
    put_wall = net.loc[net["net_gex"].idxmin()]
    primary = zgl_info.get("primary")
    zgl = primary["level"] if primary else None

    print("=" * 52)
    print("  SPX GEX SUMMARY")
    print("=" * 52)
    print(f"  Spot               : {spot:,.2f}")
    if zgl:
        print(f"  Zero Gamma Level   : {zgl:,.1f}  (BS repricing)")
        print(f"  Pts from ZGL       : {zgl_info['pts_from_zgl']:+.1f}")
    print(f"  GEX Regime         : {zgl_info['regime']}")
    print(f"  Total Call GEX     : ${total_call / 1e6:.2f}M")
    print(f"  Total Put GEX      : ${total_put / 1e6:.2f}M")
    print(f"  Net GEX (snapshot) : ${(total_call + total_put) / 1e6:.2f}M")
    print(f"  Call Wall          : {int(call_wall['strike'])}  (${call_wall['net_gex'] / 1e6:.2f}M)")
    print(f"  Put Wall           : {int(put_wall['strike'])}  (${put_wall['net_gex'] / 1e6:.2f}M)")
    print("=" * 52)


# ── Charting ──────────────────────────────────────────────────────────────────


def _fmt_gex(val, _=None) -> str:
    abs_v, sign = abs(val), "-" if val < 0 else ""
    if abs_v >= 1e6:
        return f"{sign}${abs_v / 1e6:.0f}M"
    if abs_v >= 1e3:
        return f"{sign}${abs_v / 1e3:.0f}K"
    return f"{sign}${abs_v:.0f}"


def _ref_vline(ax, x, color, label, y_anchor=0.98, ha="left"):
    ax.axvline(x, color=color, linewidth=1.8, linestyle=(0, (5, 4)), zorder=5)
    ax.text(
        x,
        y_anchor,
        f" {label}" if ha == "left" else f"{label} ",
        color=color,
        fontsize=9,
        fontweight="bold",
        ha=ha,
        va="top",
        transform=ax.get_xaxis_transform(),
    )


def plot_net_gex_histogram(ax, net: pd.DataFrame, spot: float, zgl: float | None):
    colors = [COLOR_POS if v >= 0 else COLOR_NEG for v in net["net_gex"]]
    ax.bar(net["strike"], net["net_gex"], width=BAR_WIDTH * 0.8, color=colors, zorder=3, linewidth=0)
    ax.axhline(0, color=GRID_COLOR, linewidth=0.8, zorder=2)

    _ref_vline(ax, spot, COLOR_SPOT, f"Spot {spot:,.0f}", ha="left")
    if zgl is not None:
        _ref_vline(ax, zgl, COLOR_ZGL, f"ZGL {zgl:,.0f}", ha="right", y_anchor=0.94)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_gex))
    ax.set_xlabel("Strike", fontsize=10)
    ax.set_ylabel("Net GEX", fontsize=10)
    ax.set_title("Net GEX by Strike", fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.5, zorder=0)

    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color=COLOR_POS, label="Positive GEX (call-heavy)"),
            Patch(color=COLOR_NEG, label="Negative GEX (put-heavy)"),
        ],
        fontsize=8,
        loc="upper left",
        framealpha=0.7,
    )


def plot_gex_curve(ax, zgl_info: dict, spot: float):
    """
    Plots the continuous GEX(spot) curve produced by the BS repricing sweep.
    This is what the ZGL root-finding actually operates on.
    """
    sweep = zgl_info["sweep"]
    gex_curve = zgl_info["gex_curve"]
    primary = zgl_info.get("primary")
    zgl = primary["level"] if primary else None
    regime = zgl_info.get("regime", "")

    ax.plot(sweep, gex_curve, color=COLOR_CUMGEX, linewidth=2, zorder=4)
    ax.fill_between(sweep, gex_curve, 0, where=(gex_curve >= 0), color=FILL_POS, alpha=0.5, zorder=2)
    ax.fill_between(sweep, gex_curve, 0, where=(gex_curve < 0), color=FILL_NEG, alpha=0.5, zorder=2)
    ax.axhline(0, color=GRID_COLOR, linewidth=0.8, zorder=3)

    _ref_vline(ax, spot, COLOR_SPOT, f"Spot {spot:,.0f}", ha="left")
    if zgl is not None:
        _ref_vline(ax, zgl, COLOR_ZGL, f"ZGL {zgl:,.0f}", ha="right", y_anchor=0.94)
        # Mark the root precisely
        ax.plot(zgl, 0, "o", color=COLOR_ZGL, markersize=7, zorder=6)

    regime_color = COLOR_POS if regime == "POSITIVE" else COLOR_NEG
    pts_str = (
        f"  —  Regime: {regime} ({zgl_info['pts_from_zgl']:+.0f} pts from ZGL)"
        if zgl_info["pts_from_zgl"] is not None
        else ""
    )
    ax.set_title(
        f"Total GEX vs Hypothetical Spot (BS repricing){pts_str}",
        fontsize=11,
        fontweight="bold",
        pad=10,
        color=regime_color,
    )

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_gex))
    ax.set_xlabel("Hypothetical SPX spot", fontsize=10)
    ax.set_ylabel("Total dealer net GEX", fontsize=10)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.5, zorder=0)


def plot(
    df: pd.DataFrame,
    net: pd.DataFrame,
    spot: float,
    zgl_info: dict,
    title_suffix: str = "",
    save_path: str | None = None,
):
    zgl = zgl_info["primary"]["level"] if zgl_info["primary"] else None
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGSIZE)
    fig.suptitle(f"SPX GEX Analysis{title_suffix}", fontsize=14, fontweight="bold", y=1.01)

    plot_net_gex_histogram(ax1, net, spot, zgl)
    plot_gex_curve(ax2, zgl_info, spot)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Chart saved → {save_path}")
    else:
        plt.show()

    return fig


# ── Main ──────────────────────────────────────────────────────────────────────


def run(csv_path: str, save_path: str | None = None):
    df_raw = load_chain(csv_path)
    spot, T, r = derive_params(df_raw)

    # Static snapshot GEX (for histogram)
    df_gex = df_raw.copy()
    df_gex["gex"] = df_gex["gamma"] * df_gex["open_interest"] * spot**2 * 0.01
    df_gex.loc[df_gex["right"] == "P", "gex"] *= -1
    net = calc_static_gex(df_raw, spot)

    # Proper ZGL via BS repricing sweep
    print("Computing ZGL via BS repricing sweep…")
    zgl_info = find_zero_gamma(df_raw, spot, T, r)

    stem = Path(csv_path).stem
    parts = stem.split("_")
    date_label = f"  [{parts[2]}]" if len(parts) >= 3 else ""

    summary_stats(df_gex, net, spot, zgl_info)
    plot(df_gex, net, spot, zgl_info, title_suffix=date_label, save_path=save_path)


if __name__ == "__main__":
    run("output/spx_chain_20260629_193146.csv", save_path="output/spx_gex_20260629_193146.png")
