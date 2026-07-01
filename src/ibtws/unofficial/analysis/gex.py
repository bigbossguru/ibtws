"""
GEX Calculator
==============
A class-based GEX (Gamma Exposure) calculator for options chains.

Computes:
    - Per-strike net GEX (spot gamma histogram)
    - Full GEX profile via Black-Scholes repricing sweep
    - Zero Gamma Level (ZGL) via Brent's root-finding
    - Call Wall / Put Wall
    - GEX regime (positive/negative)

Usage:
    from gex_calculator import GexCalculator

    calc = GexCalculator("spx_chain_20260629_193146.csv")
    calc.compute()
    calc.summary()
    calc.plot(save_path="gex_chart.png")

    # Access results programmatically
    print(calc.zero_gamma_level)
    print(calc.regime)
    print(calc.total_gex)
"""

import io
import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq


@dataclass
class ZeroCrossing:
    """A point where GEX crosses zero."""

    level: float
    direction: str  # "neg→pos" or "pos→neg"


@dataclass
class GexResult:
    """Complete result of a GEX calculation."""

    spot: float
    zero_gamma_level: float | None
    regime: str | None  # "POSITIVE" or "NEGATIVE"
    pts_from_zgl: float | None
    total_gex: float  # net GEX at spot ($)
    call_gex: float  # total call GEX ($)
    put_gex: float  # total put GEX ($)
    call_wall: float  # strike with max positive GEX
    put_wall: float  # strike with max negative GEX
    all_crossings: list[ZeroCrossing] = field(default_factory=list)
    sweep_levels: np.ndarray = field(default_factory=lambda: np.array([]))
    sweep_gex: np.ndarray = field(default_factory=lambda: np.array([]))
    net_gex_by_strike: pd.DataFrame = field(default_factory=pd.DataFrame)


class GexCalculator:
    """
    Gamma Exposure calculator using Black-Scholes repricing.

    Parameters
    ----------
    df : pd.DataFrame
        Options chain data.
    risk_free_rate : float
        Annualised risk-free rate (default: 0.045).
    sweep_points : int
        Resolution of the GEX profile sweep (default: 500).
    bar_width : float
        Bar width for histogram chart (default: 4.0).
    """

    REQUIRED_COLUMNS = {"strike", "right", "gamma", "open_interest", "underlying_price", "iv", "expiry", "timestamp"}

    def __init__(
        self,
        risk_free_rate: float = 0.045,
        sweep_points: int = 500,
        bar_width: float = 4.0,
    ):
        self.risk_free_rate = risk_free_rate
        self.sweep_points = sweep_points
        self.bar_width = bar_width

        # Results (populated after compute())
        self.result: GexResult | None = None

    def compute(self, df: pd.DataFrame) -> GexResult:
        """Run full GEX computation: static GEX, profile sweep, ZGL."""
        # Load and validate data
        self._df = df
        self.spot, self._T, self._r = self._derive_params()

        net = self._calc_static_gex()
        df_gex = self._compute_gex_column()
        zgl_info = self._find_zero_gamma()

        # Extract key metrics
        call_gex = df_gex[df_gex["right"] == "C"]["gex"].sum()
        put_gex = df_gex[df_gex["right"] == "P"]["gex"].sum()
        total_gex = call_gex + put_gex

        call_wall_row = net.loc[net["net_gex"].idxmax()]
        put_wall_row = net.loc[net["net_gex"].idxmin()]

        crossings = [ZeroCrossing(level=c["level"], direction=c["direction"]) for c in zgl_info["all_crossings"]]

        primary = zgl_info["primary"]
        zgl = primary["level"] if primary else None

        self.result = GexResult(
            spot=self.spot,
            zero_gamma_level=zgl,
            regime=zgl_info["regime"],
            pts_from_zgl=zgl_info["pts_from_zgl"],
            total_gex=total_gex,
            call_gex=call_gex,
            put_gex=put_gex,
            call_wall=float(call_wall_row["strike"]),
            put_wall=float(put_wall_row["strike"]),
            all_crossings=crossings,
            sweep_levels=zgl_info["sweep"],
            sweep_gex=zgl_info["gex_curve"],
            net_gex_by_strike=net,
        )
        return self.result

    @property
    def zero_gamma_level(self) -> float | None:
        """Zero Gamma Level (compute first)."""
        return self.result.zero_gamma_level if self.result else None

    @property
    def regime(self) -> str | None:
        """Current GEX regime: 'POSITIVE' or 'NEGATIVE'."""
        return self.result.regime if self.result else None

    @property
    def total_gex(self) -> float | None:
        """Total net GEX at current spot ($)."""
        return self.result.total_gex if self.result else None

    def summary(self) -> str:
        """Return formatted summary string. Also prints to stdout."""
        if self.result is None:
            raise RuntimeError("Call compute() first")

        r = self.result
        lines = [
            "=" * 52,
            "  SPX GEX SUMMARY",
            "=" * 52,
            f"  Spot               : {r.spot:,.2f}",
        ]
        if r.zero_gamma_level is not None:
            lines.append(f"  Zero Gamma Level   : {r.zero_gamma_level:,.1f}  (BS repricing)")
            lines.append(f"  Pts from ZGL       : {r.pts_from_zgl:+.1f}")
        lines.extend([
            f"  GEX Regime         : {r.regime}",
            f"  Total Call GEX     : ${r.call_gex / 1e6:.2f}M",
            f"  Total Put GEX      : ${r.put_gex / 1e6:.2f}M",
            f"  Net GEX (snapshot) : ${r.total_gex / 1e6:.2f}M",
            f"  Call Wall          : {int(r.call_wall)}",
            f"  Put Wall           : {int(r.put_wall)}",
            "=" * 52,
        ])

        text = "\n".join(lines)
        print(text)
        return text

    def plot(self, save_path: str | None = None, title_suffix: str = "") -> bytes:
        """
        Generate combined GEX chart: profile curve + per-strike histogram.

        Parameters
        ----------
        save_path : str, optional
            If provided, saves chart to file instead of showing.
        title_suffix : str, optional
            Additional text for chart title.
        """
        if self.result is None:
            raise RuntimeError("Call compute() first")

        r = self.result
        fig = self._render_chart(r, save_path=save_path, title_suffix=title_suffix)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    def _derive_params(self):
        spot = self._df["underlying_price"].iloc[-1]
        # Expiry = 4pm close on the expiry date, not midnight — critical for 0DTE,
        # where midnight vs. close is the difference between T>0 and T<0.
        expiry = datetime.strptime(str(self._df["expiry"].iloc[-1]), "%Y%m%d").replace(hour=22, minute=0, second=0)
        now = datetime.fromtimestamp(self._df["timestamp"].iloc[-1])
        T = (expiry - now).total_seconds() / (365.25 * 24 * 3600)
        if T <= 0:
            raise ValueError(
                f"Non-positive time to expiry ({T * 365.25:.3f} days): snapshot ({now}) is at/after "
                f"the 4pm close on the expiry date ({expiry}). Check the timestamp's timezone matches system local time."
            )
        return spot, T, self.risk_free_rate

    def _compute_gex_column(self) -> pd.DataFrame:
        """Add GEX column to dataframe."""
        df = self._df.copy()
        df["gex"] = df["gamma"] * df["open_interest"] * self.spot**2 * 0.01
        df.loc[df["right"] == "P", "gex"] *= -1
        return df

    def _calc_static_gex(self) -> pd.DataFrame:
        """Per-strike net GEX using pre-computed gammas."""
        df = self._compute_gex_column()
        net = (
            df
            .groupby("strike")["gex"]
            .sum()
            .reset_index()
            .rename(columns={"gex": "net_gex"})
            .sort_values("strike")
            .reset_index(drop=True)
        )
        return net

    @staticmethod
    def _bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Black-Scholes gamma (identical for calls and puts)."""
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    def _total_gex_at(self, S: float) -> float:
        """Reprice all options at hypothetical spot S, return total net GEX."""
        total = 0.0
        for _, row in self._df.iterrows():
            g = self._bs_gamma(S, row["strike"], self._T, self._r, row["iv"])
            gex = g * row["open_interest"] * S**2 * 0.01
            total += gex if row["right"] == "C" else -gex
        return total

    def _find_zero_gamma(self) -> dict:
        """Sweep spot prices and find ZGL via Brent's method."""
        strikes = sorted(self._df["strike"].unique())
        S_min = strikes[0] - 100
        S_max = strikes[-1] + 100
        sweep = np.linspace(S_min, S_max, self.sweep_points)
        gex_curve = np.array([self._total_gex_at(s) for s in sweep])

        crossings = []
        for i in range(1, len(sweep)):
            if gex_curve[i - 1] * gex_curve[i] <= 0:
                try:
                    zgl = brentq(self._total_gex_at, sweep[i - 1], sweep[i], xtol=0.1)
                    direction = "neg→pos" if gex_curve[i - 1] < 0 else "pos→neg"
                    crossings.append({"level": zgl, "direction": direction})
                except Exception:
                    pass

        neg_pos = [c for c in crossings if c["direction"] == "neg→pos"]
        primary = (
            min(neg_pos, key=lambda c: abs(c["level"] - self.spot))
            if neg_pos
            else (crossings[0] if crossings else None)
        )

        regime = None
        pts_from_zgl = None
        if primary:
            pts_from_zgl = self.spot - primary["level"]
            regime = "POSITIVE" if pts_from_zgl > 0 else "NEGATIVE"

        return {
            "all_crossings": crossings,
            "primary": primary,
            "regime": regime,
            "pts_from_zgl": pts_from_zgl,
            "sweep": sweep,
            "gex_curve": gex_curve,
        }

    def _render_chart(self, r: GexResult, save_path: str | None, title_suffix: str) -> plt.Figure:
        """Render combined GEX chart."""
        net = r.net_gex_by_strike
        sweep = r.sweep_levels
        gex_curve = r.sweep_gex
        zgl = r.zero_gamma_level

        fig, ax = plt.subplots(1, 1, figsize=(14, 7))

        # Secondary y-axis: per-strike histogram
        ax2 = ax.twinx()
        bar_colors = ["green" if v >= 0 else "red" for v in net["net_gex"]]
        ax2.bar(
            net["strike"],
            net["net_gex"] / 1e6,
            width=self.bar_width,
            color=bar_colors,
            alpha=0.35,
            zorder=2,
            linewidth=0.2,
            edgecolor="k",
            label="Spot GEX per Strike ($mm)",
        )
        ax2.set_ylabel("Spot GEX per Strike ($ Millions per 1% move)", fontsize=11, color="grey")
        ax2.tick_params(axis="y", labelcolor="grey", labelsize=9)
        ax2.legend(fontsize=10, loc="upper right")

        # Primary y-axis: GEX profile curve
        gex_bn = gex_curve / 1e9
        ax.plot(sweep, gex_bn, color="blue", linewidth=2.5, zorder=5, label="GEX Profile (All Expiries)")
        ax.fill_between(sweep, gex_bn, 0, where=(gex_bn >= 0), color="green", alpha=0.08, zorder=1)
        ax.fill_between(sweep, gex_bn, 0, where=(gex_bn < 0), color="red", alpha=0.08, zorder=1)
        ax.axhline(0, color="grey", linewidth=0.5, zorder=3)

        # Key levels
        ax.axvline(x=r.spot, color="red", linestyle="--", linewidth=1.5, label=f"Spot: {r.spot:.0f}", zorder=6)
        if zgl is not None:
            ax.axvline(x=zgl, color="green", linestyle="-.", linewidth=1.5, label=f"Zero Gamma: {zgl:.0f}", zorder=6)

        # Formatting
        ax.set_xlabel("Index Price", fontsize=12)
        ax.set_ylabel("GEX Profile ($ Billions per 1% move)", fontsize=11, color="blue")
        ax.tick_params(axis="y", labelcolor="blue")

        if zgl is not None:
            gex_at_spot = gex_bn[np.argmin(np.abs(sweep - r.spot))]
            ax.set_title(
                f"SPX GEX Analysis (BS Repricing){title_suffix}\n"
                f"GEX at Spot: ${gex_at_spot:.3f} Bn | Zero Gamma: {zgl:.0f}",
                fontsize=13,
                fontweight="bold",
            )
        else:
            ax.set_title(f"SPX GEX Analysis (BS Repricing){title_suffix}", fontsize=13, fontweight="bold")

        ax.legend(fontsize=10, loc="lower right")
        ax.grid(True, alpha=0.3, zorder=0)

        # X-axis range
        strike_min = net["strike"].min()
        strike_max = net["strike"].max()
        x_pad = (strike_max - strike_min) * 0.05
        ax.set_xlim(strike_min - x_pad, strike_max + x_pad)

        # Align zero on both y-axes
        x_lo, x_hi = ax.get_xlim()
        mask = (sweep >= x_lo) & (sweep <= x_hi)
        visible_gex = gex_bn[mask] if mask.any() else gex_bn
        # Filter out NaN/Inf
        visible_gex = visible_gex[np.isfinite(visible_gex)]
        if len(visible_gex) == 0:
            prof_max, prof_min = 1.0, -1.0
        else:
            prof_max = visible_gex.max()
            prof_min = visible_gex.min()

        bar_vals = net["net_gex"].values / 1e6
        bar_vals_finite = bar_vals[np.isfinite(bar_vals)]
        bar_max = bar_vals_finite.max() if len(bar_vals_finite) > 0 else 1.0

        pad = 1.15
        prof_max = max(prof_max, 0.1) * pad
        prof_min = min(prof_min, -0.1) * pad
        bar_max = max(bar_max, 0.1) * pad

        frac = abs(prof_min) / (abs(prof_min) + prof_max)
        ax.set_ylim(prof_min, prof_max)
        ax2.set_ylim(-frac * bar_max / (1 - frac), bar_max)

        # Regime annotations
        if zgl is not None:
            x_lo_vis = strike_min - x_pad
            x_hi_vis = strike_max + x_pad
            y_pos = prof_min + (prof_max - prof_min) * 0.88
            neg_x = x_lo_vis + (zgl - x_lo_vis) * 0.5
            pos_x = zgl + (x_hi_vis - zgl) * 0.5
            ax.text(
                neg_x,
                y_pos,
                "Negative Gamma\n(Amplifying)",
                fontsize=11,
                color="red",
                ha="center",
                fontweight="bold",
                alpha=0.7,
            )
            ax.text(
                pos_x,
                y_pos,
                "Positive Gamma\n(Stabilizing)",
                fontsize=11,
                color="green",
                ha="center",
                fontweight="bold",
                alpha=0.7,
            )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logging.info(f"Chart saved → {save_path}")
        else:
            plt.show()

        return fig
