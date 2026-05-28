"""07 — Pre-market Volatility Regime.

Assess the current volatility environment for 0DTE SPX premium-selling strategies.
Scores 6 independent components (0–100) and classifies into GREEN / YELLOW / RED.

Run before market open (~08:45 EST) to decide whether to trade today.

Components (by weight):
  C1: VIX absolute level (20 pts)
  C2: VIX/VIX3M term structure (25 pts) — THE leading indicator
  C3: VIX1D/VIX ratio (20 pts) — is today priced hot?
  C4: VVIX level (20 pts) — vol-of-vol early warning
  C5: VVIX divergence (10 pts) — hidden institutional hedging
  C6: IV/RV spread (5 pts) — variance risk premium (hard override if negative)
"""

from __future__ import annotations

import asyncio
import logging

from ibtws.config import IBKRConfig
from ibtws.unofficial.client import IBKRClient
from ibtws.unofficial.analysis.volatility import VolRegimeDetector, VolRegimeResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")


def _print(r: VolRegimeResult) -> None:
    emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}[r.regime]
    print("\n=== PREMARKET VOL REGIME ===\n")
    print("Inputs:")
    print(f"  VIX          : {r.vix:.2f}  [LIVE]")
    print(f"  VIX prev     : {r.vix_prev_close:.2f}")
    print(f"  VIX1D        : {r.vix1d:.2f}")
    print(f"  VIX3M        : {r.vix3m:.2f}")
    print(f"  VVIX         : {r.vvix:.2f}  (prev: {r.vvix_prev_close:.2f})")
    print(f"  VX front     : {r.vx_front:.2f}  [LIVE]")
    print(f"  RV 20d       : {r.rv_20d:.2f}%")
    print(f"  VRP (IV-RV)  : {r.vix - r.rv_20d:+.2f} pts")
    print()
    print("Components (higher = more dangerous):")
    print(f"  [C1] VIX absolute      : {r.vix_absolute.value:<8.2f}  → {r.vix_absolute.score:2d}/20")
    print(
        f"  [C2] Term structure     : {r.term_structure.value:<8.3f}  → {r.term_structure.score:2d}/25  {'⚠️  BACKWARDATION' if r.term_structure.value >= 1.0 else ''}"
    )
    print(
        f"  [C3] VIX1D/VIX ratio    : {r.vix1d_vix_ratio.value:<8.3f}  → {r.vix1d_vix_ratio.score:2d}/20  {'⚠️  HOT DAY' if r.vix1d_vix_ratio.value >= 1.0 else ''}"
    )
    print(
        f"  [C4] VVIX level         : {r.vvix_level.value:<8.1f}  → {r.vvix_level.score:2d}/20  {'⚠️  ELEVATED' if r.vvix_level.value >= 120 else ''}"
    )
    print(
        f"  [C5] VVIX divergence    : {r.vvix_divergence.value:+.1f}%{'':<4}  → {r.vvix_divergence.score:2d}/10  {'⚠️  DIVERGING' if r.vvix_divergence.score >= 6 else ''}"
    )
    print(
        f"  [C6] IV/RV spread (VRP) : {r.iv_rv_spread.value:<8.2f}  → {r.iv_rv_spread.score:2d}/5   {'⚠️  NO EDGE' if r.iv_rv_spread.value < 0 else ''}"
    )
    print()
    print(f"{'─' * 50}")
    print(f"  TOTAL SCORE : {r.score} / 100")
    print(f"  REGIME      : {emoji} {r.regime}")
    if r.vrp_override:
        print("  VRP OVERRIDE: ⚠️  Negative VRP — no statistical edge")
    print(f"  ACTION      : {r.action}")
    print(f"{'─' * 50}")
    print(f"\nNote: {r.data_note}")


async def main() -> None:
    config = IBKRConfig(host="192.168.0.129", port=7497, client_id=14)

    async with IBKRClient(config) as client:
        await client.connect()

        detector = VolRegimeDetector(client)
        result = await detector.detect()
        _print(result)


if __name__ == "__main__":
    asyncio.run(main())
