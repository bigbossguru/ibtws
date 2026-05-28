"""07 — Pre-market Volatility Regime.

Assess the current volatility environment for 0DTE SPX Iron Condor strategies.
Scores 7 components (0–100) and classifies into GREEN / YELLOW / RED regime.

Run before market open (~08:45 EST) to decide whether to trade today.
All data is fetched from IBKR (VIX live, VIX1D/VIX9D prev close, VX futures,
52-week range, SPX realized volatility).
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
    print(f"  VIX prev     : {r.vix_prev_close:.2f}  [prev close]")
    print(f"  VIX1D        : {r.vix1d:.2f}  [prev close]")
    print(f"  VIX9D        : {r.vix9d:.2f}  [prev close]")
    print(f"  VX front     : {r.vx_front:.2f}  [LIVE]")
    print(f"  IV Rank      : {r.ivr:.1f}")
    print(f"  RV 20d       : {r.rv_20d:.2f}%")
    print()
    print("Components:")
    print(f"  [C1] VIX1D absolute    : {r.vix1d_absolute.value:<12} → {r.vix1d_absolute.score:2d} pts")
    print(f"  [C2] VIX1D/VIX ratio   : {r.vix1d_vix_ratio.value:<12} → {r.vix1d_vix_ratio.score:2d} pts")
    print(f"  [C3] Term Structure    : {r.term_structure.value:<12} → {r.term_structure.score:2d} pts")
    print(f"  [C4] VIX9D/VIX ratio   : {r.vix9d_vix_ratio.value:<12} → {r.vix9d_vix_ratio.score:2d} pts")
    print(f"  [C5] Overnight VIX chg : {r.overnight_vix_chg.value}%{'':<6} → {r.overnight_vix_chg.score:2d} pts")
    print(f"  [C6] IV Rank (IVR)     : {r.iv_rank.value:<12} → {r.iv_rank.score:2d} pts")
    print(f"  [C7] IV/RV Spread      : {r.iv_rv_spread.value:<12} → {r.iv_rv_spread.score:2d} pts")
    print()
    print(f"TOTAL SCORE : {r.score} / 100")
    print(f"REGIME      : {emoji} {r.regime}")
    print(f"ACTION      : {r.action}")
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
