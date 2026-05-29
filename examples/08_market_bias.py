"""08 — Market Structure & Bias Detection.

Determines market condition (Bullish/Bearish/Neutral) and confidence level
by scoring three components — all sourced from IBKR API:

  C1: Price Structure — swing HH/HL/LH/LL on daily bars (±2 pts)
  C2: Volatility Regime — composite GREEN/YELLOW/RED     (±1 pt)
  C3: Gap & VWAP positioning                             (±1 pt)

Total score: -4 to +4 → Strong Bearish … Neutral … Strong Bullish.

Signal inherits TRADE/NOTRADE from VolRegimeResult — RED means sit out.

Run before market open to establish directional bias for the session.
"""

from __future__ import annotations

import asyncio
import logging

from ibtws.config import IBKRConfig
from ibtws.unofficial.client import IBKRClient
from ibtws.unofficial.analysis.market_bias import MarketBiasDetector, MarketBiasResult, Bias

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")


def _print(r: MarketBiasResult) -> None:
    emoji = {
        Bias.STRONG_BULLISH: "🟢🟢",
        Bias.LEAN_BULLISH: "🟢",
        Bias.NEUTRAL: "⚪",
        Bias.LEAN_BEARISH: "🔴",
        Bias.STRONG_BEARISH: "🔴🔴",
    }[r.bias]

    print("\n=== MARKET STRUCTURE & BIAS ===\n")
    print(f"  Spot         : {r.spot:.2f}")
    print(f"  Prior Close  : {r.prior_close:.2f}")
    print(f"  VWAP         : {r.vwap:.2f}" if r.vwap else "  VWAP         : N/A (pre-market)")
    print(f"  Structure    : {r.structure.value}")
    if r.invalidation_level:
        print(f"  Invalidation : {r.invalidation_level:.2f}  ← bias flips if broken")
    print()
    print("Component Scores:")
    print(f"  [C1] Structure (HH/HL/LH/LL) : {r.structure_score:+d}")
    print(f"  [C2] Vol Regime              : {r.vol_regime_score:+d}")
    print(f"  [C3] Gap & VWAP              : {r.gap_vwap_score:+d}")
    print()
    print(f"{'─' * 50}")
    print(f"  TOTAL SCORE  : {r.score:+d} / ±4")
    print(f"  BIAS         : {emoji} {r.bias.value}")
    print(f"  CONFIDENCE   : {r.confidence}")
    print(f"  SIGNAL       : {'✅ ' + r.signal if r.signal == 'TRADE' else '🚫 ' + r.signal}")
    print(f"  VOL CONTEXT  : {'HIGH VOL' if r.high_vol else 'LOW/NORMAL VOL'}")
    print(f"  ACTION       : {r.action}")
    print(f"{'─' * 50}")

    if r.vol_regime:
        vr = r.vol_regime
        print(f"\n  Vol Regime   : {vr.regime} (score {vr.score}/100)")
        print(f"  VIX          : {vr.vix:.2f}")
        print(f"  Term Struct  : {vr.term_structure.value:.3f} (VIX/VIX3M)")


async def main() -> None:
    config = IBKRConfig(host="192.168.0.129", port=7497, client_id=15)

    async with IBKRClient(config) as client:
        await client.connect()
        detector = MarketBiasDetector(client)
        result = await detector.detect()
        _print(result)


if __name__ == "__main__":
    asyncio.run(main())
