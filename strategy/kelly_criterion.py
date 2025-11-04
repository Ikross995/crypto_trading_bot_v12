"""
Kelly Criterion Position Sizing
================================

Optimal position sizing based on historical performance statistics.

The Kelly Criterion calculates the optimal percentage of capital to risk
on each trade based on:
- Win rate (probability of winning)
- Profit/Loss ratio (average win / average loss)

Formula: Kelly % = (bp - q) / b
Where:
  b = profit/loss ratio (avg_win / avg_loss)
  p = win rate (probability of winning)
  q = 1 - p (probability of losing)

IMPORTANT: Full Kelly is aggressive and leads to high drawdowns.
Use fractional Kelly (0.25 = Quarter Kelly, 0.5 = Half Kelly) for safer results.

Research shows:
- Full Kelly (1.0): Optimal growth but high volatility (not recommended)
- Half Kelly (0.5): 75% of growth, 50% of volatility
- Quarter Kelly (0.25): 50% of growth, 25% of volatility (recommended for crypto)
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class KellyResult:
    """Result from Kelly Criterion calculation"""
    kelly_percentage: float  # Optimal % of capital to risk
    win_rate: float  # Historical win rate
    profit_loss_ratio: float  # Avg win / avg loss
    avg_win: float  # Average winning trade
    avg_loss: float  # Average losing trade
    total_trades: int  # Number of trades analyzed
    confidence: float  # Statistical confidence (0-1)
    recommendation: str  # Trading recommendation
    fractional_multiplier: float  # Applied fractional Kelly


class KellyCriterionCalculator:
    """
    Calculates optimal position sizing using Kelly Criterion.

    Usage:
        calculator = KellyCriterionCalculator(use_fractional=0.25)
        result = await calculator.calculate_kelly_size(trade_history)
        position_size = account_balance * result.kelly_percentage
    """

    def __init__(
        self,
        use_fractional: float = 0.25,
        min_trades_required: int = 30,
        max_kelly_percentage: float = 0.10,
        min_kelly_percentage: float = 0.005,
        lookback_days: Optional[int] = None
    ):
        """
        Initialize Kelly Criterion calculator.

        Args:
            use_fractional: Fractional Kelly multiplier
                          0.25 = Quarter Kelly (conservative, recommended)
                          0.50 = Half Kelly (moderate)
                          1.00 = Full Kelly (aggressive, NOT recommended)
            min_trades_required: Minimum trades for statistical significance
            max_kelly_percentage: Maximum % of capital to risk (safety cap)
            min_kelly_percentage: Minimum % of capital to risk
            lookback_days: Only use trades from last N days (None = all trades)
        """
        self.use_fractional = use_fractional
        self.min_trades_required = min_trades_required
        self.max_kelly_percentage = max_kelly_percentage
        self.min_kelly_percentage = min_kelly_percentage
        self.lookback_days = lookback_days

        logger.info(
            f"Kelly Criterion initialized: "
            f"fractional={use_fractional}, "
            f"min_trades={min_trades_required}, "
            f"lookback_days={lookback_days}"
        )

    async def calculate_kelly_size(
        self,
        trade_history: List[Dict],
        regime: Optional[str] = None
    ) -> KellyResult:
        """
        Calculate optimal Kelly position size.

        Args:
            trade_history: List of trades with 'pnl', 'timestamp', 'regime' keys
            regime: Optional filter for specific market regime

        Returns:
            KellyResult with optimal position size and statistics
        """
        # Filter trades
        filtered_trades = self._filter_trades(trade_history, regime)

        # Check if we have enough data
        if len(filtered_trades) < self.min_trades_required:
            logger.warning(
                f"Insufficient trades ({len(filtered_trades)}/{self.min_trades_required}). "
                "Using conservative default."
            )
            return self._default_result(len(filtered_trades))

        # Calculate win/loss statistics
        wins = [t for t in filtered_trades if t['pnl'] > 0]
        losses = [t for t in filtered_trades if t['pnl'] < 0]
        breakevens = [t for t in filtered_trades if t['pnl'] == 0]

        if not wins or not losses:
            logger.warning("No wins or no losses in history. Using conservative default.")
            return self._default_result(len(filtered_trades))

        # Calculate statistics
        win_rate = len(wins) / (len(wins) + len(losses))  # Exclude breakevens
        loss_rate = 1 - win_rate

        avg_win = sum(t['pnl'] for t in wins) / len(wins)
        avg_loss = abs(sum(t['pnl'] for t in losses) / len(losses))

        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0

        # Kelly formula: (bp - q) / b
        # Where b = profit/loss ratio, p = win rate, q = loss rate
        kelly_pct = (profit_loss_ratio * win_rate - loss_rate) / profit_loss_ratio

        # Handle negative Kelly (system has negative expectancy)
        if kelly_pct <= 0:
            logger.warning(
                f"Negative Kelly ({kelly_pct:.4f}). System has negative expectancy! "
                "DO NOT TRADE."
            )
            return KellyResult(
                kelly_percentage=0.0,
                win_rate=win_rate,
                profit_loss_ratio=profit_loss_ratio,
                avg_win=avg_win,
                avg_loss=avg_loss,
                total_trades=len(filtered_trades),
                confidence=0.0,
                recommendation="DO NOT TRADE - Negative expectancy",
                fractional_multiplier=self.use_fractional
            )

        # Apply fractional Kelly
        fractional_kelly = kelly_pct * self.use_fractional

        # Apply caps (safety limits)
        kelly_capped = max(
            self.min_kelly_percentage,
            min(self.max_kelly_percentage, fractional_kelly)
        )

        # Calculate confidence based on sample size
        confidence = self._calculate_confidence(len(filtered_trades))

        # Generate recommendation
        recommendation = self._generate_recommendation(
            kelly_capped,
            win_rate,
            profit_loss_ratio,
            confidence
        )

        result = KellyResult(
            kelly_percentage=kelly_capped,
            win_rate=win_rate,
            profit_loss_ratio=profit_loss_ratio,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_trades=len(filtered_trades),
            confidence=confidence,
            recommendation=recommendation,
            fractional_multiplier=self.use_fractional
        )

        logger.info(
            f"Kelly Calculation: {kelly_capped:.2%} "
            f"(WR: {win_rate:.1%}, P/L: {profit_loss_ratio:.2f}, "
            f"Trades: {len(filtered_trades)}, Confidence: {confidence:.1%})"
        )

        return result

    def _filter_trades(
        self,
        trade_history: List[Dict],
        regime: Optional[str] = None
    ) -> List[Dict]:
        """Filter trades by regime and lookback period"""
        filtered = trade_history.copy()

        # Filter by regime
        if regime:
            filtered = [t for t in filtered if t.get('regime') == regime]

        # Filter by lookback period
        if self.lookback_days:
            cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
            filtered = [
                t for t in filtered
                if datetime.fromisoformat(t.get('timestamp', '2000-01-01')) > cutoff_date
            ]

        return filtered

    def _default_result(self, trade_count: int) -> KellyResult:
        """Return conservative default when insufficient data"""
        return KellyResult(
            kelly_percentage=0.01,  # Conservative 1%
            win_rate=0.5,
            profit_loss_ratio=1.0,
            avg_win=0.0,
            avg_loss=0.0,
            total_trades=trade_count,
            confidence=0.0,
            recommendation="Insufficient data - using conservative 1% sizing",
            fractional_multiplier=self.use_fractional
        )

    def _calculate_confidence(self, sample_size: int) -> float:
        """
        Calculate statistical confidence based on sample size.

        Uses a logarithmic scale:
        - 30 trades: 50% confidence
        - 100 trades: 75% confidence
        - 500+ trades: 95% confidence
        """
        if sample_size < self.min_trades_required:
            return 0.0
        elif sample_size >= 500:
            return 0.95
        elif sample_size >= 200:
            return 0.85
        elif sample_size >= 100:
            return 0.75
        elif sample_size >= 50:
            return 0.60
        else:
            return 0.50

    def _generate_recommendation(
        self,
        kelly_pct: float,
        win_rate: float,
        pl_ratio: float,
        confidence: float
    ) -> str:
        """Generate trading recommendation based on Kelly results"""

        # Check for excellent system
        if kelly_pct >= 0.05 and win_rate >= 0.55 and pl_ratio >= 2.0:
            return "EXCELLENT - Strong edge detected"

        # Check for good system
        elif kelly_pct >= 0.03 and win_rate >= 0.50 and pl_ratio >= 1.5:
            return "GOOD - Positive edge, continue trading"

        # Check for marginal system
        elif kelly_pct >= 0.02:
            if confidence < 0.60:
                return "MARGINAL - Need more data for confidence"
            else:
                return "MARGINAL - Slight edge, use with caution"

        # Low Kelly but might be OK
        elif kelly_pct >= 0.01:
            return "WEAK - Very small edge, monitor closely"

        # Very low Kelly
        else:
            return "POOR - Minimal/no edge detected"

    async def calculate_kelly_by_regime(
        self,
        trade_history: List[Dict]
    ) -> Dict[str, KellyResult]:
        """
        Calculate Kelly for each market regime separately.

        Returns:
            Dict mapping regime name to KellyResult
        """
        # Find all unique regimes
        regimes = set(t.get('regime', 'UNKNOWN') for t in trade_history)

        results = {}
        for regime in regimes:
            result = await self.calculate_kelly_size(trade_history, regime=regime)
            results[regime] = result

            logger.info(
                f"[{regime}] Kelly: {result.kelly_percentage:.2%}, "
                f"WR: {result.win_rate:.1%}, P/L: {result.profit_loss_ratio:.2f}"
            )

        return results

    def adjust_for_drawdown(
        self,
        kelly_percentage: float,
        current_drawdown: float,
        max_drawdown_threshold: float = 0.15
    ) -> float:
        """
        Reduce position size during drawdowns.

        Args:
            kelly_percentage: Calculated Kelly %
            current_drawdown: Current drawdown as decimal (0.10 = 10%)
            max_drawdown_threshold: Start reducing at this drawdown level

        Returns:
            Adjusted Kelly percentage
        """
        if current_drawdown <= 0:
            return kelly_percentage

        # If drawdown exceeds threshold, reduce Kelly
        if current_drawdown >= max_drawdown_threshold:
            reduction_factor = 1 - (current_drawdown / max_drawdown_threshold)
            reduction_factor = max(0.25, min(1.0, reduction_factor))  # 25-100%

            adjusted = kelly_percentage * reduction_factor

            logger.warning(
                f"Drawdown adjustment: {current_drawdown:.1%} drawdown, "
                f"reducing Kelly from {kelly_percentage:.2%} to {adjusted:.2%}"
            )

            return adjusted

        return kelly_percentage

    def get_position_size(
        self,
        account_balance: float,
        kelly_result: KellyResult,
        current_drawdown: float = 0.0
    ) -> Dict:
        """
        Calculate actual position size in dollars.

        Args:
            account_balance: Total account balance
            kelly_result: Result from calculate_kelly_size()
            current_drawdown: Optional current drawdown for adjustment

        Returns:
            Dict with position size and details
        """
        # Adjust Kelly for drawdown
        adjusted_kelly = self.adjust_for_drawdown(
            kelly_result.kelly_percentage,
            current_drawdown
        )

        # Calculate position size
        position_size = account_balance * adjusted_kelly

        # Safety check: never risk more than 10% of account
        max_position = account_balance * 0.10
        if position_size > max_position:
            logger.warning(
                f"Position size ${position_size:.2f} exceeds 10% cap. "
                f"Capping at ${max_position:.2f}"
            )
            position_size = max_position

        return {
            'position_size': position_size,
            'position_percentage': adjusted_kelly,
            'original_kelly': kelly_result.kelly_percentage,
            'account_balance': account_balance,
            'drawdown_adjustment_applied': current_drawdown > 0,
            'current_drawdown': current_drawdown
        }

    def log_kelly_analysis(self, result: KellyResult):
        """Log detailed Kelly analysis"""
        logger.info("=" * 70)
        logger.info("📊 KELLY CRITERION ANALYSIS")
        logger.info("=" * 70)
        logger.info(f"Optimal Position Size: {result.kelly_percentage:.2%}")
        logger.info(f"Fractional Multiplier: {result.fractional_multiplier}x")
        logger.info("")
        logger.info(f"Win Rate: {result.win_rate:.2%}")
        logger.info(f"Profit/Loss Ratio: {result.profit_loss_ratio:.2f}")
        logger.info(f"Average Win: ${result.avg_win:.2f}")
        logger.info(f"Average Loss: ${result.avg_loss:.2f}")
        logger.info(f"Total Trades: {result.total_trades}")
        logger.info(f"Confidence: {result.confidence:.1%}")
        logger.info("")
        logger.info(f"Recommendation: {result.recommendation}")
        logger.info("=" * 70)
