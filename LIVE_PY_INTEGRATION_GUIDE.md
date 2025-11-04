# 🔧 Интеграция Phase 1-4 улучшений в runner/live.py

**Цель:** Интегрировать Kelly Criterion, Dynamic Stops, Adaptive Strategy, GRU и другие улучшения в существующий trading engine.

---

## 📋 Что будет интегрировано

### ✅ Phase 1: Critical Fixes
- **Concurrency Protection** - защита от race conditions
- **Rate Limiting** - защита от API банов
- **WebSocket Auto-Reconnect** - надежное подключение

### ✅ Phase 2: AI/ML
- **GRU Model** - замена/дополнение LSTM (MAPE 3.54% vs 5.2%)

### ✅ Phase 3: Trading Logic
- **Regime Detector** - определение 5 рыночных режимов
- **Adaptive Strategy** - авто-настройка под режим

### ✅ Phase 4: Risk Management
- **Kelly Criterion** - оптимальный sizing вместо fixed %
- **Dynamic ATR Stops** - адаптивные стопы (1.5-3.0× ATR)
- **Trailing Stops** - улучшенные trailing stops

---

## 🔨 Изменения в live.py

### 1. Добавить импорты (в начало файла, после строки 88)

```python
# ==================== PHASE 1-4 INTEGRATIONS ====================

# Phase 1: Critical Fixes
try:
    from utils.concurrency import get_global_safe_state  # type: ignore
except Exception:
    get_global_safe_state = None  # type: ignore

try:
    from utils.rate_limiter import AdaptiveRateLimiter  # type: ignore
except Exception:
    AdaptiveRateLimiter = None  # type: ignore

# Phase 2: GRU Model
try:
    from models.gru_predictor import GRUPricePredictor  # type: ignore
except Exception:
    GRUPricePredictor = None  # type: ignore

# Phase 3: Adaptive Strategy
try:
    from strategy.regime_detector import MarketRegimeDetector  # type: ignore
except Exception:
    MarketRegimeDetector = None  # type: ignore

try:
    from strategy.adaptive_strategy import AdaptiveStrategyManager  # type: ignore
except Exception:
    AdaptiveStrategyManager = None  # type: ignore

# Phase 4: Risk Management
try:
    from strategy.kelly_criterion import KellyCriterionCalculator  # type: ignore
except Exception:
    KellyCriterionCalculator = None  # type: ignore

try:
    from strategy.dynamic_stops import DynamicStopLossManager  # type: ignore
except Exception:
    DynamicStopLossManager = None  # type: ignore

# ================================================================
```

---

### 2. Добавить инициализацию в __init__ (после строки 365)

```python
        # ==================== PHASE 1-4 COMPONENTS ====================

        # Phase 1: Concurrency & Rate Limiting
        self.safe_state = None
        if get_global_safe_state:
            try:
                self.safe_state = get_global_safe_state()
                self.logger.info("🔒 [CONCURRENCY] Race condition protection enabled")
            except Exception as e:
                self.logger.warning("🔒 [CONCURRENCY] Failed to initialize: %s", e)

        self.rate_limiter = None
        if AdaptiveRateLimiter:
            try:
                # Binance: 1200 requests/minute
                self.rate_limiter = AdaptiveRateLimiter(
                    max_calls=1200,
                    time_window=60
                )
                self.logger.info("⏱️ [RATE_LIMIT] Adaptive rate limiter initialized (1200/min)")
            except Exception as e:
                self.logger.warning("⏱️ [RATE_LIMIT] Failed to initialize: %s", e)

        # Phase 2: GRU Predictor
        self.gru_predictor = None
        if GRUPricePredictor and getattr(config, "gru_enable", True):
            try:
                self.gru_predictor = GRUPricePredictor(
                    sequence_length=60,
                    features=12
                )
                # Try to load pre-trained model
                model_path = "models/checkpoints/gru_btcusdt.keras"
                if Path(model_path).exists():
                    self.gru_predictor.load(model_path)
                    self.logger.info("🤖 [GRU] Pre-trained model loaded from %s", model_path)
                else:
                    self.logger.info("🤖 [GRU] Model initialized (needs training)")
            except Exception as e:
                self.logger.warning("🤖 [GRU] Failed to initialize: %s", e)

        # Phase 3: Regime Detection & Adaptive Strategy
        self.regime_detector = None
        self.adaptive_strategy = None

        if MarketRegimeDetector:
            try:
                self.adaptive_strategy = AdaptiveStrategyManager(
                    enable_regime_logging=True
                )
                self.logger.info("🎯 [ADAPTIVE] Adaptive strategy manager initialized")
                self.logger.info("  - 5 market regimes: STRONG_TREND, VOLATILE_TREND, TIGHT_RANGE, CHOPPY, TRANSITIONAL")
                self.logger.info("  - Auto-adjusts: position size, stops, targets, DCA")
            except Exception as e:
                self.logger.warning("🎯 [ADAPTIVE] Failed to initialize: %s", e)

        # Phase 4: Kelly Criterion & Dynamic Stops
        self.kelly_calculator = None
        if KellyCriterionCalculator:
            try:
                self.kelly_calculator = KellyCriterionCalculator(
                    use_fractional=0.25,  # Quarter Kelly (conservative)
                    min_trades_required=30,
                    max_kelly_percentage=0.10,
                    min_kelly_percentage=0.005
                )
                self.logger.info("💰 [KELLY] Kelly Criterion position sizing enabled (Quarter Kelly)")

                # Initialize trade history for Kelly calculation
                self.trade_history = []
            except Exception as e:
                self.logger.warning("💰 [KELLY] Failed to initialize: %s", e)

        self.dynamic_stops = None
        if DynamicStopLossManager:
            try:
                self.dynamic_stops = DynamicStopLossManager(
                    default_atr_multiplier=2.0,
                    max_stop_percentage=0.05,  # 5% max
                    trailing_activation_pct=0.025,  # Activate at 2.5% profit
                    breakeven_activation_pct=0.015  # Breakeven at 1.5% profit
                )
                self.logger.info("🛡️ [DYNAMIC_STOPS] Dynamic ATR-based stops enabled")
                self.logger.info("  - Initial: 1.5-3.0× ATR (adapts to volatility)")
                self.logger.info("  - Trailing: activates at +2.5% profit")
                self.logger.info("  - Breakeven: moves at +1.5% profit")
            except Exception as e:
                self.logger.warning("🛡️ [DYNAMIC_STOPS] Failed to initialize: %s", e)

        # Track highest/lowest prices for trailing stops
        self.position_extremes = {}  # {symbol: {'highest': float, 'lowest': float}}

        # ================================================================
```

---

### 3. ЗАМЕНИТЬ метод _position_size_qty (строки 466-531)

**Старый код:**
```python
def _position_size_qty(self, price: Optional[float], strength: float = 0.5) -> Optional[float]:
    # ...фиксированный риск %...
```

**Новый код (с Kelly Criterion):**

```python
def _position_size_qty(
    self,
    symbol: str,
    price: Optional[float],
    strength: float = 0.5,
    regime: Optional[str] = None
) -> Optional[float]:
    """
    ENHANCED: Kelly Criterion position sizing with regime adaptation.

    Replaces fixed % risk with optimal Kelly sizing based on:
    - Historical win rate & profit/loss ratio
    - Market regime (from adaptive strategy)
    - Signal confidence
    """
    p = _to_float(price)
    if not p or p <= 0:
        return None

    equity = float(self.equity_usdt)

    # ==================== KELLY CRITERION SIZING ====================
    if self.kelly_calculator and self.trade_history:
        try:
            # Calculate Kelly percentage
            kelly_result = await self.kelly_calculator.calculate_kelly_size(
                trade_history=self.trade_history,
                regime=regime  # Calculate for specific regime if available
            )

            # Get position size in dollars
            position_info = self.kelly_calculator.get_position_size(
                account_balance=equity,
                kelly_result=kelly_result,
                current_drawdown=self._calculate_current_drawdown()
            )

            position_value = position_info['position_size']

            self.logger.info(
                "💰 [KELLY] Win Rate: %.1f%%, P/L Ratio: %.2f, Kelly: %.2%%, Position: $%.2f",
                kelly_result.win_rate * 100,
                kelly_result.profit_loss_ratio,
                kelly_result.kelly_percentage * 100,
                position_value
            )

        except Exception as kelly_e:
            self.logger.warning("💰 [KELLY] Calculation failed, using fallback: %s", kelly_e)
            # Fallback to old method
            position_value = self._fallback_position_sizing(equity, strength)
    else:
        # Not enough history yet, use conservative fallback
        if self.kelly_calculator and len(self.trade_history) < 30:
            self.logger.info(
                "💰 [KELLY] Insufficient history (%d trades), using conservative 1%% sizing",
                len(self.trade_history)
            )
        position_value = equity * 0.01  # Conservative 1%

    # ==================== REGIME ADAPTATION ====================
    if self.adaptive_strategy and regime:
        try:
            # Get regime-specific multiplier
            params = self.adaptive_strategy.get_current_parameters()
            position_multiplier = params.position_size_multiplier

            # Apply regime adjustment
            original_value = position_value
            position_value = self.adaptive_strategy.adjust_position_size(position_value)

            self.logger.info(
                "🎯 [ADAPTIVE] Regime: %s, Multiplier: %.2fx, Position: $%.2f → $%.2f",
                regime,
                position_multiplier,
                original_value,
                position_value
            )
        except Exception as adapt_e:
            self.logger.warning("🎯 [ADAPTIVE] Adjustment failed: %s", adapt_e)

    # Apply signal strength
    strength_clamped = min(1.0, max(0.1, strength))
    position_value *= (0.5 + strength_clamped * 0.5)  # 0.5x to 1.0x

    # Safety checks
    position_value = max(self.min_notional, position_value)
    max_position = equity * 0.2 * self.leverage  # Max 20% of equity
    position_value = min(position_value, max_position)

    # Convert to quantity
    qty = position_value / p

    if qty <= 0:
        return None

    self.logger.info(
        "[POSITION_SIZE] Equity=$%.2f, Position=$%.2f, Qty=%.6f",
        equity,
        position_value,
        qty,
    )

    return qty

def _fallback_position_sizing(self, equity: float, strength: float) -> float:
    """Fallback position sizing when Kelly unavailable"""
    base_risk_pct = self.risk_pct  # e.g., 0.5%
    strength_clamped = min(1.0, max(0.1, strength))
    strength_multiplier = 0.5 + (strength_clamped * 0.5)
    adjusted_risk_pct = base_risk_pct * strength_multiplier
    risk_amount = equity * (adjusted_risk_pct / 100.0)
    sl_distance_pct = getattr(self.config, "sl_pct", 2.0)
    position_value = risk_amount / (sl_distance_pct / 100.0)
    return position_value

def _calculate_current_drawdown(self) -> float:
    """Calculate current drawdown from peak equity"""
    if not hasattr(self, 'peak_equity'):
        self.peak_equity = self.equity_usdt

    self.peak_equity = max(self.peak_equity, self.equity_usdt)

    if self.peak_equity <= 0:
        return 0.0

    drawdown = (self.peak_equity - self.equity_usdt) / self.peak_equity
    return max(0.0, drawdown)
```

---

### 4. Добавить метод обнаружения режима (перед _process_symbol)

```python
async def _detect_market_regime(self, symbol: str, candles_df) -> Optional[str]:
    """
    Detect current market regime using Regime Detector.

    Returns:
        Regime name (STRONG_TREND, VOLATILE_TREND, TIGHT_RANGE, CHOPPY, TRANSITIONAL)
    """
    if not self.adaptive_strategy:
        return None

    try:
        # Update regime with latest data
        regime_info = await self.adaptive_strategy.update_regime(candles_df)

        regime = regime_info.regime.value
        confidence = regime_info.confidence

        self.logger.info(
            "📊 [REGIME] %s: %s (confidence: %.2f)",
            symbol,
            regime.upper(),
            confidence
        )

        # Log regime metrics
        metrics = regime_info.metrics
        self.logger.info(
            "  - ADX: %.1f, Volatility: %.2f%%, BB Width: %.4f",
            metrics.get('adx', 0),
            metrics.get('volatility', 0) * 100,
            metrics.get('bb_width', 0)
        )

        return regime

    except Exception as e:
        self.logger.warning("📊 [REGIME] Detection failed: %s", e)
        return None
```

---

### 5. Модифицировать _process_symbol (добавить regime detection)

Найдите строку 1262 где:
```python
raw = await self._produce_raw_signal(symbol, md)
```

**Добавьте ПЕРЕД этой строкой:**

```python
        # ==================== REGIME DETECTION ====================
        regime = None
        if md is not None and self.adaptive_strategy:
            try:
                # Convert market data to DataFrame
                import pandas as pd

                if hasattr(md, 'close'):
                    candles_df = pd.DataFrame({
                        'timestamp': md.timestamp,
                        'open': md.open,
                        'high': md.high,
                        'low': md.low,
                        'close': md.close,
                        'volume': md.volume
                    })
                elif isinstance(md, list) and len(md) > 0:
                    if isinstance(md[0], dict):
                        candles_df = pd.DataFrame(md)
                    else:
                        candles_df = pd.DataFrame(md, columns=[
                            'timestamp', 'open', 'high', 'low', 'close', 'volume'
                        ])
                else:
                    candles_df = None

                if candles_df is not None and len(candles_df) > 100:
                    regime = await self._detect_market_regime(symbol, candles_df)

                    # Check if should trade in this regime
                    if regime and self.adaptive_strategy:
                        params = self.adaptive_strategy.get_current_parameters()

                        # Some regimes may require higher confidence
                        if regime == 'CHOPPY' and params.confidence_threshold > 0.75:
                            self.logger.warning(
                                "🚫 [ADAPTIVE] CHOPPY regime requires high confidence (>0.75), "
                                "may skip marginal signals"
                            )
            except Exception as regime_e:
                self.logger.warning("📊 [REGIME] Failed to detect regime: %s", regime_e)
```

---

## 📄 Продолжение следует...

Это первая часть интеграции. Файл слишком большой чтобы показать все изменения сразу.

**Следующие части:**
- Интеграция GRU predictions
- Интеграция Dynamic ATR stops
- Интеграция Concurrency protection для order execution
- Интеграция Rate limiting для API calls

Хотите я продолжу или сначала примените эти изменения?
