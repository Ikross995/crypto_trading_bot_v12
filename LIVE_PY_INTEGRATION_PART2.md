# 🔧 Интеграция Phase 1-4: Часть 2

## Продолжение интеграции в runner/live.py

---

### 6. Интеграция GRU Predictions (заменить/дополнить LSTM)

Найдите строку где используется LSTM (примерно строка 276-284 в __init__):

```python
# LSTM Integration
self.lstm_predictor = None
if LSTMPredictor and getattr(config, "lstm_enable", False):
    try:
        self.lstm_predictor = LSTMPredictor(config)
        self.logger.info("LSTM predictor initialized...")
    except Exception as e:
        self.logger.warning("LSTMPredictor init failed: %s", e)
```

**ДОПОЛНИТЕ (не заменяйте, GRU уже добавлен в Part 1):**

Добавьте новый метод после _detect_market_regime:

```python
async def _get_gru_prediction(self, symbol: str, candles_df, current_price: float) -> Optional[Dict]:
    """
    Get GRU model price prediction.

    Returns:
        Dict with predicted_price, expected_change_pct, confidence
    """
    if not self.gru_predictor:
        return None

    try:
        # Prepare features (last 60 candles)
        if len(candles_df) < 60:
            return None

        feature_columns = [
            'open', 'high', 'low', 'close', 'volume'
        ]

        # Add technical indicators if available
        if 'rsi' in candles_df.columns:
            feature_columns.extend(['rsi', 'macd', 'sma_20', 'ema_50'])

        # Get last 60 candles
        last_sequence = candles_df[feature_columns].iloc[-60:].values

        # Scale features
        last_sequence_scaled = self.gru_predictor.scaler.transform(last_sequence)

        # Reshape for GRU: (1, sequence_length, features)
        X = last_sequence_scaled.reshape(1, 60, len(feature_columns))

        # Predict
        predicted_price = (await self.gru_predictor.predict(X))[0]

        # Calculate expected change
        expected_change_pct = ((predicted_price - current_price) / current_price) * 100

        # Calculate confidence (based on prediction magnitude)
        confidence = min(abs(expected_change_pct) / 2.0, 1.0)  # Max at 2% move

        self.logger.info(
            "🤖 [GRU] %s: Current $%.2f → Predicted $%.2f (%.2f%%), Confidence: %.2f",
            symbol,
            current_price,
            predicted_price,
            expected_change_pct,
            confidence
        )

        return {
            'predicted_price': predicted_price,
            'expected_change_pct': expected_change_pct,
            'confidence': confidence,
            'direction': 'BUY' if expected_change_pct > 0.3 else 'SELL' if expected_change_pct < -0.3 else 'NEUTRAL'
        }

    except Exception as e:
        self.logger.warning("🤖 [GRU] Prediction failed: %s", e)
        return None
```

**Использование в _process_symbol (после regime detection):**

```python
        # ==================== GRU PREDICTION ====================
        gru_prediction = None
        if self.gru_predictor and candles_df is not None:
            gru_prediction = await self._get_gru_prediction(symbol, candles_df, price)

            if gru_prediction:
                # Combine with regime information
                if regime and self.adaptive_strategy:
                    # Check if GRU prediction aligns with regime
                    should_trade, reason = self.adaptive_strategy.should_take_trade(
                        signal_confidence=gru_prediction['confidence'],
                        signal_direction=gru_prediction['direction']
                    )

                    if not should_trade:
                        self.logger.warning(
                            "🚫 [GRU+ADAPTIVE] Trade rejected: %s",
                            reason
                        )
                        return  # Skip this trade

                # Enhance signal strength with GRU prediction
                if abs(gru_prediction['expected_change_pct']) > 0.5:
                    sig.strength = min(2.0, sig.strength * (1 + gru_prediction['confidence']))
                    self.logger.info(
                        "🤖 [GRU] Enhanced signal strength: %.2f (GRU confidence: %.2f)",
                        sig.strength,
                        gru_prediction['confidence']
                    )
```

---

### 7. Интеграция Dynamic ATR Stops

**Добавьте новый метод для расчета стопов:**

```python
async def _calculate_dynamic_stops(
    self,
    symbol: str,
    entry_price: float,
    side: str,
    market_data: Dict,
    position_size: float,
    regime: Optional[str] = None
) -> Dict:
    """
    Calculate dynamic ATR-based stop-loss and take-profit levels.

    Returns:
        Dict with stop_loss, take_profit, risk_reward
    """
    if not self.dynamic_stops:
        # Fallback to fixed stops
        sl_pct = getattr(self.config, "sl_fixed_pct", 2.0) / 100
        tp_pct = getattr(self.config, "tp_levels", [2.0])[0] / 100

        if side == 'BUY':
            stop_loss = entry_price * (1 - sl_pct)
            take_profit = entry_price * (1 + tp_pct)
        else:
            stop_loss = entry_price * (1 + sl_pct)
            take_profit = entry_price * (1 - tp_pct)

        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward': tp_pct / sl_pct,
            'type': 'FIXED'
        }

    try:
        # Calculate initial stop
        initial_stop = await self.dynamic_stops.calculate_initial_stop(
            entry_price=entry_price,
            side=side,
            market_data=market_data,
            position_size=position_size,
            regime=regime
        )

        # Calculate take profit (R:R based on stop distance)
        stop_distance = initial_stop.stop_distance
        tp_distance = stop_distance * 2.5  # Default 1:2.5 R:R

        if side == 'BUY':
            take_profit = entry_price + tp_distance
        else:
            take_profit = entry_price - tp_distance

        risk_reward = tp_distance / stop_distance

        self.logger.info(
            "🛡️ [DYNAMIC_STOP] %s: SL=$%.2f (%.2f%%), TP=$%.2f, R:R=1:%.2f, ATR Mult=%.1fx",
            symbol,
            initial_stop.stop_price,
            initial_stop.stop_percentage * 100,
            take_profit,
            risk_reward,
            initial_stop.atr_multiplier
        )

        return {
            'stop_loss': initial_stop.stop_price,
            'take_profit': take_profit,
            'risk_reward': risk_reward,
            'atr_value': initial_stop.atr_value,
            'atr_multiplier': initial_stop.atr_multiplier,
            'type': 'DYNAMIC_ATR'
        }

    except Exception as e:
        self.logger.error("🛡️ [DYNAMIC_STOP] Calculation failed: %s", e)
        # Fallback to fixed
        return await self._calculate_dynamic_stops(symbol, entry_price, side, {}, position_size, regime)
```

**Использование перед размещением ордера (в _process_symbol):**

Найдите где размещается ордер (примерно строка 1429+) и добавьте расчет стопов:

```python
        # Calculate position size
        qty = self._position_size_qty(symbol, price, sig.strength, regime)
        if not qty:
            return

        # ==================== CALCULATE DYNAMIC STOPS ====================
        # Prepare market data for stop calculation
        atr = None
        if md and hasattr(md, 'atr'):
            atr = md.atr
        elif candles_df is not None and len(candles_df) > 14:
            # Calculate ATR from candles
            high_low = candles_df['high'] - candles_df['low']
            high_close = abs(candles_df['high'] - candles_df['close'].shift())
            low_close = abs(candles_df['low'] - candles_df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]

        market_data_for_stops = {
            'atr_14': atr if atr else price * 0.02,  # Fallback to 2% of price
            'close': price
        }

        position_value = qty * price
        stops_data = await self._calculate_dynamic_stops(
            symbol=symbol,
            entry_price=price,
            side=sig.side,
            market_data=market_data_for_stops,
            position_size=position_value,
            regime=regime
        )

        stop_loss_price = stops_data['stop_loss']
        take_profit_price = stops_data['take_profit']

        self.logger.info(
            "[ENTRY] %s %s @ %.2f, Qty=%.6f, SL=%.2f, TP=%.2f (R:R=1:%.2f)",
            sig.side,
            symbol,
            price,
            qty,
            stop_loss_price,
            take_profit_price,
            stops_data.get('risk_reward', 0)
        )
```

---

### 8. Интеграция Concurrency Protection

**Замените места где размещаются ордера на protected версию:**

Найдите код размещения ордера (примерно после строки 1429):

```python
if not self.dry_run:
    try:
        # ... existing order placement code ...
```

**ОБЕРНИТЕ в concurrency protection:**

```python
if not self.dry_run:
    # ==================== CONCURRENCY PROTECTION ====================
    async def execute_order_protected():
        """Protected order execution"""
        try:
            # EXISTING ORDER PLACEMENT CODE HERE
            # Place the order
            order = await self.client.create_order(
                symbol=symbol,
                side=sig.side,
                order_type="MARKET",
                quantity=qty,
            )

            # Set stop-loss
            if stop_loss_price:
                await self.client.create_order(
                    symbol=symbol,
                    side="SELL" if sig.side == "BUY" else "BUY",
                    order_type="STOP_MARKET",
                    quantity=qty,
                    stop_price=stop_loss_price
                )

            # Set take-profit
            if take_profit_price:
                await self.client.create_order(
                    symbol=symbol,
                    side="SELL" if sig.side == "BUY" else "BUY",
                    order_type="TAKE_PROFIT_MARKET",
                    quantity=qty,
                    stop_price=take_profit_price
                )

            return order

        except Exception as order_e:
            self.logger.error("[ORDER_ERROR] %s: %s", symbol, order_e)
            return None

    # Execute with concurrency protection
    if self.safe_state:
        async with self.safe_state.atomic_trade_operation():
            order = await execute_order_protected()
    else:
        order = await execute_order_protected()

    # Update position tracking
    if order:
        # Initialize extremes tracking for trailing stops
        self.position_extremes[symbol] = {
            'highest': price,
            'lowest': price,
            'entry_price': price,
            'side': sig.side,
            'stop_loss': stop_loss_price,
            'atr': atr
        }

        # Record trade for Kelly calculation
        self.trade_history.append({
            'pnl': 0,  # Will update on close
            'regime': regime,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'entry': price,
            'symbol': symbol,
            'side': sig.side,
            'size': position_value
        })
```

---

### 9. Интеграция Rate Limiting

**Оберните ВСЕ API вызовы rate limiter:**

Создайте helper метод:

```python
async def _api_call_with_limit(self, api_func, *args, **kwargs):
    """
    Wrap API call with rate limiting.

    Usage:
        result = await self._api_call_with_limit(
            self.client.get_account
        )
    """
    if self.rate_limiter:
        await self.rate_limiter.acquire()

    try:
        return await api_func(*args, **kwargs)
    except Exception as e:
        # Check if 429 error
        if '429' in str(e) or 'rate limit' in str(e).lower():
            if self.rate_limiter:
                self.rate_limiter.reduce_limit(percentage=0.8)
                self.logger.warning("⏱️ [RATE_LIMIT] 429 error, reduced limit to 80%%")
        raise
```

**Используйте во всех API вызовах:**

```python
# Вместо:
order = await self.client.create_order(...)

# Используйте:
order = await self._api_call_with_limit(
    self.client.create_order,
    symbol=symbol,
    side=sig.side,
    ...
)
```

---

### 10. Обновление Trailing Stops (улучшенная версия)

**Добавьте метод для обновления trailing stops:**

```python
async def _update_trailing_stops(self, symbol: str):
    """
    Update trailing stops for active position using Dynamic Stop Manager.
    """
    if symbol not in self.position_extremes:
        return

    if symbol not in self.active_positions:
        return

    position = self.active_positions[symbol]
    extremes = self.position_extremes[symbol]

    try:
        # Get current price
        current_price = await self._latest_price(symbol)
        if not current_price:
            return

        # Update highest/lowest
        if position['side'] == 'BUY':
            extremes['highest'] = max(extremes['highest'], current_price)
        else:
            extremes['lowest'] = min(extremes['lowest'], current_price)

        # Calculate updated stop
        if self.dynamic_stops:
            market_data = {
                'atr_14': extremes.get('atr', current_price * 0.02),
                'close': current_price
            }

            updated_stop = await self.dynamic_stops.update_stop(
                entry_price=extremes['entry_price'],
                current_price=current_price,
                highest_price=extremes['highest'],
                lowest_price=extremes['lowest'],
                side=position['side'],
                market_data=market_data,
                current_stop=extremes['stop_loss'],
                position_size=position.get('notional', 0)
            )

            # Check if stop moved
            if updated_stop.stop_price != extremes['stop_loss']:
                self.logger.info(
                    "🔔 [TRAILING] %s: Stop moved $%.2f → $%.2f (%s)",
                    symbol,
                    extremes['stop_loss'],
                    updated_stop.stop_price,
                    updated_stop.stop_type.value
                )

                # Update stop on exchange
                if not self.dry_run:
                    # Cancel old stop
                    # Create new stop
                    # (implementation depends on your exchange client)
                    pass

                extremes['stop_loss'] = updated_stop.stop_price

    except Exception as e:
        self.logger.error("[TRAILING] %s error: %s", symbol, e)
```

**Вызывайте в основном цикле (уже есть в строке 1182):**

```python
# Monitor trailing stops after processing all symbols
if self.trailing_stop_manager or self.dynamic_stops:
    try:
        for symbol in self.active_positions.keys():
            await self._update_trailing_stops(symbol)
    except Exception as trail_e:
        self.logger.error("[TRAIL_SL] Monitoring error: %s", trail_e)
```

---

## 🎯 Итоговая интеграция

После всех изменений ваш live.py будет:

✅ **Использовать Kelly Criterion** для optimal sizing
✅ **Определять market regime** перед каждой сделкой
✅ **Адаптировать параметры** под текущий режим
✅ **Получать GRU predictions** для подтверждения сигналов
✅ **Устанавливать dynamic ATR stops** вместо fixed %
✅ **Обновлять trailing stops** автоматически
✅ **Защищать от race conditions** через concurrency locks
✅ **Предотвращать API bans** через rate limiting

---

## 📝 Следующие шаги

1. **Примените изменения** из Part 1 и Part 2
2. **Протестируйте** на testnet
3. **Запустите примеры** для проверки
4. **Мониторьте логи** для выявления проблем

Готово! 🚀
