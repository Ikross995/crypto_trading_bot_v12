#!/usr/bin/env python3
"""
Telegram Bot Integration for Trading Dashboard
Отправка обновлений дашборда в Telegram группу/канал

Использование:
    bot = TelegramDashboardBot(token, chat_id)
    await bot.send_dashboard_update(dashboard_data)
"""

import asyncio
import aiohttp
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger


class TelegramDashboardBot:
    """Telegram бот для отправки обновлений дашборда."""

    def __init__(self, token: str, chat_id: str):
        """
        Инициализация Telegram бота.

        Args:
            token: Bot token от @BotFather
            chat_id: ID группы/канала (можно получить от @userinfobot)
        """
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"

    async def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        Отправить текстовое сообщение.

        Args:
            text: Текст сообщения (поддерживает HTML/Markdown)
            parse_mode: "HTML" или "Markdown"

        Returns:
            True если успешно отправлено
        """
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/sendMessage"
                data = {
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                    "disable_web_page_preview": True
                }

                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        logger.info("📤 [TELEGRAM] Message sent successfully")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ [TELEGRAM] Failed to send message: {error_text}")
                        return False

        except Exception as e:
            logger.error(f"❌ [TELEGRAM] Error sending message: {e}")
            return False

    async def send_document(self, file_path: Path, caption: str = "") -> bool:
        """
        Отправить файл (например HTML дашборд).

        Args:
            file_path: Путь к файлу
            caption: Подпись к файлу

        Returns:
            True если успешно отправлено
        """
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/sendDocument"

                with open(file_path, 'rb') as file:
                    form = aiohttp.FormData()
                    form.add_field('chat_id', self.chat_id)
                    form.add_field('document', file, filename=file_path.name)
                    if caption:
                        form.add_field('caption', caption)

                    async with session.post(url, data=form) as response:
                        if response.status == 200:
                            logger.info(f"📤 [TELEGRAM] Document sent: {file_path.name}")
                            return True
                        else:
                            error_text = await response.text()
                            logger.error(f"❌ [TELEGRAM] Failed to send document: {error_text}")
                            return False

        except Exception as e:
            logger.error(f"❌ [TELEGRAM] Error sending document: {e}")
            return False

    async def send_dashboard_update(self, dashboard_data: Any) -> bool:
        """
        Отправить обновление дашборда в красивом формате.

        Args:
            dashboard_data: DashboardData объект с метриками

        Returns:
            True если успешно отправлено
        """
        try:
            # Формируем красивое сообщение
            message = self._format_dashboard_message(dashboard_data)

            # Отправляем
            return await self.send_message(message, parse_mode="HTML")

        except Exception as e:
            logger.error(f"❌ [TELEGRAM] Error sending dashboard update: {e}")
            return False

    def _format_dashboard_message(self, data: Any) -> str:
        """Форматирует данные дашборда в красивое HTML сообщение."""

        # Эмодзи для статуса
        roi_emoji = "🟢" if data.roi_pct >= 0 else "🔴"
        risk_emoji = "🟢" if data.risk_score < 30 else "🟡" if data.risk_score < 70 else "🔴"
        pnl_emoji = "💰" if data.total_pnl >= 0 else "📉"

        # Streak emoji
        if data.win_streak > 0:
            streak_emoji = "🔥"
            streak_text = f"Win Streak: {data.win_streak}"
        elif data.loss_streak > 0:
            streak_emoji = "❄️"
            streak_text = f"Loss Streak: {data.loss_streak}"
        else:
            streak_emoji = "⚪"
            streak_text = "No Active Streak"

        message = f"""
<b>🚀 Trading Dashboard Update</b>
<i>{data.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</i>

━━━━━━━━━━━━━━━━━━━━
<b>💰 ACCOUNT BALANCE</b>
━━━━━━━━━━━━━━━━━━━━

Balance: <b>${data.account_balance:,.2f}</b>
Equity: <b>${data.equity:,.2f}</b>
Total P&L: <b>{pnl_emoji} ${data.total_pnl:+,.2f}</b> ({data.roi_pct:+.2f}%)
Hourly P&L: <b>${data.hourly_pnl:+,.2f}/hr</b>

━━━━━━━━━━━━━━━━━━━━
<b>📊 TRADING STATS</b>
━━━━━━━━━━━━━━━━━━━━

Total Trades: <b>{data.total_trades}</b>
Win Rate: <b>{data.win_rate:.1%}</b> ({data.winning_trades}W/{data.losing_trades}L)
Profit Factor: <b>{data.profit_factor:.2f}x</b>
Sharpe Ratio: <b>{data.sharpe_ratio:.2f}</b>

Best Trade: <b>💎 ${data.best_trade:,.2f}</b>
Worst Trade: <b>💔 ${data.worst_trade:,.2f}</b>

{streak_emoji} <b>{streak_text}</b>
Best Win Streak: <b>🏆 {data.max_win_streak}</b>
Worst Loss Streak: <b>💀 {data.max_loss_streak}</b>

━━━━━━━━━━━━━━━━━━━━
<b>⚠️ RISK METRICS</b>
━━━━━━━━━━━━━━━━━━━━

Risk Score: <b>{risk_emoji} {data.risk_score:.0f}/100</b>
Margin Used: <b>${data.total_margin_used:,.2f}</b> ({data.margin_usage_pct:.1f}%)
Free Margin: <b>${data.free_margin:,.2f}</b>

━━━━━━━━━━━━━━━━━━━━
<b>📈 OPEN POSITIONS ({len(data.open_positions_details)})</b>
━━━━━━━━━━━━━━━━━━━━
"""

        # Добавляем открытые позиции
        if data.open_positions_details:
            for pos in data.open_positions_details[:5]:  # Первые 5 позиций
                side_emoji = "🟢" if pos['side'] == 'LONG' else "🔴"
                pnl_emoji = "💚" if pos['pnl'] >= 0 else "💔"

                message += f"""
<b>{pos['symbol']}</b> {side_emoji} {pos['leverage']:.0f}x
Entry: ${pos['entry_price']:,.2f} → ${pos['current_price']:,.2f}
P&L: {pnl_emoji} <b>${pos['pnl']:+,.2f}</b> ({pos['pnl_pct']:+.2f}%)
Margin: ${pos['margin_used']:,.2f}
"""

            if len(data.open_positions_details) > 5:
                message += f"\n<i>... and {len(data.open_positions_details) - 5} more</i>\n"
        else:
            message += "\n<i>No open positions</i>\n"

        message += "\n━━━━━━━━━━━━━━━━━━━━"

        return message

    async def send_trade_opened(self, trade_info: Dict[str, Any]) -> bool:
        """
        Отправить уведомление об открытии новой позиции.

        Args:
            trade_info: {
                'symbol': str,
                'side': str (LONG/SHORT),
                'entry_price': float,
                'quantity': float,
                'leverage': float,
                'notional': float,
                'margin_used': float,
                'stop_loss': float (optional),
                'take_profit': float (optional),
                'reason': str (optional)
            }

        Returns:
            True если успешно отправлено
        """
        try:
            message = self._format_trade_opened_message(trade_info)
            return await self.send_message(message, parse_mode="HTML")
        except Exception as e:
            logger.error(f"❌ [TELEGRAM] Error sending trade opened notification: {e}")
            return False

    async def send_trade_closed(self, trade_info: Dict[str, Any]) -> bool:
        """
        Отправить уведомление о закрытии позиции.

        Args:
            trade_info: {
                'symbol': str,
                'side': str (LONG/SHORT),
                'entry_price': float,
                'exit_price': float,
                'quantity': float,
                'pnl': float,
                'pnl_pct': float,
                'duration': str (optional),
                'reason': str (optional)
            }

        Returns:
            True если успешно отправлено
        """
        try:
            message = self._format_trade_closed_message(trade_info)
            return await self.send_message(message, parse_mode="HTML")
        except Exception as e:
            logger.error(f"❌ [TELEGRAM] Error sending trade closed notification: {e}")
            return False

    async def send_position_update(self, position_info: Dict[str, Any]) -> bool:
        """
        Отправить обновление по открытой позиции.

        Args:
            position_info: {
                'symbol': str,
                'side': str,
                'entry_price': float,
                'current_price': float,
                'pnl': float,
                'pnl_pct': float,
                'margin_used': float,
                'leverage': float
            }

        Returns:
            True если успешно отправлено
        """
        try:
            message = self._format_position_update_message(position_info)
            return await self.send_message(message, parse_mode="HTML")
        except Exception as e:
            logger.error(f"❌ [TELEGRAM] Error sending position update: {e}")
            return False

    def _format_trade_opened_message(self, trade: Dict[str, Any]) -> str:
        """Форматирует сообщение об открытии позиции."""
        side_emoji = "🟢" if trade['side'] == 'LONG' else "🔴"

        message = f"""
<b>🎯 NEW TRADE OPENED</b>

<b>{side_emoji} {trade['side']} {trade['symbol']}</b>

━━━━━━━━━━━━━━━━━━━━
<b>📊 ENTRY DETAILS</b>
━━━━━━━━━━━━━━━━━━━━

Entry Price: <b>${trade['entry_price']:,.4f}</b>
Quantity: <b>{trade['quantity']:.4f}</b>
Leverage: <b>{trade.get('leverage', 1):.0f}x</b>

Notional: <b>${trade.get('notional', 0):,.2f}</b>
Margin Used: <b>${trade.get('margin_used', 0):,.2f}</b>
"""

        # Stop Loss / Take Profit
        if trade.get('stop_loss'):
            sl_dist = abs((trade['stop_loss'] - trade['entry_price']) / trade['entry_price'] * 100)
            message += f"\n🛡️ Stop Loss: <b>${trade['stop_loss']:,.4f}</b> ({sl_dist:.2f}%)"

        if trade.get('take_profit'):
            tp_dist = abs((trade['take_profit'] - trade['entry_price']) / trade['entry_price'] * 100)
            message += f"\n💎 Take Profit: <b>${trade['take_profit']:,.4f}</b> ({tp_dist:.2f}%)"

        # Reason
        if trade.get('reason'):
            message += f"\n\n📝 Reason: <i>{trade['reason']}</i>"

        message += "\n\n━━━━━━━━━━━━━━━━━━━━"

        return message

    def _format_trade_closed_message(self, trade: Dict[str, Any]) -> str:
        """Форматирует сообщение о закрытии позиции."""
        side_emoji = "🟢" if trade['side'] == 'LONG' else "🔴"
        pnl_emoji = "💰" if trade['pnl'] >= 0 else "📉"
        result_emoji = "✅" if trade['pnl'] >= 0 else "❌"

        message = f"""
<b>{result_emoji} TRADE CLOSED</b>

<b>{side_emoji} {trade['side']} {trade['symbol']}</b>

━━━━━━━━━━━━━━━━━━━━
<b>📊 TRADE SUMMARY</b>
━━━━━━━━━━━━━━━━━━━━

Entry: <b>${trade['entry_price']:,.4f}</b>
Exit: <b>${trade['exit_price']:,.4f}</b>
Quantity: <b>{trade['quantity']:.4f}</b>

━━━━━━━━━━━━━━━━━━━━
<b>{pnl_emoji} RESULT</b>
━━━━━━━━━━━━━━━━━━━━

P&L: <b>${trade['pnl']:+,.2f}</b>
P&L %: <b>{trade['pnl_pct']:+.2f}%</b>
"""

        # Duration
        if trade.get('duration'):
            message += f"\n⏱️ Duration: <b>{trade['duration']}</b>"

        # Exit reason
        if trade.get('reason'):
            message += f"\n📝 Reason: <i>{trade['reason']}</i>"

        message += "\n\n━━━━━━━━━━━━━━━━━━━━"

        return message

    def _format_position_update_message(self, pos: Dict[str, Any]) -> str:
        """Форматирует сообщение об обновлении позиции."""
        side_emoji = "🟢" if pos['side'] == 'LONG' else "🔴"
        pnl_emoji = "💚" if pos['pnl'] >= 0 else "💔"

        price_change = pos['current_price'] - pos['entry_price']
        price_change_pct = (price_change / pos['entry_price']) * 100

        message = f"""
<b>📊 POSITION UPDATE</b>

<b>{side_emoji} {pos['side']} {pos['symbol']}</b>

Entry: <b>${pos['entry_price']:,.4f}</b>
Current: <b>${pos['current_price']:,.4f}</b>
Change: <b>{price_change:+,.4f}</b> ({price_change_pct:+.2f}%)

Leverage: <b>{pos.get('leverage', 1):.0f}x</b>
Margin: <b>${pos.get('margin_used', 0):,.2f}</b>

{pnl_emoji} <b>P&L: ${pos['pnl']:+,.2f} ({pos['pnl_pct']:+.2f}%)</b>
"""

        return message

    async def test_connection(self) -> bool:
        """
        Тест соединения с Telegram.

        Returns:
            True если бот работает
        """
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/getMe"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        bot_info = data.get('result', {})
                        logger.info(f"✅ [TELEGRAM] Bot connected: @{bot_info.get('username')}")
                        return True
                    else:
                        logger.error(f"❌ [TELEGRAM] Bot connection failed: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"❌ [TELEGRAM] Connection test failed: {e}")
            return False


# Пример использования
async def main():
    """Пример использования Telegram бота."""
    import os
    from dotenv import load_dotenv

    # Загружаем .env
    load_dotenv()

    # Читаем настройки из .env
    BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
    CHAT_ID = os.getenv("TG_CHAT_ID", "")

    if not BOT_TOKEN or not CHAT_ID:
        print("❌ Error: TG_BOT_TOKEN and TG_CHAT_ID must be set in .env file")
        print("\nAdd to your .env file:")
        print("TG_BOT_TOKEN=your_bot_token_from_botfather")
        print("TG_CHAT_ID=your_chat_id_from_userinfobot")
        print("\nSee TELEGRAM_SETUP.md for detailed instructions")
        return

    print(f"🤖 Testing Telegram bot...")
    print(f"Token: {BOT_TOKEN[:10]}...{BOT_TOKEN[-5:]}")
    print(f"Chat ID: {CHAT_ID}")
    print()

    # Создаем бота
    bot = TelegramDashboardBot(BOT_TOKEN, CHAT_ID)

    # Тестируем соединение
    if await bot.test_connection():
        print()
        # Отправляем тестовое сообщение
        success = await bot.send_message(
            "🤖 <b>Trading Bot Connected!</b>\n\n"
            "✅ Telegram integration is working!\n"
            "Dashboard updates will be sent to this chat.\n\n"
            "<i>This is a test message from your trading bot.</i>",
            parse_mode="HTML"
        )

        if success:
            print("✅ Test message sent successfully!")
            print("📱 Check your Telegram group/chat for the message")
        else:
            print("❌ Failed to send test message")
            print("Check that bot is added to the group and has send permissions")
    else:
        print("❌ Failed to connect to Telegram bot")
        print("Check your TG_BOT_TOKEN in .env file")


if __name__ == "__main__":
    asyncio.run(main())
