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
from typing import Optional, Dict, Any, Callable
from loguru import logger


class TelegramUpdateHandler:
    """Handles incoming updates from Telegram (commands, callbacks, messages)."""

    def __init__(self, bot: 'TelegramDashboardBot', trading_engine: Any = None):
        self.bot = bot
        self.trading_engine = trading_engine
        self.last_update_id = 0
        self.running = False

        # Callback handlers
        self.callback_handlers: Dict[str, Callable] = {}
        self.command_handlers: Dict[str, Callable] = {}

        # Register default handlers
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register default command and callback handlers."""
        # Commands
        self.command_handlers['/start'] = self.handle_start_command
        self.command_handlers['/help'] = self.handle_help_command
        self.command_handlers['/menu'] = self.handle_menu_command
        self.command_handlers['/status'] = self.handle_status_command
        self.command_handlers['/positions'] = self.handle_positions_command
        self.command_handlers['/stats'] = self.handle_stats_command

        # Callback queries
        self.callback_handlers['menu_main'] = self.handle_menu_main
        self.callback_handlers['menu_portfolio'] = self.handle_menu_portfolio
        self.callback_handlers['menu_stats'] = self.handle_menu_stats
        self.callback_handlers['menu_trades'] = self.handle_menu_trades
        self.callback_handlers['menu_history'] = self.handle_menu_history
        self.callback_handlers['menu_settings'] = self.handle_menu_settings
        self.callback_handlers['menu_wallet'] = self.handle_menu_wallet
        self.callback_handlers['menu_refresh'] = self.handle_menu_refresh

        # Portfolio sub-menus
        self.callback_handlers['portfolio_details'] = self.handle_portfolio_details
        self.callback_handlers['portfolio_chart'] = self.handle_portfolio_chart

        # Stats sub-menus
        self.callback_handlers['stats_best'] = self.handle_stats_best
        self.callback_handlers['stats_worst'] = self.handle_stats_worst
        self.callback_handlers['stats_daily'] = self.handle_stats_daily
        self.callback_handlers['stats_weekly'] = self.handle_stats_weekly

    async def start_polling(self):
        """Start polling for updates."""
        self.running = True
        logger.info("📱 [TELEGRAM] Starting update polling...")

        while self.running:
            try:
                updates = await self.get_updates()

                for update in updates:
                    await self.process_update(update)

            except Exception as e:
                logger.error(f"📱 [TELEGRAM] Error in polling: {e}")

            await asyncio.sleep(1)  # Poll every second

    async def stop_polling(self):
        """Stop polling for updates."""
        self.running = False
        logger.info("📱 [TELEGRAM] Stopped update polling")

    async def get_updates(self) -> list:
        """Get updates from Telegram."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.bot.base_url}/getUpdates"
                params = {
                    "offset": self.last_update_id + 1,
                    "timeout": 30,
                }

                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=35)) as response:
                    if response.status == 200:
                        data = await response.json()
                        updates = data.get("result", [])

                        if updates:
                            self.last_update_id = updates[-1]["update_id"]

                        return updates
                    return []
        except Exception as e:
            logger.debug(f"📱 [TELEGRAM] Get updates error: {e}")
            return []

    async def process_update(self, update: Dict[str, Any]):
        """Process incoming update."""
        try:
            # Handle callback query (button press)
            if "callback_query" in update:
                await self.handle_callback_query(update["callback_query"])

            # Handle command
            elif "message" in update:
                message = update["message"]
                if "text" in message and message["text"].startswith("/"):
                    await self.handle_command(message)

        except Exception as e:
            logger.error(f"📱 [TELEGRAM] Error processing update: {e}")

    async def handle_callback_query(self, callback: Dict[str, Any]):
        """Handle callback query from inline button."""
        try:
            callback_id = callback["id"]
            data = callback.get("data", "")
            message = callback.get("message", {})
            message_id = message.get("message_id")

            logger.info(f"📱 [TELEGRAM] Callback: {data}")

            # Answer callback query (removes loading state)
            await self.answer_callback_query(callback_id)

            # Execute handler
            if data in self.callback_handlers:
                await self.callback_handlers[data](message_id)
            else:
                logger.warning(f"📱 [TELEGRAM] No handler for callback: {data}")

        except Exception as e:
            logger.error(f"📱 [TELEGRAM] Error handling callback: {e}")

    async def handle_command(self, message: Dict[str, Any]):
        """Handle command message."""
        try:
            text = message.get("text", "")
            command = text.split()[0].lower()

            logger.info(f"📱 [TELEGRAM] Command: {command}")

            if command in self.command_handlers:
                await self.command_handlers[command]()
            else:
                await self.bot.send_message(f"Unknown command: {command}\nUse /help for available commands")

        except Exception as e:
            logger.error(f"📱 [TELEGRAM] Error handling command: {e}")

    async def answer_callback_query(self, callback_id: str, text: str = ""):
        """Answer callback query."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.bot.base_url}/answerCallbackQuery"
                data = {"callback_query_id": callback_id}
                if text:
                    data["text"] = text

                await session.post(url, json=data)
        except Exception as e:
            logger.debug(f"Error answering callback: {e}")

    # Command Handlers
    async def handle_start_command(self):
        """Handle /start command."""
        welcome_text = """
🤖 <b>Welcome to AI Trading Bot!</b>

This bot helps you monitor and manage your automated trading.

<b>Available commands:</b>
/menu - Show main menu
/status - Current bot status
/positions - Open positions
/stats - Trading statistics
/help - Show this help

Use /menu to start navigating with buttons!
        """
        await self.bot.send_message(welcome_text)

    async def handle_help_command(self):
        """Handle /help command."""
        help_text = """
<b>📚 Available Commands:</b>

/start - Welcome message
/menu - Show interactive menu
/status - Bot status and account info
/positions - List open positions
/stats - Trading statistics
/help - This help message

<b>🎯 Tips:</b>
• Use /menu for button navigation
• Dashboard updates automatically every 5 min
• Tap buttons to navigate menus
        """
        await self.bot.send_message(help_text)

    async def handle_menu_command(self):
        """Handle /menu command."""
        await self.bot.send_main_menu(webapp_url=self.bot.webapp_url)

    async def handle_status_command(self):
        """Handle /status command."""
        if not self.trading_engine:
            await self.bot.send_message("❌ Trading engine not available")
            return

        try:
            balance = getattr(self.trading_engine, 'equity_usdt', 0.0)
            running = getattr(self.trading_engine, 'running', False)

            status_text = f"""
<b>🤖 Bot Status</b>

Status: {'🟢 Running' if running else '🔴 Stopped'}
Balance: ${balance:,.2f} USDT

Use /menu for detailed info
            """
            await self.bot.send_message(status_text)
        except Exception as e:
            await self.bot.send_message(f"Error getting status: {e}")

    async def handle_positions_command(self):
        """Handle /positions command."""
        await self.handle_menu_trades(None)

    async def handle_stats_command(self):
        """Handle /stats command."""
        await self.handle_menu_stats(None)

    # Callback Handlers
    async def handle_menu_main(self, message_id: Optional[int]):
        """Handle main menu callback."""
        await self.bot.send_main_menu(webapp_url=self.bot.webapp_url)

    async def handle_menu_portfolio(self, message_id: Optional[int]):
        """Handle portfolio menu callback."""
        if not self.trading_engine:
            return

        try:
            # Get current balance (updated value)
            balance = getattr(self.trading_engine, 'equity_usdt', 0.0)

            # Get initial balance for ROI calculation
            initial = getattr(self.trading_engine, 'initial_equity', None)
            if initial is None:
                initial = getattr(self.trading_engine.config, 'paper_equity', 1000.0)

            # Calculate P&L and ROI
            total_pnl = balance - initial
            roi_pct = ((balance - initial) / initial * 100) if initial > 0 else 0.0

            portfolio_data = {
                'balance': balance,
                'equity': balance,
                'total_pnl': total_pnl,
                'roi_pct': roi_pct,
            }

            # Try to get real data from portfolio tracker
            if hasattr(self.trading_engine, 'portfolio_tracker') and self.trading_engine.portfolio_tracker:
                try:
                    stats = self.trading_engine.portfolio_tracker.get_stats()
                    if stats:
                        portfolio_data['total_pnl'] = stats.get('total_pnl', total_pnl)
                        # portfolio_tracker might have better P&L data
                except Exception:
                    pass

            await self.bot.send_portfolio_menu(portfolio_data)
        except Exception as e:
            await self.bot.send_message(f"Error loading portfolio: {e}")

    async def handle_menu_stats(self, message_id: Optional[int]):
        """Handle stats menu callback."""
        if not self.trading_engine:
            return

        try:
            stats_data = {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
            }

            if hasattr(self.trading_engine, 'portfolio_tracker') and self.trading_engine.portfolio_tracker:
                try:
                    stats = self.trading_engine.portfolio_tracker.get_stats()
                    if stats:
                        stats_data['total_trades'] = stats.get('total_trades', 0)
                        winning = stats.get('winning_trades', 0)
                        total = stats_data['total_trades']
                        stats_data['win_rate'] = (winning / total) if total > 0 else 0.0
                        # FIXED: Добавлено извлечение profit_factor и sharpe_ratio
                        stats_data['profit_factor'] = stats.get('profit_factor', 0.0)
                        stats_data['sharpe_ratio'] = stats.get('sharpe_ratio', 0.0)
                except Exception as e:
                    logger.error(f"📱 [TELEGRAM] Error getting stats from portfolio_tracker: {e}")

            await self.bot.send_stats_menu(stats_data)
        except Exception as e:
            await self.bot.send_message(f"Error loading stats: {e}")

    async def handle_menu_trades(self, message_id: Optional[int]):
        """Handle active trades menu callback."""
        if not self.trading_engine:
            await self.bot.send_message("❌ Trading engine not available")
            return

        try:
            positions = []
            if hasattr(self.trading_engine, 'active_positions'):
                positions = list(self.trading_engine.active_positions.keys())

            if positions:
                text = "<b>📝 ACTIVE POSITIONS</b>\n\n"
                for symbol in positions[:10]:  # Max 10
                    text += f"• {symbol}\n"

                if len(positions) > 10:
                    text += f"\n<i>... and {len(positions) - 10} more</i>"
            else:
                text = "<b>📝 ACTIVE POSITIONS</b>\n\n<i>No open positions</i>"

            keyboard = [[{"text": "🔙 Назад", "callback_data": "menu_main"}]]
            await self.bot.send_message_with_keyboard(text, keyboard)
        except Exception as e:
            await self.bot.send_message(f"Error loading positions: {e}")

    async def handle_menu_history(self, message_id: Optional[int]):
        """Handle history menu callback."""
        if not self.trading_engine:
            text = "<b>📜 TRADE HISTORY</b>\n\n<i>Trading engine not available</i>"
            keyboard = [[{"text": "🔙 Назад", "callback_data": "menu_main"}]]
            await self.bot.send_message_with_keyboard(text, keyboard)
            return

        try:
            # Try to get trade history from portfolio tracker
            recent_trades = []
            if hasattr(self.trading_engine, 'portfolio_tracker') and self.trading_engine.portfolio_tracker:
                try:
                    stats = self.trading_engine.portfolio_tracker.get_stats()
                    if stats:
                        total_trades = stats.get('total_trades', 0)
                        winning_trades = stats.get('winning_trades', 0)
                        losing_trades = stats.get('losing_trades', 0)
                    else:
                        total_trades = winning_trades = losing_trades = 0
                except Exception:
                    total_trades = winning_trades = losing_trades = 0
            else:
                total_trades = winning_trades = losing_trades = 0

            text = f"""
<b>📜 TRADE HISTORY</b>

<b>📊 Overall Statistics:</b>
Total Trades: {total_trades}
✅ Winning: {winning_trades}
❌ Losing: {losing_trades}

━━━━━━━━━━━━━━━━━━━━
💡 For detailed trade history, check:
• Enhanced Dashboard (/dashboard)
• Portfolio History (data/portfolio_history.json)
            """
            keyboard = [[{"text": "🔙 Назад", "callback_data": "menu_main"}]]
            await self.bot.send_message_with_keyboard(text, keyboard)
        except Exception as e:
            text = f"<b>📜 TRADE HISTORY</b>\n\n<i>Error loading history: {e}</i>"
            keyboard = [[{"text": "🔙 Назад", "callback_data": "menu_main"}]]
            await self.bot.send_message_with_keyboard(text, keyboard)

    async def handle_menu_settings(self, message_id: Optional[int]):
        """Handle settings menu callback."""
        if not self.trading_engine:
            text = "<b>⚙️ SETTINGS</b>\n\n<i>Trading engine not available</i>"
            keyboard = [[{"text": "🔙 Назад", "callback_data": "menu_main"}]]
            await self.bot.send_message_with_keyboard(text, keyboard)
            return

        try:
            # Get bot configuration
            config = self.trading_engine.config
            leverage = getattr(config, 'leverage', 'N/A')
            symbols = getattr(config, 'symbols', [])
            mode = getattr(config, 'paper_mode', True)

            mode_emoji = "📝" if mode else "💰"
            mode_text = "Paper Trading" if mode else "Live Trading"

            text = f"""
<b>⚙️ BOT SETTINGS</b>

<b>🔧 Configuration:</b>
Mode: {mode_emoji} <b>{mode_text}</b>
Leverage: <b>{leverage}x</b>
Symbols: <b>{len(symbols)}</b> pairs

<b>📊 Monitored Pairs:</b>
{', '.join(symbols[:5])}
{'...' if len(symbols) > 5 else ''}

━━━━━━━━━━━━━━━━━━━━
💡 To change settings, edit your config file
            """
            keyboard = [[{"text": "🔙 Назад", "callback_data": "menu_main"}]]
            await self.bot.send_message_with_keyboard(text, keyboard)
        except Exception as e:
            text = f"<b>⚙️ SETTINGS</b>\n\n<i>Error loading settings: {e}</i>"
            keyboard = [[{"text": "🔙 Назад", "callback_data": "menu_main"}]]
            await self.bot.send_message_with_keyboard(text, keyboard)

    async def handle_menu_wallet(self, message_id: Optional[int]):
        """Handle wallet menu callback."""
        if not self.trading_engine:
            return

        balance = getattr(self.trading_engine, 'equity_usdt', 0.0)
        initial_equity = getattr(self.trading_engine, 'initial_equity', balance)
        total_pnl = balance - initial_equity
        roi_pct = ((balance - initial_equity) / initial_equity * 100) if initial_equity > 0 else 0.0
        open_positions = len(getattr(self.trading_engine, 'active_positions', {}))

        pnl_emoji = "💰" if total_pnl >= 0 else "📉"
        roi_emoji = "🟢" if roi_pct >= 0 else "🔴"

        text = f"""
<b>💰 WALLET OVERVIEW</b>

━━━━━━━━━━━━━━━━━━━━
<b>💵 Current Balance</b>
${balance:,.2f} USDT

<b>💎 Initial Equity</b>
${initial_equity:,.2f} USDT

<b>{pnl_emoji} Total P&L</b>
${total_pnl:+,.2f} USDT

<b>{roi_emoji} ROI</b>
{roi_pct:+.2f}%

<b>📊 Open Positions</b>
{open_positions} position{'s' if open_positions != 1 else ''}

━━━━━━━━━━━━━━━━━━━━
💡 Use /dashboard for real-time updates
        """
        keyboard = [[{"text": "🔙 Назад", "callback_data": "menu_main"}]]
        await self.bot.send_message_with_keyboard(text, keyboard)

    async def handle_menu_refresh(self, message_id: Optional[int]):
        """Handle refresh callback."""
        await self.bot.send_message("🔄 Refreshing...")
        await self.handle_menu_main(None)

    async def handle_portfolio_details(self, message_id: Optional[int]):
        """Handle portfolio details callback."""
        if not self.trading_engine:
            await self.bot.send_message("❌ Trading engine not available")
            return

        try:
            balance = getattr(self.trading_engine, 'equity_usdt', 0.0)

            # Get positions from active_positions
            positions = getattr(self.trading_engine, 'active_positions', {})

            text = f"""
<b>💼 PORTFOLIO DETAILS</b>

<b>💵 Balance:</b> ${balance:,.2f} USDT
<b>📊 Open Positions:</b> {len(positions)}

<b>📈 Recent Activity:</b>
<i>Monitoring {len(getattr(self.trading_engine.config, 'symbols', []))} symbols</i>

━━━━━━━━━━━━━━━━━━━━
            """

            keyboard = [[{"text": "🔙 Назад", "callback_data": "menu_portfolio"}]]
            await self.bot.send_message_with_keyboard(text, keyboard)
        except Exception as e:
            await self.bot.send_message(f"Error loading details: {e}")

    async def handle_portfolio_chart(self, message_id: Optional[int]):
        """Handle portfolio chart callback."""
        text = """
<b>📈 PORTFOLIO CHART</b>

✅ <b>Real-time charts are available!</b>

Use the <b>🚀 Enhanced Dashboard</b> button in the main menu to view:
• 📊 Live equity chart
• 📈 Real-time position updates
• 💰 P&L tracking
• 🔴 WebSocket streaming (< 1 sec latency)

Or visit: /dashboard

━━━━━━━━━━━━━━━━━━━━
        """
        keyboard = [[{"text": "🔙 Назад", "callback_data": "menu_portfolio"}]]
        await self.bot.send_message_with_keyboard(text, keyboard)

    async def handle_stats_best(self, message_id: Optional[int]):
        """Handle best trades callback."""
        if not self.trading_engine:
            await self.bot.send_message("❌ Trading engine not available")
            return

        try:
            text = """
<b>🏆 BEST TRADES</b>

<i>Loading best trades data...</i>

━━━━━━━━━━━━━━━━━━━━
💡 For detailed trade analysis, check:
• Enhanced Dashboard (/dashboard)
• Trade History in Portfolio section
            """
            keyboard = [[{"text": "🔙 Назад", "callback_data": "menu_stats"}]]
            await self.bot.send_message_with_keyboard(text, keyboard)
        except Exception as e:
            await self.bot.send_message(f"Error loading best trades: {e}")

    async def handle_stats_worst(self, message_id: Optional[int]):
        """Handle worst trades callback."""
        if not self.trading_engine:
            await self.bot.send_message("❌ Trading engine not available")
            return

        try:
            text = """
<b>💔 WORST TRADES</b>

<i>Loading worst trades data...</i>

━━━━━━━━━━━━━━━━━━━━
💡 For detailed trade analysis, check:
• Enhanced Dashboard (/dashboard)
• Trade History in Portfolio section
            """
            keyboard = [[{"text": "🔙 Назад", "callback_data": "menu_stats"}]]
            await self.bot.send_message_with_keyboard(text, keyboard)
        except Exception as e:
            await self.bot.send_message(f"Error loading worst trades: {e}")

    async def handle_stats_daily(self, message_id: Optional[int]):
        """Handle daily stats callback."""
        if not self.trading_engine:
            await self.bot.send_message("❌ Trading engine not available")
            return

        try:
            text = """
<b>📅 DAILY STATISTICS</b>

<i>Loading daily performance data...</i>

━━━━━━━━━━━━━━━━━━━━
💡 For detailed daily breakdown, check:
• Enhanced Dashboard (/dashboard)
• Performance charts and history
            """
            keyboard = [[{"text": "🔙 Назад", "callback_data": "menu_stats"}]]
            await self.bot.send_message_with_keyboard(text, keyboard)
        except Exception as e:
            await self.bot.send_message(f"Error loading daily stats: {e}")

    async def handle_stats_weekly(self, message_id: Optional[int]):
        """Handle weekly stats callback."""
        if not self.trading_engine:
            await self.bot.send_message("❌ Trading engine not available")
            return

        try:
            text = """
<b>📆 WEEKLY STATISTICS</b>

<i>Loading weekly performance data...</i>

━━━━━━━━━━━━━━━━━━━━
💡 For detailed weekly breakdown, check:
• Enhanced Dashboard (/dashboard)
• Performance charts and history
            """
            keyboard = [[{"text": "🔙 Назад", "callback_data": "menu_stats"}]]
            await self.bot.send_message_with_keyboard(text, keyboard)
        except Exception as e:
            await self.bot.send_message(f"Error loading weekly stats: {e}")


class TelegramDashboardBot:
    """Telegram бот для отправки обновлений дашборда."""

    def __init__(self, token: str, chat_id: str, webapp_url: str = None):
        """
        Инициализация Telegram бота.

        Args:
            token: Bot token от @BotFather
            chat_id: ID группы/канала (можно получить от @userinfobot)
            webapp_url: URL Telegram Web App для интерактивного дашборда (опционально)
        """
        self.token = token
        self.chat_id = chat_id
        self.webapp_url = webapp_url
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

    async def send_tp_sl_triggered(self, order_info: Dict[str, Any]) -> bool:
        """
        Отправить уведомление о срабатывании TP/SL ордера.

        Args:
            order_info: {
                'symbol': str,
                'side': str (LONG/SHORT),
                'order_type': str (TP/SL),
                'trigger_price': float,
                'entry_price': float,
                'quantity': float,
                'pnl': float (estimated),
                'level': int (TP1, TP2, etc)
            }

        Returns:
            True если успешно отправлено
        """
        try:
            message = self._format_tp_sl_triggered_message(order_info)
            return await self.send_message(message, parse_mode="HTML")
        except Exception as e:
            logger.error(f"❌ [TELEGRAM] Error sending TP/SL notification: {e}")
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
        """Форматирует сообщение об открытии позиции с мотивирующим дизайном."""
        from datetime import datetime

        side_emoji = "🟢" if trade['side'] == 'LONG' else "🔴"
        direction_emoji = "📈" if trade['side'] == 'LONG' else "📉"

        message = f"""╔════════════════════════╗
║  <b>⚡ NEW TRADE LIVE!</b>  ║
╚════════════════════════╝

<b>{side_emoji} {trade['symbol']}  │  {trade['side']} {direction_emoji}</b>
⏰ {datetime.now().strftime('%H:%M:%S UTC')}
💬 <i>Position is now active and monitored!</i>

╭─────────────────────╮
│ <b>📊 ENTRY INFO</b>       │
╰─────────────────────╯

<b>Entry Price:</b> ${trade['entry_price']:,.4f}
<b>Quantity:</b> {trade['quantity']:.4f}
<b>Position Size:</b> ${trade.get('notional', 0):,.2f}
"""

        # Show leverage info prominently
        leverage = trade.get('leverage', 1)
        margin = trade.get('margin_used', 0)
        if leverage > 1:
            message += f"\n<b>⚡ Leverage:</b> {leverage:.0f}x"
            if margin > 0:
                message += f" | <b>Margin:</b> ${margin:,.2f}"
                # Show potential
                potential_profit = margin * leverage * 0.05  # 5% move example
                message += f"\n<i>💡 5% move = ~${potential_profit:,.2f}</i>"

        # Protection orders with visual appeal
        has_protections = trade.get('stop_loss') or trade.get('take_profit')
        if has_protections:
            message += "\n\n╭─────────────────────╮\n│ <b>🛡️ PROTECTION SETUP</b> │\n╰─────────────────────╯\n"

            if trade.get('take_profit'):
                tp_dist = trade.get('tp_distance', 0)
                tp_count = trade.get('tp_count', 1)
                tp_badge = f" ({tp_count} levels)" if tp_count > 1 else ""

                # Calculate potential profit
                if margin > 0 and tp_dist != 0:
                    potential_roi = abs(tp_dist) * leverage
                    potential_profit = margin * potential_roi / 100
                    message += f"\n💎 <b>Take Profit{tp_badge}:</b> ${trade['take_profit']:,.4f}"
                    message += f"\n   🎯 Target: <b>{tp_dist:+.2f}%</b> → ROI: ~{potential_roi:.1f}% (~${potential_profit:,.2f})"
                else:
                    message += f"\n💎 <b>Take Profit{tp_badge}:</b> ${trade['take_profit']:,.4f}"
                    message += f"\n   🎯 Target: <b>{tp_dist:+.2f}%</b>"

            if trade.get('stop_loss'):
                sl_dist = trade.get('sl_distance', 0)

                # Calculate risk
                if margin > 0 and sl_dist != 0:
                    risk_roi = abs(sl_dist) * leverage
                    risk_amount = margin * risk_roi / 100
                    message += f"\n\n🛡️ <b>Stop Loss:</b> ${trade['stop_loss']:,.4f}"
                    message += f"\n   ⚠️ Distance: <b>{sl_dist:+.2f}%</b> → Risk: ~{risk_roi:.1f}% (~${risk_amount:,.2f})"
                else:
                    message += f"\n\n🛡️ <b>Stop Loss:</b> ${trade['stop_loss']:,.4f}"
                    message += f"\n   ⚠️ Distance: <b>{sl_dist:+.2f}%</b>"

        # ALWAYS show balance
        message += "\n\n╭─────────────────────╮\n│ <b>💰 ACCOUNT STATUS</b>   │\n╰─────────────────────╯\n"
        balance = trade.get('account_balance')
        if balance is not None:
            message += f"\n<b>💵 Balance:</b> {balance:,.2f} USDT"
            # Show position as % of balance
            if margin > 0 and balance > 0:
                position_pct = (margin / balance * 100)
                message += f"\n<b>📊 Position Size:</b> {position_pct:.2f}% of balance"
        else:
            message += f"\n<b>💵 Balance:</b> <i>Loading...</i>"

        # Signal info
        if trade.get('reason'):
            message += f"\n\n📡 <i>{trade['reason']}</i>"

        message += "\n\n╚════════════════════════╝"

        return message

    def _format_trade_closed_message(self, trade: Dict[str, Any]) -> str:
        """Форматирует сообщение о закрытии позиции с мотивирующим дизайном."""
        from datetime import datetime

        # Smart emojis and motivational messages based on ROI
        pnl = trade.get('pnl', 0)
        pnl_pct = trade.get('pnl_pct', 0)
        roi_pct = trade.get('roi_pct', pnl_pct)

        if roi_pct >= 0:
            if roi_pct > 15:
                result_emoji = "🌟"
                status = "INCREDIBLE WIN"
                motivation = "Outstanding execution! 🏆"
                bar_color = "█"
            elif roi_pct > 10:
                result_emoji = "🚀"
                status = "HUGE WIN"
                motivation = "Excellent profit! Keep it up! 💪"
                bar_color = "█"
            elif roi_pct > 5:
                result_emoji = "🎯"
                status = "GREAT WIN"
                motivation = "Solid performance! 👍"
                bar_color = "█"
            elif roi_pct > 2:
                result_emoji = "✅"
                status = "WIN"
                motivation = "Nice profit! Building wealth! 💰"
                bar_color = "▓"
            else:
                result_emoji = "✔️"
                status = "PROFIT"
                motivation = "Every win counts! 📈"
                bar_color = "▓"
        else:
            if roi_pct < -10:
                result_emoji = "🛡️"
                status = "STOPPED"
                motivation = "Protected capital. Next one! 🎯"
                bar_color = "░"
            elif roi_pct < -5:
                result_emoji = "⚠️"
                status = "CLOSED"
                motivation = "Learning opportunity! 📚"
                bar_color = "░"
            else:
                result_emoji = "📊"
                status = "EXITED"
                motivation = "Small setback. Stay focused! 🎓"
                bar_color = "░"

        side_emoji = "🟢" if trade['side'] == 'LONG' else "🔴"

        message = f"""╔════════════════════════╗
║  <b>{result_emoji} {status}</b>  ║
╚════════════════════════╝

<b>{side_emoji} {trade['symbol']}  │  {trade['side']}</b>
⏰ {datetime.now().strftime('%H:%M:%S UTC')}

╭─────────────────────╮
│ <b>📊 PERFORMANCE</b>      │
╰─────────────────────╯
"""

        # Result with beautiful progress bar
        pnl_bar = ""
        bar_length = 15
        if roi_pct >= 0:
            filled = min(int(abs(roi_pct) / 1.5), bar_length)
            pnl_bar = bar_color * filled + "░" * (bar_length - filled)
        else:
            filled = min(int(abs(roi_pct) / 1.5), bar_length)
            pnl_bar = bar_color * filled + "·" * (bar_length - filled)

        pnl_emoji = "💰" if pnl >= 0 else "📉"
        message += f"\n{pnl_emoji} <b>Profit/Loss:</b> ${pnl:+,.2f} USDT"
        message += f"\n📈 <b>ROI:</b> {roi_pct:+.2f}%"
        message += f"\n[{pnl_bar}] {abs(roi_pct):.1f}%"
        message += f"\n\n💬 <i>{motivation}</i>"

        # Trade details
        message += "\n\n╭─────────────────────╮\n│ <b>📋 TRADE DETAILS</b>   │\n╰─────────────────────╯\n"
        message += f"\n<b>Entry Price:</b> ${trade['entry_price']:,.4f}"
        message += f"\n<b>Exit Price:</b> ${trade['exit_price']:,.4f}"

        # Price movement with direction
        price_change = trade['exit_price'] - trade['entry_price']
        price_change_pct = (price_change / trade['entry_price'] * 100) if trade['entry_price'] > 0 else 0
        if trade['side'] == 'LONG':
            if price_change >= 0:
                change_emoji = "⬆️"
                change_status = "In our favor"
            else:
                change_emoji = "⬇️"
                change_status = "Against us"
        else:  # SHORT
            if price_change <= 0:
                change_emoji = "⬇️"
                change_status = "In our favor"
            else:
                change_emoji = "⬆️"
                change_status = "Against us"
        message += f"\n{change_emoji} <b>Move:</b> {price_change_pct:+.2f}% <i>({change_status})</i>"

        message += f"\n<b>Quantity:</b> {trade['quantity']:.4f}"

        # Show leverage and margin
        if trade.get('leverage'):
            message += f"\n<b>Leverage:</b> {trade['leverage']}x"
            if trade.get('margin_used'):
                message += f" | <b>Margin:</b> ${trade['margin_used']:.2f}"

        # Duration
        if trade.get('duration'):
            message += f"\n⏱️ <b>Duration:</b> {trade['duration']}"

        # Show protection orders that were set
        tp_orders = trade.get('tp_orders', [])
        sl_orders = trade.get('sl_orders', [])
        if tp_orders or sl_orders:
            message += "\n\n╭─────────────────────╮\n│ <b>🛡️ PROTECTIONS SET</b> │\n╰─────────────────────╯\n"

            if tp_orders:
                tp_count = len(tp_orders)
                tp_badge = f" ({tp_count} levels)" if tp_count > 1 else ""
                first_tp = tp_orders[0] if tp_orders else 0
                tp_dist = ((first_tp - trade['entry_price']) / trade['entry_price'] * 100) if trade['entry_price'] > 0 else 0
                message += f"\n💎 <b>Take Profit{tp_badge}:</b> ${first_tp:,.4f} ({tp_dist:+.2f}%)"

            if sl_orders:
                first_sl = sl_orders[0] if sl_orders else 0
                sl_dist = ((first_sl - trade['entry_price']) / trade['entry_price'] * 100) if trade['entry_price'] > 0 else 0
                message += f"\n🛡️ <b>Stop Loss:</b> ${first_sl:,.4f} ({sl_dist:+.2f}%)"

        # Exit reason with emoji
        if trade.get('reason'):
            reason = trade['reason']
            if "Manual" in reason:
                reason_emoji = "👤"
                reason_text = "Manual Close"
            elif "Profit" in reason:
                reason_emoji = "🎯"
                reason_text = "Take Profit Triggered"
            elif "Stop" in reason:
                reason_emoji = "🛡️"
                reason_text = "Stop Loss Triggered"
            else:
                reason_emoji = "📝"
                reason_text = reason
            message += f"\n\n{reason_emoji} <b>Exit Reason:</b> <i>{reason_text}</i>"

        # ALWAYS show balance - this is critical!
        message += "\n\n╭─────────────────────╮\n│ <b>💰 ACCOUNT STATUS</b>   │\n╰─────────────────────╯\n"
        balance = trade.get('account_balance')
        if balance is not None:
            message += f"\n<b>💵 Balance:</b> {balance:,.2f} USDT"
            # Show P&L impact
            balance_impact = (pnl / balance * 100) if balance > 0 else 0
            if abs(balance_impact) >= 0.01:
                impact_emoji = "📈" if balance_impact > 0 else "📉"
                message += f"\n{impact_emoji} <b>Impact:</b> {balance_impact:+.2f}% of balance"
        else:
            message += f"\n<b>💵 Balance:</b> <i>Loading...</i>"

        message += "\n\n╚════════════════════════╝"

        return message

    def _format_tp_sl_triggered_message(self, order: Dict[str, Any]) -> str:
        """Форматирует сообщение о срабатывании TP/SL ордера."""
        from datetime import datetime

        order_type = order.get('order_type', 'TP')
        side_emoji = "🎯" if order['side'] == 'LONG' else "🎲"

        if order_type == 'TP':
            event_emoji = "💎"
            event_title = "TAKE PROFIT HIT"
            level_info = f"TP{order.get('level', 1)}"
        else:
            event_emoji = "🛡️"
            event_title = "STOP LOSS HIT"
            level_info = "SL"

        pnl = order.get('pnl', 0)
        entry_price = order.get('entry_price', 0)
        trigger_price = order.get('trigger_price', 0)
        quantity = order.get('quantity', 0)

        # Calculate percentage
        pnl_pct = (pnl / (entry_price * quantity) * 100) if entry_price * quantity > 0 else 0

        message = f"""╔════════════════════════╗
║  <b>{event_emoji} {event_title}</b>  ║
╚════════════════════════╝

<b>{side_emoji}  {order['side']}  {order['symbol']}</b>  {level_info}
⏰ <i>{datetime.now().strftime('%H:%M:%S UTC')}</i>

╭─────────────────────╮
│ <b>📊 ORDER DETAILS</b>    │
╰─────────────────────╯

<b>Entry:</b> ${entry_price:,.4f}
<b>Trigger:</b> ${trigger_price:,.4f}
<b>Qty:</b> {quantity:.4f}
"""

        # Price movement
        price_change = trigger_price - entry_price
        price_change_pct = (price_change / entry_price * 100) if entry_price > 0 else 0
        if price_change >= 0:
            change_emoji = "⬆️" if price_change_pct > 2 else "↗️"
        else:
            change_emoji = "⬇️" if price_change_pct < -2 else "↘️"
        message += f"\n{change_emoji} <b>Move:</b> ${price_change:+,.4f} ({price_change_pct:+.2f}%)"

        # PnL estimate
        pnl_emoji = "💰" if pnl >= 0 else "📉"
        message += f"\n\n╭─────────────────────╮\n│ <b>{pnl_emoji} PARTIAL RESULT</b>  │\n╰─────────────────────╯\n"
        message += f"\n<b>Est. P&L:</b> ${pnl:+,.2f} USDT"
        message += f"\n<b>Est. ROI:</b> {pnl_pct:+.2f}%"

        # ALWAYS show balance at the bottom
        message += "\n\n╭─────────────────────╮\n│ <b>💰 ACCOUNT STATUS</b>   │\n╰─────────────────────╯\n"
        if order.get('account_balance'):
            message += f"\n<b>Balance:</b> {order['account_balance']:,.2f} USDT"
        else:
            message += f"\n<b>Balance:</b> <i>Not available</i>"

        message += "\n\n╚════════════════════════╝"

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

    async def send_message_with_keyboard(
        self, text: str, keyboard: list, parse_mode: str = "HTML"
    ) -> bool:
        """
        Отправить сообщение с Inline клавиатурой.

        Args:
            text: Текст сообщения
            keyboard: Inline клавиатура (список списков кнопок)
                     Формат: [[{"text": "Button 1", "callback_data": "btn1"}, ...], ...]
            parse_mode: Режим парсинга ("HTML" или "Markdown")

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
                    "reply_markup": {"inline_keyboard": keyboard},
                }

                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        logger.info("📤 [TELEGRAM] Message with keyboard sent successfully")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"❌ [TELEGRAM] Failed to send message with keyboard: {error_text}"
                        )
                        return False

        except Exception as e:
            logger.error(f"❌ [TELEGRAM] Error sending message with keyboard: {e}")
            return False

    async def send_main_menu(self, webapp_url: str = None) -> bool:
        """Отправить главное меню с Inline клавиатурой."""
        menu_text = """
<b>🤖 Trading Bot Menu</b>

Выберите действие:
        """

        keyboard = [
            [
                {"text": "📊 Портфолио", "callback_data": "menu_portfolio"},
                {"text": "📈 Статистика", "callback_data": "menu_stats"},
            ],
            [
                {"text": "📝 Активные сделки", "callback_data": "menu_trades"},
                {"text": "📜 История", "callback_data": "menu_history"},
            ],
            [
                {"text": "⚙️ Настройки", "callback_data": "menu_settings"},
                {"text": "💰 Кошелек", "callback_data": "menu_wallet"},
            ],
        ]

        # Add Web App buttons if URL is provided
        if webapp_url:
            # Enhanced dashboard (теперь на главной странице)
            keyboard.insert(0, [
                {"text": "🚀 Enhanced Dashboard", "web_app": {"url": webapp_url}}
            ])
            # Simple dashboard (простая версия на /simple)
            simple_url = webapp_url.rstrip('/') + '/simple'
            keyboard.insert(1, [
                {"text": "📱 Простой Dashboard", "web_app": {"url": simple_url}}
            ])

        keyboard.append([
            {"text": "🔄 Обновить", "callback_data": "menu_refresh"},
        ])

        return await self.send_message_with_keyboard(menu_text, keyboard)

    async def send_portfolio_menu(self, portfolio_data: Dict[str, Any]) -> bool:
        """Отправить меню портфолио."""
        balance = portfolio_data.get("balance", 0.0)
        equity = portfolio_data.get("equity", 0.0)
        pnl = portfolio_data.get("total_pnl", 0.0)
        roi = portfolio_data.get("roi_pct", 0.0)

        pnl_emoji = "💰" if pnl >= 0 else "📉"
        roi_emoji = "🟢" if roi >= 0 else "🔴"

        text = f"""
<b>💼 ПОРТФОЛИО</b>

💵 <b>Баланс:</b> ${balance:,.2f} USDT
💎 <b>Equity:</b> ${equity:,.2f} USDT
{pnl_emoji} <b>P&L:</b> ${pnl:+,.2f} ({roi:+.2f}%)
{roi_emoji} <b>ROI:</b> {roi:+.2f}%

━━━━━━━━━━━━━━━━━━━━
        """

        keyboard = [
            [
                {"text": "📊 Детали", "callback_data": "portfolio_details"},
                {"text": "📈 График", "callback_data": "portfolio_chart"},
            ],
            [
                {"text": "🔙 Назад", "callback_data": "menu_main"},
            ],
        ]

        return await self.send_message_with_keyboard(text, keyboard)

    async def send_stats_menu(self, stats_data: Dict[str, Any]) -> bool:
        """Отправить меню статистики."""
        total_trades = stats_data.get("total_trades", 0)
        win_rate = stats_data.get("win_rate", 0.0) * 100
        profit_factor = stats_data.get("profit_factor", 0.0)
        sharpe = stats_data.get("sharpe_ratio", 0.0)

        text = f"""
<b>📊 СТАТИСТИКА</b>

🔢 <b>Всего сделок:</b> {total_trades}
📈 <b>Win Rate:</b> {win_rate:.1f}%
💹 <b>Profit Factor:</b> {profit_factor:.2f}
📉 <b>Sharpe Ratio:</b> {sharpe:.2f}

━━━━━━━━━━━━━━━━━━━━
        """

        keyboard = [
            [
                {"text": "🏆 Лучшие сделки", "callback_data": "stats_best"},
                {"text": "💔 Худшие сделки", "callback_data": "stats_worst"},
            ],
            [
                {"text": "📅 По дням", "callback_data": "stats_daily"},
                {"text": "📆 По неделям", "callback_data": "stats_weekly"},
            ],
            [
                {"text": "🔙 Назад", "callback_data": "menu_main"},
            ],
        ]

        return await self.send_message_with_keyboard(text, keyboard)

    async def edit_message(
        self, message_id: int, text: str, keyboard: list = None, parse_mode: str = "HTML"
    ) -> bool:
        """
        Редактировать существующее сообщение.

        Args:
            message_id: ID сообщения для редактирования
            text: Новый текст
            keyboard: Новая Inline клавиатура (опционально)
            parse_mode: Режим парсинга

        Returns:
            True если успешно отредактировано
        """
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/editMessageText"
                data = {
                    "chat_id": self.chat_id,
                    "message_id": message_id,
                    "text": text,
                    "parse_mode": parse_mode,
                }

                if keyboard:
                    data["reply_markup"] = {"inline_keyboard": keyboard}

                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        logger.info("📝 [TELEGRAM] Message edited successfully")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ [TELEGRAM] Failed to edit message: {error_text}")
                        return False

        except Exception as e:
            logger.error(f"❌ [TELEGRAM] Error editing message: {e}")
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
