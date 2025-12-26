#!/usr/bin/env python3
"""
Real-time Trading Bot Monitor
Monitors log files and displays trading activity in real-time
"""

import time
import os
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import re

class BotMonitor:
    def __init__(self):
        self.log_files = {
            'main': 'logs/trading_bot.log',
            'errors': 'logs/errors.log',
            'trades': 'logs/trades.log',
            'performance': 'logs/performance.log',
            'root': 'trading_bot.log'
        }

        self.positions = {}
        self.file_positions = {}
        self.stats = defaultdict(int)

        # ANSI color codes
        self.colors = {
            'green': '\033[92m',
            'red': '\033[91m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'cyan': '\033[96m',
            'magenta': '\033[95m',
            'white': '\033[97m',
            'bold': '\033[1m',
            'end': '\033[0m'
        }

    def color(self, text, color_name):
        """Apply color to text"""
        return f"{self.colors.get(color_name, '')}{text}{self.colors['end']}"

    def parse_line(self, line, source):
        """Parse log line and extract important information"""

        # Trading signals
        if 'SIGNAL' in line or '📊' in line or 'Signal generated' in line.lower():
            self.stats['signals'] += 1
            return self.color(f"📊 SIGNAL: {line.strip()}", 'cyan')

        # Position opened
        if 'POSITION_OPENED' in line or 'opened long' in line.lower() or 'opened short' in line.lower():
            self.stats['positions_opened'] += 1
            match = re.search(r'(LONG|SHORT|long|short).*?(\w+USDT)', line, re.IGNORECASE)
            if match:
                side, symbol = match.groups()
                return self.color(f"🟢 POSITION OPENED: {side.upper()} {symbol}", 'green')
            return self.color(f"🟢 POSITION OPENED: {line.strip()}", 'green')

        # Position closed
        if 'POSITION_CLOSED' in line or 'closed position' in line.lower():
            self.stats['positions_closed'] += 1
            # Try to extract PnL
            pnl_match = re.search(r'pnl[:\s=]+(-?\d+\.?\d*)', line, re.IGNORECASE)
            if pnl_match:
                pnl = float(pnl_match.group(1))
                color = 'green' if pnl > 0 else 'red'
                symbol = '💰' if pnl > 0 else '📉'
                return self.color(f"{symbol} POSITION CLOSED: PnL = ${pnl:.2f} | {line.strip()}", color)
            return self.color(f"🔴 POSITION CLOSED: {line.strip()}", 'yellow')

        # Trade executed
        if 'TRADE' in line and ('BUY' in line or 'SELL' in line or 'LONG' in line or 'SHORT' in line):
            return self.color(f"💱 TRADE: {line.strip()}", 'magenta')

        # AI/ML updates
        if any(x in line for x in ['🧠', 'ML', 'AI_LEARNING', 'model', 'COMBO']):
            return self.color(f"🤖 AI: {line.strip()}", 'blue')

        # Errors
        if source == 'errors' or 'ERROR' in line or '❌' in line:
            self.stats['errors'] += 1
            return self.color(f"❌ ERROR: {line.strip()}", 'red')

        # Warnings
        if 'WARNING' in line or '⚠️' in line:
            self.stats['warnings'] += 1
            return self.color(f"⚠️  WARNING: {line.strip()}", 'yellow')

        # Balance/Account info
        if any(x in line.lower() for x in ['balance', 'account', 'equity', 'pnl']):
            return self.color(f"💰 {line.strip()}", 'green')

        # Info messages (show only important ones)
        if any(x in line for x in ['Starting', 'Stopped', 'Connected', 'Initialized', 'Ready']):
            return self.color(f"ℹ️  {line.strip()}", 'white')

        return None

    def tail_file(self, filepath, source):
        """Tail a log file and yield new lines"""
        if not os.path.exists(filepath):
            return

        # Initialize file position
        if filepath not in self.file_positions:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(0, 2)  # Go to end of file
                self.file_positions[filepath] = f.tell()

        # Read new lines
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            f.seek(self.file_positions[filepath])
            new_lines = f.readlines()
            self.file_positions[filepath] = f.tell()

            for line in new_lines:
                if line.strip():
                    parsed = self.parse_line(line, source)
                    if parsed:
                        yield parsed

    def print_stats(self):
        """Print statistics"""
        print("\n" + "="*80)
        print(self.color("📊 TRADING BOT MONITOR - STATISTICS", 'bold'))
        print("="*80)
        print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Signals: {self.color(str(self.stats['signals']), 'cyan')}")
        print(f"🟢 Positions Opened: {self.color(str(self.stats['positions_opened']), 'green')}")
        print(f"🔴 Positions Closed: {self.color(str(self.stats['positions_closed']), 'yellow')}")
        print(f"❌ Errors: {self.color(str(self.stats['errors']), 'red')}")
        print(f"⚠️  Warnings: {self.color(str(self.stats['warnings']), 'yellow')}")
        print("="*80 + "\n")

    def monitor(self, show_stats_interval=60):
        """Main monitoring loop"""
        print(self.color("="*80, 'bold'))
        print(self.color("🚀 TRADING BOT REAL-TIME MONITOR STARTED", 'bold'))
        print(self.color("="*80, 'bold'))
        print(f"\nMonitoring files:")

        active_files = []
        for name, path in self.log_files.items():
            if os.path.exists(path):
                size = os.path.getsize(path)
                print(f"  ✅ {name}: {path} ({size:,} bytes)")
                active_files.append((path, name))
            else:
                print(f"  ⚠️  {name}: {path} (not found)")

        if not active_files:
            print(self.color("\n❌ No log files found! Bot may not be running.", 'red'))
            return

        print(f"\n{self.color('Press Ctrl+C to stop monitoring', 'yellow')}\n")
        print("="*80 + "\n")

        last_stats_time = time.time()

        try:
            while True:
                has_output = False

                # Check all log files
                for filepath, source in active_files:
                    for line in self.tail_file(filepath, source):
                        print(line)
                        has_output = True

                # Print stats periodically
                if time.time() - last_stats_time > show_stats_interval:
                    self.print_stats()
                    last_stats_time = time.time()

                # Sleep to avoid high CPU usage
                time.sleep(0.5)

        except KeyboardInterrupt:
            print(f"\n\n{self.color('🛑 Monitoring stopped', 'yellow')}")
            self.print_stats()

if __name__ == '__main__':
    monitor = BotMonitor()
    monitor.monitor(show_stats_interval=120)  # Show stats every 2 minutes
