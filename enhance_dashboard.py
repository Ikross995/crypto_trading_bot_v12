#!/usr/bin/env python3
"""
Enhance the existing dashboard.html with:
1. ROI and Sharpe Ratio in status bar (6 metrics total)
2. Color-coded status values
3. Drawdown and Timeline charts (6 charts total)
"""

import re

# Read the dashboard
with open('telegram_webapp/dashboard.html', 'r', encoding='utf-8') as f:
    content = f.read()

print("🔧 Enhancing dashboard...")

# 1. Add ROI and Sharpe Ratio to status bar with color coding
old_status = r'''                    <div class="status-item">
                        <div class="status-value">\$\{formatNumber\(data\.profitFactor \|\| 0, 2\)\}</div>
                        <div class="status-label">Profit Factor</div>
                    </div>
                </div>

                <div class="main-grid">'''

new_status = '''                    <div class="status-item">
                        <div class="status-value ${getClass(data.profitFactor - 1)}">${formatNumber(data.profitFactor || 0, 2)}</div>
                        <div class="status-label">Profit Factor</div>
                    </div>
                    <div class="status-item">
                        <div class="status-value ${getClass(data.roiPct || 0)}">${formatPercent(data.roiPct || 0)}</div>
                        <div class="status-label">ROI</div>
                    </div>
                    <div class="status-item">
                        <div class="status-value ${getClass(data.sharpeRatio || 0)}">${formatNumber(data.sharpeRatio || 0, 2)}</div>
                        <div class="status-label">Sharpe Ratio</div>
                    </div>
                </div>

                <div class="main-grid">'''

content = re.sub(old_status, new_status, content, flags=re.MULTILINE)
print("✅ Added ROI and Sharpe Ratio to status bar")

# 2. Add color to Win Rate
content = content.replace(
    '<div class="status-value">${formatNumber(data.winRate || 0, 1)}%</div>',
    '<div class="status-value ${getClass(data.winRate - 50)}">${formatNumber(data.winRate || 0, 1)}%</div>'
)
print("✅ Added color coding to Win Rate")

# 3. Add Drawdown and Timeline chart tabs
old_tabs = r'''                        <button class="chart-tab" onclick="switchChart\('trades'\)" id="tab-trades">
                            📈 Trades
                        </button>
                    </div>'''

new_tabs = '''                        <button class="chart-tab" onclick="switchChart('trades')" id="tab-trades">
                            📈 Trades
                        </button>
                        <button class="chart-tab" onclick="switchChart('drawdown')" id="tab-drawdown">
                            📉 Drawdown
                        </button>
                        <button class="chart-tab" onclick="switchChart('timeline')" id="tab-timeline">
                            ⏱️ Timeline
                        </button>
                    </div>'''

if 'drawdown' not in content:
    content = re.sub(old_tabs, new_tabs, content, flags=re.MULTILINE)
    print("✅ Added Drawdown and Timeline chart tabs")
else:
    print("⚠️  Charts already exist, skipping")

# Write enhanced dashboard
with open('telegram_webapp/dashboard.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("\n🎉 Dashboard enhanced successfully!")
print("📊 Status bar: 6 metrics (added ROI and Sharpe Ratio)")
print("📈 Charts: Added Drawdown and Timeline tabs")
print("\n🔄 Refresh your browser with Ctrl+F5 to see changes")
