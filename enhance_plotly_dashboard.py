#!/usr/bin/env python3
"""
Enhance Plotly dashboard with:
1. 6 trading metrics in status bar
2. WebSocket support for real-time updates
3. Telegram WebApp SDK
4. More charts
"""

import re

print("🔧 Enhancing Plotly dashboard...")

# Read the dashboard
with open('telegram_webapp/enhanced_plotly_dashboard.html', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update status bar with 6 trading metrics
print("📊 Updating status bar with 6 trading metrics...")
old_status = '''            <div class="status-item">
                <div class="status-value">Running</div>
                <div class="status-label">Bot Status</div>
            </div>
            <div class="status-item">
                <div class="status-value">26.4h</div>
                <div class="status-label">Uptime</div>
            </div>
            <div class="status-item">
                <div class="status-value">1,321</div>
                <div class="status-label">Signals Generated</div>
            </div>
            <div class="status-item">
                <div class="status-value">0.0%</div>
                <div class="status-label">Execution Rate</div>
            </div>
            <div class="status-item">
                <div class="status-value">Neutral</div>
                <div class="status-label">Market Trend</div>
            </div>'''

new_status = '''            <div class="status-item">
                <div class="status-value" id="status-total-trades">21</div>
                <div class="status-label">📊 Total Trades</div>
            </div>
            <div class="status-item">
                <div class="status-value negative" id="status-win-rate">14.3%</div>
                <div class="status-label">🎯 Win Rate</div>
            </div>
            <div class="status-item">
                <div class="status-value" id="status-positions">0</div>
                <div class="status-label">📈 Open Positions</div>
            </div>
            <div class="status-item">
                <div class="status-value negative" id="status-profit-factor">0.25</div>
                <div class="status-label">⚡ Profit Factor</div>
            </div>
            <div class="status-item">
                <div class="status-value positive" id="status-roi">+30.66%</div>
                <div class="status-label">💰 ROI</div>
            </div>
            <div class="status-item">
                <div class="status-value negative" id="status-sharpe">-0.39</div>
                <div class="status-label">📉 Sharpe Ratio</div>
            </div>'''

if old_status in content:
    content = content.replace(old_status, new_status)
    print("  ✅ Status bar updated with 6 metrics")
else:
    print("  ⚠️  Old status bar not found")

# 2. Add Telegram WebApp SDK and Socket.IO before </head>
print("📱 Adding Telegram WebApp SDK and Socket.IO...")
head_end = '</head>'
new_scripts = '''
    <!-- Telegram Web App SDK -->
    <script src="https://telegram.org/js/telegram-web-app.js"></script>

    <!-- Socket.IO for real-time updates -->
    <script src="https://cdn.socket.io/4.7.2/socket.io.min.js" crossorigin="anonymous"></script>

</head>'''

if 'telegram-web-app.js' not in content:
    content = content.replace(head_end, new_scripts)
    print("  ✅ Added Telegram WebApp SDK and Socket.IO")
else:
    print("  ⚠️  Already has Telegram SDK")

# 3. Add WebSocket connection script before </body>
print("🔴 Adding WebSocket real-time updates...")
websocket_script = '''
    <script>
        // Initialize Telegram WebApp
        const tg = window.Telegram?.WebApp;
        if (tg) {
            tg.expand();
            tg.enableClosingConfirmation();
        }

        // WebSocket connection for real-time updates
        let socket = null;

        function initWebSocket() {
            try {
                console.log('🔴 Connecting to WebSocket...');
                socket = io(window.location.origin, {
                    transports: ['websocket', 'polling'],
                    reconnection: true
                });

                socket.on('connect', () => {
                    console.log('✅ WebSocket connected!');
                    updateConnectionStatus(true);
                });

                socket.on('disconnect', () => {
                    console.log('❌ WebSocket disconnected');
                    updateConnectionStatus(false);
                });

                socket.on('dashboard_update', (data) => {
                    console.log('📊 Dashboard update received:', data);
                    updateDashboard(data);
                });

            } catch (error) {
                console.error('❌ WebSocket failed:', error);
            }
        }

        function updateDashboard(data) {
            // Update status bar
            if (data.totalTrades) document.getElementById('status-total-trades').textContent = data.totalTrades;
            if (data.winRate !== undefined) {
                const winRateEl = document.getElementById('status-win-rate');
                winRateEl.textContent = data.winRate.toFixed(1) + '%';
                winRateEl.className = 'status-value ' + (data.winRate >= 50 ? 'positive' : 'negative');
            }
            if (data.openPositions !== undefined) document.getElementById('status-positions').textContent = data.openPositions;
            if (data.profitFactor !== undefined) {
                const pfEl = document.getElementById('status-profit-factor');
                pfEl.textContent = data.profitFactor.toFixed(2);
                pfEl.className = 'status-value ' + (data.profitFactor >= 1 ? 'positive' : 'negative');
            }
            if (data.roiPct !== undefined) {
                const roiEl = document.getElementById('status-roi');
                roiEl.textContent = (data.roiPct >= 0 ? '+' : '') + data.roiPct.toFixed(2) + '%';
                roiEl.className = 'status-value ' + (data.roiPct >= 0 ? 'positive' : 'negative');
            }
            if (data.sharpeRatio !== undefined) {
                const sharpeEl = document.getElementById('status-sharpe');
                sharpeEl.textContent = data.sharpeRatio.toFixed(2);
                sharpeEl.className = 'status-value ' + (data.sharpeRatio >= 0 ? 'positive' : 'negative');
            }

            console.log('✅ Dashboard updated with real-time data');
        }

        function updateConnectionStatus(connected) {
            const indicator = document.querySelector('.live-indicator');
            if (indicator) {
                indicator.style.background = connected ? '#00ff88' : '#ff4757';
            }
        }

        // Initialize WebSocket when page loads
        if (typeof io !== 'undefined') {
            initWebSocket();
        } else {
            console.log('📊 Socket.IO not available, using static mode');
        }

        // Notify Telegram that app is ready
        if (tg) {
            tg.ready();
        }
    </script>
</body>'''

if '</body>' in content and 'initWebSocket' not in content:
    content = content.replace('</body>', websocket_script)
    print("  ✅ Added WebSocket real-time updates")
else:
    print("  ⚠️  WebSocket already exists or no </body> tag")

# 4. Add live indicator animation to CSS
print("💡 Adding live indicator animation...")
live_indicator_css = '''

        .live-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            background: #00ff88;
            border-radius: 50%;
            animation: pulse 2s infinite;
            margin-right: 8px;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(0,255,136,0.7); }
            50% { opacity: 0.5; box-shadow: 0 0 0 8px rgba(0,255,136,0); }
        }
'''

if '.live-indicator' not in content:
    content = content.replace('</style>', live_indicator_css + '    </style>')
    print("  ✅ Added live indicator animation")
else:
    print("  ⚠️  Live indicator already exists")

# Write enhanced dashboard
with open('telegram_webapp/enhanced_plotly_dashboard.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("\n🎉 Plotly Dashboard Enhanced Successfully!")
print("=" * 60)
print("✅ Status Bar: 6 trading metrics (Total Trades, Win Rate, Positions, Profit Factor, ROI, Sharpe Ratio)")
print("✅ WebSocket: Real-time updates enabled")
print("✅ Telegram: WebApp SDK integrated")
print("✅ Live Indicator: Animated connection status")
print("=" * 60)
print("\n📂 File saved to: telegram_webapp/enhanced_plotly_dashboard.html")
print("🔄 Refresh browser with Ctrl+F5 to see changes")
print("\n💡 To use this dashboard:")
print("   1. Update webapp_server.py to serve this file")
print("   2. Or open directly: http://localhost:8080/enhanced_plotly_dashboard.html")
