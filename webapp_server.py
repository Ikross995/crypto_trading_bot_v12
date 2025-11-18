#!/usr/bin/env python3
"""
Telegram Web App Server
Локальный сервер для хостинга дашборда и API с real-time данными через WebSocket
"""

import asyncio
import json
import os
from pathlib import Path
from datetime import datetime
from flask import Flask, jsonify, send_from_directory, request, make_response
from flask_cors import CORS
from flask_socketio import SocketIO, emit

app = Flask(__name__, static_folder='telegram_webapp', static_url_path='')
CORS(app)  # Enable CORS for Telegram Web App

# Initialize SocketIO for real-time updates
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    logger=False,
    engineio_logger=False
)


# Middleware для обработки ngrok предупреждений
@app.before_request
def add_ngrok_headers():
    """Добавляет заголовки для обхода страницы предупреждения ngrok."""
    # Получаем запрос с правильными заголовками
    pass


@app.after_request
def after_request(response):
    """Добавляет необходимые заголовки в ответ."""
    # Добавляем заголовок для обхода ngrok warning
    response.headers['ngrok-skip-browser-warning'] = 'true'
    # Добавляем кастомный User-Agent если нужно
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,ngrok-skip-browser-warning'
    return response

# Global state - будет обновляться из торгового бота
dashboard_data = {
    'balance': 0.0,
    'equity': 0.0,
    'totalPnl': 0.0,
    'roiPct': 0.0,
    'openPositions': 0,
    'totalTrades': 0,
    'winRate': 0.0,
    'profitFactor': 0.0,
    'positions': [],
    'equityHistory': {
        'labels': [],
        'values': []
    },
    'lastUpdate': None
}


def load_dashboard_data_from_file():
    """Загрузить данные из файла (если есть)."""
    try:
        data_file = Path('data/dashboard_state.json')
        if data_file.exists():
            with open(data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load dashboard data: {e}")
    return None


def update_dashboard_data(new_data):
    """Обновить данные дашборда."""
    global dashboard_data
    dashboard_data.update(new_data)
    dashboard_data['lastUpdate'] = datetime.now().isoformat()
    # Emit update to all connected WebSocket clients
    emit_dashboard_update(new_data)


def emit_dashboard_update(data):
    """Отправить обновление дашборда всем подключенным клиентам через WebSocket."""
    try:
        socketio.emit('dashboard_update', data, namespace='/')
    except Exception as e:
        print(f"Warning: Could not emit dashboard update: {e}")


def emit_trade_update(trade_data):
    """Отправить обновление о сделке через WebSocket."""
    try:
        socketio.emit('trade_update', trade_data, namespace='/')
        print(f"📡 Trade update emitted: {trade_data.get('symbol', 'N/A')}")
    except Exception as e:
        print(f"Warning: Could not emit trade update: {e}")


def emit_position_update(position_data):
    """Отправить обновление позиций через WebSocket."""
    try:
        socketio.emit('position_update', position_data, namespace='/')
        print(f"📡 Position update emitted")
    except Exception as e:
        print(f"Warning: Could not emit position update: {e}")


def emit_price_update(price_data):
    """Отправить обновление цены через WebSocket."""
    try:
        socketio.emit('price_update', price_data, namespace='/')
    except Exception as e:
        print(f"Warning: Could not emit price update: {e}")


@app.route('/')
def index():
    """Главная страница - Enhanced дашборд с WebSocket."""
    try:
        enhanced_path = Path('data/learning_reports/enhanced_dashboard.html')
        if enhanced_path.exists():
            return send_from_directory('data/learning_reports', 'enhanced_dashboard.html')
        else:
            # Fallback to simple dashboard if enhanced not found
            return send_from_directory('telegram_webapp', 'dashboard.html')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/simple')
def simple_dashboard():
    """Простой дашборд (старая версия)."""
    return send_from_directory('telegram_webapp', 'dashboard.html')


@app.route('/api/dashboard')
def get_dashboard():
    """API endpoint для получения данных дашборда."""
    # Попытаться загрузить свежие данные из файла
    file_data = load_dashboard_data_from_file()
    if file_data:
        update_dashboard_data(file_data)

    return jsonify(dashboard_data)


@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'dataAvailable': dashboard_data['lastUpdate'] is not None
    })


@app.route('/api/positions')
def get_positions():
    """API endpoint для получения открытых позиций."""
    return jsonify({
        'positions': dashboard_data.get('positions', []),
        'count': dashboard_data.get('openPositions', 0)
    })


# ==================== WebSocket Events ====================

@socketio.on('connect')
def handle_connect():
    """Обработка подключения нового клиента."""
    print(f"✅ Client connected: {request.sid}")
    # Отправляем текущее состояние дашборда новому клиенту
    emit('dashboard_update', dashboard_data)


@socketio.on('disconnect')
def handle_disconnect():
    """Обработка отключения клиента."""
    print(f"❌ Client disconnected: {request.sid}")


@socketio.on('request_update')
def handle_request_update():
    """Обработка запроса на обновление данных."""
    # Загружаем свежие данные из файла
    file_data = load_dashboard_data_from_file()
    if file_data:
        update_dashboard_data(file_data)
    emit('dashboard_update', dashboard_data)


def run_server(host='0.0.0.0', port=8080):
    """Запустить Flask сервер с WebSocket поддержкой."""
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║       📱 Telegram Web App Server Started                 ║
║              🔴 Real-Time WebSocket Enabled              ║
╚═══════════════════════════════════════════════════════════╝

🌐 Local URL:     http://localhost:{port}
🌐 Network URL:   http://{host}:{port}

📊 Dashboards:
   • Main (Enhanced):  http://localhost:{port}/
   • Simple:           http://localhost:{port}/simple

🔌 API:           http://localhost:{port}/api/dashboard
💚 Health:        http://localhost:{port}/api/health
⚡ WebSocket:     ws://localhost:{port}/socket.io/

💡 Для доступа из Telegram используй ngrok:
   ngrok http {port}

   Затем добавь в .env:
   TG_WEBAPP_URL=https://your-ngrok-url.ngrok-free.app

🔄 Режимы обновления:
   • WebSocket:  Real-time streaming (< 1 сек)
   • Fallback:   HTTP polling (30 сек)
   • File:       data/dashboard_state.json

Нажми Ctrl+C для остановки сервера
    """)

    # Run with SocketIO support
    socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)


def initialize_websocket_bridge():
    """Инициализировать WebSocket bridge для интеграции с торговым движком."""
    try:
        from utils.websocket_bridge import ws_bridge
        import webapp_server as self_module

        # Setup callbacks
        ws_bridge.setup_callbacks(self_module)
        print("✅ WebSocket Bridge initialized")
    except Exception as e:
        print(f"Warning: Could not initialize WebSocket bridge: {e}")


if __name__ == '__main__':
    # Initialize WebSocket bridge
    initialize_websocket_bridge()
    run_server()
