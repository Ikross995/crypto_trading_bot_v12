#!/usr/bin/env python3
"""
Telegram Web App Server
Локальный сервер для хостинга дашборда и API с real-time данными
"""

import asyncio
import json
import os
from pathlib import Path
from datetime import datetime
from flask import Flask, jsonify, send_from_directory, request, make_response
from flask_cors import CORS

app = Flask(__name__, static_folder='telegram_webapp', static_url_path='')
CORS(app)  # Enable CORS for Telegram Web App


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


@app.route('/')
def index():
    """Главная страница - дашборд."""
    return send_from_directory('telegram_webapp', 'dashboard.html')


@app.route('/enhanced')
def enhanced_dashboard():
    """Enhanced дашборд из data/learning_reports."""
    try:
        enhanced_path = Path('data/learning_reports/enhanced_dashboard.html')
        if enhanced_path.exists():
            return send_from_directory('data/learning_reports', 'enhanced_dashboard.html')
        else:
            return jsonify({'error': 'Enhanced dashboard not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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


def run_server(host='0.0.0.0', port=8080):
    """Запустить Flask сервер."""
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║       📱 Telegram Web App Server Started                 ║
╚═══════════════════════════════════════════════════════════╝

🌐 Local URL:     http://localhost:{port}
🌐 Network URL:   http://{host}:{port}

📊 Dashboards:
   • Main:        http://localhost:{port}/
   • Enhanced:    http://localhost:{port}/enhanced

🔌 API:           http://localhost:{port}/api/dashboard
💚 Health:        http://localhost:{port}/api/health

💡 Для доступа из Telegram используй ngrok:
   ngrok http {port}

   Затем добавь в .env:
   TG_WEBAPP_URL=https://your-ngrok-url.ngrok-free.app

🔄 Автоматическое обновление данных из файла:
   data/dashboard_state.json

Нажми Ctrl+C для остановки сервера
    """)

    app.run(host=host, port=port, debug=False)


if __name__ == '__main__':
    run_server()
