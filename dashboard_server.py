#!/usr/bin/env python3
"""
🚀 Enhanced Dashboard Web Server
Простой веб-сервер для отображения трейдинг-дашборда
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
import webbrowser
import threading
import time

class DashboardHandler(SimpleHTTPRequestHandler):
    """Кастомный handler для обслуживания дашборда"""

    def __init__(self, *args, **kwargs):
        # Устанавливаем корневую директорию
        super().__init__(*args, directory=os.path.dirname(os.path.abspath(__file__)), **kwargs)

    def end_headers(self):
        # Добавляем CORS headers для локальной разработки
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

    def log_message(self, format, *args):
        # Красивый лог
        print(f"🌐 {self.address_string()} - {format % args}")

def open_browser(url, delay=1.5):
    """Открывает браузер через delay секунд"""
    time.sleep(delay)
    print(f"\n🚀 Открываю браузер: {url}")
    webbrowser.open(url)

def run_server(port=8080):
    """Запускает веб-сервер"""

    # Проверяем наличие файла дашборда
    dashboard_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'data', 'learning_reports', 'enhanced_dashboard.html'
    )

    if not os.path.exists(dashboard_path):
        print(f"❌ Файл дашборда не найден: {dashboard_path}")
        return

    server_address = ('', port)
    httpd = HTTPServer(server_address, DashboardHandler)

    url = f"http://localhost:{port}/data/learning_reports/enhanced_dashboard.html"

    print("=" * 70)
    print("🚀 Enhanced Trading Dashboard Server")
    print("=" * 70)
    print(f"\n✅ Сервер запущен!")
    print(f"🌐 URL: {url}")
    print(f"📁 Корневая директория: {os.path.dirname(os.path.abspath(__file__))}")
    print(f"\n💡 Откройте в браузере: {url}")
    print(f"⚠️  Для остановки нажмите Ctrl+C")
    print("=" * 70 + "\n")

    # Автоматически открываем браузер
    threading.Thread(target=open_browser, args=(url,), daemon=True).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n🛑 Остановка сервера...")
        httpd.shutdown()
        print("✅ Сервер остановлен!")

if __name__ == '__main__':
    import sys

    # Позволяем указать порт через аргумент
    port = 8080
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"❌ Неверный порт: {sys.argv[1]}, использую 8080")

    run_server(port)
