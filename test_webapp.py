#!/usr/bin/env python3
"""
Скрипт для тестирования веб-приложения
Проверяет все компоненты системы
"""

import json
import os
from pathlib import Path
import requests
import time


def print_header(text):
    """Красивый заголовок."""
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print('=' * 60)


def test_files():
    """Проверка наличия необходимых файлов."""
    print_header("ТЕСТ 1: Проверка файлов")

    files = {
        'webapp_server.py': 'Веб-сервер',
        'telegram_webapp/dashboard.html': 'HTML дашборд',
        'telegram_webapp/test.html': 'Тестовая страница',
        'data/dashboard_state.json': 'Файл данных',
    }

    all_ok = True
    for file_path, description in files.items():
        exists = Path(file_path).exists()
        status = "✅" if exists else "❌"
        print(f"{status} {description}: {file_path}")

        if not exists:
            all_ok = False
            if file_path == 'data/dashboard_state.json':
                print(f"   ⚠️  Запустите: python update_dashboard_data.py")

    return all_ok


def test_dashboard_data():
    """Проверка данных dashboard_state.json."""
    print_header("ТЕСТ 2: Проверка данных")

    data_file = Path('data/dashboard_state.json')

    if not data_file.exists():
        print("❌ Файл data/dashboard_state.json не найден")
        return False

    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print("✅ Файл загружен успешно")
        print(f"\n📊 Содержимое:")
        print(f"   Balance: ${data.get('balance', 0):.2f}")
        print(f"   Total P&L: ${data.get('totalPnl', 0):.2f}")
        print(f"   ROI: {data.get('roiPct', 0):.2f}%")
        print(f"   Trades: {data.get('totalTrades', 0)}")
        print(f"   Win Rate: {data.get('winRate', 0):.1f}%")
        print(f"   Last Update: {data.get('lastUpdate', 'N/A')}")

        # Проверка структуры
        required_fields = ['balance', 'equity', 'totalPnl', 'roiPct', 'positions', 'equityHistory']
        missing = [f for f in required_fields if f not in data]

        if missing:
            print(f"\n⚠️  Отсутствуют поля: {', '.join(missing)}")
            return False

        return True

    except json.JSONDecodeError as e:
        print(f"❌ Ошибка парсинга JSON: {e}")
        return False
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False


def test_webserver():
    """Проверка веб-сервера."""
    print_header("ТЕСТ 3: Проверка веб-сервера")

    urls = [
        'http://localhost:8080/',
        'http://localhost:8080/api/dashboard',
        'http://localhost:8080/api/health',
    ]

    all_ok = True

    for url in urls:
        try:
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                print(f"✅ {url}")

                # Показать данные для API endpoints
                if 'api' in url:
                    try:
                        data = response.json()
                        print(f"   📄 Response: {json.dumps(data, indent=2)[:200]}...")
                    except:
                        print(f"   📄 Response: {response.text[:100]}...")
            else:
                print(f"❌ {url} - HTTP {response.status_code}")
                all_ok = False

        except requests.exceptions.ConnectionError:
            print(f"❌ {url} - Не удалось подключиться")
            print(f"   ⚠️  Запустите: python webapp_server.py")
            all_ok = False
        except Exception as e:
            print(f"❌ {url} - {e}")
            all_ok = False

    return all_ok


def test_ngrok():
    """Проверка ngrok."""
    print_header("ТЕСТ 4: Проверка ngrok")

    try:
        # Попробуем получить информацию о туннелях
        response = requests.get('http://localhost:4040/api/tunnels', timeout=2)

        if response.status_code == 200:
            data = response.json()
            tunnels = data.get('tunnels', [])

            if tunnels:
                print("✅ ngrok работает")
                for tunnel in tunnels:
                    print(f"\n   🌐 Public URL: {tunnel['public_url']}")
                    print(f"   → {tunnel['config']['addr']}")
                return True
            else:
                print("⚠️  ngrok запущен, но нет активных туннелей")
                return False

    except requests.exceptions.ConnectionError:
        print("❌ ngrok не запущен")
        print("   ⚠️  Запустите: ngrok http 8080")
        return False
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False


def print_summary(results):
    """Вывести итоги."""
    print_header("ИТОГИ")

    total = len(results)
    passed = sum(results.values())

    for test_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {test_name}")

    print(f"\n📊 Пройдено: {passed}/{total}")

    if passed == total:
        print("\n🎉 Все тесты пройдены! Веб-приложение должно работать.")
        print("\n📝 Следующие шаги:")
        print("   1. Откройте тестовую страницу: http://localhost:8080/test.html")
        print("   2. Если тесты в браузере проходят, откройте через ngrok в Telegram")
    else:
        print("\n⚠️  Некоторые тесты не пройдены. Проверьте сообщения выше.")

        if not results.get('Файлы'):
            print("\n💡 Запустите: python update_dashboard_data.py")

        if not results.get('Веб-сервер'):
            print("\n💡 Запустите: python webapp_server.py")

        if not results.get('ngrok'):
            print("\n💡 Запустите: ngrok http 8080")


def main():
    """Главная функция."""
    print("""
╔═══════════════════════════════════════════════════════════╗
║       🧪 Telegram Web App Test Suite                     ║
╚═══════════════════════════════════════════════════════════╝
    """)

    results = {}

    # Запуск тестов
    results['Файлы'] = test_files()
    results['Данные'] = test_dashboard_data()
    results['Веб-сервер'] = test_webserver()
    results['ngrok'] = test_ngrok()

    # Итоги
    print_summary(results)


if __name__ == '__main__':
    main()
