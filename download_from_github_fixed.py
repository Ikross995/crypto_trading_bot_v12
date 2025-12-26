#!/usr/bin/env python3
"""
Скрипт для скачивания Phase 1-4 файлов с GitHub
"""

import os
import sys
import requests
from pathlib import Path

# GitHub raw URL
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/Ikross995/crypto_trading_bot_v12/claude/initial-setup-011CUm6mzSUTnLX4H2dj5D3Y"

# Список файлов для скачивания
FILES_TO_DOWNLOAD = [
    # Phase 1: Critical Fixes
    "utils/concurrency.py",
    "utils/rate_limiter.py",
    "exchange/websocket_manager.py",
    "tests/test_concurrency.py",

    # Phase 2: GRU Model
    "models/gru_predictor.py",
    "examples/gru_training_example.py",

    # Phase 3: Adaptive Strategy
    "strategy/regime_detector.py",
    "strategy/adaptive_strategy.py",

    # Phase 4: Risk Management
    "strategy/kelly_criterion.py",
    "strategy/dynamic_stops.py",

    # Examples
    "examples/adaptive_trading_integration.py",
    "examples/risk_management_example.py",
    "examples/websocket_example.py",

    # Documentation
    "IMPLEMENTATION_COMPLETE.md",
    "IMPROVEMENT_ROADMAP.md",
    "LIVE_PY_INTEGRATION_GUIDE.md",
    "LIVE_PY_INTEGRATION_PART2.md",
    "FILES_CHECKLIST.md",
]

def download_file(file_path: str) -> bool:
    """Download a single file from GitHub"""
    try:
        # Build GitHub URL
        url = f"{GITHUB_RAW_BASE}/{file_path}"

        # Download
        response = requests.get(url, timeout=30)

        if response.status_code == 404:
            print(f"⚠️  File not found on GitHub: {file_path}")
            return False

        response.raise_for_status()

        # Get directory path
        dir_path = os.path.dirname(file_path)

        # Create directory only if it's not empty
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        # Write file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

        return True

    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading {file_path}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error saving {file_path}: {e}")
        return False

def main():
    print("=" * 70)
    print("Скачивание файлов Phase 1-4 с GitHub")
    print("=" * 70)
    print()

    # Check if we're in the right directory
    if not os.path.exists("runner"):
        print("❌ Ошибка: папка 'runner' не найдена.")
        print("   Запустите скрипт из корня проекта crypto_trading_bot_v12")
        sys.exit(1)

    success_count = 0
    fail_count = 0
    skip_count = 0

    for file_path in FILES_TO_DOWNLOAD:
        print(f"Downloading {file_path}...", end=" ")

        # Skip if file already exists
        if os.path.exists(file_path):
            print("⏭️  (already exists)")
            skip_count += 1
            continue

        if download_file(file_path):
            print("✅")
            success_count += 1
        else:
            print("❌")
            fail_count += 1

    print()
    print("=" * 70)
    print("📊 Результаты:")
    print(f"   ✅ Успешно скачано: {success_count}")
    print(f"   ⏭️  Пропущено (уже есть): {skip_count}")
    print(f"   ❌ Ошибок: {fail_count}")
    print("=" * 70)

    if fail_count > 0:
        print()
        print("⚠️  Некоторые файлы не удалось скачать.")
        print("   Возможные причины:")
        print("   1. Файлы ещё не загружены на GitHub")
        print("   2. Неверная ветка или URL")
        print("   3. Проблемы с интернет-соединением")
        print()
        print("   Попросите Claude показать содержимое файлов напрямую:")
        print('   "покажи содержимое strategy/kelly_criterion.py"')

    if success_count > 0:
        print()
        print("✅ Файлы успешно скачаны!")
        print()
        print("🔧 СЛЕДУЮЩИЕ ШАГИ:")
        print()
        print("1️⃣  Проверьте установку:")
        print('    python -c "from strategy.kelly_criterion import KellyCriterionCalculator; print(\'OK\')"')
        print()
        print("2️⃣  (Опционально) Обучите GRU модель:")
        print("    python examples/gru_training_example.py")
        print()
        print("3️⃣  Скачайте изменения в runner/live.py:")
        print("    git pull origin claude/initial-setup-011CUm6mzSUTnLX4H2dj5D3Y")
        print()
        print("4️⃣  Запустите бота:")
        print("    python runner/__init__.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Прервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Критическая ошибка: {e}")
        sys.exit(1)
