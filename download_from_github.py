#!/usr/bin/env python3

"""

Скрипт для скачивания файлов с GitHub

"""

 

import urllib.request

import os

 

GITHUB_RAW_URL = "https://raw.githubusercontent.com/Ikross995/crypto_trading_bot_v12/claude/initial-setup-011CUm6mzSUTnLX4H2dj5D3Y/"

 

FILES = [

    "utils/concurrency.py",

    "utils/rate_limiter.py",

    "exchange/websocket_manager.py",

    "tests/test_concurrency.py",

    "models/gru_predictor.py",

    "examples/gru_training_example.py",

    "strategy/regime_detector.py",

    "strategy/adaptive_strategy.py",

    "strategy/kelly_criterion.py",

    "strategy/dynamic_stops.py",

    "examples/adaptive_trading_integration.py",

    "examples/risk_management_example.py",

    "examples/websocket_example.py",

    "IMPLEMENTATION_COMPLETE.md",

    "IMPROVEMENT_ROADMAP.md",

    "INTEGRATION_EXAMPLE.md",

    "FILES_CHECKLIST.md",

]

 

def download_file(file_path):

    """Download file from GitHub"""

    url = GITHUB_RAW_URL + file_path

 

    # Create directory if needed

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

 

    try:

        print(f"Downloading {file_path}...", end=" ")

        urllib.request.urlretrieve(url, file_path)

        print("✅")

        return True

    except Exception as e:

        print(f"❌ Error: {e}")

        return False

 

if __name__ == "__main__":

    print("=" * 70)

    print("Скачивание файлов с GitHub")

    print("=" * 70)

    print()

 

    success = 0

    failed = 0

 

    for file_path in FILES:

        if download_file(file_path):

            success += 1

        else:

            failed += 1

 

    print()

    print("=" * 70)

    print(f"✅ Успешно: {success}")

    print(f"❌ Ошибок: {failed}")

    print("=" * 70)

