#!/bin/bash
###############################################################################
# 📦 УСТАНОВКА ЗАВИСИМОСТЕЙ ЛОКАЛЬНО
###############################################################################
#
# Этот скрипт устанавливает все необходимые зависимости для бота,
# включая TensorFlow для ML персистентности
#
###############################################################################

set -e  # Остановка при ошибке

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║          📦 УСТАНОВКА ЗАВИСИМОСТЕЙ CRYPTO TRADING BOT                   ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

# Проверка Python
echo -e "${BLUE}🔍 Проверка Python...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 не найден!${NC}"
    echo "Установите Python 3.8+ перед запуском этого скрипта"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}✅ Python ${PYTHON_VERSION}${NC}"

# Проверка pip
echo -e "${BLUE}🔍 Проверка pip...${NC}"
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}❌ pip3 не найден!${NC}"
    echo "Установите pip: https://pip.pypa.io/en/stable/installation/"
    exit 1
fi

echo -e "${GREEN}✅ pip установлен${NC}"
echo ""

# Рекомендация использовать виртуальное окружение
echo -e "${YELLOW}💡 РЕКОМЕНДАЦИЯ: Использовать виртуальное окружение${NC}"
echo ""
echo "Виртуальное окружение изолирует зависимости проекта."
echo ""
read -p "Создать виртуальное окружение? (y/n): " CREATE_VENV

if [[ "$CREATE_VENV" =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${BLUE}📦 Создание виртуального окружения...${NC}"

    if [ -d "venv" ]; then
        echo -e "${YELLOW}⚠️  Папка venv уже существует${NC}"
        read -p "Удалить и пересоздать? (y/n): " RECREATE
        if [[ "$RECREATE" =~ ^[Yy]$ ]]; then
            rm -rf venv
            python3 -m venv venv
            echo -e "${GREEN}✅ Виртуальное окружение пересоздано${NC}"
        fi
    else
        python3 -m venv venv
        echo -e "${GREEN}✅ Виртуальное окружение создано${NC}"
    fi

    echo ""
    echo -e "${BLUE}🔌 Активация виртуального окружения...${NC}"

    # Определяем систему
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        # Windows
        source venv/Scripts/activate
    else
        # Linux/Mac
        source venv/bin/activate
    fi

    echo -e "${GREEN}✅ Виртуальное окружение активировано${NC}"
    echo ""
fi

# Обновление pip
echo -e "${BLUE}🔄 Обновление pip...${NC}"
pip install --upgrade pip setuptools wheel
echo -e "${GREEN}✅ pip обновлен${NC}"
echo ""

# Установка зависимостей
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                      УСТАНОВКА ЗАВИСИМОСТЕЙ                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

if [ -f "requirements_fixed.txt" ]; then
    echo -e "${BLUE}📦 Установка из requirements_fixed.txt...${NC}"
    echo ""

    pip install -r requirements_fixed.txt

    echo ""
    echo -e "${GREEN}✅ Все зависимости установлены!${NC}"
else
    echo -e "${RED}❌ Файл requirements_fixed.txt не найден!${NC}"
    echo ""
    echo -e "${YELLOW}Установка основных зависимостей вручную...${NC}"

    pip install tensorflow>=2.16.0
    pip install scikit-learn>=1.3.0
    pip install pandas>=2.0.0
    pip install numpy>=1.24.0
    pip install python-binance>=1.0.19
    pip install ccxt>=4.0.0
    pip install python-dotenv>=1.0.0
    pip install typer>=0.9.0
    pip install rich>=13.0.0

    echo ""
    echo -e "${GREEN}✅ Основные зависимости установлены${NC}"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                          ПРОВЕРКА УСТАНОВКИ                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

# Проверка ключевых библиотек
echo -e "${BLUE}🧪 Проверка установленных пакетов...${NC}"
echo ""

python3 << 'PYEOF'
import sys

packages = {
    'tensorflow': 'TensorFlow',
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'sklearn': 'scikit-learn',
    'binance': 'python-binance',
    'ccxt': 'CCXT',
}

all_ok = True
for module, name in packages.items():
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✅ {name:20s} {version}")
    except ImportError:
        print(f"❌ {name:20s} NOT INSTALLED")
        all_ok = False

print()
if all_ok:
    print("🎉 Все ключевые пакеты установлены!")

    # Тест TensorFlow
    import tensorflow as tf
    print("\n🧪 Тест TensorFlow:")
    x = tf.constant([1, 2, 3])
    print(f"   Тензор создан: {x.numpy()}")
    print("\n✅ TensorFlow работает корректно!")
    sys.exit(0)
else:
    print("⚠️  Некоторые пакеты не установлены")
    sys.exit(1)
PYEOF

if [ $? -eq 0 ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════════╗"
    echo "║                    ✅ УСТАНОВКА ЗАВЕРШЕНА                                ║"
    echo "╚══════════════════════════════════════════════════════════════════════════╝"
    echo ""

    if [[ "$CREATE_VENV" =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}📌 ВАЖНО: Виртуальное окружение создано${NC}"
        echo ""
        echo "Для активации в будущем используйте:"
        echo ""
        if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
            echo -e "  ${GREEN}venv\\Scripts\\activate${NC}  (Windows)"
        else
            echo -e "  ${GREEN}source venv/bin/activate${NC}  (Linux/Mac)"
        fi
        echo ""
    fi

    echo "🚀 Следующие шаги:"
    echo ""
    echo "  1. Настройте .env файл с API ключами"
    echo "  2. Проверьте ML статус:"
    echo -e "     ${GREEN}python3 check_ml_status.py${NC}"
    echo "  3. Запустите бота:"
    echo -e "     ${GREEN}./START_TRADING.sh${NC}"
    echo ""
else
    echo ""
    echo -e "${RED}❌ Установка завершилась с ошибками${NC}"
    echo ""
    echo "Попробуйте установить вручную:"
    echo "  pip install -r requirements_fixed.txt"
    exit 1
fi
