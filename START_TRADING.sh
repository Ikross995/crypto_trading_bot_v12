#!/bin/bash
###############################################################################
# 🚀 ЗАПУСК ТОРГОВЛИ С ML ПЕРСИСТЕНТНОСТЬЮ
###############################################################################
#
# Этот скрипт запускает бота с полной поддержкой ML:
# - EnhancedAdaptiveLearningSystem (ML с персистентностью)
# - Автосохранение моделей каждые 10 сделок
# - Загрузка моделей при старте
# - 10 торговых пар для быстрого обучения
#
###############################################################################

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║           🚀 AI CRYPTO TRADING BOT - LAUNCH                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

# Проверяем, что TensorFlow установлен
if ! python3 -c "import tensorflow" 2>/dev/null; then
    echo -e "${RED}❌ TensorFlow не установлен!${NC}"
    echo -e "${YELLOW}Установите: pip install tensorflow${NC}"
    exit 1
fi

echo -e "${GREEN}✅ TensorFlow установлен${NC}"

# Проверяем ML модели
if [ -d "ml_learning_data" ]; then
    model_count=$(ls -1 ml_learning_data/*.pkl 2>/dev/null | wc -l)
    if [ $model_count -gt 0 ]; then
        echo -e "${GREEN}✅ Найдено $model_count сохраненных моделей${NC}"
        echo -e "${BLUE}   Модели будут загружены при старте${NC}"
    else
        echo -e "${YELLOW}📚 ML модели не найдены - начнется обучение с нуля${NC}"
    fi
else
    echo -e "${YELLOW}📚 ML модели не найдены - начнется обучение с нуля${NC}"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                      КОНФИГУРАЦИЯ                                        ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Режим:         Testnet (безопасное тестирование)"
echo "  Таймфрейм:     30m"
echo "  ML система:    COMBO (Ensemble + RL + Meta-Learner)"
echo "  Логи:          Verbose (подробные)"
echo "  Пары:          BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, ADAUSDT,"
echo "                 XRPUSDT, DOGEUSDT, AVAXUSDT, LINKUSDT, APTUSDT"
echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                   МОНИТОРИНГ ML ПРОГРЕССА                                ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  В другом терминале запустите:"
echo ""
echo -e "  ${GREEN}$ watch -n 10 python3 check_ml_status.py${NC}"
echo ""
echo "  Чтобы видеть прогресс обучения ML моделей в реальном времени."
echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                        ЗАПУСК                                            ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${BLUE}Для остановки нажмите Ctrl+C${NC}"
echo -e "${BLUE}ML модели будут автоматически сохранены при остановке${NC}"
echo ""
sleep 2

# Запуск бота
python3 cli.py live \
    --timeframe 30m \
    --testnet \
    --use-combo \
    --verbose \
    --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT,XRPUSDT,DOGEUSDT,AVAXUSDT,LINKUSDT,APTUSDT
