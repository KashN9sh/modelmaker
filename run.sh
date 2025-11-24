#!/bin/bash
# Скрипт для запуска приложения

echo "Запуск STL Sprue Generator..."

# Активируем виртуальное окружение, если оно существует
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Виртуальное окружение активировано"
else
    echo "ВНИМАНИЕ: Виртуальное окружение не найдено!"
    echo "Создайте его командой: python3 -m venv venv"
    echo "И установите зависимости: pip install -r requirements.txt"
fi

echo ""
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

