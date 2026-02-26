#!/bin/bash
# Компиляция диссертации: xelatex + biber + xelatex × 2
# Запускать из любого места — скрипт сам перейдёт в корень проекта

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

run_xelatex() {
    xelatex -interaction=nonstopmode -output-directory=Disser Disser/dissertation.tex
    # xelatex возвращает !=0 при warnings — проверяем, что PDF создан
    if [ ! -f Disser/dissertation.pdf ]; then
        echo "ОШИБКА: PDF не создан"
        exit 1
    fi
}

echo "=== xelatex (1/3) ==="
run_xelatex

echo "=== biber ==="
biber Disser/dissertation || { echo "ОШИБКА: biber упал"; exit 1; }

echo "=== xelatex (2/3) ==="
run_xelatex

echo "=== xelatex (3/3) ==="
run_xelatex

echo "=== Готово: Disser/dissertation.pdf ==="
