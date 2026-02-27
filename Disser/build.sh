#!/bin/bash
# Компиляция диссертации: xelatex + biber + xelatex × 2
# Запускать из любого места — скрипт сам перейдёт в Disser/

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

run_xelatex() {
    xelatex -interaction=nonstopmode dissertation.tex
    # xelatex возвращает !=0 при warnings — проверяем, что PDF создан
    if [ ! -f dissertation.pdf ]; then
        echo "ОШИБКА: PDF не создан"
        exit 1
    fi
}

echo "=== xelatex (1/3) ==="
run_xelatex

echo "=== biber ==="
biber dissertation || { echo "ОШИБКА: biber упал"; exit 1; }

echo "=== makeindex (nomenclature) ==="
makeindex dissertation.nlo -s nomencl.ist -o dissertation.nls 2>/dev/null || echo "(nomenclature: нет записей или .nlo пуст — пропуск)"

echo "=== xelatex (2/3) ==="
run_xelatex

echo "=== xelatex (3/3) ==="
run_xelatex

echo "=== Готово: $(pwd)/dissertation.pdf ==="
