#!/usr/bin/env python
"""
Ночной прогон: 64 сессии NOF (16 мышей × 4 дня).

Стадии (каждая может быть запущена отдельно или последовательно):
  1. caches      — построение mean_matrix (real + shuffled) для каждой сессии,
                   skip если уже закэшировано.
  2. geometry    — geometry_cohort.py --days all (FA2 layout + 4 метрики)
  3. rqa         — per_window_analysis.py --days all (RQA окна, PCA, per-metric)
  4. plots       — финальные fig_method/fig_geometry/fig_dynamics с новыми данными

Параметры FA2 и бинаризации унаследованы из geometry_cohort.py
(scalingRatio=1.0, gravity=0.5, бинаризация 95-й перцентиль).

Запуск:
    python run_night.py                 # все стадии
    python run_night.py --skip-caches   # пропустить кэширование (если уже есть)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import time
import argparse
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

# Импортируем функцию кэширования из run_full_1d
from run_full_1d import build_one, list_1d_sessions


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-caches', action='store_true',
                    help='Пропустить построение кэшей (если уже есть)')
    ap.add_argument('--skip-geometry', action='store_true')
    ap.add_argument('--skip-rqa', action='store_true')
    ap.add_argument('--skip-plots', action='store_true')
    return ap.parse_args()


def stage_caches():
    """Построение mean_matrix real + shuffled для всех 64 NOF сессий."""
    sessions = list_1d_sessions(cohort='nof', days='all')
    print(f'\n{"="*70}')
    print(f'  STAGE 1: build caches for {len(sessions)} NOF sessions (all days)')
    print(f'{"="*70}')
    t_total = time.time()
    ok, fail, skipped = 0, 0, 0
    for i, s in enumerate(sessions, 1):
        print(f'\n[{i}/{len(sessions)}] {s}', flush=True)
        try:
            res = build_one(s)
            if res:
                ok += 1
            else:
                fail += 1
        except Exception as e:
            print(f'  FAILED — {type(e).__name__}: {e}', flush=True)
            fail += 1
    dt = (time.time() - t_total) / 60
    print(f'\n{"="*70}')
    print(f'  STAGE 1 done: {ok} ok, {fail} failed, {dt:.1f} min')
    print(f'{"="*70}')


def stage_geometry():
    print(f'\n{"="*70}')
    print(f'  STAGE 2: geometry_cohort.py --days all')
    print(f'{"="*70}')
    t = time.time()
    res = subprocess.run(
        [sys.executable, '-u', str(HERE / 'geometry_cohort.py'),
         '--days', 'all'],
        cwd=HERE, check=False)
    print(f'\n  STAGE 2 done in {(time.time()-t)/60:.1f} min, '
          f'exit code = {res.returncode}')


def stage_rqa():
    print(f'\n{"="*70}')
    print(f'  STAGE 3: per_window_analysis.py --days all')
    print(f'{"="*70}')
    t = time.time()
    res = subprocess.run(
        [sys.executable, '-u', str(HERE / 'per_window_analysis.py'),
         '--days', 'all'],
        cwd=HERE, check=False)
    print(f'\n  STAGE 3 done in {(time.time()-t)/60:.1f} min, '
          f'exit code = {res.returncode}')


def stage_plots():
    print(f'\n{"="*70}')
    print(f'  STAGE 4: rebuild figures with new data')
    print(f'{"="*70}')
    for script in ('fig_method.py', 'fig_geometry.py', 'fig_dynamics.py',
                   'plot_per_metric.py'):
        t = time.time()
        print(f'\n  >>> {script}', flush=True)
        res = subprocess.run([sys.executable, '-u', str(HERE / script)],
                             cwd=HERE, check=False)
        print(f'  ({time.time()-t:.0f}s, exit {res.returncode})')


def main():
    args = parse_args()
    t_total = time.time()

    if not args.skip_caches:
        stage_caches()
    if not args.skip_geometry:
        stage_geometry()
    if not args.skip_rqa:
        stage_rqa()
    if not args.skip_plots:
        stage_plots()

    dt = (time.time() - t_total) / 60
    print(f'\n{"="*70}')
    print(f'  ВСЕГО: {dt:.1f} мин ({dt/60:.1f} ч)')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
