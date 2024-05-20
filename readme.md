### Датасет
generate_promp.ipynb - ноутбук разметки анекдотов (добавления к ним запроса)
data/dataset.jsonl - собраный SFT датасет для модели 

### Тренировка
sftDataSet.py - датасет, для sft тренировки
train.py - основной скрипт тренировки
train.sbatch - скрипт для запуска train на кластере slurm

### Оценка
eval.py - основной скрипт оценки модели
eval.sbatch - скрипт для запуска eval на кластере slurm

test_checkpoints.ipynb - скрипт для оценки качества генерации анекдотов, полученных из eval.py

### Util
Makefile - makefile...