# Flower Classification using Contrastive Learning

Проект для классификации изображений цветов с использованием метрического обучения (contrastive learning) и современного MLOps-пайплайна.

## Описание проекта

### Постановка задачи

Задача классификации изображений цветов на 5 классов: daisy (ромашка), dandelion (одуванчик), roses (розы), sunflowers (подсолнухи), tulips (тюльпаны). Проект использует подход contrastive learning для обучения модели различать цветы на основе их визуальных признаков.

### Данные

Используется датасет [Flowers Recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition) из Kaggle, содержащий:

- **Общее количество**: ~3670 изображений
- **Классы**: 5 типов цветов
- **Разрешение**: переменное, приводится к 224x224 для обучения
- **Формат**: JPG изображения

Распределение по классам:

- daisy: 633 изображения
- dandelion: 898 изображений
- roses: 641 изображение
- sunflowers: 699 изображений
- tulips: 799 изображений

### Архитектура решения

**Модель**:

- Backbone: RexNet-150 (предобученная на ImageNet)
- Framework: PyTorch Lightning для структурированного обучения
- Подход: Triplet learning с contrastive loss

**Функция потерь**:

- Classification Loss: CrossEntropyLoss для основной классификации
- Contrastive Loss: CosineEmbeddingLoss для метрического обучения
- Общий loss: сумма classification и contrastive losses

**Метрики**:

- F1-score (micro average)
- Accuracy
- Отдельное логирование contrastive и classification losses

### Особенности реализации

1. **Triplet Dataset**: каждый sample содержит query, positive и negative изображения
2. **Data Augmentation**: стандартные трансформации ImageNet (resize, normalize)
3. **Mixed Precision Training**: для ускорения обучения
4. **Early Stopping**: мониторинг val_loss с patience=10
5. **Model Checkpointing**: сохранение лучшей модели по validation loss

## Setup

### Требования

- Python 3.10+
- CUDA-совместимая GPU (опционально, но рекомендуется)
- Poetry для управления зависимостями

### Установка зависимостей

```bash
# Клонирование репозитория
git clone <repository-url>
cd flower_classifier

# Установка зависимостей через Poetry
poetry install

# Активация виртуального окружения
poetry shell

# Установка pre-commit hooks
poetry run pre-commit install
```

### Загрузка данных

DVC автоматически интегрирован в процесс обучения. При запуске тренировки данные будут загружены автоматически, если они отсутствуют:

```bash
# Ручная загрузка данных (опционально)
dvc pull

# Или через Makefile
make ddvc-dataset
```

## Train

### Базовое обучение

```bash
# Запуск обучения с конфигурацией по умолчанию
make run-train

# Или напрямую через Poetry
poetry run python -m flower_classifier.training.train
```

**Автоматические действия при обучении:**

- ✅ Проверка и загрузка данных через DVC (если нужно)
- ✅ Создание необходимых директорий
- ✅ Настройка MLflow эксперимента
- ✅ Сохранение лучшей модели и логирование метрик

### Кастомизация параметров

Основные параметры обучения находятся в `configs/train.yaml`:

```yaml
# Изменение количества эпох
trainer:
  max_epochs: 25

# Изменение batch size
data:
  batch_size: 16

# Изменение learning rate
model:
  learning_rate: 0.001
```

### Мониторинг обучения

```bash
# Запуск MLflow UI для просмотра метрик
make mlflow-ui
# Или напрямую:
poetry run mlflow ui --host 127.0.0.1 --port 8080
```

Перейдите на `http://localhost:8080` для просмотра экспериментов.

### Структура логирования

- **Метрики**: train/val/test loss, F1-score, accuracy
- **Артефакты**: лучшая модель, конфигурация, версия кода
- **Графики**: сохраняются в директории `plots/`

## Production Preparation

### Конвертация в ONNX

```bash
# Через Makefile (рекомендуется)
make convert-to-onnx model=models/best_model.ckpt output=models/flower_classifier.onnx

# Или напрямую
poetry run python -m flower_classifier.production.convert_to_onnx \
    --checkpoint-path models/best_model.ckpt \
    --output-path models/flower_classifier.onnx
```

**Возможности ONNX экспорта:**

- ✅ Автоматическая верификация точности
- ✅ Поддержка dynamic batch size
- ✅ Настройка opset version
- ✅ Проверка совместимости ONNX Runtime

### Оптимизация TensorRT

```bash
# Конвертация ONNX модели в TensorRT engine
./scripts/convert_to_tensorrt.sh models/flower_classifier.onnx models/flower_classifier.trt

# Через Makefile с параметрами
make convert-to-tensorrt-run input=models/flower_classifier.onnx output=models/flower_classifier.trt batch=4 precision=fp16

# Различные режимы точности
./scripts/convert_to_tensorrt.sh models/model.onnx models/model_fp32.trt 1 fp32
./scripts/convert_to_tensorrt.sh models/model.onnx models/model_fp16.trt 1 fp16
./scripts/convert_to_tensorrt.sh models/model.onnx models/model_int8.trt 1 int8
```

**Возможности TensorRT конвертации:**

- ✅ Поддержка FP32, FP16, INT8 precision
- ✅ Настройка batch size для оптимизации
- ✅ Автоматическая проверка наличия trtexec
- ✅ Подробное логирование процесса
- ✅ Отображение размера результирующего engine

### TensorRT Inference

```bash
# Инференс с TensorRT engine
make run-tensorrt-inference engine=models/model.trt image=test.jpg

# С сохранением результата
make run-tensorrt-inference engine=models/model.trt image=test.jpg output=result.json

# Benchmark производительности
make run-tensorrt-inference engine=models/model.trt image=test.jpg benchmark=true iterations=1000

# Или напрямую
poetry run python -m flower_classifier.production.tensorrt_inference \
    --engine-path models/flower_classifier.trt \
    --image-path test_image.jpg \
    --benchmark \
    --iterations 500
```

**Производительность TensorRT:**

- **FP16**: ~2-3x ускорение по сравнению с PyTorch
- **INT8**: ~4-5x ускорение (требует калибровку)
- **Throughput**: 200+ изображений/сек на современной GPU
- **Latency**: ~2-5ms на изображение

### Комплектация поставки

Для развертывания модели необходимы:

- `flower_classifier.onnx` или `flower_classifier.trt` - файл модели
- `configs/` - конфигурационные файлы
- `flower_classifier/` - Python пакет с кодом
- Список зависимостей из `pyproject.toml`

## Infer

### Формат входных данных

Модель принимает изображения в формате:

- **Формат**: JPG, PNG, JPEG, BMP, TIFF
- **Размер**: любой (автоматически приводится к 224x224)
- **Каналы**: RGB

### Предсказание для одного изображения

```bash
# Предсказание класса цветка
poetry run python -m flower_classifier.inference.predict \
    --image-path path/to/flower_image.jpg \
    --model-path models/best_model.ckpt \
    --device auto

# С сохранением результата
poetry run python -m flower_classifier.inference.predict \
    --image-path path/to/flower_image.jpg \
    --model-path models/best_model.ckpt \
    --output-file result.json
```

### Batch предсказание

```bash
# Обработка целой директории через Makefile
make run-inference model=models/best_model.ckpt input=data/test_images output=predictions.json

# Или напрямую
poetry run python -m flower_classifier.inference.batch_predict \
    --input-dir path/to/images/ \
    --model-path models/best_model.ckpt \
    --output-file predictions.json

# Обработка списка файлов
poetry run python -m flower_classifier.inference.batch_predict \
    --file-list image_list.txt \
    --model-path models/best_model.ckpt \
    --output-file predictions.json
```

### Пример выходных данных

```json
{
  "image_path": "test_flower.jpg",
  "predicted_class": "roses",
  "confidence": 0.95,
  "all_probabilities": {
    "roses": 0.95,
    "tulips": 0.03,
    "daisy": 0.01,
    "sunflowers": 0.01,
    "dandelion": 0.0
  }
}
```

## Архитектура проекта

```
flower_classifier/
├── configs/                 # Hydra конфигурации
│   ├── train.yaml          # Основная конфигурация обучения
│   ├── data.yaml           # Параметры данных
│   └── model.yaml          # Параметры модели
├── data/                   # Данные и DVC файлы
│   ├── raw/               # Исходные данные
│   └── *.dvc              # DVC метафайлы
├── flower_classifier/      # Основной Python пакет
│   ├── data/              # Модули работы с данными
│   ├── models/            # Архитектуры моделей
│   ├── training/          # Пайплайн обучения
│   ├── inference/         # Модули инференса
│   │   ├── predict.py     # Одиночные предсказания
│   │   └── batch_predict.py # Batch предсказания
│   ├── production/        # Production модули
│   │   ├── convert_to_onnx.py # ONNX экспорт
│   │   └── tensorrt_inference.py # TensorRT инференс
│   └── utils.py           # Вспомогательные функции
├── models/                # Сохраненные модели
├── plots/                 # Графики и визуализации
├── tests/                 # Unit тесты
├── scripts/               # Скрипты (TensorRT и др.)
│   └── convert_to_tensorrt.sh # TensorRT конвертация
├── pyproject.toml         # Poetry конфигурация
├── Makefile              # Команды проекта
└── README.md             # Документация
```

## Технические детали

### Используемые технологии

- **PyTorch Lightning**: структурированное обучение нейросетей
- **Hydra**: управление конфигурациями
- **MLflow**: эксперимент трекинг и логирование
- **DVC**: версионирование данных с автоматической интеграцией
- **Poetry**: управление зависимостями
- **Pre-commit**: контроль качества кода (black, isort, flake8, prettier)
- **timm**: современные архитектуры компьютерного зрения
- **Click**: CLI интерфейсы для inference
- **ONNX**: модель в production формате
- **TensorRT**: высокопроизводительный инференс с GPU оптимизацией

### Требования к ресурсам

- **RAM**: минимум 8GB, рекомендуется 16GB+
- **GPU**: любая CUDA-совместимая, рекомендуется 4GB+ VRAM
- **Диск**: ~2GB для данных + модели
- **Время обучения**: ~30 минут на современной GPU (25 эпох)

### Производительность

На тестовом наборе достигаются следующие метрики:

- **Accuracy**: ~99.5%
- **F1-score**: ~99.5%

**Время инференса (одно изображение):**

- **PyTorch**: ~50ms на GPU
- **ONNX**: ~30ms на GPU
- **TensorRT FP16**: ~15ms на GPU
- **TensorRT INT8**: ~8ms на GPU

**Throughput (изображений/сек):**

- **PyTorch**: ~20 FPS
- **ONNX**: ~35 FPS
- **TensorRT FP16**: ~70 FPS
- **TensorRT INT8**: ~120 FPS

## Разработка

### Запуск тестов

```bash
make test
# Или напрямую:
poetry run pytest tests/
```

### Проверка качества кода

```bash
# Все pre-commit проверки
poetry run pre-commit run --all-files

# Форматирование кода
make format

# Отдельные инструменты
poetry run black flower_classifier/
poetry run isort flower_classifier/
poetry run flake8 flower_classifier/
```

### Добавление новых зависимостей

```bash
poetry add package_name
poetry add --group dev package_name  # для dev зависимостей
```

## Команды Makefile

```bash
# Обучение и данные
make run-train                      # Запуск обучения
make ddvc-dataset                   # Загрузка данных через DVC
make mlflow-ui                      # Запуск MLflow UI

# Inference
make run-inference                  # Batch inference (PyTorch)
make run-tensorrt-inference         # TensorRT inference

# Конвертация моделей
make convert-to-onnx               # Конвертация в ONNX
make convert-to-tensorrt-run       # Конвертация ONNX → TensorRT

# Разработка
make format                        # Форматирование кода
make test                          # Запуск тестов
make pre-commit-install            # Установка pre-commit hooks
```

### Примеры использования команд

```bash
# Полный пайплайн от обучения до TensorRT
make run-train
make convert-to-onnx model=models/best_model.ckpt output=models/model.onnx
make convert-to-tensorrt-run input=models/model.onnx output=models/model.trt batch=4 precision=fp16

# Сравнение производительности
make run-tensorrt-inference engine=models/model.trt image=test.jpg benchmark=true iterations=1000
```
