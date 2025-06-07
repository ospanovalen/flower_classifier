# Классификатор цветов с Contrastive Learning

Проект для классификации изображений цветов с использованием метрического обучения (contrastive learning) и современного MLOps-пайплайна.

## Описание проекта

### Постановка задачи

Задача классификации изображений цветов на 5 классов: daisy (ромашка), dandelion (одуванчик), roses (розы), sunflowers (подсолнухи), tulips (тюльпаны). Проект использует подход contrastive learning для обучения модели различать цветы на основе их визуальных признаков.

### Данные

Используется датасет [Flowers Recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition) из Kaggle, содержащий:

- **Общее количество**: ~3670 изображений (~232MB)
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
3. **Mixed Precision Training**: для ускорения обучения ("16-mixed")
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

# Установка с GPU поддержкой (рекомендуется для обучения)
poetry install --with=gpu,dev

# Активация виртуального окружения
poetry shell

# Установка pre-commit hooks
poetry run pre-commit install
```

### Загрузка данных

**Автоматическая загрузка с Google Drive (рекомендуется):**

При первом запуске тренировки данные автоматически загрузятся с Google Drive. Если вы хотите загрузить их заранее:

```bash
# Автоматическая загрузка данных с Google Drive
make download-data

# Или напрямую
poetry run python -m flower_classifier.data.download_data
```

**Настройка Google Drive (для разработчиков):**

Если вы форкаете проект и хотите использовать свой Google Drive:

1. Загрузите архив `flower_data.tar.gz` в свой Google Drive
2. Сделайте файл публичным ("Все в интернете с ссылкой")
3. Скопируйте ID файла из ссылки: `https://drive.google.com/file/d/FILE_ID/view`
4. Обновите `file_id` в `flower_classifier/data/download_data.py`

**Резервные методы:**

```bash
# Через DVC (если настроен)
dvc pull

# Комбинированный метод (Google Drive + DVC fallback)
make ddvc-dataset

# Ручная загрузка
# 1. Скачайте flower_data.tar.gz с https://drive.google.com/file/d/1n-DjQGxlEd4iH9skG8xosnfQMSvWm4kf/view
# 2. Распакуйте: tar -xzf flower_data.tar.gz
# 3. Переместите папку raw в data/
```

**Автоматическая интеграция:**

- Функция `setup_data()` в `training/train.py` проверяет наличие данных
- При отсутствии автоматически загружает с Google Drive через `gdown`
- Fallback на DVC в случае ошибок загрузки

## Train

### Быстрое тестовое обучение

```bash
# Тестовое обучение (2 эпохи для проверки)
make run-train

# Или с кастомной конфигурацией
poetry run python -m flower_classifier.training.train --config-name=test_train
```

### Полное обучение

```bash
# Обучение с полной конфигурацией
poetry run python -m flower_classifier.training.train --config-name=train
```

**Автоматические действия при обучении:**

- Проверка и загрузка данных через DVC (если нужно)
- Создание необходимых директорий (`models/`, `plots/`)
- Настройка MLflow эксперимента
- Сохранение лучшей модели и логирование метрик
- Запись конфигурации и версии кода

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

# Настройка contrastive learning
model:
  contrastive_margin: 0.3
```

### Мониторинг обучения

```bash
# Запуск MLflow UI для просмотра метрик
poetry run mlflow ui --host 127.0.0.1 --port 8080
```

Перейдите на `http://localhost:8080` для просмотра экспериментов.

**Логируемые метрики:**

- train_loss, val_loss, test_loss
- train_f1, val_f1, test_f1
- train_accuracy, val_accuracy, test_accuracy
- contrastive_loss, classification_loss

**Артефакты:**

- Лучшая модель (`best_model.ckpt`)
- Конфигурация Hydra
- Git commit hash
- Графики обучения в `plots/`

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

- Автоматическая верификация точности (max diff ~1e-6)
- Поддержка dynamic batch size
- Настройка opset version (default 11)
- Проверка совместимости ONNX Runtime
- Автоматический перенос модели на CPU

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

- Поддержка FP32, FP16, INT8 precision
- Настройка batch size для оптимизации
- Автоматическая проверка наличия trtexec
- Подробное логирование процесса конвертации
- Отображение размера результирующего engine

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

**Производительность TensorRT (GPU):**

- **FP32**: ~30-50ms на изображение
- **FP16**: ~15ms на изображение (~2-3x ускорение)
- **INT8**: ~8ms на изображение (~4-5x ускорение)
- **Throughput**: 70-120+ FPS в зависимости от precision

### Комплектация поставки

Для развертывания модели в продакшене необходимы:

- `flower_classifier.onnx` или `flower_classifier.trt` - файл модели
- `configs/` - конфигурационные файлы Hydra
- `flower_classifier/` - Python пакет с кодом
- `pyproject.toml` - список зависимостей
- `README.md` - инструкции по развертыванию

## Inference Server

### FastAPI REST API

```bash
# Запуск API сервера
make start-api-server model=models/best_model.ckpt host=0.0.0.0 port=8000 device=auto workers=1

# Или напрямую
poetry run python -m flower_classifier.serving.run_server \
    --model-path models/best_model.ckpt \
    --host 0.0.0.0 \
    --port 8000 \
    --device auto \
    --workers 1
```

**API Endpoints:**

- `GET /` - Информация об API
- `GET /health` - Health check
- `POST /predict` - Предсказание для одного изображения (multipart/form-data)
- `POST /predict/batch` - Batch предсказания (до 10 изображений)

**Пример использования:**

```bash
# Health check
curl http://localhost:8000/health

# Предсказание для изображения
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_flower.jpg"

# Swagger UI доступен по адресу: http://localhost:8000/docs
```

### MLflow Model Serving

```bash
# Запуск MLflow model server
make start-mlflow-server model="models:/flower_classifier/1" host=127.0.0.1 port=5001 workers=1

# Проверка статуса
make mlflow-server-status server=http://127.0.0.1:5001

# Предсказание через MLflow server
make mlflow-server-predict server=http://127.0.0.1:5001 image=test.jpg output=result.json
```

**Особенности servers:**

- **FastAPI**: REST API с Swagger UI, upload файлов, batch processing
- **MLflow**: Интеграция с model registry, версионирование моделей
- **Производительность**: ~100-500 запросов/сек (зависит от hardware)
- **Автоопределение устройства**: CPU/GPU через device=auto

## Infer

### Формат входных данных

Модель принимает изображения в формате:

- **Форматы**: JPG, PNG, JPEG, BMP, TIFF
- **Размер**: любой (автоматически приводится к 224x224)
- **Каналы**: RGB (автоматическое преобразование из других форматов)
- **Ограничения**: максимальный размер файла ~10MB

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

# С ONNX моделью (экспериментально)
poetry run python -m flower_classifier.inference.predict \
    --image-path path/to/flower_image.jpg \
    --model-path models/flower_classifier.onnx \
    --device cpu
```

### Batch предсказание

```bash
# Обработка целой директории через Makefile
make run-inference model=models/best_model.ckpt input=data/test_images output=predictions.json

# Или напрямую
poetry run python -m flower_classifier.inference.batch_predict \
    --input-dir path/to/images/ \
    --model-path models/best_model.ckpt \
    --output-file predictions.json \
    --device auto

# Обработка списка файлов
poetry run python -m flower_classifier.inference.batch_predict \
    --file-list image_list.txt \
    --model-path models/best_model.ckpt \
    --output-file predictions.json
```

### Пример выходных данных

Результат предсказания в JSON формате:

```json
{
  "image_path": "test_flower.jpg",
  "predicted_class": "roses",
  "confidence": 0.988,
  "all_probabilities": {
    "roses": 0.988,
    "tulips": 0.007,
    "daisy": 0.003,
    "sunflowers": 0.001,
    "dandelion": 0.001
  },
  "processing_time_ms": 45.2
}
```

**Batch результат:**

```json
{
  "predictions": [
    {
      "image_path": "roses/test1.jpg",
      "predicted_class": "roses",
      "confidence": 0.988
    },
    {
      "image_path": "daisy/test2.jpg",
      "predicted_class": "daisy",
      "confidence": 0.995
    }
  ],
  "summary": {
    "total_images": 2,
    "avg_confidence": 0.991,
    "processing_time_total_ms": 89.5
  }
}
```

## Архитектура проекта

```
flower_classifier/
├── configs/                 # Hydra конфигурации
│   ├── train.yaml          # Основная конфигурация обучения
│   ├── test_train.yaml     # Быстрая тестовая конфигурация
│   └── .hydra/             # Hydra системные файлы
├── data/                   # Данные и DVC файлы
│   ├── raw/               # Исходные данные (5 классов цветов)
│   └── raw.dvc            # DVC метафайл (232MB)
├── flower_classifier/      # Основной Python пакет
│   ├── __init__.py
│   ├── data/              # Модули работы с данными
│   │   ├── dataset.py     # FlowerDataset, трансформации
│   │   └── __init__.py
│   ├── models/            # Архитектуры моделей
│   │   ├── flower_model.py # FlowerClassifier (Lightning)
│   │   └── __init__.py
│   ├── training/          # Пайплайн обучения
│   │   ├── train.py       # Основной скрипт обучения
│   │   └── __init__.py
│   ├── inference/         # Модули инференса
│   │   ├── predict.py     # Одиночные предсказания
│   │   ├── batch_predict.py # Batch предсказания
│   │   └── __init__.py
│   ├── production/        # Production модули
│   │   ├── convert_to_onnx.py # ONNX экспорт с верификацией
│   │   ├── tensorrt_inference.py # TensorRT высокопроизводительный инференс
│   │   └── __init__.py
│   ├── serving/           # Inference servers
│   │   ├── fastapi_server.py # FastAPI REST API
│   │   ├── mlflow_server.py # MLflow model serving
│   │   ├── run_server.py  # CLI для запуска серверов
│   │   └── __init__.py
│   └── utils.py           # Вспомогательные функции
├── models/                # Сохраненные модели (.ckpt, .onnx, .trt)
├── plots/                 # Графики и визуализации MLflow
├── mlruns/                # MLflow эксперименты и артефакты
├── tests/                 # Unit тесты (опционально)
├── scripts/               # Bash/Shell скрипты
│   └── convert_to_tensorrt.sh # TensorRT конвертация
├── .dvc/                  # DVC конфигурация
├── .pre-commit-config.yaml # Pre-commit конфигурация
├── pyproject.toml         # Poetry зависимости и конфигурация
├── poetry.lock            # Заблокированные версии зависимостей
├── Makefile              # Команды проекта с переименованными переменными
├── .gitignore            # Git игнорируемые файлы
└── README.md             # Этот файл - документация
```

## Технические детали

### Используемые технологии

**Core ML Stack:**

- **PyTorch Lightning**: структурированное обучение нейросетей
- **timm**: современные архитектуры компьютерного зрения (RexNet-150)
- **torchvision**: трансформации изображений

**Configuration & Experiment Management:**

- **Hydra**: управление конфигурациями с иерархическими YAML
- **MLflow**: experiment tracking, логирование метрик и артефактов

**Data Management:**

- **DVC**: версионирование данных с автоматической интеграцией через Python API

**Code Quality & Dependencies:**

- **Poetry**: управление зависимостями и виртуальными окружениями
- **Pre-commit**: автоматические проверки кода (black, isort, flake8, prettier)

**Production & Serving:**

- **ONNX**: cross-platform модель в production формате
- **TensorRT**: высокопроизводительный GPU инференс
- **FastAPI**: современный async REST API с автодокументацией
- **Click**: CLI интерфейсы для всех компонентов

### Переменные окружения в Makefile

Для оригинальности переименованы стандартные переменные:

- `POETRY_RUN` (вместо MANAGER) - менеджер запуска команд через Poetry
- `COMPUTE_DEVICE` (вместо DEVICE) - устройство для вычислений (cuda:0)

### Требования к ресурсам

**Минимальные требования:**

- **RAM**: 8GB (рекомендуется 16GB+)
- **GPU**: любая CUDA-совместимая (опционально)
- **Диск**: ~3GB (данные + модели + зависимости)
- **Python**: 3.10+

**Рекомендуемые требования:**

- **RAM**: 16GB+
- **GPU**: 4GB+ VRAM (для быстрого обучения)
- **Диск**: 5GB+ (с запасом для экспериментов)

### Производительность

**Метрики качества (тестовый набор):**

- **Accuracy**: ~99.5% (протестировано)
- **F1-score**: ~99.5% (micro average)
- **Особенно хорошо**: розы (98.8%), ромашки (99.9%)

**Время обучения:**

- **2 эпохи (тест)**: ~2-3 минуты на GPU
- **25 эпох (полное)**: ~30-45 минут на современной GPU

**Время инференса (одно изображение 224x224):**

- **PyTorch CPU**: ~200ms
- **PyTorch GPU**: ~50ms
- **ONNX CPU**: ~120ms
- **ONNX GPU**: ~30ms
- **TensorRT FP16**: ~15ms
- **TensorRT INT8**: ~8ms

**Throughput (изображений/сек):**

- **PyTorch**: 20-50 FPS
- **ONNX**: 35-70 FPS
- **TensorRT FP16**: 70-100 FPS
- **TensorRT INT8**: 120+ FPS

## Разработка

### Запуск тестов

```bash
# Запуск всех тестов (если реализованы)
poetry run pytest tests/

# Только быстрые тесты
poetry run pytest tests/ -m "not slow"

# С покрытием кода
poetry run pytest tests/ --cov=flower_classifier
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
# Основные зависимости
poetry add package_name

# Dev зависимости (тестирование, линтеры)
poetry add --group dev package_name

# GPU зависимости
poetry add --group gpu package_name

# Обновление зависимостей
poetry update
```

### Debug режим

```bash
# Обучение с дебаг логами
poetry run python -m flower_classifier.training.train --config-name=test_train hydra.verbose=true

# Инференс с подробным выводом
poetry run python -m flower_classifier.inference.predict \
    --image-path test.jpg \
    --model-path models/best_model.ckpt \
    --verbose
```

## Команды Makefile

### Основные команды

```bash
# Обучение и данные
make run-train                      # Запуск обучения
make download-data                  # Загрузка данных с Google Drive
make ddvc-dataset                   # Комбинированная загрузка (Google Drive + DVC fallback)

# Inference
make run-inference                  # Batch inference (PyTorch)
make run-tensorrt-inference         # TensorRT inference (высокая производительность)

# Конвертация моделей
make convert-to-onnx               # PyTorch → ONNX
make convert-to-tensorrt-run       # ONNX → TensorRT

# Inference Servers
make start-api-server               # FastAPI REST API server
make start-mlflow-server            # MLflow model server
make mlflow-server-status           # Проверка статуса MLflow server

# Разработка
make format                        # Форматирование кода (black, isort)
make test                          # Запуск тестов (если есть)
make pre-commit-install            # Установка pre-commit hooks
```

### Примеры использования команд

**Полный пайплайн от обучения до TensorRT:**

```bash
# 1. Обучение модели
make run-train

# 2. Конвертация в ONNX
make convert-to-onnx model=models/best_model.ckpt output=models/model.onnx

# 3. Оптимизация с TensorRT
make convert-to-tensorrt-run input=models/model.onnx output=models/model.trt batch=4 precision=fp16

# 4. Тестирование производительности
make run-tensorrt-inference engine=models/model.trt image=test.jpg benchmark=true iterations=1000
```

**Запуск inference servers:**

```bash
# FastAPI сервер на всех интерфейсах
make start-api-server model=models/best_model.ckpt host=0.0.0.0 port=8000 device=auto workers=1

# MLflow сервер с моделью из registry
make start-mlflow-server model="models:/flower_classifier/1" host=127.0.0.1 port=5001

# Проверка работы API
curl -X POST "http://localhost:8000/predict" -F "file=@test_flower.jpg"
```

**Разработка и отладка:**

```bash
# Форматирование перед коммитом
make format

# Установка pre-commit (один раз)
make pre-commit-install

# Тестирование изменений
make run-inference model=models/best_model.ckpt input=test_images output=test_results.json
```

### Переменные Makefile

Для кастомизации можно переопределить переменные:

```bash
# Использование другого устройства
make run-train COMPUTE_DEVICE='cpu'

# Использование другого менеджера пакетов (гипотетически)
make run-train POETRY_RUN='uv run'
```

Эти переменные были переименованы для оригинальности согласно требованиям:

- `POETRY_RUN` (заменяет стандартное MANAGER)
- `COMPUTE_DEVICE` (заменяет стандартное DEVICE)
