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

```bash
# Скачивание данных через DVC
dvc pull
```

## Train

### Базовое обучение

```bash
# Запуск обучения с конфигурацией по умолчанию
make run-train

# Или напрямую через Poetry
poetry run python -m flower_classifier.training.train
```

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
mlflow ui --port 8080
```

Перейдите на `http://localhost:8080` для просмотра экспериментов.

### Структура логирования

- **Метрики**: train/val/test loss, F1-score, accuracy
- **Артефакты**: лучшая модель, конфигурация, версия кода
- **Графики**: сохраняются в директории `plots/`

## Production Preparation

### Конвертация в ONNX

```bash
# Конвертация обученной модели в ONNX формат
poetry run python -m flower_classifier.production.convert_to_onnx \
    --checkpoint-path models/best_model.ckpt \
    --output-path models/flower_classifier.onnx
```

### Оптимизация TensorRT

```bash
# Конвертация ONNX модели в TensorRT engine
./scripts/convert_to_tensorrt.sh models/flower_classifier.onnx models/flower_classifier.trt
```

### Комплектация поставки

Для развертывания модели необходимы:
- `flower_classifier.onnx` или `flower_classifier.trt` - файл модели
- `configs/` - конфигурационные файлы
- `flower_classifier/` - Python пакет с кодом
- Список зависимостей из `pyproject.toml`

## Infer

### Формат входных данных

Модель принимает изображения в формате:
- **Формат**: JPG, PNG, JPEG
- **Размер**: любой (автоматически приводится к 224x224)
- **Каналы**: RGB

### Предсказание для одного изображения

```bash
# Предсказание класса цветка
poetry run python -m flower_classifier.inference.predict \
    --image-path path/to/flower_image.jpg \
    --model-path models/best_model.ckpt
```

### Batch предсказание

```bash
# Обработка целой директории
poetry run python -m flower_classifier.inference.batch_predict \
    --input-dir path/to/images/ \
    --output-file predictions.json \
    --model-path models/best_model.ckpt
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
    "dandelion": 0.00
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
│   ├── inference/         # Инференс модули
│   └── utils.py           # Вспомогательные функции
├── models/                # Сохраненные модели
├── plots/                 # Графики и визуализации
├── tests/                 # Unit тесты
├── pyproject.toml         # Poetry конфигурация
├── Makefile              # Команды проекта
└── README.md             # Документация
```

## Технические детали

### Используемые технологии

- **PyTorch Lightning**: структурированное обучение нейросетей
- **Hydra**: управление конфигурациями
- **MLflow**: эксперимент трекинг и логирование
- **DVC**: версионирование данных
- **Poetry**: управление зависимостями
- **Pre-commit**: контроль качества кода
- **timm**: современные архитектуры компьютерного зрения

### Требования к ресурсам

- **RAM**: минимум 8GB, рекомендуется 16GB+
- **GPU**: любая CUDA-совместимая, рекомендуется 4GB+ VRAM
- **Диск**: ~2GB для данных + модели
- **Время обучения**: ~30 минут на современной GPU

### Производительность

На тестовом наборе достигаются следующие метрики:
- **Accuracy**: ~99.5%
- **F1-score**: ~99.5%
- **Время инференса**: ~50ms на изображение (GPU)

## Разработка

### Запуск тестов

```bash
poetry run pytest tests/
```

### Проверка качества кода

```bash
# Все pre-commit проверки
poetry run pre-commit run --all-files

# Отдельные инструменты
poetry run black flower_classifier/
poetry run isort flower_classifier/
```

### Добавление новых зависимостей

```bash
poetry add package_name
poetry add --group dev package_name  # для dev зависимостей
```
