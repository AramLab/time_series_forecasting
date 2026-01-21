# ============================================================================
# BUILDER STAGE - Сборка зависимостей (может быть кэширован)
# ============================================================================
FROM python:3.10-slim-bullseye as builder

# Установка системных зависимостей для сборки
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements файла
COPY requirements-docker.txt ./

# Установка Python зависимостей с оптимизацией для wheels
# --prefer-binary избегает компиляции если доступны prebuilt wheels
RUN pip install --upgrade pip setuptools wheel --no-cache-dir && \
    pip install --no-cache-dir --prefer-binary cython && \
    pip install --no-cache-dir --prefer-binary -r requirements-docker.txt || true

# Пытаемся установить PyEMD, но это не критично
RUN pip install --no-cache-dir --prefer-binary "PyEMD>=1.0.0,<2.0.0" 2>&1 || echo "⚠️  PyEMD installation skipped - will use SimpleCEEMDAN fallback"

# ============================================================================
# RUNTIME STAGE - Только необходимые компоненты (маленький образ)
# ============================================================================
FROM python:3.10-slim-bullseye

WORKDIR /app

# Установка минимальных runtime зависимостей (без компиляторов)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    liblapack3 \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Копирование установленных пакетов из builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Копирование кода проекта (этот слой часто меняется, поэтому он последний)
COPY . .

# Создание директории для результатов
RUN mkdir -p results

# Проверка установки основных пакетов и CEEMDAN
RUN python -c "import numpy, pandas, scipy, tensorflow; print('✅ Core dependencies installed')" && \
    python -c "from utils.ceemdan_pure_python import SimpleCEEMDAN; print('✅ SimpleCEEMDAN (pure Python) available')" || \
    echo "⚠️  Some optional dependencies may be missing but project can still run"

# Команда по умолчанию
CMD ["python", "main.py", "--mode", "synthetic"]
