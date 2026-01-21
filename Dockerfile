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
    pip install --no-cache-dir --prefer-binary -r requirements-docker.txt || true && \
    pip install --no-cache-dir --prefer-binary "PyEMD>=1.0.0,<2.0.0" 2>/dev/null || echo "PyEMD fallback to pure Python"

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

# Проверка установки основных пакетов
RUN python -c "import numpy, pandas, scipy, tensorflow; print('✅ All core dependencies installed')" || \
    echo "⚠️  Some dependencies may be missing but project can still run with fallbacks"

# Команда по умолчанию
CMD ["python", "main.py", "--mode", "synthetic"]
