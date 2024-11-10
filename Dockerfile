# Используем базовый образ Python
FROM python:3.12.4

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем все файлы в контейнер
COPY . .
RUN apt-get update && apt-get install -y libgl1-mesa-glx
# Устанавливаем зависимости
RUN pip install --no-cache-dir --default-timeout=10000 -r requirements.txt

# Устанавливаем переменные окружения
ENV PYTHONUNBUFFERED 1
ENV PORT=8011

# Запускаем сервер Uvicorn
CMD ["./wait-for-it.sh", "db:5432", "--", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8011"]
