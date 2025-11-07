#!/bin/bash
# старт

set -e

echo "🚀 Starting DobbyLearn..."
echo ""

# Проверка .env
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "📝 Copy .env.example to .env and fill in your credentials:"
    echo "   cp .env.example .env"
    echo "   nano .env"
    exit 1
fi

# Проверка Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not installed!"
    echo "📦 Install Docker: curl -fsSL https://get.docker.com | sh"
    exit 1
fi

# Проверка Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not installed!"
    echo "📦 Install: apt install docker-compose"
    exit 1
fi

echo "✅ Prerequisites check passed"
echo ""

# Создать data директорию
mkdir -p data

# Остановить старый контейнер если есть
if [ "$(docker ps -q -f name=dobbylearn)" ]; then
    echo "🛑 Stopping old container..."
    docker-compose down
fi

# Запустить
echo "🐳 Starting Docker containers..."
docker-compose up -d --build

echo ""
echo "⏳ Waiting for services to start..."
sleep 5

# Проверить статус
if [ "$(docker ps -q -f name=dobbylearn)" ]; then
    echo ""
    echo "✅ DobbyLearn is running!"
    echo ""
    echo "📊 Check logs: docker-compose logs -f"
    echo "🔍 Check health: curl http://localhost:8000/health"
    echo ""
    echo "🌐 Next steps:"
    echo "   1. Setup Cloudflare Tunnel for HTTPS"
    echo "   2. Update Telegram Bot URL in @BotFather"
    echo ""
    echo "📖 Full guide: cat DEPLOY.md"
else
    echo ""
    echo "❌ Failed to start!"
    echo "📋 Check logs: docker-compose logs"
    exit 1
fi
