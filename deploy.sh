#!/bin/bash
# 배포 스크립트 (Linux/Mac)

set -e

echo "=========================================="
echo "🚀 뉴스 온도계 배포 스크립트"
echo "=========================================="
echo ""

# Docker 설치 확인
if ! command -v docker &> /dev/null; then
    echo "❌ Docker가 설치되어 있지 않습니다."
    echo "   Docker 설치: https://docs.docker.com/get-docker/"
    exit 1
fi

# Docker Compose 설치 확인
if ! command -v docker-compose &> /dev/null; then
    echo "⚠️  Docker Compose가 설치되어 있지 않습니다."
    echo "   docker-compose 대신 docker compose를 사용합니다."
    USE_DOCKER_COMPOSE=false
else
    USE_DOCKER_COMPOSE=true
fi

echo "✅ Docker 확인 완료"
echo ""

# 이미지 빌드
echo "📦 Docker 이미지 빌드 중..."
docker build -t news-thermometer:latest .

if [ $? -eq 0 ]; then
    echo "✅ 이미지 빌드 완료"
else
    echo "❌ 이미지 빌드 실패"
    exit 1
fi

echo ""

# 기존 컨테이너 중지 및 제거
echo "🛑 기존 컨테이너 중지 중..."
if [ "$USE_DOCKER_COMPOSE" = true ]; then
    docker-compose down 2>/dev/null || true
else
    docker compose down 2>/dev/null || true
fi

# 컨테이너 실행
echo "🚀 컨테이너 시작 중..."
if [ "$USE_DOCKER_COMPOSE" = true ]; then
    docker-compose up -d
else
    docker compose up -d
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 배포 완료!"
    echo "=========================================="
    echo ""
    echo "📍 접속 주소: http://localhost:8000"
    echo "📍 API 문서: http://localhost:8000/docs"
    echo "📍 헬스 체크: http://localhost:8000/api/health"
    echo ""
    echo "📋 로그 확인: docker-compose logs -f"
    echo "🛑 중지: docker-compose down"
    echo ""
else
    echo "❌ 컨테이너 시작 실패"
    exit 1
fi

