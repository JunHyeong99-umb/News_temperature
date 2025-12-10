@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ==========================================
echo 🚀 뉴스 온도계 배포 스크립트
echo ==========================================
echo.

REM Docker 설치 확인
where docker >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Docker가 설치되어 있지 않습니다.
    echo    Docker 설치: https://docs.docker.com/get-docker/
    pause
    exit /b 1
)

echo ✅ Docker 확인 완료
echo.

REM 이미지 빌드
echo 📦 Docker 이미지 빌드 중...
docker build -t news-thermometer:latest .

if %ERRORLEVEL% NEQ 0 (
    echo ❌ 이미지 빌드 실패
    pause
    exit /b 1
)

echo ✅ 이미지 빌드 완료
echo.

REM 기존 컨테이너 중지 및 제거
echo 🛑 기존 컨테이너 중지 중...
docker-compose down 2>nul

REM 컨테이너 실행
echo 🚀 컨테이너 시작 중...
docker-compose up -d

if %ERRORLEVEL% NEQ 0 (
    echo ❌ 컨테이너 시작 실패
    pause
    exit /b 1
)

echo.
echo ==========================================
echo ✅ 배포 완료!
echo ==========================================
echo.
echo 📍 접속 주소: http://localhost:8000
echo 📍 API 문서: http://localhost:8000/docs
echo 📍 헬스 체크: http://localhost:8000/api/health
echo.
echo 📋 로그 확인: docker-compose logs -f
echo 🛑 중지: docker-compose down
echo.
pause

