@echo off
chcp 65001 >nul
cd /d "%~dp0"

REM 가상환경 활성화
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
    echo ✅ 가상환경 활성화 완료
) else (
    echo ⚠️  가상환경이 없습니다. Anaconda base 환경을 사용합니다.
)

echo ============================================================
echo 🚀 네이버 API 테스트 서버 시작
echo ============================================================
echo.
python app.py
pause

