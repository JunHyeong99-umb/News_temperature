@echo off
REM .venv 가상환경 활성화 스크립트
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
    echo ✅ 가상환경 활성화 완료
    echo.
    echo Python 경로 확인:
    python -c "import sys; print(sys.executable)"
    echo.
    echo PyTorch CUDA 확인:
    python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
) else (
    echo ❌ .venv 폴더를 찾을 수 없습니다.
    echo 가상환경을 생성하려면: python -m venv .venv
    pause
    exit /b 1
)

