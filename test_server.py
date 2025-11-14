"""서버 실행 테스트 스크립트"""
import sys
import os

print("=" * 60)
print("서버 실행 테스트 시작")
print("=" * 60)

# Python 버전 확인
print(f"\n1. Python 버전: {sys.version}")

# 필요한 패키지 확인
print("\n2. 패키지 확인 중...")
try:
    import fastapi
    print(f"   ✓ FastAPI 설치됨: {fastapi.__version__}")
except ImportError:
    print("   ✗ FastAPI가 설치되지 않았습니다.")
    print("   실행: pip install fastapi uvicorn")
    sys.exit(1)

try:
    import uvicorn
    print(f"   ✓ Uvicorn 설치됨: {uvicorn.__version__}")
except ImportError:
    print("   ✗ Uvicorn이 설치되지 않았습니다.")
    print("   실행: pip install uvicorn")
    sys.exit(1)

# app.py 임포트 테스트
print("\n3. app.py 임포트 테스트 중...")
try:
    from app import app
    print("   ✓ app.py 임포트 성공")
except Exception as e:
    print(f"   ✗ app.py 임포트 실패: {e}")
    sys.exit(1)

# 서버 시작
print("\n4. 서버 시작 중...")
print("=" * 60)
print("📍 접속 주소: http://localhost:8000")
print("📍 API 문서: http://localhost:8000/docs")
print("=" * 60)
print("\n⏹️  서버를 종료하려면 Ctrl+C를 누르세요.\n")

try:
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
except KeyboardInterrupt:
    print("\n\n서버가 종료되었습니다.")
except Exception as e:
    print(f"\n\n❌ 서버 실행 오류: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

