# 뉴스 온도계 📰🌡️

뉴스 감정 분석 및 요약 서비스입니다. 네이버 뉴스 API를 통해 뉴스를 수집하고, AI 모델을 활용하여 감정 분석과 요약을 제공합니다.

## 주요 기능

- 🔍 **뉴스 검색**: 네이버 뉴스 API를 통한 실시간 뉴스 검색
- 😊 **감정 분석**: 뉴스 기사의 감정을 분석하여 긍정/부정/중립 판단
- 📝 **기사 요약**: AI 모델을 활용한 뉴스 기사 자동 요약
- 📊 **시각화**: 감정 분석 결과를 차트로 시각화
- 🔐 **인증 시스템**: 네이버 API 인증을 통한 보안 접근

## 기술 스택

- **Backend**: FastAPI, Python 3.11
- **AI/ML**: Transformers, OpenAI API (선택사항)
- **배포**: Docker, Docker Compose
- **기타**: Uvicorn, Gunicorn

## 시작하기

### 사전 요구사항

- Python 3.11 이상
- Docker 및 Docker Compose (Docker 배포 시)
- 네이버 뉴스 API Client ID 및 Client Secret

### 설치 방법

#### 1. 저장소 클론

```bash
git clone <repository-url>
cd news2
```

#### 2. 가상 환경 생성 및 활성화

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

#### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

#### 4. 환경 변수 설정 (선택사항)

`.env` 파일을 생성하여 다음 변수를 설정할 수 있습니다:

```env
PORT=8000
```

### 실행 방법

#### 방법 1: Docker 사용 (권장)

**Windows:**
```cmd
deploy.bat
```

**Linux/Mac:**
```bash
chmod +x deploy.sh
./deploy.sh
```

또는 수동으로:

```bash
# 이미지 빌드
docker build -t news-thermometer .

# 컨테이너 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 중지
docker-compose down
```

#### 방법 2: 로컬에서 직접 실행

**Windows:**
```cmd
start_server.bat
```

**Linux/Mac:**
```bash
python app.py
```

또는:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

#### 방법 3: 테스트 스크립트 사용

```bash
python test_server.py
```

### 접속

서버 실행 후 다음 주소로 접속할 수 있습니다:

- **메인 페이지**: http://localhost:8000
- **로그인 페이지**: http://localhost:8000/login
- **API 문서**: http://localhost:8000/docs
- **헬스 체크**: http://localhost:8000/api/health

## 사용 방법

### 1. 로그인

1. 브라우저에서 `http://localhost:8000/login` 접속
2. 네이버 뉴스 API의 Client ID와 Client Secret 입력
3. 로그인 버튼 클릭

### 2. 뉴스 검색 및 분석

1. 로그인 후 메인 페이지에서 검색어 입력
2. 검색 옵션 설정:
   - 최대 결과 수
   - 검색 기간 (일 단위)
   - 정렬 방식 (날짜순/조회수순)
   - 모델 모드 (로컬 모델/OpenAI API)
3. "검색 및 분석" 버튼 클릭
4. 결과 확인:
   - 뉴스 기사 목록
   - 감정 분석 결과
   - 요약된 내용
   - 시각화 차트

### 3. API 사용

#### 헬스 체크

```bash
curl http://localhost:8000/api/health
```

#### 뉴스 검색 및 분석 (인증 필요)

```bash
curl -X POST "http://localhost:8000/api/test" \
  -H "Content-Type: application/json" \
  -H "Cookie: session_id=YOUR_SESSION_ID" \
  -d '{
    "query": "검색어",
    "max_results": 10,
    "days": 1,
    "include_full_text": true,
    "sort_by": "date",
    "model_mode": "local"
  }'
```

자세한 API 문서는 `/docs` 엔드포인트에서 확인할 수 있습니다.

## 프로젝트 구조

```
news2/
├── app.py                 # FastAPI 메인 애플리케이션
├── requirements.txt       # Python 의존성 목록
├── Dockerfile            # Docker 이미지 빌드 설정
├── docker-compose.yml    # Docker Compose 설정
├── deploy.bat            # Windows 배포 스크립트
├── deploy.sh             # Linux/Mac 배포 스크립트
├── start_server.bat      # Windows 서버 시작 스크립트
├── test_server.py        # 서버 테스트 스크립트
├── src/
│   ├── crawl_naver_api.py    # 네이버 뉴스 API 크롤러
│   ├── crawl_naver_link.py   # 네이버 뉴스 링크 크롤러
│   └── sentiment_analyzer.py # 감정 분석기
├── kosum-v1-tuned/       # 요약 모델 파일
├── sentiment_model/      # 감정 분석 모델 파일
├── static/               # 정적 파일 (CSS, 이미지 등)
└── temp/                 # 임시 파일 저장소
```

## 모델 정보

### 요약 모델
- 위치: `kosum-v1-tuned/`
- 용도: 뉴스 기사 요약

### 감정 분석 모델
- 위치: `sentiment_model/`
- 용도: 뉴스 기사 감정 분석 (긍정/부정/중립)

### OpenAI API 지원
- OpenAI API를 사용한 감정 분석도 지원합니다
- `model_mode`를 `'openai'`로 설정하고 `openai_api_key`를 제공하면 사용 가능합니다

## 배포

### 클라우드 배포 옵션

자세한 배포 가이드는 다음 문서를 참조하세요:

- [빠른 시작 가이드](QUICK_START.md)
- [상세 배포 가이드](DEPLOY.md)

### 주요 배포 플랫폼

- **Render**: Docker 기반 배포 지원
- **Railway**: GitHub 연동 자동 배포
- **Google Cloud Run**: 서버리스 컨테이너 배포
- **AWS**: ECS, EC2 등
- **VPS**: DigitalOcean, Linode 등

### 주의사항

- **메모리 요구사항**: 최소 2GB, 권장 4GB 이상
- **모델 파일 크기**: Docker 이미지가 5GB 이상일 수 있음
- **포트 설정**: 기본 포트는 8000 (환경 변수 `PORT`로 변경 가능)

## 문제 해결

### 포트가 이미 사용 중인 경우

```bash
# 다른 포트 사용
docker run -p 8001:8000 news-thermometer
```

또는 환경 변수 설정:
```bash
export PORT=8001
```

### 메모리 부족 오류

- Docker Desktop 설정에서 메모리 할당량을 4GB 이상으로 증가
- 배포 플랫폼에서 메모리 제한 확인

### 모델 로딩 실패

```bash
# 모델 파일 확인
docker run --rm news-thermometer ls -lh kosum-v1-tuned/
docker run --rm news-thermometer ls -lh sentiment_model/
```

### 의존성 설치 오류

```bash
# pip 업그레이드 후 재시도
pip install --upgrade pip
pip install -r requirements.txt
```

## 개발

### 코드 스타일

- Python 코드는 PEP 8 스타일 가이드를 따릅니다
- 타입 힌트를 사용합니다

### 테스트

```bash
# 서버 테스트
python test_server.py
```

## 기여

버그 리포트, 기능 제안, Pull Request를 환영합니다!

## 문의

문제가 발생하거나 질문이 있으시면 이슈를 등록해주세요.

---

**뉴스 온도계**로 뉴스의 감정을 측정해보세요! 📰🌡️

