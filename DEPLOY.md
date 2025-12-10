# 뉴스 온도계 배포 가이드

이 문서는 뉴스 온도계 웹 애플리케이션을 배포하는 방법을 설명합니다.

## 배포 옵션

### 1. Docker를 사용한 배포 (권장)

#### 로컬에서 Docker 이미지 빌드 및 실행

```bash
# Docker 이미지 빌드
docker build -t news-thermometer .

# Docker 컨테이너 실행
docker run -d -p 8000:8000 --name news-thermometer news-thermometer

# 또는 docker-compose 사용
docker-compose up -d
```

#### Docker Hub에 푸시하여 배포

```bash
# Docker Hub에 로그인
docker login

# 이미지 태그 지정 (your-username을 실제 사용자명으로 변경)
docker tag news-thermometer your-username/news-thermometer:latest

# Docker Hub에 푸시
docker push your-username/news-thermometer:latest
```

### 2. Render 배포

1. [Render](https://render.com)에 가입 및 로그인
2. "New +" → "Web Service" 선택
3. GitHub 저장소 연결 또는 Docker 이미지 URL 입력
4. 설정:
   - **Name**: news-thermometer
   - **Environment**: Docker
   - **Dockerfile Path**: Dockerfile
   - **Port**: 8000
5. "Create Web Service" 클릭

**참고**: Render의 무료 플랜은 모델 파일 크기 제한이 있을 수 있습니다. 유료 플랜을 고려하세요.

### 3. Railway 배포

1. [Railway](https://railway.app)에 가입 및 로그인
2. "New Project" → "Deploy from GitHub" 선택
3. 저장소 선택 후 배포
4. 환경 변수 설정 (필요시)

Railway는 Dockerfile을 자동으로 인식합니다.

### 4. Google Cloud Run 배포

```bash
# Google Cloud SDK 설치 및 로그인
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Docker 이미지 빌드
docker build -t gcr.io/YOUR_PROJECT_ID/news-thermometer .

# Google Container Registry에 푸시
docker push gcr.io/YOUR_PROJECT_ID/news-thermometer

# Cloud Run에 배포
gcloud run deploy news-thermometer \
  --image gcr.io/YOUR_PROJECT_ID/news-thermometer \
  --platform managed \
  --region asia-northeast3 \
  --allow-unauthenticated \
  --memory 4Gi \
  --timeout 300
```

### 5. AWS 배포

#### AWS ECS (Elastic Container Service)

1. AWS ECR에 Docker 이미지 푸시
2. ECS 클러스터 생성
3. 태스크 정의 생성 (최소 4GB 메모리 권장)
4. 서비스 생성 및 배포

#### AWS EC2

```bash
# EC2 인스턴스에 SSH 접속 후
git clone YOUR_REPO_URL
cd news2
docker-compose up -d
```

### 6. VPS 서버 배포 (DigitalOcean, Linode 등)

```bash
# 서버에 SSH 접속
ssh user@your-server-ip

# Docker 설치 (Ubuntu/Debian)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 프로젝트 클론
git clone YOUR_REPO_URL
cd news2

# Docker Compose로 실행
docker-compose up -d

# Nginx 리버스 프록시 설정 (선택사항)
sudo apt-get install nginx
# /etc/nginx/sites-available/news-thermometer 설정 파일 생성
```

## 주의사항

### 모델 파일 크기

이 프로젝트는 대용량 모델 파일을 포함하고 있습니다:
- `kosum-v1-tuned/`: 요약 모델
- `sentiment_model/`: 감정 분석 모델

Docker 이미지 크기가 5GB 이상일 수 있습니다. 배포 플랫폼의 저장소 제한을 확인하세요.

### 메모리 요구사항

- 최소: 2GB RAM
- 권장: 4GB RAM 이상
- 모델 로딩 시 추가 메모리 필요

### 포트 설정

기본 포트는 8000입니다. 환경 변수 `PORT`로 변경 가능합니다.

### 세션 관리

현재는 메모리 기반 세션을 사용합니다. 프로덕션 환경에서는 Redis를 사용하는 것을 권장합니다.

## 환경 변수

필요한 환경 변수:
- `PORT`: 서버 포트 (기본값: 8000)

## 헬스 체크

애플리케이션은 `/api/health` 엔드포인트를 제공합니다:
```bash
curl http://localhost:8000/api/health
```

## 문제 해결

### Docker 빌드 실패
- 메모리 부족: Docker Desktop의 메모리 할당량 증가
- 네트워크 오류: pip 설치 시 타임아웃 발생 가능, 재시도

### 모델 로딩 실패
- 메모리 부족: 컨테이너 메모리 제한 확인
- 파일 경로: 모델 파일 경로 확인

### 포트 충돌
- 다른 포트 사용: `docker run -p 8001:8000` 또는 환경 변수 `PORT=8001` 설정

## 모니터링

프로덕션 환경에서는 다음을 고려하세요:
- 로그 수집 (CloudWatch, Datadog 등)
- 성능 모니터링
- 에러 추적 (Sentry 등)
- 자동 스케일링 설정

## 보안

프로덕션 배포 시:
- HTTPS 사용 (Let's Encrypt 등)
- 환경 변수로 민감한 정보 관리
- CORS 설정 확인
- Rate limiting 추가 고려

