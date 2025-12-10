# 빠른 시작 가이드

## 로컬에서 Docker로 실행하기

### 1. Docker 설치 확인

```bash
docker --version
docker-compose --version
```

Docker가 설치되어 있지 않다면: https://docs.docker.com/get-docker/

### 2. 배포 스크립트 실행

**Windows:**
```cmd
deploy.bat
```

**Linux/Mac:**
```bash
chmod +x deploy.sh
./deploy.sh
```

### 3. 수동으로 실행하기

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

## 클라우드 배포

### Render (가장 간단)

1. [Render](https://render.com) 가입
2. "New +" → "Web Service"
3. GitHub 저장소 연결
4. 설정:
   - **Environment**: Docker
   - **Dockerfile Path**: Dockerfile
   - **Plan**: Standard 이상 (모델 크기 고려)
5. 배포 시작

### Railway

1. [Railway](https://railway.app) 가입
2. "New Project" → "Deploy from GitHub"
3. 저장소 선택
4. 자동 배포 시작

### Google Cloud Run

```bash
# 프로젝트 설정
gcloud config set project YOUR_PROJECT_ID

# 이미지 빌드 및 푸시
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/news-thermometer

# 배포
gcloud run deploy news-thermometer \
  --image gcr.io/YOUR_PROJECT_ID/news-thermometer \
  --platform managed \
  --region asia-northeast3 \
  --memory 4Gi \
  --timeout 300 \
  --allow-unauthenticated
```

## 접속 확인

배포 완료 후 다음 주소로 접속:

- 메인 페이지: `http://your-domain/login`
- API 문서: `http://your-domain/docs`
- 헬스 체크: `http://your-domain/api/health`

## 문제 해결

### 포트가 이미 사용 중인 경우

```bash
# 다른 포트 사용
docker run -p 8001:8000 news-thermometer
```

### 메모리 부족 오류

Docker Desktop 설정에서 메모리 할당량을 4GB 이상으로 증가하세요.

### 모델 로딩 실패

모델 파일이 제대로 포함되었는지 확인:
```bash
docker run --rm news-thermometer ls -lh kosum-v1-tuned/
```

## 다음 단계

자세한 배포 정보는 [DEPLOY.md](DEPLOY.md)를 참조하세요.

