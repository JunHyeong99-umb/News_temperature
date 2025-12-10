# 다음 단계 가이드

## 현재 상태
- ✅ Git 저장소 초기화됨
- ✅ 원격 저장소 연결됨 (origin/main)
- ⚠️ Docker 설치 필요 (로컬 테스트용)

## 단계별 진행 방법

### 1단계: 배포 파일들을 Git에 추가하기

```bash
# 변경된 파일과 새 파일 추가
git add .

# 커밋
git commit -m "배포 설정 파일 추가 (Docker, Render, Railway)"

# GitHub에 푸시
git push origin main
```

### 2단계: 클라우드 배포 선택하기

#### 옵션 A: Render (가장 간단, 추천) ⭐

1. **Render 가입**
   - https://render.com 접속
   - GitHub 계정으로 로그인

2. **새 웹 서비스 생성**
   - "New +" 버튼 클릭
   - "Web Service" 선택
   - GitHub 저장소 선택 (news2)

3. **설정**
   - **Name**: news-thermometer
   - **Environment**: Docker
   - **Dockerfile Path**: Dockerfile (자동 인식됨)
   - **Plan**: Standard 이상 선택 (모델 크기 고려)
   - **Region**: Singapore 또는 가장 가까운 지역

4. **배포 시작**
   - "Create Web Service" 클릭
   - 자동으로 빌드 및 배포 시작 (10-20분 소요)

5. **완료 후**
   - 배포 완료되면 URL이 생성됨 (예: https://news-thermometer.onrender.com)
   - 해당 URL로 접속하여 테스트

#### 옵션 B: Railway (간단)

1. **Railway 가입**
   - https://railway.app 접속
   - GitHub 계정으로 로그인

2. **새 프로젝트 생성**
   - "New Project" 클릭
   - "Deploy from GitHub" 선택
   - 저장소 선택

3. **자동 배포**
   - Railway가 자동으로 Dockerfile 인식
   - 자동 배포 시작

4. **완료 후**
   - 배포 완료되면 URL 생성됨
   - Settings에서 Custom Domain 설정 가능

#### 옵션 C: Google Cloud Run (고급)

```bash
# Google Cloud SDK 설치 필요
# https://cloud.google.com/sdk/docs/install

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

### 3단계: 로컬에서 Docker 테스트 (선택사항)

Docker를 설치하고 싶다면:

1. **Docker Desktop 설치**
   - https://docs.docker.com/get-docker/
   - Windows용 Docker Desktop 다운로드 및 설치

2. **로컬 테스트**
   ```cmd
   deploy.bat
   ```

3. **접속 확인**
   - http://localhost:8000

## 빠른 시작 (GitHub 푸시만 하면 됨)

```bash
# 1. 파일 추가 및 커밋
git add .
git commit -m "배포 설정 추가"

# 2. GitHub에 푸시
git push origin main

# 3. Render 또는 Railway에서 배포 시작
```

## 배포 후 확인사항

1. **헬스 체크**
   ```
   https://your-app-url/api/health
   ```

2. **메인 페이지**
   ```
   https://your-app-url/login
   ```

3. **API 문서**
   ```
   https://your-app-url/docs
   ```

## 문제 해결

### 배포 실패 시
- 로그 확인: Render/Railway 대시보드에서 로그 확인
- 메모리 부족: 플랜 업그레이드 고려
- 빌드 타임아웃: 모델 파일 크기로 인해 시간이 오래 걸릴 수 있음

### 모델 파일이 너무 큰 경우
- Git LFS 사용 고려
- 또는 모델 파일을 별도 스토리지에 저장하고 런타임에 다운로드

## 추천 배포 플랫폼 비교

| 플랫폼 | 난이도 | 무료 플랜 | 추천도 |
|--------|--------|-----------|--------|
| Render | ⭐ 쉬움 | 있음 (제한적) | ⭐⭐⭐⭐⭐ |
| Railway | ⭐⭐ 보통 | 있음 | ⭐⭐⭐⭐ |
| Cloud Run | ⭐⭐⭐ 어려움 | 있음 (제한적) | ⭐⭐⭐ |
| AWS | ⭐⭐⭐⭐ 어려움 | 없음 | ⭐⭐ |

**추천**: Render가 가장 간단하고 빠르게 시작할 수 있습니다!

