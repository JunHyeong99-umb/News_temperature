# 모델 파일 다운로드 가이드

## 문제
GitHub에 모델 파일(2GB+)을 푸시할 수 없어서, 모델 파일은 로컬에만 보관하고 코드만 배포합니다.

## 배포 시 모델 파일 처리 방법

### 옵션 1: 로컬에서 Docker 빌드 (권장)

모델 파일이 있는 로컬에서 Docker 이미지를 빌드하면 모델이 포함됩니다:

```bash
# 로컬에서 빌드
docker build -t news-thermometer .

# 이미지를 Docker Hub나 다른 레지스트리에 푸시
docker tag news-thermometer your-username/news-thermometer
docker push your-username/news-thermometer
```

### 옵션 2: 배포 플랫폼에서 모델 다운로드

배포 플랫폼의 빌드 단계에서 모델을 다운로드하도록 설정:

#### Render/Railway에서 사용할 수 있는 방법:

1. **환경 변수에 모델 다운로드 URL 설정**
2. **Dockerfile 수정하여 빌드 시 모델 다운로드**

예시 Dockerfile 추가 스크립트:
```dockerfile
# 모델 다운로드 (Google Drive, Dropbox 등)
RUN curl -L "YOUR_MODEL_DOWNLOAD_URL" -o kosum-v1-tuned/model.safetensors
```

### 옵션 3: 모델을 별도 스토리지에 저장

- Google Drive
- AWS S3
- Dropbox
- GitHub Releases (LFS 사용)

## 현재 권장 방법

**로컬에서 Docker 이미지를 빌드한 후 Docker Hub에 푸시하는 방법**이 가장 간단합니다:

1. 로컬에서 `docker build` 실행 (모델 파일 포함)
2. Docker Hub에 푸시
3. Render/Railway에서 Docker Hub 이미지 사용

이렇게 하면 GitHub에 모델 파일을 푸시할 필요가 없습니다!

