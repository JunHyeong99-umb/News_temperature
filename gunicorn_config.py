# Gunicorn 설정 파일 (프로덕션 환경용)
# 사용법: gunicorn -c gunicorn_config.py app:app

import multiprocessing
import os

# 서버 소켓
bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"
backlog = 2048

# Worker 프로세스
workers = int(os.getenv('WORKERS', multiprocessing.cpu_count() * 2 + 1))
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = 120
keepalive = 5

# 로깅
accesslog = "-"  # stdout
errorlog = "-"   # stderr
loglevel = os.getenv('LOG_LEVEL', 'info')
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# 프로세스 이름
proc_name = "news-thermometer"

# 서버 메커니즘
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# 성능
preload_app = True
max_requests = 1000
max_requests_jitter = 50

