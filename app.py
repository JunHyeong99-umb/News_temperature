"""
뉴스 온도계 - 뉴스 감정 분석 및 요약 서비스
"""
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
import secrets
from datetime import datetime, timedelta
from src.crawl_naver_api import NaverNewsAPICrawler
from src.sentiment_analyzer import SentimentAnalyzer

app = FastAPI(title="뉴스 온도계", description="뉴스 감정 분석 및 요약 서비스")

# 세션 저장소 (실제 운영 환경에서는 Redis나 DB 사용 권장)
sessions = {}

# 감정 분석기 초기화 (지연 로딩)
sentiment_analyzer = None
sentiment_analyzer_openai = None  # OpenAI API를 사용하는 감정 분석기

def get_sentiment_analyzer(openai_api_key: Optional[str] = None, use_openai: bool = False):
    """감정 분석기 인스턴스를 가져옵니다 (지연 로딩)"""
    global sentiment_analyzer, sentiment_analyzer_openai
    
    if use_openai and openai_api_key:
        # OpenAI API 사용
        if sentiment_analyzer_openai is None:
            try:
                sentiment_analyzer_openai = SentimentAnalyzer(
                    openai_api_key=openai_api_key,
                    use_openai=True
                )
                print("OpenAI API 감정 분석기 초기화 완료")
            except Exception as e:
                print(f"OpenAI API 감정 분석기 초기화 실패: {e}")
                sentiment_analyzer_openai = None
        return sentiment_analyzer_openai
    else:
        # 로컬 모델 사용
        if sentiment_analyzer is None:
            try:
                sentiment_analyzer = SentimentAnalyzer()
                print("로컬 감정 분석기 초기화 완료")
            except Exception as e:
                print(f"로컬 감정 분석기 초기화 실패: {e}")
                sentiment_analyzer = None
        return sentiment_analyzer

# temp 폴더 생성
os.makedirs('temp', exist_ok=True)

# 정적 파일 서빙 (이미지 파일)
app.mount("/temp", StaticFiles(directory="temp"), name="temp")
os.makedirs('static', exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


class LoginRequest(BaseModel):
    """로그인 요청 모델"""
    client_id: str
    client_secret: str


class TestRequest(BaseModel):
    """API 테스트 요청 모델"""
    query: str
    
    max_results: int = 10
    days: int = 1
    include_full_text: bool = True
    sort_by: str = 'date'  # 'date': 날짜순, 'view': 조회수순
    model_mode: str = 'local'  # 'local': 로컬 모델 사용, 'openai': OpenAI API 사용
    openai_api_key: Optional[str] = None  # OpenAI API 키 (model_mode가 'openai'일 때 필요)


def get_session(request: Request) -> Optional[dict]:
    """세션 정보를 가져옵니다"""
    session_id = request.cookies.get("session_id")
    if session_id and session_id in sessions:
        session = sessions[session_id]
        # 세션 만료 확인 (24시간)
        if datetime.now() - session["created_at"] < timedelta(hours=24):
            return session
        else:
            # 만료된 세션 삭제
            del sessions[session_id]
    return None


def require_login(request: Request) -> dict:
    """로그인이 필요한 엔드포인트에서 사용하는 의존성"""
    session = get_session(request)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="로그인이 필요합니다"
        )
    return session


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """메인 페이지 - 로그인 체크 후 테스트 인터페이스"""
    session = get_session(request)
    if not session:
        return RedirectResponse(url="/login", status_code=302)
    
    # 로그인된 사용자의 Client ID와 Secret 사용
    client_id = session["client_id"]
    client_secret = session["client_secret"]
    html_content = """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>뉴스 온도계</title>
        <link rel="icon" href="/static/favicon.svg" type="image/svg+xml">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', 'Helvetica Neue', Arial, sans-serif;
                background: #f5f5f7;
                min-height: 100vh;
                padding: 0;
                color: #1d1d1f;
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
            }
            .header {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: saturate(180%) blur(20px);
                -webkit-backdrop-filter: saturate(180%) blur(20px);
                border-bottom: 1px solid rgba(0, 0, 0, 0.08);
                padding: 20px 0;
                position: sticky;
                top: 0;
                z-index: 100;
            }
            .header-content {
                max-width: 1200px;
                margin: 0 auto;
                padding: 0 40px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .header h1 {
                font-size: 28px;
                font-weight: 600;
                color: #1d1d1f;
                letter-spacing: -0.5px;
                display: flex;
                align-items: center;
                gap: 12px;
            }
            .logo-icon {
                width: 32px;
                height: 32px;
                display: inline-block;
                object-fit: contain;
            }
            .user-info {
                display: flex;
                align-items: center;
                gap: 15px;
                font-size: 14px;
                color: #86868b;
            }
            .logout-btn {
                background: transparent;
                border: 1px solid #d2d2d7;
                color: #1d1d1f;
                padding: 8px 16px;
                border-radius: 20px;
                font-size: 14px;
                font-weight: 400;
                cursor: pointer;
                transition: all 0.2s ease;
                pointer-events: auto;
                position: relative;
                z-index: 10;
            }
            .logout-btn:hover {
                background: #1d1d1f;
                color: white;
                border-color: #1d1d1f;
            }
            .logout-btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 60px 40px;
            }
            .search-section {
                background: white;
                border-radius: 24px;
                padding: 50px;
                margin-bottom: 40px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
            }
            h2 {
                font-size: 48px;
                font-weight: 600;
                color: #1d1d1f;
                margin-bottom: 12px;
                letter-spacing: -1px;
            }
            .subtitle {
                font-size: 21px;
                color: #86868b;
                margin-bottom: 50px;
                font-weight: 400;
            }
            .form-group {
                margin-bottom: 30px;
            }
            label {
                display: block;
                margin-bottom: 10px;
                color: #1d1d1f;
                font-weight: 500;
                font-size: 17px;
            }
            input[type="text"], input[type="number"], select {
                width: 100%;
                padding: 14px 18px;
                border: 1px solid #d2d2d7;
                border-radius: 16px;
                font-size: 17px;
                background: #fbfbfd;
                transition: all 0.2s ease;
                font-family: inherit;
            }
            select {
                appearance: none;
                -webkit-appearance: none;
                -moz-appearance: none;
                background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%23333' d='M6 9L1 4h10z'/%3E%3C/svg%3E");
                background-repeat: no-repeat;
                background-position: right 18px center;
                padding-right: 45px;
                cursor: pointer;
            }
            .custom-select-wrapper {
                position: relative;
            }
            .custom-select {
                appearance: none;
                -webkit-appearance: none;
                -moz-appearance: none;
                background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%23333' d='M6 9L1 4h10z'/%3E%3C/svg%3E");
                background-repeat: no-repeat;
                background-position: right 18px center;
                padding-right: 45px;
                cursor: pointer;
            }
            input:focus, select:focus {
                outline: none;
                border-color: #0071e3;
                background: white;
                box-shadow: 0 0 0 4px rgba(0, 113, 227, 0.1);
            }
            .custom-select:focus {
                background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%230071e3' d='M6 9L1 4h10z'/%3E%3C/svg%3E");
            }
            /* 드롭다운 리스트 둥글게 만들기 시도 */
            .custom-select::-ms-expand {
                display: none;
            }
            /* 드롭다운 리스트 스타일 개선 (브라우저 제한으로 완벽하지 않을 수 있음) */
            .custom-select option {
                padding: 12px 18px;
                border-radius: 8px;
            }
            /* 커스텀 드롭다운 스타일 */
            .custom-dropdown {
                position: relative;
                width: 100%;
            }
            .custom-dropdown-selected {
                width: 100%;
                padding: 14px 18px;
                border: 1px solid #d2d2d7;
                border-radius: 16px;
                font-size: 17px;
                background: #fbfbfd;
                cursor: pointer;
                display: flex;
                justify-content: space-between;
                align-items: center;
                transition: all 0.2s ease;
                font-family: inherit;
            }
            .custom-dropdown-selected:hover {
                border-color: #0071e3;
                background: white;
            }
            .custom-dropdown-selected.active {
                border-color: #0071e3;
                background: white;
                box-shadow: 0 0 0 4px rgba(0, 113, 227, 0.1);
            }
            .dropdown-arrow {
                color: #86868b;
                font-size: 12px;
                transition: transform 0.3s ease;
            }
            .custom-dropdown-selected.active .dropdown-arrow {
                transform: rotate(180deg);
            }
            .custom-dropdown-list {
                position: absolute;
                top: calc(100% + 8px);
                left: 0;
                right: 0;
                background: white;
                border: 1px solid #d2d2d7;
                border-radius: 16px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                overflow: hidden;
                display: none;
                z-index: 1000;
                padding: 8px;
            }
            .custom-dropdown-list.show {
                display: block;
            }
            .custom-dropdown-option {
                padding: 12px 18px;
                border-radius: 12px;
                cursor: pointer;
                transition: all 0.2s ease;
                color: #1d1d1f;
                font-size: 17px;
                margin-bottom: 4px;
            }
            .custom-dropdown-option:last-child {
                margin-bottom: 0;
            }
            .custom-dropdown-option:hover {
                background: #f5f5f7;
            }
            .custom-dropdown-option.selected {
                background: #0071e3;
                color: white;
            }
            .form-row {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }
            button {
                background: #0071e3;
                color: white;
                padding: 16px 32px;
                border: none;
                border-radius: 16px;
                font-size: 17px;
                font-weight: 500;
                cursor: pointer;
                width: 100%;
                transition: all 0.2s ease;
                font-family: inherit;
            }
            button:hover {
                background: #0077ed;
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(0, 113, 227, 0.3);
            }
            button:active {
                transform: translateY(0);
            }
            button:disabled {
                background: #d2d2d7;
                cursor: not-allowed;
                transform: none;
                box-shadow: none;
            }
            .loading {
                display: none;
                text-align: center;
                margin: 60px 0;
            }
            .loading.active {
                display: block;
            }
            .spinner {
                border: 3px solid #f5f5f7;
                border-top: 3px solid #0071e3;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 0.8s linear infinite;
                margin: 0 auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .results {
                margin-top: 40px;
                display: none;
            }
            .results.active {
                display: block;
            }
            .results h2 {
                font-size: 40px;
                margin-bottom: 30px;
            }
            .results-grid {
                display: grid;
                grid-template-columns: 1fr;
                gap: 24px;
                margin-top: 30px;
            }
            .result-card {
                background: white;
                border-radius: 20px;
                padding: 28px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                cursor: default;
                border: 1px solid transparent;
                overflow: visible !important;
                overflow-x: visible !important;
                overflow-y: visible !important;
            }
            .result-card:hover {
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            }
            .result-card h3 {
                color: #1d1d1f;
                margin-bottom: 12px;
                font-size: 20px;
                font-weight: 600;
                line-height: 1.4;
                word-wrap: break-word;
                overflow-wrap: break-word;
                letter-spacing: -0.3px;
                white-space: normal !important;
                display: block !important;
                width: 100% !important;
                overflow: visible !important;
                text-overflow: clip !important;
                max-height: none !important;
                min-height: auto !important;
                height: auto !important;
                -webkit-line-clamp: unset !important;
                line-clamp: unset !important;
                text-overflow: unset !important;
                overflow-x: visible !important;
                overflow-y: visible !important;
            }
            .result-card .meta {
                color: #86868b;
                font-size: 14px;
                margin-bottom: 16px;
                display: flex;
                flex-wrap: wrap;
                gap: 12px;
            }
            .result-card .meta span {
                display: inline-block;
            }
            .result-card .summary {
                color: #515154;
                font-size: 15px;
                line-height: 1.6;
                margin-bottom: 20px;
                word-wrap: break-word;
            }
            .sentiment-box {
                background: #f5f5f7;
                border-radius: 16px;
                padding: 20px;
                margin: 20px 0;
                display: flex;
                align-items: center;
                gap: 20px;
            }
            .sentiment-image {
                width: 120px;
                height: auto;
                min-height: 80px;
                max-height: 150px;
                object-fit: contain;
                border-radius: 12px;
                flex-shrink: 0;
            }
            .sentiment-info {
                flex: 1;
            }
            .sentiment-label {
                font-size: 16px;
                font-weight: 500;
                margin-bottom: 8px;
                color: #1d1d1f;
            }
            .sentiment-temperature {
                font-size: 32px;
                font-weight: 600;
                color: #0071e3;
                letter-spacing: -0.5px;
            }
            .result-card a {
                color: #0071e3;
                text-decoration: none;
                font-size: 14px;
                font-weight: 500;
                display: inline-flex;
                align-items: center;
                gap: 6px;
            }
            .result-card a:hover {
                text-decoration: underline;
            }
            .error {
                background: #fff3f3;
                border: 1px solid #ff3b30;
                color: #d70015;
                padding: 20px;
                border-radius: 16px;
                margin-top: 20px;
                font-size: 15px;
            }
            /* 커스텀 Alert 모달 */
            .custom-alert-overlay {
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0, 0, 0, 0.4);
                backdrop-filter: blur(4px);
                z-index: 10000;
                align-items: center;
                justify-content: center;
            }
            .custom-alert-overlay.show {
                display: flex;
            }
            .custom-alert-modal {
                background: white;
                border-radius: 20px;
                padding: 0;
                max-width: 400px;
                width: 90%;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
                animation: modalSlideIn 0.3s ease-out;
                overflow: hidden;
            }
            @keyframes modalSlideIn {
                from {
                    opacity: 0;
                    transform: translateY(-20px) scale(0.95);
                }
                to {
                    opacity: 1;
                    transform: translateY(0) scale(1);
                }
            }
            .custom-alert-content {
                padding: 32px 24px 24px 24px;
            }
            .custom-alert-title {
                font-size: 20px;
                font-weight: 600;
                color: #1d1d1f;
                margin-bottom: 12px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .custom-alert-message {
                font-size: 17px;
                color: #515154;
                line-height: 1.5;
                margin-bottom: 24px;
            }
            .custom-alert-buttons {
                display: flex;
                justify-content: flex-end;
                gap: 12px;
                padding-top: 20px;
                border-top: 1px solid #e5e5e7;
            }
            .custom-alert-btn {
                padding: 12px 24px;
                border: none;
                border-radius: 12px;
                font-size: 17px;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.2s ease;
                font-family: inherit;
            }
            .custom-alert-btn-primary {
                background: #0071e3;
                color: white;
            }
            .custom-alert-btn-primary:hover {
                background: #0077ed;
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(0, 113, 227, 0.3);
            }
            .custom-alert-btn-primary:active {
                transform: translateY(0);
            }
            .info {
                background: #f5f5f7;
                border-radius: 16px;
                padding: 24px;
                margin-bottom: 30px;
                font-size: 15px;
                line-height: 1.6;
                color: #515154;
            }
            .info strong {
                color: #1d1d1f;
                display: block;
                margin-bottom: 8px;
                font-size: 17px;
            }
            .accordion {
                margin-bottom: 20px;
            }
            .accordion-header {
                background: #f5f5f7;
                border-radius: 16px;
                padding: 16px 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                cursor: pointer;
                user-select: none;
                transition: background-color 0.2s ease;
            }
            .accordion-header:hover {
                background: #e5e5e7;
            }
            .accordion-header strong {
                color: #1d1d1f;
                font-size: 17px;
            }
            .accordion-icon {
                color: #86868b;
                font-size: 14px;
                transition: transform 0.3s ease;
            }
            .accordion-content {
                max-height: 0;
                overflow: hidden;
                transition: max-height 0.3s ease-out;
            }
            .checkbox-group {
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .checkbox-group input[type="checkbox"] {
                width: auto;
            }
            input[type="radio"] {
                width: auto;
                margin-right: 10px;
                cursor: pointer;
                accent-color: #0071e3;
            }
            label[for] {
                cursor: pointer;
            }
            .radio-group {
                display: flex;
                flex-direction: column;
                gap: 12px;
                margin-top: 12px;
            }
            .radio-group label {
                display: flex;
                align-items: center;
                cursor: pointer;
                font-weight: 400;
                font-size: 17px;
            }
            .result-count {
                color: #86868b;
                font-size: 17px;
                margin-bottom: 30px;
            }
            #openai_key_group {
                transition: all 0.3s ease;
                overflow: hidden;
            }
            @media (max-width: 768px) {
                .container {
                    padding: 30px 20px;
                }
                .search-section {
                    padding: 30px 20px;
                }
                h2 {
                    font-size: 36px;
                }
                .subtitle {
                    font-size: 18px;
                }
                .form-row {
                    grid-template-columns: 1fr;
                }
                .results-grid {
                    grid-template-columns: 1fr;
                }
                .header-content {
                    padding: 0 20px;
                }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <div class="header-content">
                <h1>
                    <img src="/temp/logo.png" alt="뉴스 온도계" class="logo-icon" />
                    뉴스 온도계
                </h1>
                <div class="user-info">
                    <span>${client_id}</span>
                    <button type="button" id="logoutBtn" class="logout-btn">로그아웃</button>
                </div>
            </div>
        </div>
        
        <div class="container">
            <div class="search-section">
                <h2>뉴스 검색</h2>
                <p class="subtitle">검색어를 입력하고 AI가 분석한 감정 온도와 요약을 확인하세요</p>
                
                <div class="info">
                    <strong>사용 방법</strong>
                    <ol style="margin: 12px 0; padding-left: 20px; line-height: 1.8;">
                        <li>검색어를 입력하세요</li>
                        <li>최대 결과 수와 검색 기간을 설정하세요</li>
                        <li>정렬 기준을 선택하세요</li>
                        <li>모델 선택에서 로컬 모델 또는 OpenAI API를 선택하세요</li>
                        <li>검색 버튼을 클릭하여 검색 결과를 확인하세요</li>
                    </ol>
                </div>
                
                <form id="testForm">
                    <div class="form-group">
                        <label for="query">검색어</label>
                        <input type="text" id="query" name="query" required 
                               placeholder="예: AI, 인공지능, 삼성전자" value="AI">
                    </div>
                    
                    <div class="form-row">
                        <div class="form-group">
                            <label for="max_results">최대 결과 수</label>
                            <input type="number" id="max_results" name="max_results" 
                                   min="1" max="1000" value="10">
                        </div>
                        <div class="form-group">
                            <label for="days">최근 며칠간</label>
                            <input type="number" id="days" name="days" 
                                   min="1" max="30" value="1">
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label for="sort_by">정렬 기준</label>
                        <div class="custom-dropdown">
                            <div class="custom-dropdown-selected" id="sort_by_display" onclick="toggleDropdown('sort_by')">
                                <span>날짜순 (최신순)</span>
                                <span class="dropdown-arrow">▼</span>
                            </div>
                            <div class="custom-dropdown-list" id="sort_by_list">
                                <div class="custom-dropdown-option selected" data-value="date" onclick="selectOption('sort_by', 'date', '날짜순 (최신순)')">날짜순 (최신순)</div>
                                <div class="custom-dropdown-option" data-value="view" onclick="selectOption('sort_by', 'view', '조회수순 (높은순)')">조회수순 (높은순)</div>
                            </div>
                            <select id="sort_by" name="sort_by" style="display: none;">
                                <option value="date" selected>날짜순 (최신순)</option>
                                <option value="view">조회수순 (높은순)</option>
                            </select>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label>모델 선택</label>
                        <div class="radio-group">
                            <label>
                                <input type="radio" name="model_mode" value="local" checked>
                                <span>로컬 모델 - 정확도 낮음</span>
                            </label>
                            <label>
                                <input type="radio" name="model_mode" value="openai">
                                <span>OpenAI API - 정확도 높음</span>
                            </label>
                        </div>
                        <p style="margin-top: 12px; font-size: 14px; color: #86868b; line-height: 1.6;">
                            <strong>로컬 모델:</strong> kosum-v1-tuned (요약) + 파인튜닝된 감정 분석 모델<br>
                            <strong>OpenAI API:</strong> OpenAI API (요약) + OpenAI API (감정 분석)
                        </p>
                    </div>
                    
                    <div class="form-group" id="openai_key_group" style="display: none;">
                        <label for="openai_api_key">OpenAI API 키</label>
                        <input type="text" id="openai_api_key" name="openai_api_key" 
                               placeholder="sk-... (OpenAI API 키를 입력하세요)">
                        <p style="margin-top: 8px; font-size: 14px; color: #86868b;">
                            OpenAI API를 선택한 경우 필요합니다. <a href="https://platform.openai.com/api-keys" target="_blank" style="text-decoration: none;">키 발급 받기</a>
                        </p>
                    </div>
                    
                    <button type="button" id="submitBtn">검색</button>
                </form>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p style="margin-top: 16px; color: #86868b; font-size: 17px;">뉴스를 검색하고 있습니다...</p>
                </div>
            </div>
            
            <div class="results" id="results">
                <h2>검색 결과</h2>
                <div id="resultContent"></div>
            </div>
        </div>
        
        <!-- 커스텀 Alert 모달 -->
        <div class="custom-alert-overlay" id="customAlertOverlay">
            <div class="custom-alert-modal">
                <div class="custom-alert-content">
                    <div class="custom-alert-title">
                        <span>⚠️ 알림</span>
                    </div>
                    <div class="custom-alert-message" id="customAlertMessage"></div>
                    <div class="custom-alert-buttons">
                        <button type="button" class="custom-alert-btn custom-alert-btn-primary" id="customAlertOkBtn">확인</button>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            function toggleDropdown(name) {
                var list = document.getElementById(name + '_list');
                var selected = document.getElementById(name + '_display');
                
                // 다른 드롭다운 닫기
                document.querySelectorAll('.custom-dropdown-list').forEach(function(dropdown) {
                    if (dropdown !== list) {
                        dropdown.classList.remove('show');
                        dropdown.parentElement.querySelector('.custom-dropdown-selected').classList.remove('active');
                    }
                });
                
                // 현재 드롭다운 토글
                list.classList.toggle('show');
                selected.classList.toggle('active');
            }
            
            function selectOption(name, value, text) {
                var select = document.getElementById(name);
                var display = document.getElementById(name + '_display');
                var list = document.getElementById(name + '_list');
                
                select.value = value;
                display.querySelector('span:first-child').textContent = text;
                
                // 옵션 선택 상태 업데이트
                list.querySelectorAll('.custom-dropdown-option').forEach(function(option) {
                    option.classList.remove('selected');
                });
                list.querySelector('[data-value="' + value + '"]').classList.add('selected');
                
                // 드롭다운 닫기
                list.classList.remove('show');
                display.classList.remove('active');
            }
            
            // 외부 클릭 시 드롭다운 닫기
            document.addEventListener('click', function(event) {
                if (!event.target.closest('.custom-dropdown')) {
                    document.querySelectorAll('.custom-dropdown-list').forEach(function(list) {
                        list.classList.remove('show');
                        list.parentElement.querySelector('.custom-dropdown-selected').classList.remove('active');
                    });
                }
            });
            
            // 커스텀 Alert 함수
            function customAlert(message) {
                var overlay = document.getElementById('customAlertOverlay');
                var messageEl = document.getElementById('customAlertMessage');
                var okBtn = document.getElementById('customAlertOkBtn');
                
                if (!overlay || !messageEl || !okBtn) {
                    // 폴백: 기본 alert 사용
                    alert(message);
                    return;
                }
                
                messageEl.textContent = message;
                overlay.classList.add('show');
                
                // 확인 버튼 클릭 시 닫기
                var closeAlert = function() {
                    overlay.classList.remove('show');
                    okBtn.removeEventListener('click', closeAlert);
                    overlay.removeEventListener('click', handleOverlayClick);
                };
                
                var handleOverlayClick = function(e) {
                    if (e.target === overlay) {
                        closeAlert();
                    }
                };
                
                okBtn.addEventListener('click', closeAlert);
                overlay.addEventListener('click', handleOverlayClick);
                
                // ESC 키로 닫기
                var handleEscKey = function(e) {
                    if (e.key === 'Escape') {
                        closeAlert();
                        document.removeEventListener('keydown', handleEscKey);
                    }
                };
                document.addEventListener('keydown', handleEscKey);
            }
            
            function escapeHtml(text) {
                if (!text) return '';
                var div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }
            
            function decodeHtmlEntities(text) {
                if (!text) return '';
                var textarea = document.createElement('textarea');
                textarea.innerHTML = text;
                return textarea.value;
            }
            
            function formatText(text) {
                if (!text) return '';
                var decoded = decodeHtmlEntities(String(text));
                var escaped = escapeHtml(decoded);
                
                // 먼저 모든 줄바꿈을 공백으로 변환
                var newlineRegex = new RegExp('[\\r\\n]+', 'g');
                var textWithoutNewlines = escaped.replace(newlineRegex, ' ');
                
                // 연속된 공백을 하나로 정리
                var spaceRegex = new RegExp('\\s+', 'g');
                var normalizedText = textWithoutNewlines.replace(spaceRegex, ' ');
                
                // 마침표(.) 뒤에 줄바꿈 추가 (한 문장마다)
                // 마침표 뒤에 공백이 있으면 공백을 <br>로, 없으면 <br> 추가
                var formattedText = normalizedText.replace(/\\.\\s+/g, '.<br>');
                // 마지막 문장이 마침표로 끝나면 줄바꿈 추가
                formattedText = formattedText.replace(/\\.$/, '.<br>');
                
                return formattedText.trim();
            }
            
            // 로그아웃 함수 정의
            function handleLogout(e) {
                console.log('handleLogout 호출됨');
                if (e) {
                    e.preventDefault();
                    e.stopPropagation();
                }
                
                var logoutBtn = document.getElementById('logoutBtn');
                if (!logoutBtn) {
                    console.error('로그아웃 버튼을 찾을 수 없습니다');
                    window.location.href = '/login';
                    return false;
                }
                
                if (logoutBtn.disabled) {
                    return false;
                }
                
                logoutBtn.disabled = true;
                logoutBtn.textContent = '로그아웃 중...';
                
                fetch('/api/logout', { 
                    method: 'POST',
                    credentials: 'include',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                })
                .then(function(response) {
                    console.log('로그아웃 응답:', response.status);
                    window.location.href = '/login';
                })
                .catch(function(error) {
                    console.error('로그아웃 오류:', error);
                    window.location.href = '/login';
                });
                
                return false;
            }
            
            // 검색 함수 정의
            function handleTestClick() {
                console.log('테스트 버튼 클릭됨!');
                
                var form = document.getElementById('testForm');
                var submitBtn = document.getElementById('submitBtn');
                
                if (!form || !submitBtn) {
                    customAlert('폼 또는 버튼을 찾을 수 없습니다.');
                    console.error('폼 또는 버튼을 찾을 수 없습니다.');
                    return;
                }
                
                var formData = new FormData(form);
                var query = formData.get('query');
                
                // 검색어 검증
                if (!query || !query.trim()) {
                    customAlert('검색어를 입력해주세요.');
                    return;
                }
                
                var modelModeRadio = document.querySelector('input[name="model_mode"]:checked');
                var modelMode = modelModeRadio ? modelModeRadio.value : 'local';
                
                // OpenAI API 키가 필요한 경우 확인
                var needsOpenAIKey = (modelMode === 'openai');
                var openaiApiKey = needsOpenAIKey ? (formData.get('openai_api_key') || null) : null;
                
                // OpenAI API 키 검증
                if (needsOpenAIKey && (!openaiApiKey || !openaiApiKey.trim())) {
                    customAlert('OpenAI API 키를 입력해주세요.');
                    return;
                }
                
                var data = {
                    query: query.trim(),
                    max_results: parseInt(formData.get('max_results')) || 10,
                    days: parseInt(formData.get('days')) || 1,
                    include_full_text: true,
                    sort_by: formData.get('sort_by') || 'date',
                    model_mode: modelMode,
                    openai_api_key: openaiApiKey
                };
                
                if (isNaN(data.max_results) || data.max_results < 1) {
                    customAlert('최대 결과 수는 1 이상이어야 합니다.');
                    return;
                }
                if (isNaN(data.days) || data.days < 1) {
                    customAlert('날짜는 1 이상이어야 합니다.');
                    return;
                }
                
                var loading = document.getElementById('loading');
                var results = document.getElementById('results');
                var resultContent = document.getElementById('resultContent');
                
                if (!loading || !results || !resultContent) {
                    customAlert('페이지 오류가 발생했습니다.');
                    return;
                }
                
                loading.classList.add('active');
                results.classList.remove('active');
                submitBtn.disabled = true;
                resultContent.innerHTML = '';
                
                console.log('API 요청 시작:', data);
                
                fetch('/api/test', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                })
                .then(function(response) {
                    console.log('응답 상태:', response.status);
                    if (!response.ok) {
                        return response.text().then(function(text) {
                            throw new Error('서버 오류 (' + response.status + '): ' + text);
                        });
                    }
                    return response.json();
                })
                .then(function(result) {
                    console.log('응답 데이터:', result);
                    
                    if (result.success) {
                        if (result.data && result.data.length > 0) {
                            var html = '<p class="result-count">총 <strong>' + result.data.length + '</strong>개의 기사를 찾았습니다.</p>';
                            html += '<div class="results-grid">';
                            
                            result.data.forEach(function(item, index) {
                                var title = item.title || '제목 없음';
                                var titleOriginal = title;
                                title = decodeHtmlEntities(title);
                                title = escapeHtml(title);
                                titleOriginal = escapeHtml(titleOriginal);
                                
                                html += '<div class="result-card">';
                                html += '<h3 title="' + titleOriginal + '" style="white-space: normal !important; overflow: visible !important; overflow-x: visible !important; overflow-y: visible !important; text-overflow: clip !important; text-overflow: unset !important; max-height: none !important; min-height: auto !important; height: auto !important; display: block !important; width: 100% !important; -webkit-line-clamp: unset !important; line-clamp: unset !important; word-break: break-word !important;">' + title + '</h3>';
                                
                                html += '<div class="meta">';
                                if (item.source) {
                                    html += '<span>' + escapeHtml(item.source) + '</span>';
                                }
                                if (item.pubDate) {
                                    html += '<span>' + escapeHtml(item.pubDate) + '</span>';
                                }
                                if (item.view_count !== undefined) {
                                    html += '<span>' + item.view_count.toLocaleString() + '회 조회</span>';
                                }
                                html += '</div>';
                                
                                if (item.sentiment) {
                                    html += '<div class="sentiment-box" onclick="event.stopPropagation()">';
                                    var imgPath = escapeHtml(item.sentiment.image_path);
                                    html += '<img src="/' + imgPath + '" alt="감정 분석" class="sentiment-image" onerror="this.style.display=&quot;none&quot;">';
                                    html += '<div class="sentiment-info">';
                                    html += '<div class="sentiment-label">' + escapeHtml(item.sentiment.label) + '</div>';
                                    html += '<div class="sentiment-temperature">' + escapeHtml(item.sentiment.temperature) + '°C</div>';
                                    html += '</div></div>';
                                }
                                
                                if (item.text) {
                                    var formattedText = formatText(item.text);
                                    html += '<div class="summary">' + formattedText + '</div>';
                                }
                                
                                html += '<a href="' + escapeHtml(item.link) + '" target="_blank" onclick="event.stopPropagation()">기사 보기 →</a>';
                                html += '</div>';
                            });
                            
                            html += '</div>';
                            resultContent.innerHTML = html;
                        } else {
                            resultContent.innerHTML = '<div class="error">검색 결과가 없습니다.</div>';
                        }
                        results.classList.add('active');
                    } else {
                        resultContent.innerHTML = '<div class="error">오류: ' + escapeHtml(result.error || '알 수 없는 오류가 발생했습니다.') + '</div>';
                        results.classList.add('active');
                    }
                })
                .catch(function(error) {
                    console.error('요청 오류:', error);
                    var resultContent = document.getElementById('resultContent');
                    var results = document.getElementById('results');
                    
                    if (resultContent) {
                        resultContent.innerHTML = '<div class="error"><strong>요청 중 오류가 발생했습니다:</strong><br>' + escapeHtml(error.message) + '<br><br>브라우저 콘솔(F12)에서 자세한 오류를 확인하세요.</div>';
                    }
                    if (results) {
                        results.classList.add('active');
                    }
                })
                .finally(function() {
                    loading.classList.remove('active');
                    submitBtn.disabled = false;
                });
            }
            
            // 전역으로 할당
            window.handleTestClick = handleTestClick;
            window.handleLogout = handleLogout;
            
            console.log('스크립트 로드됨 - 함수 정의 완료');
            
            document.addEventListener('DOMContentLoaded', function() {
                console.log('DOM 로드 완료');
                
                // 검색 버튼 이벤트 리스너 등록
                var submitBtn = document.getElementById('submitBtn');
                if (submitBtn) {
                    console.log('검색 버튼 찾음 - 이벤트 리스너 등록');
                    submitBtn.addEventListener('click', function(e) {
                        e.preventDefault();
                        e.stopPropagation();
                        console.log('검색 버튼 클릭 이벤트 발생');
                        
                        // 함수가 정의되어 있으면 호출, 없으면 직접 실행
                        if (typeof window.handleTestClick === 'function') {
                            window.handleTestClick();
                        } else {
                            console.log('handleTestClick 함수를 직접 호출');
                            // 함수 내용을 직접 실행
                            var form = document.getElementById('testForm');
                            if (!form) {
                                customAlert('폼을 찾을 수 없습니다.');
                                return;
                            }
                            
                            var formData = new FormData(form);
                            var query = formData.get('query');
                            
                            // 검색어 검증
                            if (!query || !query.trim()) {
                                customAlert('검색어를 입력해주세요.');
                                return;
                            }
                            
                            var modelModeRadio = document.querySelector('input[name="model_mode"]:checked');
                            var modelMode = modelModeRadio ? modelModeRadio.value : 'local';
                            
                            // OpenAI API 키가 필요한 경우 확인
                            var needsOpenAIKey = (modelMode === 'openai');
                            var openaiApiKey = needsOpenAIKey ? (formData.get('openai_api_key') || null) : null;
                            
                            // OpenAI API 키 검증
                            if (needsOpenAIKey && (!openaiApiKey || !openaiApiKey.trim())) {
                                customAlert('OpenAI API 키를 입력해주세요.');
                                return;
                            }
                            
                            var data = {
                                query: query.trim(),
                                max_results: parseInt(formData.get('max_results')) || 10,
                                days: parseInt(formData.get('days')) || 1,
                                include_full_text: true,
                                sort_by: formData.get('sort_by') || 'date',
                                model_mode: modelMode,
                                openai_api_key: openaiApiKey
                            };
                            
                            if (isNaN(data.max_results) || data.max_results < 1) {
                                customAlert('최대 결과 수는 1 이상이어야 합니다.');
                                return;
                            }
                            if (isNaN(data.days) || data.days < 1) {
                                customAlert('날짜는 1 이상이어야 합니다.');
                                return;
                            }
                            
                            var loading = document.getElementById('loading');
                            var results = document.getElementById('results');
                            var resultContent = document.getElementById('resultContent');
                            
                            if (!loading || !results || !resultContent) {
                                customAlert('페이지 오류가 발생했습니다.');
                                return;
                            }
                            
                            submitBtn.disabled = true;
                            loading.classList.add('active');
                            results.classList.remove('active');
                            resultContent.innerHTML = '';
                            
                            console.log('API 요청 시작:', data);
                            
                            fetch('/api/test', {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json',
                                },
                                body: JSON.stringify(data)
                            })
                            .then(function(response) {
                                console.log('응답 상태:', response.status);
                                if (!response.ok) {
                                    return response.text().then(function(text) {
                                        throw new Error('서버 오류 (' + response.status + '): ' + text);
                                    });
                                }
                                return response.json();
                            })
                            .then(function(result) {
                                console.log('응답 데이터:', result);
                                
                                if (result.success) {
                                    if (result.data && result.data.length > 0) {
                                        var html = '<p class="result-count">총 <strong>' + result.data.length + '</strong>개의 기사를 찾았습니다.</p>';
                                        html += '<div class="results-grid">';
                                        
                                        result.data.forEach(function(item, index) {
                                            var title = item.title || '제목 없음';
                                            var titleOriginal = title;
                                            title = decodeHtmlEntities(title);
                                            title = escapeHtml(title);
                                            titleOriginal = escapeHtml(titleOriginal);
                                            
                                            html += '<div class="result-card">';
                                            html += '<h3 title="' + titleOriginal + '" style="white-space: normal !important; overflow: visible !important; overflow-x: visible !important; overflow-y: visible !important; text-overflow: clip !important; text-overflow: unset !important; max-height: none !important; min-height: auto !important; height: auto !important; display: block !important; width: 100% !important; -webkit-line-clamp: unset !important; line-clamp: unset !important; word-break: break-word !important;">' + title + '</h3>';
                                            
                                            html += '<div class="meta">';
                                            if (item.source) {
                                                html += '<span>' + escapeHtml(item.source) + '</span>';
                                            }
                                            if (item.pubDate) {
                                                html += '<span>' + escapeHtml(item.pubDate) + '</span>';
                                            }
                                            if (item.view_count !== undefined) {
                                                html += '<span>' + item.view_count.toLocaleString() + '회 조회</span>';
                                            }
                                            html += '</div>';
                                            
                                            if (item.sentiment) {
                                                html += '<div class="sentiment-box" onclick="event.stopPropagation()">';
                                                var imgPath = escapeHtml(item.sentiment.image_path);
                                                html += '<img src="/' + imgPath + '" alt="감정 분석" class="sentiment-image" onerror="this.style.display=&quot;none&quot;">';
                                                html += '<div class="sentiment-info">';
                                                html += '<div class="sentiment-label">' + escapeHtml(item.sentiment.label) + '</div>';
                                                html += '<div class="sentiment-temperature">' + escapeHtml(item.sentiment.temperature) + '°C</div>';
                                                html += '</div></div>';
                                            }
                                            
                                            if (item.text) {
                                                var formattedText = formatText(item.text);
                                                html += '<div class="summary">' + formattedText + '</div>';
                                            }
                                            
                                            html += '<a href="' + escapeHtml(item.link) + '" target="_blank" onclick="event.stopPropagation()">기사 보기 →</a>';
                                            html += '</div>';
                                        });
                                        
                                        html += '</div>';
                                        resultContent.innerHTML = html;
                                    } else {
                                        resultContent.innerHTML = '<div class="error">검색 결과가 없습니다.</div>';
                                    }
                                    results.classList.add('active');
                                } else {
                                    resultContent.innerHTML = '<div class="error">오류: ' + escapeHtml(result.error || '알 수 없는 오류가 발생했습니다.') + '</div>';
                                    results.classList.add('active');
                                }
                            })
                            .catch(function(error) {
                                console.error('요청 오류:', error);
                                var resultContent = document.getElementById('resultContent');
                                var results = document.getElementById('results');
                                
                                if (resultContent) {
                                    resultContent.innerHTML = '<div class="error"><strong>요청 중 오류가 발생했습니다:</strong><br>' + escapeHtml(error.message) + '<br><br>브라우저 콘솔(F12)에서 자세한 오류를 확인하세요.</div>';
                                }
                                if (results) {
                                    results.classList.add('active');
                                }
                            })
                            .finally(function() {
                                loading.classList.remove('active');
                                submitBtn.disabled = false;
                            });
                        }
                    });
                } else {
                    console.error('검색 버튼을 찾을 수 없습니다');
                }
                
                // 로그아웃 버튼 이벤트 리스너 등록
                var logoutBtn = document.getElementById('logoutBtn');
                if (logoutBtn) {
                    console.log('로그아웃 버튼 찾음 - 이벤트 리스너 등록');
                    logoutBtn.addEventListener('click', function(e) {
                        e.preventDefault();
                        e.stopPropagation();
                        console.log('로그아웃 버튼 클릭 이벤트 발생');
                        
                        // 함수가 정의되어 있으면 호출, 없으면 직접 실행
                        if (typeof window.handleLogout === 'function') {
                            window.handleLogout(e);
                        } else {
                            console.log('handleLogout 함수를 직접 실행');
                            // 직접 로그아웃 처리
                            logoutBtn.disabled = true;
                            logoutBtn.textContent = '로그아웃 중...';
                            
                            fetch('/api/logout', { 
                                method: 'POST',
                                credentials: 'include',
                                headers: {
                                    'Content-Type': 'application/json'
                                }
                            })
                            .then(function(response) {
                                console.log('로그아웃 응답 받음:', response.status);
                                window.location.href = '/login';
                            })
                            .catch(function(error) {
                                console.error('로그아웃 오류:', error);
                                window.location.href = '/login';
                            });
                        }
                    });
                } else {
                    console.error('로그아웃 버튼을 찾을 수 없습니다');
                }
                
                // 라디오 버튼 이벤트 처리
                var openaiKeyGroup = document.getElementById('openai_key_group');
                var modelModeRadios = document.querySelectorAll('input[name="model_mode"]');
                
                function updateOpenAIKeyVisibility() {
                    if (openaiKeyGroup) {
                        var selectedModelMode = document.querySelector('input[name="model_mode"]:checked');
                        
                        var needsOpenAIKey = false;
                        if (selectedModelMode && selectedModelMode.value === 'openai') {
                            needsOpenAIKey = true;
                        }
                        
                        if (needsOpenAIKey) {
                            openaiKeyGroup.style.display = 'block';
                        } else {
                            openaiKeyGroup.style.display = 'none';
                        }
                    }
                }
                
                // 라디오 버튼 변경 이벤트 리스너 등록
                modelModeRadios.forEach(function(radio) {
                    radio.addEventListener('change', function() {
                        updateOpenAIKeyVisibility();
                    });
                    radio.addEventListener('click', function() {
                        updateOpenAIKeyVisibility();
                    });
                });
                
                // 초기 상태 설정 (약간의 지연을 두어 DOM이 완전히 로드된 후 실행)
                setTimeout(function() {
                    updateOpenAIKeyVisibility();
                }, 100);
                
                console.log('이벤트 리스너 등록 완료');
            });
        </script>
    </body>
    </html>
    """
    # HTML에서 변수 치환
    html_content = html_content.replace("${client_id}", client_id[:10] + "..." if len(client_id) > 10 else client_id)
    return html_content


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """로그인 페이지"""
    # 이미 로그인된 경우 메인 페이지로 리다이렉트
    session = get_session(request)
    if session:
        return RedirectResponse(url="/", status_code=302)
    
    html_content = """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>로그인 - 뉴스 온도계</title>
        <link rel="icon" href="/static/favicon.svg" type="image/svg+xml">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', 'Helvetica Neue', Arial, sans-serif;
                background: #f5f5f7;
                min-height: 100vh;
                padding: 0;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #1d1d1f;
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
            }
            .container {
                max-width: 1100px;
                width: 100%;
                background: white;
                border-radius: 24px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
                padding: 0;
                display: flex;
                overflow: hidden;
                margin: 20px;
            }
            .left-section {
                flex: 1;
                padding: 60px;
                background: transparent;
                display: flex;
                flex-direction: column;
                justify-content: center;
            }
            .right-section {
                flex: 1;
                padding: 60px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                background: transparent;
            }
            .left-section h1 {
                color: #1d1d1f;
                margin-bottom: 16px;
                font-size: 48px;
                font-weight: 600;
                letter-spacing: -1px;
                display: flex;
                align-items: center;
                gap: 16px;
            }
            .left-section .logo-icon-large {
                width: 48px;
                height: 48px;
                display: inline-block;
                object-fit: contain;
            }
            .left-section .subtitle {
                color: #86868b;
                margin-bottom: 40px;
                font-size: 21px;
                font-weight: 400;
                line-height: 1.5;
            }
            .right-section h2 {
                color: #1d1d1f;
                margin-bottom: 12px;
                font-size: 40px;
                font-weight: 600;
                letter-spacing: -0.5px;
            }
            .right-section .subtitle {
                color: #86868b;
                margin-bottom: 40px;
                font-size: 17px;
                font-weight: 400;
            }
            .form-group {
                margin-bottom: 28px;
            }
            label {
                display: block;
                margin-bottom: 10px;
                color: #1d1d1f;
                font-weight: 500;
                font-size: 17px;
            }
            input[type="text"], input[type="password"] {
                width: 100%;
                padding: 14px 18px;
                border: 1px solid #d2d2d7;
                border-radius: 16px;
                font-size: 17px;
                background: #fbfbfd;
                color: #1d1d1f;
                transition: all 0.2s ease;
                font-family: inherit;
            }
            input::placeholder {
                color: #86868b;
            }
            input:focus {
                outline: none;
                border-color: #0071e3;
                background: white;
                box-shadow: 0 0 0 4px rgba(0, 113, 227, 0.1);
            }
            button {
                background: #0071e3;
                color: white;
                padding: 16px 32px;
                border: none;
                border-radius: 16px;
                font-size: 17px;
                font-weight: 500;
                cursor: pointer;
                width: 100%;
                transition: all 0.2s ease;
                font-family: inherit;
            }
            button:hover {
                background: #0077ed;
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(0, 113, 227, 0.3);
            }
            button:active {
                transform: translateY(0);
            }
            button:disabled {
                background: #d2d2d7;
                cursor: not-allowed;
                transform: none;
                box-shadow: none;
            }
            .error {
                background: #fff3f3;
                border: 1px solid #ff3b30;
                color: #d70015;
                padding: 16px;
                border-radius: 16px;
                margin-bottom: 24px;
                display: none;
                font-size: 15px;
            }
            .error.active {
                display: block;
            }
            /* 커스텀 Alert 모달 */
            .custom-alert-overlay {
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0, 0, 0, 0.4);
                backdrop-filter: blur(4px);
                z-index: 10000;
                align-items: center;
                justify-content: center;
            }
            .custom-alert-overlay.show {
                display: flex;
            }
            .custom-alert-modal {
                background: white;
                border-radius: 20px;
                padding: 0;
                max-width: 400px;
                width: 90%;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
                animation: modalSlideIn 0.3s ease-out;
                overflow: hidden;
            }
            @keyframes modalSlideIn {
                from {
                    opacity: 0;
                    transform: translateY(-20px) scale(0.95);
                }
                to {
                    opacity: 1;
                    transform: translateY(0) scale(1);
                }
            }
            .custom-alert-content {
                padding: 32px 24px 24px 24px;
            }
            .custom-alert-title {
                font-size: 20px;
                font-weight: 600;
                color: #1d1d1f;
                margin-bottom: 12px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .custom-alert-message {
                font-size: 17px;
                color: #515154;
                line-height: 1.5;
                margin-bottom: 24px;
            }
            .custom-alert-buttons {
                display: flex;
                justify-content: flex-end;
                gap: 12px;
                padding-top: 20px;
                border-top: 1px solid #e5e5e7;
            }
            .custom-alert-btn {
                padding: 12px 24px;
                border: none;
                border-radius: 12px;
                font-size: 17px;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.2s ease;
                font-family: inherit;
            }
            .custom-alert-btn-primary {
                background: #0071e3;
                color: white;
            }
            .custom-alert-btn-primary:hover {
                background: #0077ed;
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(0, 113, 227, 0.3);
            }
            .custom-alert-btn-primary:active {
                transform: translateY(0);
            }
            .info {
                background: #f5f5f7;
                border-radius: 16px;
                padding: 24px;
                margin-bottom: 24px;
                font-size: 15px;
                line-height: 1.6;
                color: #515154;
                border: 1px solid #e5e5e7;
            }
            .info strong {
                display: block;
                margin-bottom: 8px;
                color: #1d1d1f;
                font-size: 17px;
            }
            .info a {
                color: #0071e3;
                text-decoration: none;
                transition: all 0.2s ease;
            }
            .info a:hover {
                text-decoration: underline;
                color: #0077ed;
            }
            .form-group p a,
            .form-group p a:link,
            .form-group p a:visited,
            .form-group p a:active,
            .form-group p a:focus {
                color: #0071e3;
                text-decoration: none !important;
                transition: all 0.2s ease;
            }
            .form-group p a:hover {
                color: #0077ed;
                text-decoration: underline !important;
            }
            @media (max-width: 768px) {
                .container {
                    flex-direction: column;
                    margin: 0;
                    border-radius: 0;
                }
                .left-section, .right-section {
                    padding: 40px 30px;
                }
                .left-section h1 {
                    font-size: 36px;
                }
                .right-section h2 {
                    font-size: 32px;
                }
            }
            .accordion {
                margin-bottom: 20px;
            }
            .accordion-header {
                background: #e3f2fd;
                border-left: 4px solid #2196f3;
                padding: 15px;
                border-radius: 8px;
                cursor: pointer;
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-weight: 600;
                color: #333;
                transition: background-color 0.3s;
            }
            .accordion-header:hover {
                background: #bbdefb;
            }
            .accordion-content {
                max-height: 0;
                overflow: hidden;
                transition: max-height 0.3s ease-out;
                background: #f5f5f5;
                border-left: 4px solid #2196f3;
                border-radius: 0 0 8px 8px;
            }
            .accordion-content.active {
                max-height: 500px;
                padding: 15px;
                margin-bottom: 20px;
            }
            .accordion-icon {
                transition: transform 0.3s;
                font-size: 18px;
            }
            .accordion-icon.rotated {
                transform: rotate(180deg);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <!-- 왼쪽 섹션: 설명 -->
            <div class="left-section">
                <h1>
                    <img src="/temp/logo.png" alt="뉴스 온도계" class="logo-icon-large" />
                    뉴스 온도계
                </h1>
                <p class="subtitle">
                    뉴스를 검색하고 AI가 분석한 감정 온도와 요약을 확인하세요
                </p>
                
                <div class="info">
                    <strong>뉴스 온도계란?</strong>
                    네이버 뉴스를 검색하고, 각 기사의 감정을 온도로 표현하며, 
                    AI가 요약한 내용을 제공하는 서비스입니다. 
                    뉴스의 분위기와 핵심 내용을 한눈에 파악할 수 있습니다.
                </div>
                
                <div class="info">
                    <strong>사용 방법</strong>
                    1. <a href="https://developers.naver.com" target="_blank">네이버 개발자 센터</a>에서 애플리케이션 등록<br>
                    2. Client ID와 Client Secret 발급<br>
                </div>
            </div>
            
            <!-- 오른쪽 섹션: 로그인 폼 -->
            <div class="right-section">
                <h2>로그인</h2>
                <p class="subtitle">네이버 API 키를 입력하여 로그인하세요</p>

                <div class="error" id="errorMsg"></div>
                
                <form id="loginForm" novalidate>
                    <div class="form-group">
                        <label for="client_id">Client ID</label>
                        <input type="text" id="client_id" name="client_id" 
                               placeholder="네이버 Client ID 입력">
                    </div>
                    <div class="form-group">
                        <label for="client_secret">Client Secret</label>
                        <input type="password" id="client_secret" name="client_secret" 
                               placeholder="네이버 Client Secret 입력">
                    </div>
                    
                    <button type="submit" id="submitBtn">로그인</button>
                </form>
            </div>
        </div>
        
        <!-- 커스텀 Alert 모달 -->
        <div class="custom-alert-overlay" id="customAlertOverlay">
            <div class="custom-alert-modal">
                <div class="custom-alert-content">
                    <div class="custom-alert-title">
                        <span>⚠️ 알림</span>
                    </div>
                    <div class="custom-alert-message" id="customAlertMessage"></div>
                    <div class="custom-alert-buttons">
                        <button type="button" class="custom-alert-btn custom-alert-btn-primary" id="customAlertOkBtn">확인</button>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // 커스텀 Alert 함수
            function customAlert(message) {
                var overlay = document.getElementById('customAlertOverlay');
                var messageEl = document.getElementById('customAlertMessage');
                var okBtn = document.getElementById('customAlertOkBtn');
                
                if (!overlay || !messageEl || !okBtn) {
                    // 폴백: 기본 alert 사용
                    alert(message);
                    return;
                }
                
                messageEl.textContent = message;
                overlay.classList.add('show');
                
                // 확인 버튼 클릭 시 닫기
                var closeAlert = function() {
                    overlay.classList.remove('show');
                    okBtn.removeEventListener('click', closeAlert);
                    overlay.removeEventListener('click', handleOverlayClick);
                };
                
                var handleOverlayClick = function(e) {
                    if (e.target === overlay) {
                        closeAlert();
                    }
                };
                
                okBtn.addEventListener('click', closeAlert);
                overlay.addEventListener('click', handleOverlayClick);
                
                // ESC 키로 닫기
                var handleEscKey = function(e) {
                    if (e.key === 'Escape') {
                        closeAlert();
                        document.removeEventListener('keydown', handleEscKey);
                    }
                };
                document.addEventListener('keydown', handleEscKey);
            }
            
            function toggleAccordion(id) {
                var content = document.getElementById(id);
                var icon = document.getElementById(id + 'Icon');
                
                if (content.classList.contains('active')) {
                    content.classList.remove('active');
                    icon.classList.remove('rotated');
                } else {
                    content.classList.add('active');
                    icon.classList.add('rotated');
                }
            }
            
            document.addEventListener('DOMContentLoaded', function() {
                const form = document.getElementById('loginForm');
                const errorMsg = document.getElementById('errorMsg');
                const submitBtn = document.getElementById('submitBtn');
                
                form.addEventListener('submit', async function(e) {
                    e.preventDefault();
                    
                    const formData = new FormData(e.target);
                    const clientId = formData.get('client_id');
                    const clientSecret = formData.get('client_secret');
                    
                    // 네이버 Client ID 검증
                    if (!clientId || !clientId.trim()) {
                        customAlert('네이버 Client ID를 입력해주세요.');
                        return;
                    }
                    
                    // 네이버 Client Secret 검증
                    if (!clientSecret || !clientSecret.trim()) {
                        customAlert('네이버 Client Secret을 입력해주세요.');
                        return;
                    }
                    
                    const data = {
                        client_id: clientId.trim(),
                        client_secret: clientSecret.trim()
                    };
                    
                    submitBtn.disabled = true;
                    errorMsg.classList.remove('active');
                    
                    try {
                        const response = await fetch('/api/login', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify(data)
                        });
                        
                        const result = await response.json();
                        
                        if (result.success) {
                            window.location.href = '/';
                        } else {
                            errorMsg.textContent = result.error || '로그인에 실패했습니다.';
                            errorMsg.classList.add('active');
                            submitBtn.disabled = false;
                        }
                    } catch (error) {
                        errorMsg.textContent = '요청 중 오류가 발생했습니다: ' + error.message;
                        errorMsg.classList.add('active');
                        submitBtn.disabled = false;
                    }
                });
            });
        </script>
    </body>
    </html>
    """
    return html_content


@app.post("/api/login")
async def login(request: LoginRequest):
    """로그인 API - Client ID와 Secret 검증"""
    try:
        # 간단한 검증: 실제로 API를 호출해서 키가 유효한지 확인
        test_crawler = NaverNewsAPICrawler(
            client_id=request.client_id,
            client_secret=request.client_secret,
            delay=0.1
        )
        
        # 테스트 검색으로 키 유효성 확인
        try:
            test_crawler.get_recent_news(query="테스트", days=1, max_results=1)
        except Exception as e:
            # API 키가 유효하지 않은 경우
            error_msg = str(e)
            if "401" in error_msg or "Unauthorized" in error_msg or "인증" in error_msg:
                return JSONResponse({
                    "success": False,
                    "error": "Client ID 또는 Client Secret이 올바르지 않습니다."
                }, status_code=401)
            else:
                # 다른 오류는 일단 통과 (네트워크 오류 등일 수 있음)
                pass
        
        # 세션 생성
        session_id = secrets.token_urlsafe(32)
        sessions[session_id] = {
            "client_id": request.client_id,
            "client_secret": request.client_secret,
            "created_at": datetime.now()
        }
        
        response = JSONResponse({
            "success": True,
            "message": "로그인 성공"
        })
        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            max_age=86400,  # 24시간
            samesite="lax"
        )
        
        return response
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"로그인 오류 발생: {str(e)}")
        print(f"상세 오류:\n{error_detail}")
        return JSONResponse({
            "success": False,
            "error": f"로그인 중 오류가 발생했습니다: {str(e)}"
        }, status_code=500)


@app.post("/api/logout")
async def logout(request: Request):
    """로그아웃 API"""
    session_id = request.cookies.get("session_id")
    if session_id and session_id in sessions:
        del sessions[session_id]
    
    response = JSONResponse({"success": True, "message": "로그아웃 성공"})
    response.delete_cookie(key="session_id")
    return response


@app.post("/api/test")
async def test_api(request: TestRequest, session: dict = Depends(require_login)):
    """뉴스 검색 및 분석 API 엔드포인트"""
    try:
        # 모델 모드에 따라 요약 모드와 감정 분석 모드 결정
        if request.model_mode == 'openai':
            summary_mode = 'openai'
            use_openai_sentiment = True
        else:  # 'local'
            summary_mode = 'kosum-v1-tuned'
            use_openai_sentiment = False
        
        # 세션에서 Client ID와 Secret 가져오기
        crawler = NaverNewsAPICrawler(
            client_id=session["client_id"],
            client_secret=session["client_secret"],
            delay=0.1,
            openai_api_key=request.openai_api_key if request.model_mode == 'openai' else None,
            summary_mode=summary_mode
        )
        
        results = crawler.get_recent_news(
            query=request.query,
            days=request.days,
            max_results=request.max_results,
            sort_by=request.sort_by,
            exclude_english=True  # 항상 영어 뉴스 제외
        )
        
        # 항상 본문 전체 추출 (include_full_text는 항상 True)
        
        # 감정 분석 수행 (오류가 발생해도 뉴스는 표시)
        analyzer = get_sentiment_analyzer(
            openai_api_key=request.openai_api_key if use_openai_sentiment else None,
            use_openai=use_openai_sentiment
        )
        if analyzer:
            for idx, result in enumerate(results):
                # 전체 본문이 있으면 전체 본문 사용, 없으면 요약본 사용
                text_for_analysis = result.get('full_text') or result.get('text', '')
                
                if text_for_analysis:
                    try:
                        # 전체 본문으로 감정 분석 수행
                        sentiment_result = analyzer.analyze(text_for_analysis, article_id=idx + 1)
                        result['sentiment'] = sentiment_result
                        mode_str = 'OpenAI API' if use_openai_sentiment else '로컬 모델 (파인튜닝된 감정 분석 모델)'
                        print(f"[감정 분석] 기사 {idx + 1}: {mode_str} 사용, {'전체 본문' if result.get('full_text') else '요약본'} 사용 ({len(text_for_analysis)}자)")
                    except Exception as e:
                        print(f"감정 분석 오류 (기사 {idx + 1}): {e}")
                        # 감정 분석 실패 시 sentiment 필드 없이 진행
                        import traceback
                        traceback.print_exc()
        
        return JSONResponse({
            "success": True,
            "data": results,
            "count": len(results)
        })
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"API 오류 발생: {str(e)}")
        print(f"상세 오류:\n{error_detail}")
        return JSONResponse({
            "success": False,
            "error": str(e),
            "detail": error_detail
        }, status_code=500)


@app.get("/api/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "ok", "message": "뉴스 온도계 서버가 정상 작동 중입니다."}


if __name__ == "__main__":
    import sys
    
    # 포트 번호를 명령줄 인자로 받을 수 있음
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("⚠️  포트 번호가 올바르지 않습니다. 기본 포트 8000을 사용합니다.")
    
    print("=" * 60)
    print("🚀 뉴스 온도계 서버 시작")
    print("=" * 60)
    print(f"📍 접속 주소: http://localhost:{port}")
    print(f"📍 API 문서: http://localhost:{port}/docs")
    print(f"📍 헬스 체크: http://localhost:{port}/api/health")
    print("=" * 60)
    print(f"\n💡 네이버 개발자 센터에서 서비스 URL을 다음으로 설정하세요:")
    print(f"   http://localhost:{port}")
    print("\n⏹️  서버를 종료하려면 Ctrl+C를 누르세요.\n")
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=port, reload=True)
    except OSError as e:
        if "address already in use" in str(e).lower() or "포트" in str(e).lower():
            print(f"\n❌ 오류: 포트 {port}가 이미 사용 중입니다.")
            print(f"💡 다른 포트로 실행하려면: python app.py 8001")
            print(f"💡 또는 사용 중인 프로세스를 종료하세요.\n")
        else:
            print(f"\n❌ 오류 발생: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류 발생: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

