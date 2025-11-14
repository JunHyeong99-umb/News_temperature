"""
네이버 API 테스트를 위한 FastAPI 웹 애플리케이션
"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
from src.crawl_naver_api import NaverNewsAPICrawler

app = FastAPI(title="네이버 API 테스트", description="네이버 뉴스 API 크롤링 테스트 도구")


class TestRequest(BaseModel):
    """API 테스트 요청 모델"""
    client_id: str
    client_secret: str
    query: str
    
    max_results: int = 10
    days: int = 1
    include_full_text: bool = True
    sort_by: str = 'date'  # 'date': 날짜순, 'view': 조회수순


@app.get("/", response_class=HTMLResponse)
async def home():
    """메인 페이지 - API 키 입력 및 테스트 인터페이스"""
    html_content = """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>네이버 API 테스트</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                padding: 40px;
            }
            h1 {
                color: #333;
                margin-bottom: 10px;
                text-align: center;
            }
            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 30px;
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 8px;
                color: #333;
                font-weight: 600;
            }
            input[type="text"], input[type="number"], select {
                width: 100%;
                padding: 12px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 14px;
                transition: border-color 0.3s;
            }
            input:focus, select:focus {
                outline: none;
                border-color: #667eea;
            }
            .form-row {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }
            button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 14px 30px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                width: 100%;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }
            button:disabled {
                background: #ccc;
                cursor: not-allowed;
                transform: none;
            }
            .loading {
                display: none;
                text-align: center;
                margin: 20px 0;
            }
            .loading.active {
                display: block;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .results {
                margin-top: 30px;
                display: none;
            }
            .results.active {
                display: block;
            }
            .result-item {
                background: #f8f9fa;
                border-left: 4px solid #667eea;
                padding: 20px;
                margin-bottom: 15px;
                border-radius: 8px;
            }
            .result-item h3 {
                color: #333;
                margin-bottom: 10px;
            }
            .result-item p {
                color: #666;
                margin: 5px 0;
                line-height: 1.6;
            }
            .result-item a {
                color: #667eea;
                text-decoration: none;
            }
            .result-item a:hover {
                text-decoration: underline;
            }
            .error {
                background: #fee;
                border-left-color: #f00;
                color: #c00;
                padding: 15px;
                border-radius: 8px;
                margin-top: 20px;
            }
            .info {
                background: #e3f2fd;
                border-left: 4px solid #2196f3;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
            }
            .checkbox-group {
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .checkbox-group input[type="checkbox"] {
                width: auto;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 네이버 뉴스 API 테스트</h1>
            <p class="subtitle">네이버 개발자 센터에서 발급받은 API 키로 뉴스 크롤링을 테스트하세요</p>
            
            <div class="info">
                <strong>📌 사용 방법:</strong><br>
                1. 네이버 개발자 센터(https://developers.naver.com)에서 애플리케이션 등록<br>
                2. Client ID와 Client Secret 발급<br>
                3. 서비스 URL에 <code>http://localhost:8000</code> 입력<br>
                4. 아래 폼에 API 키 입력 후 테스트
            </div>
            
            <form id="testForm">
                <div class="form-row">
                    <div class="form-group">
                        <label for="client_id">Client ID *</label>
                        <input type="text" id="client_id" name="client_id" required 
                               placeholder="네이버 Client ID 입력">
                    </div>
                    <div class="form-group">
                        <label for="client_secret">Client Secret *</label>
                        <input type="text" id="client_secret" name="client_secret" required 
                               placeholder="네이버 Client Secret 입력">
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="query">검색어 *</label>
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
                    <select id="sort_by" name="sort_by">
                        <option value="date">날짜순 (최신순)</option>
                        <option value="view">조회수순 (높은순)</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <div class="checkbox-group">
                        <input type="checkbox" id="include_full_text" name="include_full_text" checked>
                        <label for="include_full_text">본문 전체 추출 (시간이 오래 걸릴 수 있음)</label>
                    </div>
                </div>
                
                <button type="submit" id="submitBtn">🚀 테스트 시작</button>
            </form>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p style="margin-top: 10px;">뉴스를 검색하고 있습니다...</p>
            </div>
            
            <div class="results" id="results">
                <h2>📰 검색 결과</h2>
                <div id="resultContent"></div>
            </div>
        </div>
        
        <script>
            document.getElementById('testForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const formData = new FormData(e.target);
                const data = {
                    client_id: formData.get('client_id'),
                    client_secret: formData.get('client_secret'),
                    query: formData.get('query'),
                    max_results: parseInt(formData.get('max_results')),
                    days: parseInt(formData.get('days')),
                    include_full_text: formData.get('include_full_text') === 'on',
                    sort_by: formData.get('sort_by') || 'date'
                };
                
                const loading = document.getElementById('loading');
                const results = document.getElementById('results');
                const resultContent = document.getElementById('resultContent');
                const submitBtn = document.getElementById('submitBtn');
                
                loading.classList.add('active');
                results.classList.remove('active');
                submitBtn.disabled = true;
                resultContent.innerHTML = '';
                
                try {
                    const response = await fetch('/api/test', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        if (result.data && result.data.length > 0) {
                            let html = `<p style="margin-bottom: 20px; color: #666;">
                                총 <strong>${result.data.length}</strong>개의 기사를 찾았습니다.
                            </p>`;
                            
                            result.data.forEach((item, index) => {
                                html += `
                                    <div class="result-item">
                                        <h3>${index + 1}. ${item.title || '제목 없음'}</h3>
                                        <p><strong>출처:</strong> ${item.source || '알 수 없음'}</p>
                                        <p><strong>날짜:</strong> ${item.pubDate || '알 수 없음'}</p>
                                        ${item.view_count !== undefined ? `<p><strong>조회수:</strong> ${item.view_count.toLocaleString()}회</p>` : ''}
                                        <p><strong>본문 길이:</strong> ${item.text ? item.text.length : 0}자</p>
                                        ${item.text ? `<p><strong>본문 미리보기:</strong> ${item.text.substring(0, 200)}...</p>` : ''}
                                        <p><strong>링크:</strong> <a href="${item.link}" target="_blank">${item.link}</a></p>
                                    </div>
                                `;
                            });
                            
                            resultContent.innerHTML = html;
                        } else {
                            resultContent.innerHTML = '<div class="error">검색 결과가 없습니다.</div>';
                        }
                        results.classList.add('active');
                    } else {
                        resultContent.innerHTML = `<div class="error">오류: ${result.error || '알 수 없는 오류가 발생했습니다.'}</div>`;
                        results.classList.add('active');
                    }
                } catch (error) {
                    resultContent.innerHTML = `<div class="error">요청 중 오류가 발생했습니다: ${error.message}</div>`;
                    results.classList.add('active');
                } finally {
                    loading.classList.remove('active');
                    submitBtn.disabled = false;
                }
            });
        </script>
    </body>
    </html>
    """
    return html_content


@app.post("/api/test")
async def test_api(request: TestRequest):
    """네이버 API 테스트 엔드포인트"""
    try:
        crawler = NaverNewsAPICrawler(
            client_id=request.client_id,
            client_secret=request.client_secret,
            delay=0.1
        )
        
        results = crawler.get_recent_news(
            query=request.query,
            days=request.days,
            max_results=request.max_results,
            sort_by=request.sort_by
        )
        
        # include_full_text가 False면 본문 추출 안 함
        if not request.include_full_text:
            for result in results:
                if 'text' in result:
                    result['text'] = result.get('description', '')
        
        return JSONResponse({
            "success": True,
            "data": results,
            "count": len(results)
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@app.get("/api/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "ok", "message": "네이버 API 테스트 서버가 정상 작동 중입니다."}


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
    print("🚀 네이버 API 테스트 서버 시작")
    print("=" * 60)
    print(f"📍 접속 주소: http://localhost:{port}")
    print(f"📍 API 문서: http://localhost:{port}/docs")
    print(f"📍 헬스 체크: http://localhost:{port}/api/health")
    print("=" * 60)
    print(f"\n💡 네이버 개발자 센터에서 서비스 URL을 다음으로 설정하세요:")
    print(f"   http://localhost:{port}")
    print("\n⏹️  서버를 종료하려면 Ctrl+C를 누르세요.\n")
    
    try:
        uvicorn.run(app, host="127.0.0.1", port=port, reload=True)
    except OSError as e:
        if "address already in use" in str(e).lower() or "포트" in str(e).lower():
            print(f"\n❌ 오류: 포트 {port}가 이미 사용 중입니다.")
            print(f"💡 다른 포트로 실행하려면: python app.py 8001")
            print(f"💡 또는 사용 중인 프로세스를 종료하세요.\n")
        else:
            print(f"\n❌ 오류 발생: {e}\n")
        sys.exit(1)

