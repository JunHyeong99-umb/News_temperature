"""
네이버 검색 API를 활용한 뉴스 크롤링 모듈
네이버 검색 API를 사용하여 뉴스를 검색하고 수집합니다.
"""

import requests
import os
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time
import re
from bs4 import BeautifulSoup

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class NaverNewsAPICrawler:
    """네이버 검색 API를 사용한 뉴스 크롤링 클래스"""
    
    def __init__(self, client_id: str, client_secret: str, delay: float = 0.1, openai_api_key: Optional[str] = None):
        """
        Args:
            client_id: 네이버 개발자 센터에서 발급받은 Client ID
            client_secret: 네이버 개발자 센터에서 발급받은 Client Secret
            delay: API 요청 간 대기 시간(초). API 제한 방지용
            openai_api_key: OpenAI API 키 (요약 기능 사용 시 필요)
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.delay = delay
        self.api_url = "https://openapi.naver.com/v1/search/news.json"
        self.headers = {
            'X-Naver-Client-Id': client_id,
            'X-Naver-Client-Secret': client_secret
        }
        self.openai_api_key = openai_api_key
        self.openai_client = None
        if openai_api_key and OPENAI_AVAILABLE:
            self.openai_client = OpenAI(api_key=openai_api_key)
    
    def search_news(
        self,
        query: str,
        display: int = 100,
        start: int = 1,
        sort: str = 'date',
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> Optional[Dict]:
        """
        네이버 뉴스 검색 API를 호출합니다.
        
        Args:
            query: 검색어
            display: 한 번에 표시할 검색 결과 개수 (최대 100)
            start: 검색 시작 위치 (1부터 시작)
            sort: 정렬 옵션 ('sim': 정확도순, 'date': 날짜순)
            date_from: 시작 날짜 (YYYYMMDD 형식)
            date_to: 종료 날짜 (YYYYMMDD 형식)
            
        Returns:
            API 응답 JSON 딕셔너리 또는 None
        """
        params = {
            'query': query,
            'display': min(display, 100),  # 최대 100
            'start': start,
            'sort': sort
        }
        
        if date_from:
            params['dateFrom'] = date_from
        if date_to:
            params['dateTo'] = date_to
        
        try:
            response = requests.get(
                self.api_url,
                headers=self.headers,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            time.sleep(self.delay)
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"API 요청 오류: {e}")
            if hasattr(e.response, 'text'):
                print(f"응답 내용: {e.response.text}")
            return None
    
    def get_all_news(
        self,
        query: str,
        max_results: int = 1000,
        sort: str = 'date',
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> List[Dict]:
        """
        검색 결과를 모두 가져옵니다 (페이지네이션 처리).
        
        Args:
            query: 검색어
            max_results: 최대 수집할 기사 수
            sort: 정렬 옵션
            date_from: 시작 날짜 (YYYYMMDD)
            date_to: 종료 날짜 (YYYYMMDD)
            
        Returns:
            기사 정보 리스트
        """
        all_items = []
        start = 1
        display = 100
        
        while len(all_items) < max_results:
            result = self.search_news(
                query=query,
                display=display,
                start=start,
                sort=sort,
                date_from=date_from,
                date_to=date_to
            )
            
            if not result or 'items' not in result:
                break
            
            items = result['items']
            if not items:
                break
            
            all_items.extend(items)
            
            # 다음 페이지가 없으면 종료
            total = result.get('total', 0)
            if start + display > total or len(items) < display:
                break
            
            start += display
            
            # API 제한 방지
            time.sleep(self.delay)
        
        return all_items[:max_results]
    
    def extract_view_count(self, link: str) -> Optional[int]:
        """
        네이버 뉴스 기사 페이지에서 조회수를 추출합니다.
        
        Args:
            link: 네이버 뉴스 기사 링크
            
        Returns:
            조회수 (정수) 또는 None
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(link, headers=headers, timeout=10)
            response.raise_for_status()
            response.encoding = 'utf-8'
            
            soup = BeautifulSoup(response.text, 'html.parser')
            import re
            
            # 네이버 뉴스 조회수 추출 - 여러 패턴 시도
            view_count = None
            
            # 패턴 1: 조회수 텍스트에서 숫자 추출
            view_texts = soup.find_all(string=re.compile(r'조회\s*\d+'))
            for text in view_texts:
                numbers = re.findall(r'\d+', text)
                if numbers:
                    view_count = int(numbers[0])
                    break
            
            # 패턴 2: 특정 클래스나 ID에서 조회수 찾기
            if view_count is None:
                view_selectors = [
                    '.media_end_head_info_view_count',
                    '#viewCount',
                    '.view_count',
                    '[class*="view"]',
                    '[id*="view"]'
                ]
                for selector in view_selectors:
                    elem = soup.select_one(selector)
                    if elem:
                        text = elem.get_text()
                        numbers = re.findall(r'\d+', text.replace(',', ''))
                        if numbers:
                            view_count = int(numbers[0])
                            break
            
            # 패턴 3: 스크립트 태그에서 조회수 찾기
            if view_count is None:
                scripts = soup.find_all('script')
                for script in scripts:
                    if script.string:
                        # viewCount, view_count 등의 변수 찾기
                        match = re.search(r'(?:viewCount|view_count|조회수)[\s:=]+(\d+)', script.string)
                        if match:
                            view_count = int(match.group(1))
                            break
            
            return view_count
            
        except Exception as e:
            print(f"조회수 추출 오류 ({link}): {e}")
            return None
    
    def summarize_text(self, text: str, max_length: int = 50) -> str:
        """
        OpenAI API를 사용하여 본문 텍스트를 요약합니다.
        
        Args:
            text: 원본 본문 텍스트
            max_length: 요약 최대 길이 (기본 50자)
            
        Returns:
            요약된 텍스트 (최대 50자)
        """
        if not text or len(text.strip()) == 0:
            return text or ''
        
        # 공백 제거
        text = text.strip()
        
        # 50자 이하라도 OpenAI API를 사용할 수 있으면 요약 시도
        # (더 간결하게 만들 수 있음)
        if len(text) <= max_length and not self.openai_client:
            return text
        
        # OpenAI API를 사용할 수 있는 경우
        if self.openai_client:
            try:
                # 본문이 너무 길면 토큰 제한을 고려하여 앞부분만 사용 (약 3000자)
                text_to_summarize = text[:3000] if len(text) > 3000 else text
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",  # 비용 효율적인 모델 사용
                    messages=[
                        {
                            "role": "system",
                            "content": f"당신은 뉴스 기사 요약 전문가입니다. 주어진 뉴스 기사 본문을 {max_length}자 이내로 핵심 내용만 간결하게 요약해주세요. 요약은 기사의 주요 사실과 내용을 포함해야 합니다."
                        },
                        {
                            "role": "user",
                            "content": f"다음 뉴스 기사 본문을 {max_length}자 이내로 요약해주세요:\n\n{text_to_summarize}"
                        }
                    ],
                    max_tokens=150,
                    temperature=0.3
                )
                
                summary = response.choices[0].message.content.strip()
                
                # 50자 초과 시 자르기
                if len(summary) > max_length:
                    summary = summary[:max_length]
                    # 마지막 문장 부호나 공백 앞에서 자르기
                    last_punct = max(summary.rfind('.'), summary.rfind('!'), summary.rfind('?'), 
                                   summary.rfind('。'), summary.rfind('！'), summary.rfind('？'))
                    if last_punct > max_length * 0.7:
                        summary = summary[:last_punct + 1]
                    else:
                        summary = summary[:max_length] + '...'
                
                return summary
                
            except Exception as e:
                print(f"OpenAI API 요약 오류: {e}")
                # 오류 발생 시 기본 요약 방식으로 폴백
                return self._fallback_summarize(text, max_length)
        else:
            # OpenAI API를 사용할 수 없는 경우 기본 요약 방식 사용
            return self._fallback_summarize(text, max_length)
    
    def _fallback_summarize(self, text: str, max_length: int = 50) -> str:
        """
        OpenAI API를 사용할 수 없을 때 사용하는 기본 요약 방식
        """
        text = text.strip()
        
        if len(text) <= max_length:
            return text
        
        # 50자로 자르기 (공백을 고려하여 단어 중간에서 자르지 않도록)
        summary = text[:max_length]
        
        # 마지막 문자가 공백이 아니고, 원본 텍스트가 더 길면 "..." 추가
        if len(text) > max_length:
            # 공백이나 문장 부호 앞에서 자르기
            last_space = summary.rfind(' ')
            last_punct = max(summary.rfind('.'), summary.rfind('!'), summary.rfind('?'), 
                           summary.rfind('。'), summary.rfind('！'), summary.rfind('？'))
            
            # 문장 부호가 있으면 그 앞에서 자르기
            if last_punct > max_length * 0.7:  # 70% 이상 위치에 문장 부호가 있으면
                summary = summary[:last_punct + 1]
            elif last_space > max_length * 0.7:  # 공백이 있으면 그 앞에서 자르기
                summary = summary[:last_space]
            
            summary += '...'
        
        return summary
    
    def extract_full_text(self, link: str) -> Optional[str]:
        """
        API에서 받은 링크로 실제 기사 본문을 추출합니다.
        (API는 제목과 요약만 제공하므로, 링크로 접근하여 본문 추출)
        
        Args:
            link: 네이버 뉴스 기사 링크
            
        Returns:
            기사 본문 텍스트 또는 None
        """
        if not link:
            return None
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            response = requests.get(link, headers=headers, timeout=15, allow_redirects=True)
            response.raise_for_status()
            
            # 인코딩 자동 감지
            if response.encoding is None or response.encoding == 'ISO-8859-1':
                response.encoding = response.apparent_encoding or 'utf-8'
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 여러 선택자 시도 (우선순위 순)
            selectors = [
                '#newsct_article',
                '.news_end_body_body',
                '._article_body_contents',
                '#articleBodyContents',
                '.article_body',
                '[id*="article"]',
                '[class*="article"]',
                '[class*="body"]',
                'article',
                '.content'
            ]
            
            content_elem = None
            for selector in selectors:
                try:
                    if selector.startswith('['):
                        # 속성 선택자
                        if 'id*=' in selector:
                            content_elem = soup.find('div', {'id': lambda x: x and 'article' in x.lower()})
                        elif 'class*=' in selector:
                            content_elem = soup.find('div', {'class': lambda x: x and ('article' in str(x).lower() or 'body' in str(x).lower())})
                    else:
                        content_elem = soup.select_one(selector)
                    
                    if content_elem and content_elem.get_text(strip=True):
                        break
                except:
                    continue
            
            if content_elem:
                # 불필요한 태그 제거
                for tag in content_elem.find_all(['script', 'style', 'iframe', 'noscript', 'svg']):
                    try:
                        tag.decompose()
                    except (AttributeError, TypeError):
                        continue
                
                # 광고 관련 태그 제거
                for tag in content_elem.find_all(['div', 'section', 'aside']):
                    try:
                        # Tag 객체의 속성 안전하게 가져오기
                        classes = getattr(tag, 'get', lambda x, y: [])('class', [])
                        tag_id = getattr(tag, 'get', lambda x, y: '')('id', '')
                        
                        # classes가 None이거나 리스트가 아닌 경우 처리
                        if classes is None:
                            classes = []
                        elif not isinstance(classes, list):
                            classes = [classes] if classes else []
                        
                        if tag_id is None:
                            tag_id = ''
                        
                        # 광고 관련 키워드 확인
                        classes_str = str(classes).lower()
                        tag_id_str = str(tag_id).lower()
                        
                        if any(keyword in classes_str or keyword in tag_id_str 
                               for keyword in ['ad', 'advertisement', 'sponsor', 'promo', 'banner']):
                            tag.decompose()
                    except (AttributeError, TypeError):
                        # 태그가 예상과 다른 타입이면 건너뛰기
                        continue
                
                text = content_elem.get_text(separator='\n', strip=True)
                # 연속된 공백 정리
                text = re.sub(r'\n\s*\n+', '\n\n', text)
                text = re.sub(r' +', ' ', text)
                text = text.strip()
                
                # 최소 길이 체크 (너무 짧으면 유효하지 않은 것으로 간주)
                if len(text) < 50:
                    return None
                
                return text
            
            return None
            
        except (requests.exceptions.Timeout, requests.exceptions.RequestException):
            return None
        except Exception:
            return None
    
    def crawl_news_with_full_text(
        self,
        query: str,
        max_results: int = 100,
        include_full_text: bool = True,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        sort_by: str = 'date'
    ) -> List[Dict]:
        """
        뉴스를 검색하고 본문까지 추출하여 반환합니다.
        
        Args:
            query: 검색어
            max_results: 최대 수집할 기사 수
            include_full_text: 본문 추출 여부 (True면 시간이 오래 걸림)
            date_from: 시작 날짜 (YYYYMMDD)
            date_to: 종료 날짜 (YYYYMMDD)
            sort_by: 정렬 기준 ('date': 날짜순, 'view': 조회수순)
            
        Returns:
            {
                'title': str,
                'link': str,
                'description': str,
                'pubDate': str,
                'originallink': str,
                'view_count': int (조회수, sort_by='view'일 때만),
                'text': str (본문, include_full_text=True일 때만)
            } 형태의 딕셔너리 리스트
        """
        # 본문 추출 실패 시 대비하여 더 많은 기사를 수집 (최대 2배까지)
        # 실패한 기사를 제외하고도 충분한 수를 확보하기 위함
        items_to_fetch = max_results * 2 if include_full_text else max_results
        items = self.get_all_news(
            query=query,
            max_results=items_to_fetch,
            date_from=date_from,
            date_to=date_to
        )
        
        results = []
        for item in items:
            result = {
                'title': item.get('title', '').replace('<b>', '').replace('</b>', '').strip(),
                'link': item.get('link', ''),
                'description': item.get('description', '').replace('<b>', '').replace('</b>', '').strip(),
                'pubDate': item.get('pubDate', ''),
                'originallink': item.get('originallink', ''),
                'source': self._extract_source_from_link(item.get('originallink', '')),
            }
            
            # 조회수 추출 (조회수 순 정렬 시)
            if sort_by == 'view':
                link_to_use = result['originallink'] or result['link']
                view_count = self.extract_view_count(link_to_use)
                result['view_count'] = view_count if view_count is not None else 0
                time.sleep(self.delay)  # 조회수 추출 시 추가 대기
            
            if include_full_text:
                # 원본 링크가 있으면 원본 링크 사용, 없으면 네이버 링크 사용
                link_to_use = result['originallink'] or result['link']
                full_text = self.extract_full_text(link_to_use)
                
                if full_text:
                    # 본문을 요약하여 저장 (50자로 제한)
                    result['text'] = self.summarize_text(full_text, max_length=50)
                    # 성공한 기사만 결과에 추가
                    results.append(result)
                else:
                    # 본문 추출 실패 시 해당 기사는 건너뛰고 다음 기사로
                    # (이미 충분한 수를 확보했으면 중단)
                    if len(results) >= max_results:
                        break
                    continue  # 실패한 기사는 결과에 포함하지 않음
                time.sleep(self.delay)  # 본문 추출 시 추가 대기
            else:
                # 본문 추출을 하지 않아도 description을 요약
                description = result.get('description', '')
                if description:
                    result['text'] = self.summarize_text(description, max_length=50)
                else:
                    result['text'] = ''
                results.append(result)
            
            # 이미 충분한 수를 확보했으면 중단
            if len(results) >= max_results:
                break
        
        # 정렬 처리
        if sort_by == 'view':
            # 조회수 순으로 정렬 (내림차순)
            results.sort(key=lambda x: x.get('view_count', 0), reverse=True)
        elif sort_by == 'date':
            # 날짜 순으로 정렬 (내림차순 - 최신순)
            results.sort(key=lambda x: x.get('pubDate', ''), reverse=True)
        
        return results
    
    def _extract_source_from_link(self, link: str) -> str:
        """링크에서 출처 추출"""
        if not link:
            return "알 수 없음"
        
        try:
            from urllib.parse import urlparse
            domain = urlparse(link).netloc
            # www. 제거
            domain = domain.replace('www.', '')
            return domain
        except:
            return "알 수 없음"
    
    def get_recent_news(
        self,
        query: str,
        days: int = 1,
        max_results: int = 100,
        sort_by: str = 'date'
    ) -> List[Dict]:
        """
        최근 N일간의 뉴스를 가져옵니다.
        
        Args:
            query: 검색어
            days: 최근 며칠간 (기본 1일)
            max_results: 최대 수집할 기사 수
            sort_by: 정렬 기준 ('date': 날짜순, 'view': 조회수순)
            
        Returns:
            기사 정보 리스트
        """
        date_to = datetime.now().strftime('%Y%m%d')
        date_from = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
        
        return self.crawl_news_with_full_text(
            query=query,
            max_results=max_results,
            include_full_text=True,
            date_from=date_from,
            date_to=date_to,
            sort_by=sort_by
        )


def crawl_naver_news_api(
    query: str,
    client_id: str,
    client_secret: str,
    max_results: int = 100,
    include_full_text: bool = True,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None
) -> List[Dict]:
    """
    네이버 검색 API를 사용하여 뉴스를 크롤링하는 편의 함수
    
    Args:
        query: 검색어
        client_id: 네이버 Client ID
        client_secret: 네이버 Client Secret
        max_results: 최대 수집할 기사 수
        include_full_text: 본문 추출 여부
        date_from: 시작 날짜 (YYYYMMDD)
        date_to: 종료 날짜 (YYYYMMDD)
        
    Returns:
        기사 정보 리스트
    """
    crawler = NaverNewsAPICrawler(client_id, client_secret)
    return crawler.crawl_news_with_full_text(
        query=query,
        max_results=max_results,
        include_full_text=include_full_text,
        date_from=date_from,
        date_to=date_to
    )


if __name__ == '__main__':
    # 환경변수에서 API 키 가져오기
    CLIENT_ID = os.getenv('NAVER_CLIENT_ID', '')
    CLIENT_SECRET = os.getenv('NAVER_CLIENT_SECRET', '')
    
    if not CLIENT_ID or not CLIENT_SECRET:
        print("환경변수 NAVER_CLIENT_ID와 NAVER_CLIENT_SECRET을 설정해주세요.")
        print("또는 코드에서 직접 설정하세요.")
        exit(1)
    
    # 테스트 예시
    crawler = NaverNewsAPICrawler(CLIENT_ID, CLIENT_SECRET, delay=0.1)
    
    # 최근 1일간 'AI' 관련 뉴스 검색
    print("최근 AI 관련 뉴스 검색 중...")
    results = crawler.get_recent_news(query='AI 인공지능', days=1, max_results=10)
    
    print(f"\n총 {len(results)}개 기사 수집 완료\n")
    
    for i, result in enumerate(results, 1):
        print(f"=== 기사 {i} ===")
        print(f"제목: {result['title']}")
        print(f"출처: {result['source']}")
        print(f"날짜: {result['pubDate']}")
        print(f"본문 길이: {len(result.get('text', ''))}자")
        if result.get('text'):
            print(f"본문 미리보기: {result['text'][:200]}...")
        print(f"링크: {result['link']}")
        print()

