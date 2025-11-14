"""
네이버 검색 API를 활용한 뉴스 크롤링 모듈
네이버 검색 API를 사용하여 뉴스를 검색하고 수집합니다.
"""

import requests
import os
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time
from bs4 import BeautifulSoup


class NaverNewsAPICrawler:
    """네이버 검색 API를 사용한 뉴스 크롤링 클래스"""
    
    def __init__(self, client_id: str, client_secret: str, delay: float = 0.1):
        """
        Args:
            client_id: 네이버 개발자 센터에서 발급받은 Client ID
            client_secret: 네이버 개발자 센터에서 발급받은 Client Secret
            delay: API 요청 간 대기 시간(초). API 제한 방지용
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.delay = delay
        self.api_url = "https://openapi.naver.com/v1/search/news.json"
        self.headers = {
            'X-Naver-Client-Id': client_id,
            'X-Naver-Client-Secret': client_secret
        }
    
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
    
    def extract_full_text(self, link: str) -> Optional[str]:
        """
        API에서 받은 링크로 실제 기사 본문을 추출합니다.
        (API는 제목과 요약만 제공하므로, 링크로 접근하여 본문 추출)
        
        Args:
            link: 네이버 뉴스 기사 링크
            
        Returns:
            기사 본문 텍스트 또는 None
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(link, headers=headers, timeout=10)
            response.raise_for_status()
            response.encoding = 'utf-8'
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 네이버 뉴스 본문 선택자
            content_elem = soup.select_one('#newsct_article, .news_end_body_body, ._article_body_contents')
            if not content_elem:
                content_elem = soup.find('div', {'id': lambda x: x and 'article' in x.lower()})
            
            if content_elem:
                # 불필요한 태그 제거
                for tag in content_elem.find_all(['script', 'style', 'iframe', 'div']):
                    if 'ad' in tag.get('class', []) or 'ad' in tag.get('id', ''):
                        tag.decompose()
                
                text = content_elem.get_text(separator='\n', strip=True)
                # 연속된 공백 정리
                import re
                text = re.sub(r'\n\s*\n', '\n\n', text)
                return text.strip()
            
            return None
            
        except Exception as e:
            print(f"본문 추출 오류 ({link}): {e}")
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
        items = self.get_all_news(
            query=query,
            max_results=max_results,
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
                result['text'] = full_text or result['description']
                time.sleep(self.delay)  # 본문 추출 시 추가 대기
            else:
                result['text'] = result['description']
            
            results.append(result)
        
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

