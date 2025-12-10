"""
네이버 뉴스 링크 기반 크롤링 모듈
네이버 뉴스 URL을 직접 입력받아 본문을 추출하거나, RSS 피드에서 최신 뉴스를 자동 수집합니다.
"""

import requests
from bs4 import BeautifulSoup
from newspaper import Article
from typing import Dict, List, Optional
import re
from datetime import datetime, timedelta
import time
import feedparser
from urllib.parse import urlparse, quote


class NaverNewsLinkCrawler:
    """네이버 뉴스 링크를 통한 크롤링 클래스"""
    
    def __init__(self, delay: float = 1.0):
        """
        Args:
            delay: 요청 간 대기 시간(초). 서버 부하 방지용
            
        """
        self.delay = delay
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def extract_from_url(self, url: str) -> Optional[Dict]:
        """
        네이버 뉴스 URL에서 기사 정보를 추출합니다.
        
        Args:
            url: 네이버 뉴스 기사 URL
            
        Returns:
            {
                'title': str,
                'text': str,
                'url': str,
                'published': str,
                'source': str,
                'author': str (optional)
            } 또는 None (실패 시)
        """
        try:
            # newspaper3k 사용
            article = Article(url, language='ko')
            article.download()
            article.parse()
            
            if not article.text or len(article.text) < 100:
                # newspaper3k 실패 시 BeautifulSoup으로 재시도
                return self._extract_with_bs4(url)
            
            result = {
                'title': article.title or '',
                'text': article.text.strip(),
                'url': url,
                'published': article.publish_date.strftime('%Y-%m-%d %H:%M:%S') if article.publish_date else None,
                'source': article.source_url if hasattr(article, 'source_url') else self._extract_source_from_url(url),
                'author': article.authors[0] if article.authors else None
            }
            
            time.sleep(self.delay)
            return result
            
        except Exception as e:
            print(f"Error extracting from {url}: {e}")
            # BeautifulSoup으로 재시도
            return self._extract_with_bs4(url)
    
    def _extract_with_bs4(self, url: str) -> Optional[Dict]:
        """BeautifulSoup을 사용한 대체 추출 방법"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            response.encoding = 'utf-8'
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 네이버 뉴스 구조에 맞춘 선택자
            title_elem = soup.select_one('#title_area, .media_end_head_headline h2, h2.end_tit')
            title = title_elem.get_text(strip=True) if title_elem else ''
            
            # 본문 추출
            content_elem = soup.select_one('#newsct_article, .news_end_body_body, ._article_body_contents')
            if not content_elem:
                # 다른 가능한 선택자들
                content_elem = soup.find('div', {'id': re.compile('.*article.*', re.I)})
            
            if content_elem:
                # 불필요한 태그 제거
                for tag in content_elem.find_all(['script', 'style', 'iframe', 'div[class*="ad"]']):
                    tag.decompose()
                text = content_elem.get_text(separator='\n', strip=True)
                # 연속된 공백 정리
                text = re.sub(r'\n\s*\n', '\n\n', text)
            else:
                text = ''
            
            # 날짜 추출
            date_elem = soup.select_one('.media_end_head_info_datestamp_time, ._ARTICLE_DATE_TIME')
            published = None
            if date_elem:
                date_text = date_elem.get('data-date-time') or date_elem.get_text(strip=True)
                published = self._parse_date(date_text)
            
            # 출처 추출
            source_elem = soup.select_one('.media_end_head_top_logo img, .press_logo img')
            source = source_elem.get('alt', '') if source_elem else self._extract_source_from_url(url)
            
            # 작성자 추출
            author_elem = soup.select_one('.byline, ._ARTICLE_BYLINE')
            author = author_elem.get_text(strip=True) if author_elem else None
            
            if not title or not text or len(text) < 100:
                return None
            
            time.sleep(self.delay)
            return {
                'title': title,
                'text': text.strip(),
                'url': url,
                'published': published,
                'source': source,
                'author': author
            }
            
        except Exception as e:
            print(f"Error with BS4 extraction from {url}: {e}")
            return None
    
    def _extract_source_from_url(self, url: str) -> str:
        """URL에서 출처 추출"""
        # 네이버 뉴스 URL 패턴: https://n.news.naver.com/mnews/article/{media_code}/...
        match = re.search(r'/article/(\d+)/', url)
        if match:
            # 미디어 코드를 출처로 사용 (실제 매체명은 별도 매핑 필요)
            return f"media_{match.group(1)}"
        return "네이버 뉴스"
    
    def _parse_date(self, date_str: str) -> Optional[str]:
        """날짜 문자열 파싱"""
        if not date_str:
            return None
        
        try:
            # 다양한 날짜 형식 처리
            patterns = [
                '%Y-%m-%d %H:%M:%S',
                '%Y.%m.%d %H:%M',
                '%Y-%m-%dT%H:%M:%S',
            ]
            
            for pattern in patterns:
                try:
                    dt = datetime.strptime(date_str, pattern)
                    return dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    continue
            
            # ISO 형식
            if 'T' in date_str:
                dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            
        except Exception:
            pass
        
        return date_str
    
    def crawl_multiple(self, urls: List[str]) -> List[Dict]:
        """
        여러 URL을 순차적으로 크롤링합니다.
        
        Args:
            urls: 네이버 뉴스 URL 리스트
            
        Returns:
            추출된 기사 정보 리스트
        """
        results = []
        for url in urls:
            if not url or not url.startswith('http'):
                continue
            
            result = self.extract_from_url(url)
            if result:
                results.append(result)
        
        return results
    
    def get_news_urls_by_keyword(
        self,
        keyword: str,
        max_items: int = 20
    ) -> List[str]:
        """
        키워드로 네이버 뉴스를 검색하여 기사 URL을 가져옵니다.
        
        Args:
            keyword: 검색할 키워드
            max_items: 가져올 최대 기사 수
            
        Returns:
            네이버 뉴스 URL 리스트
        """
        urls = []
        found_links = set()
        
        try:
            # 네이버 뉴스 검색 URL
            search_url = f"https://search.naver.com/search.naver?where=news&query={quote(keyword)}&sm=tab_jum&sort=1"
            
            response = requests.get(search_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            response.encoding = 'utf-8'
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 네이버 뉴스 검색 결과에서 링크 추출
            # 여러 가능한 선택자 시도
            link_selectors = [
                'a.news_tit',  # 네이버 뉴스 검색 결과의 제목 링크
                'a.info',      # 정보 링크
                'a[href*="news.naver.com"]',
                'a[href*="/article/"]',
                '.news_area a',
                '.api_subject_bx a',
            ]
            
            for selector in link_selectors:
                links = soup.select(selector)
                for link in links:
                    if len(urls) >= max_items:
                        break
                    
                    href = link.get('href', '')
                    if not href:
                        continue
                    
                    # 네이버 뉴스 기사 링크 패턴 찾기
                    if '/article/' in href or 'n.news.naver.com' in href or 'news.naver.com' in href:
                        # 상대 경로를 절대 경로로 변환
                        if href.startswith('/'):
                            href = 'https://news.naver.com' + href
                        elif href.startswith('//'):
                            href = 'https:' + href
                        elif not href.startswith('http'):
                            continue
                        
                        # 네이버 뉴스 링크 정규화
                        if 'n.news.naver.com' not in href:
                            match = re.search(r'article/(\d+)/(\d+)', href)
                            if match:
                                href = f"https://n.news.naver.com/mnews/article/{match.group(1)}/{match.group(2)}"
                        
                        # 유효한 네이버 뉴스 링크인지 확인
                        if ('n.news.naver.com' in href or 'news.naver.com' in href) and '/article/' in href:
                            # 댓글 페이지나 기타 페이지 제외
                            if '/comment/' not in href and href not in found_links:
                                found_links.add(href)
                                urls.append(href)
                
                if len(urls) >= max_items:
                    break
            
            # 검색 결과가 부족하면 추가 페이지 시도
            if len(urls) < max_items:
                for page in range(2, 4):  # 2-3페이지까지 시도
                    if len(urls) >= max_items:
                        break
                    
                    page_url = f"https://search.naver.com/search.naver?where=news&query={quote(keyword)}&sm=tab_jum&sort=1&start={((page-1)*10)+1}"
                    try:
                        page_response = requests.get(page_url, headers=self.headers, timeout=10)
                        page_response.raise_for_status()
                        page_response.encoding = 'utf-8'
                        page_soup = BeautifulSoup(page_response.text, 'html.parser')
                        
                        for selector in link_selectors:
                            links = page_soup.select(selector)
                            for link in links:
                                if len(urls) >= max_items:
                                    break
                                
                                href = link.get('href', '')
                                if href and ('/article/' in href or 'n.news.naver.com' in href):
                                    if href.startswith('/'):
                                        href = 'https://news.naver.com' + href
                                    elif href.startswith('//'):
                                        href = 'https:' + href
                                    
                                    if 'n.news.naver.com' not in href:
                                        match = re.search(r'article/(\d+)/(\d+)', href)
                                        if match:
                                            href = f"https://n.news.naver.com/mnews/article/{match.group(1)}/{match.group(2)}"
                                    
                                    if ('n.news.naver.com' in href or 'news.naver.com' in href) and '/article/' in href:
                                        if '/comment/' not in href and href not in found_links:
                                            found_links.add(href)
                                            urls.append(href)
                    except Exception as e:
                        print(f"추가 페이지 크롤링 오류 (페이지 {page}): {e}")
                        break
            
            print(f"키워드 '{keyword}' 검색 결과: {len(urls)}개의 기사 링크를 찾았습니다.")
            
        except Exception as e:
            print(f"키워드 검색 오류: {e}")
        
        return urls[:max_items]
    
    def get_latest_news_urls_from_rss(
        self,
        category: str = 'all',
        max_items: int = 20
    ) -> List[str]:
        """
        네이버 뉴스 RSS 피드에서 최신 기사 URL을 가져옵니다.
        
        Args:
            category: 뉴스 카테고리 ('all', 'politics', 'economy', 'society', 'life', 'world', 'it')
            max_items: 가져올 최대 기사 수
            
        Returns:
            네이버 뉴스 URL 리스트
        """
        # 네이버 뉴스 RSS 피드 URL
        rss_urls = {
            'all': 'https://news.naver.com/main/rss/section.naver?sid=100',
            'politics': 'https://news.naver.com/main/rss/section.naver?sid=100',
            'economy': 'https://news.naver.com/main/rss/section.naver?sid=101',
            'society': 'https://news.naver.com/main/rss/section.naver?sid=102',
            'life': 'https://news.naver.com/main/rss/section.naver?sid=103',
            'world': 'https://news.naver.com/main/rss/section.naver?sid=104',
            'it': 'https://news.naver.com/main/rss/section.naver?sid=105',
        }
        
        # Google News RSS를 통한 네이버 뉴스 검색 (백업)
        google_rss_urls = {
            'all': 'https://news.google.com/rss/search?q=site:news.naver.com&hl=ko&gl=KR&ceid=KR:ko',
            'politics': 'https://news.google.com/rss/search?q=site:news.naver.com+정치&hl=ko&gl=KR&ceid=KR:ko',
            'economy': 'https://news.google.com/rss/search?q=site:news.naver.com+경제&hl=ko&gl=KR&ceid=KR:ko',
            'society': 'https://news.google.com/rss/search?q=site:news.naver.com+사회&hl=ko&gl=KR&ceid=KR:ko',
            'it': 'https://news.google.com/rss/search?q=site:news.naver.com+IT&hl=ko&gl=KR&ceid=KR:ko',
        }
        
        rss_url = rss_urls.get(category.lower(), rss_urls['all'])
        urls = []
        
        try:
            feed = feedparser.parse(rss_url)
            
            if not feed.entries:
                # 네이버 RSS 실패 시 Google News RSS 시도
                print("네이버 RSS 피드가 비어있습니다. Google News RSS를 시도합니다...")
                google_rss_url = google_rss_urls.get(category.lower(), google_rss_urls['all'])
                feed = feedparser.parse(google_rss_url)
            
            for entry in feed.entries[:max_items]:
                link = entry.get('link', '')
                if not link:
                    continue
                
                # Google News 링크인 경우 원본 링크 추출
                if 'news.google.com' in link:
                    # Google News 링크에서 원본 URL 추출
                    if 'url=' in link:
                        import urllib.parse
                        parsed = urllib.parse.urlparse(link)
                        params = urllib.parse.parse_qs(parsed.query)
                        if 'url' in params:
                            link = params['url'][0]
                
                if link and 'news.naver.com' in link:
                    # 네이버 뉴스 링크 정규화
                    if 'n.news.naver.com' not in link:
                        # 구형 링크를 신형으로 변환 시도
                        match = re.search(r'article/(\d+)/(\d+)', link)
                        if match:
                            link = f"https://n.news.naver.com/mnews/article/{match.group(1)}/{match.group(2)}"
                    if link not in urls:
                        urls.append(link)
            
            print(f"RSS에서 {len(urls)}개의 기사 링크를 가져왔습니다.")
            
        except Exception as e:
            print(f"RSS 피드 파싱 오류: {e}")
            # 백업: 네이버 뉴스 메인 페이지에서 링크 추출
            urls = self._get_latest_news_urls_from_main(max_items)
        
        return urls[:max_items]
    
    def _get_latest_news_urls_from_main(self, max_items: int = 20) -> List[str]:
        """
        네이버 뉴스 메인 페이지에서 최신 기사 URL을 추출합니다 (백업 방법).
        
        Args:
            max_items: 가져올 최대 기사 수
            
        Returns:
            네이버 뉴스 URL 리스트
        """
        urls = []
        found_links = set()
        
        # 여러 페이지 시도
        pages_to_try = [
            'https://news.naver.com/main/home.naver',
            'https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=100',  # 정치
            'https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=101',  # 경제
            'https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=102',  # 사회
        ]
        
        for page_url in pages_to_try:
            if len(urls) >= max_items:
                break
                
            try:
                response = requests.get(page_url, headers=self.headers, timeout=10)
                response.raise_for_status()
                response.encoding = 'utf-8'
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 모든 링크 찾기
                all_links = soup.find_all('a', href=True)
                
                for link in all_links:
                    if len(urls) >= max_items:
                        break
                        
                    href = link.get('href', '')
                    if not href:
                        continue
                    
                    # 네이버 뉴스 기사 링크 패턴 찾기
                    if '/article/' in href or 'n.news.naver.com' in href:
                        # 상대 경로를 절대 경로로 변환
                        if href.startswith('/'):
                            href = 'https://news.naver.com' + href
                        elif href.startswith('//'):
                            href = 'https:' + href
                        elif not href.startswith('http'):
                            continue
                        
                        # 네이버 뉴스 링크 정규화
                        if 'n.news.naver.com' not in href:
                            match = re.search(r'article/(\d+)/(\d+)', href)
                            if match:
                                href = f"https://n.news.naver.com/mnews/article/{match.group(1)}/{match.group(2)}"
                        
                        # 유효한 네이버 뉴스 링크인지 확인
                        if 'n.news.naver.com' in href or 'news.naver.com' in href:
                            if href not in found_links and '/article/' in href:
                                found_links.add(href)
                                urls.append(href)
                
            except Exception as e:
                print(f"페이지 크롤링 오류 ({page_url}): {e}")
                continue
        
        print(f"메인 페이지에서 {len(urls)}개의 기사 링크를 추출했습니다.")
        return urls[:max_items]
    
    def crawl_news_by_keyword(
        self,
        keyword: str,
        max_items: int = 10
    ) -> List[Dict]:
        """
        키워드로 뉴스를 검색하여 수집합니다.
        
        Args:
            keyword: 검색할 키워드
            max_items: 수집할 최대 기사 수
            
        Returns:
            추출된 기사 정보 리스트
        """
        print(f"\n키워드 '{keyword}'로 뉴스 검색 시작 (개수: {max_items})...")
        
        urls = self.get_news_urls_by_keyword(keyword, max_items)
        
        if not urls:
            print("수집할 기사 링크가 없습니다.")
            return []
        
        print(f"\n{len(urls)}개의 기사 크롤링 시작...\n")
        return self.crawl_multiple(urls)
    
    def crawl_latest_news(
        self,
        category: str = 'all',
        max_items: int = 10,
        use_rss: bool = True
    ) -> List[Dict]:
        """
        최신 뉴스를 자동으로 수집합니다.
        
        Args:
            category: 뉴스 카테고리 ('all', 'politics', 'economy', 'society', 'life', 'world', 'it')
            max_items: 수집할 최대 기사 수
            use_rss: RSS 피드 사용 여부 (False면 메인 페이지에서 추출)
            
        Returns:
            추출된 기사 정보 리스트
        """
        print(f"\n최신 뉴스 수집 시작 (카테고리: {category}, 개수: {max_items})...")
        
        if use_rss:
            urls = self.get_latest_news_urls_from_rss(category, max_items)
        else:
            urls = self._get_latest_news_urls_from_main(max_items)
        
        if not urls:
            print("수집할 기사 링크가 없습니다.")
            return []
        
        print(f"\n{len(urls)}개의 기사 크롤링 시작...\n")
        return self.crawl_multiple(urls)


def crawl_naver_news_from_links(urls: List[str], delay: float = 1.0) -> List[Dict]:
    """
    네이버 뉴스 링크 리스트를 받아 크롤링하는 편의 함수
    
    Args:
        urls: 네이버 뉴스 URL 리스트
        delay: 요청 간 대기 시간(초)
        
    Returns:
        추출된 기사 정보 리스트
    """
    crawler = NaverNewsLinkCrawler(delay=delay)
    return crawler.crawl_multiple(urls)


if __name__ == '__main__':
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='네이버 뉴스 크롤링 도구')
    parser.add_argument('urls', nargs='*', help='크롤링할 네이버 뉴스 URL 또는 검색 키워드 (선택사항)')
    parser.add_argument('--keyword', '-k', type=str, help='검색할 키워드')
    parser.add_argument('--auto', action='store_true', help='최신 뉴스 자동 수집')
    parser.add_argument('--category', default='all', 
                       choices=['all', 'politics', 'economy', 'society', 'life', 'world', 'it'],
                       help='뉴스 카테고리 (--auto 사용 시)')
    parser.add_argument('--count', type=int, default=10, help='수집할 기사 수 (기본: 10)')
    parser.add_argument('--no-rss', action='store_true', help='RSS 대신 메인 페이지에서 추출')
    
    args = parser.parse_args()
    
    crawler = NaverNewsLinkCrawler(delay=1.0)
    
    # 명령줄 인자가 없으면 기본적으로 키워드 입력 모드
    if len(sys.argv) == 1:
        print("=" * 60)
        print("네이버 뉴스 키워드 검색 크롤링")
        print("=" * 60)
        print("\n검색할 키워드를 입력하세요.")
        print("(엔터만 누르면 종료)\n")
        
        try:
            keyword = input("키워드: ").strip()
            if not keyword:
                print("\n키워드가 입력되지 않아 종료합니다.")
                sys.exit(0)
            
            count_input = input(f"수집할 기사 수 (기본: {args.count}): ").strip()
            count = int(count_input) if count_input.isdigit() else args.count
            
            results = crawler.crawl_news_by_keyword(
                keyword=keyword,
                max_items=count
            )
        except (EOFError, KeyboardInterrupt):
            print("\n\n종료합니다.")
            sys.exit(0)
    
    # 키워드 검색 모드 (--keyword 옵션)
    elif args.keyword:
        print("=" * 60)
        print("네이버 뉴스 키워드 검색 크롤링")
        print("=" * 60)
        results = crawler.crawl_news_by_keyword(
            keyword=args.keyword,
            max_items=args.count
        )
    
    # URL 직접 입력 모드 (URL로 시작하는 경우)
    elif args.urls and any(url.startswith('http') for url in args.urls):
        print("=" * 60)
        print("네이버 뉴스 링크 크롤링")
        print("=" * 60)
        print(f"\n{len(args.urls)}개의 URL 크롤링 시작...\n")
        results = crawler.crawl_multiple(args.urls)
    
    # 키워드 입력 모드 (인자가 있고 URL이 아닌 경우)
    elif args.urls and not any(url.startswith('http') for url in args.urls):
        keyword = ' '.join(args.urls)
        print("=" * 60)
        print("네이버 뉴스 키워드 검색 크롤링")
        print("=" * 60)
        results = crawler.crawl_news_by_keyword(
            keyword=keyword,
            max_items=args.count
        )
    
    # 자동 수집 모드
    elif args.auto:
        print("=" * 60)
        print("네이버 뉴스 자동 크롤링")
        print("=" * 60)
        results = crawler.crawl_latest_news(
            category=args.category,
            max_items=args.count,
            use_rss=not args.no_rss
        )
    
    # 기타 경우 (도움말 등)
    else:
        print("=" * 60)
        print("네이버 뉴스 크롤링")
        print("=" * 60)
        print("\n사용법:")
        print("  기본 실행: python src/crawl_naver_link.py")
        print("    → 키워드 입력창이 나타납니다")
        print("  키워드 검색: python src/crawl_naver_link.py --keyword <키워드> [--count 10]")
        print("  또는: python src/crawl_naver_link.py <키워드> [--count 10]")
        print("  URL 직접: python src/crawl_naver_link.py <URL1> <URL2> ...")
        print("  자동 수집: python src/crawl_naver_link.py --auto [--category all] [--count 10]")
        print("\n옵션을 보려면: python src/crawl_naver_link.py --help\n")
        sys.exit(0)
    
    # 결과 출력
    if not results:
        print("\n크롤링된 결과가 없습니다.")
    else:
        print(f"\n{'='*60}")
        print(f"총 {len(results)}개의 기사를 수집했습니다.")
        print(f"{'='*60}\n")
        
        for i, result in enumerate(results, 1):
            print(f"\n{'='*60}")
            print(f"기사 {i}/{len(results)}")
            print(f"{'='*60}")
            print(f"제목: {result['title']}")
            print(f"출처: {result['source']}")
            print(f"날짜: {result['published']}")
            print(f"본문 길이: {len(result['text'])}자")
            if result['text']:
                print(f"\n본문 미리보기:\n{result['text'][:300]}...")
            print(f"\nURL: {result['url']}")

