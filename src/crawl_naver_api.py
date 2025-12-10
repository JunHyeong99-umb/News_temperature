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

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class NaverNewsAPICrawler:
    """네이버 검색 API를 사용한 뉴스 크롤링 클래스"""
    
    def __init__(self, client_id: str, client_secret: str, delay: float = 0.1, 
                 openai_api_key: Optional[str] = None, summary_mode: str = 'kosum-v1-fast'):
        """
        Args:
            client_id: 네이버 개발자 센터에서 발급받은 Client ID
            client_secret: 네이버 개발자 센터에서 발급받은 Client Secret
            delay: API 요청 간 대기 시간(초). API 제한 방지용
            openai_api_key: OpenAI API 키 (요약 기능 사용 시 필요, summary_mode가 'openai'일 때)
            summary_mode: 요약 모드 ('kosum-v1-fast', 'kosum-v1-tuned' 또는 'openai'), 기본값은 'kosum-v1-fast'
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
        self.summary_mode = summary_mode
        
        # OpenAI 클라이언트 초기화
        if openai_api_key and OPENAI_AVAILABLE:
            self.openai_client = OpenAI(api_key=openai_api_key)
        
        # kosum-v1-fast 모델 초기화 (지연 로딩)
        self.kosum_model = None
        self.kosum_tokenizer = None
        self.kosum_device = None
        
        # kosum-v1-tuned 모델 초기화 (지연 로딩)
        self.kosum_tuned_model = None
        self.kosum_tuned_tokenizer = None
        self.kosum_tuned_device = None
    
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
        # 검색어 유효성 검사
        if not query or not query.strip():
            print("경고: 검색어가 비어있습니다.")
            return None
        
        query = query.strip()
        
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
            print(f"[검색] 검색어: '{query}', 시작 위치: {start}, 정렬: {sort}")
            response = requests.get(
                self.api_url,
                headers=self.headers,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            result = response.json()
            
            # 검색 결과 확인
            if 'items' in result:
                print(f"[검색] 검색 결과: {len(result['items'])}개 기사 발견 (전체: {result.get('total', 0)}개)")
            else:
                print(f"[검색] 경고: 검색 결과에 'items' 키가 없습니다. 응답: {result}")
            
            time.sleep(self.delay)
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"API 요청 오류: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"응답 상태 코드: {e.response.status_code}")
                print(f"응답 내용: {e.response.text}")
            return None
        except Exception as e:
            print(f"검색 중 예상치 못한 오류: {e}")
            import traceback
            traceback.print_exc()
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
        # 검색어 유효성 검사
        if not query or not query.strip():
            print("경고: 검색어가 비어있습니다.")
            return []
        
        query = query.strip()
        print(f"[get_all_news] 검색어: '{query}', 최대 결과: {max_results}, 정렬: {sort}")
        
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
            
            if not result:
                print(f"[get_all_news] 검색 결과가 None입니다. 중단합니다.")
                break
            
            if 'items' not in result:
                print(f"[get_all_news] 검색 결과에 'items' 키가 없습니다. 응답: {result}")
                break
            
            items = result['items']
            if not items:
                print(f"[get_all_news] 검색 결과가 비어있습니다. 중단합니다.")
                break
            
            all_items.extend(items)
            print(f"[get_all_news] 현재 수집된 기사 수: {len(all_items)}개")
            
            # 다음 페이지가 없으면 종료
            total = result.get('total', 0)
            if start + display > total or len(items) < display:
                print(f"[get_all_news] 더 이상 가져올 기사가 없습니다. (전체: {total}개)")
                break
            
            start += display
            
            # API 제한 방지
            time.sleep(self.delay)
        
        print(f"[get_all_news] 최종 수집된 기사 수: {len(all_items[:max_results])}개")
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
    
    def _load_kosum_model(self):
        """kosum-v1-fast 모델을 로드합니다 (지연 로딩)"""
        if self.kosum_model is not None:
            return
        
        if not TRANSFORMERS_AVAILABLE:
            print("경고: transformers가 설치되지 않았습니다. kosum-v1-fast 모델을 사용할 수 없습니다.")
            return
        
        try:
            # GPU 사용 가능 여부 확인
            self.kosum_device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # kosum-v1-fast 모델 로드 (일반적으로 한국어 요약 모델)
            # 모델 이름이 정확하지 않을 수 있으므로, 일반적인 한국어 요약 모델 사용
            # 사용자가 정확한 모델 이름을 알려주면 수정 가능
            model_name = "gogamza/kobart-summarization"  # 한국어 요약 모델
            
            print(f"kosum-v1-fast 모델 로드 중... (디바이스: {self.kosum_device})")
            self.kosum_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.kosum_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            if self.kosum_device == "cuda":
                self.kosum_model = self.kosum_model.to(self.kosum_device)
            
            self.kosum_model.eval()
            print("kosum-v1-fast 모델 로드 완료")
            
        except Exception as e:
            print(f"kosum-v1-fast 모델 로드 오류: {e}")
            print("기본 요약 방식으로 폴백합니다.")
            self.kosum_model = None
            self.kosum_tokenizer = None
    
    def _load_kosum_tuned_model(self):
        """kosum-v1-tuned 모델을 로드합니다 (지연 로딩)"""
        if self.kosum_tuned_model is not None:
            return
        
        if not TRANSFORMERS_AVAILABLE:
            print("경고: transformers가 설치되지 않았습니다. kosum-v1-tuned 모델을 사용할 수 없습니다.")
            return
        
        try:
            # GPU 사용 가능 여부 확인
            self.kosum_tuned_device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # kosum-v1-tuned 모델 로드
            # 실제 모델 이름은 사용자가 제공한 이름으로 변경 가능
            # 예: "nlpai-lab/kosmos-2-patch14-224" 또는 다른 tuned 모델
            model_name = "gogamza/kobart-summarization"  # 기본 모델 (tuned 모델 이름으로 변경 필요)
            
            print(f"kosum-v1-tuned 모델 로드 중... (디바이스: {self.kosum_tuned_device})")
            self.kosum_tuned_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.kosum_tuned_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            if self.kosum_tuned_device == "cuda":
                self.kosum_tuned_model = self.kosum_tuned_model.to(self.kosum_tuned_device)
            
            self.kosum_tuned_model.eval()
            print("kosum-v1-tuned 모델 로드 완료")
            
        except Exception as e:
            print(f"kosum-v1-tuned 모델 로드 오류: {e}")
            print("기본 요약 방식으로 폴백합니다.")
            self.kosum_tuned_model = None
            self.kosum_tuned_tokenizer = None
    
    def _summarize_with_kosum(self, text: str) -> str:
        """kosum-v1-fast 모델을 사용하여 텍스트를 요약합니다"""
        if not TRANSFORMERS_AVAILABLE or self.kosum_model is None:
            return self._fallback_summarize(text, 300)
        
        try:
            # 모델이 아직 로드되지 않았으면 로드
            if self.kosum_model is None:
                self._load_kosum_model()
            
            if self.kosum_model is None:
                return self._fallback_summarize(text, 300)
            
            # 텍스트 전처리 강화
            # 1. 불필요한 패턴 제거
            # 해시태그 제거 (#으로 시작하는 단어들)
            text = re.sub(r'#\S+', '', text)
            # 사진 = 연합뉴스 같은 패턴 제거
            text = re.sub(r'사진\s*[=:]\s*[가-힣a-zA-Z\s]+', '', text, flags=re.IGNORECASE)
            text = re.sub(r'그림\s*[=:]\s*[가-힣a-zA-Z\s]+', '', text, flags=re.IGNORECASE)
            text = re.sub(r'표\s*[=:]\s*[가-힣a-zA-Z\s]+', '', text, flags=re.IGNORECASE)
            text = re.sub(r'[/]\s*[가-힣a-zA-Z\s]+\s*제공', '', text, flags=re.IGNORECASE)
            text = re.sub(r'[/]?\s*제공\s*[=:][^\n]*', '', text, flags=re.IGNORECASE)
            text = re.sub(r'[가-힣\s]+(조감도|사진|그림|표|이미지)[\.]?\s*[/]\s*[가-힣a-zA-Z\s]+\s*제공', '', text, flags=re.IGNORECASE)
            
            # 2. 본문만 추출 (첫 문장부터 시작)
            lines = text.split('\n')
            cleaned_lines = []
            found_first_sentence = False
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 해시태그가 포함된 줄 제거
                if re.search(r'#\S+', line):
                    continue
                
                # 사진 = 연합뉴스 같은 패턴이 포함된 줄 제거
                if re.search(r'사진\s*[=:]\s*[가-힣a-zA-Z\s]+', line, re.IGNORECASE):
                    continue
                if re.search(r'그림\s*[=:]\s*[가-힣a-zA-Z\s]+', line, re.IGNORECASE):
                    continue
                if re.search(r'표\s*[=:]\s*[가-힣a-zA-Z\s]+', line, re.IGNORECASE):
                    continue
                
                # 본문 시작 확인 (실제 내용이 있는 문장)
                if not found_first_sentence:
                    # 문장 부호가 있고, 최소 길이가 있는 경우 본문 시작으로 간주
                    if re.search(r'[가-힣]{5,}', line) and ('.' in line or '다' in line or '다.' in line):
                        found_first_sentence = True
                    else:
                        # 캡션이나 제공 정보는 건너뛰기
                        if re.search(r'[/]?\s*제공|조감도|사진\s*제공', line, re.IGNORECASE):
                            continue
                        if len(line) < 20:  # 너무 짧은 줄은 건너뛰기
                            continue
                
                if found_first_sentence:
                    # 기자 정보가 나오면 중단
                    if re.search(r'[가-힣]+\s*[=:]?\s*[가-힣]*\s*기자', line) and len(line) < 50:
                        break
                    cleaned_lines.append(line)
            
            # 본문이 없으면 원본 사용
            if not cleaned_lines:
                cleaned_lines = [line.strip() for line in lines if line.strip() and len(line.strip()) > 20]
            
            text_to_summarize = '\n'.join(cleaned_lines)
            
            # 너무 길면 앞부분 사용 (본문의 핵심 부분)
            max_input_length = 512
            if len(text_to_summarize) > max_input_length:
                # 앞부분만 사용하되, 문장 단위로 자르기
                truncated = text_to_summarize[:max_input_length]
                last_period = max(truncated.rfind('.'), truncated.rfind('다.'), truncated.rfind('다'))
                if last_period > max_input_length * 0.7:
                    text_to_summarize = truncated[:last_period + 1]
                else:
                    text_to_summarize = truncated
            
            # 토크나이징
            inputs = self.kosum_tokenizer(
                text_to_summarize,
                max_length=max_input_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # 디바이스로 이동
            inputs = {k: v.to(self.kosum_device) for k, v in inputs.items()}
            
            # 요약 생성 (품질 향상을 위한 파라미터 조정)
            with torch.no_grad():
                outputs = self.kosum_model.generate(
                    **inputs,
                    max_length=250,  # 적절한 길이로 조정
                    min_length=80,   # 최소 길이 조정
                    num_beams=5,     # 4 -> 5로 증가 (더 나은 품질)
                    early_stopping=True,
                    no_repeat_ngram_size=3,  # 2 -> 3으로 증가 (반복 방지)
                    length_penalty=1.2,  # 길이 페널티 추가 (더 자연스러운 요약)
                    do_sample=False  # 결정적 생성
                )
            
            # 디코딩
            summary = self.kosum_tokenizer.decode(outputs[0], skip_special_tokens=True)
            summary = summary.strip()
            
            # 요약 결과에서 불필요한 내용 제거
            summary = self._clean_summary(summary)
            
            # 요약이 완전한 문장으로 끝나도록 처리
            if summary and not summary.endswith(('.', '!', '?', '。', '！', '？', '다')):
                # 마지막 문장 부호 찾기
                last_punct = max(
                    summary.rfind('.'),
                    summary.rfind('!'),
                    summary.rfind('?'),
                    summary.rfind('。'),
                    summary.rfind('！'),
                    summary.rfind('？'),
                    summary.rfind('다.')
                )
                if last_punct > len(summary) * 0.5:  # 중간 이후에 문장 부호가 있으면
                    summary = summary[:last_punct + 1]
                elif summary.endswith('다') and len(summary) > 10:
                    summary = summary  # '다'로 끝나면 그대로 사용
            
            return summary
            
        except Exception as e:
            print(f"kosum-v1-fast 요약 오류: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_summarize(text, 300)
    
    def _summarize_with_kosum_tuned(self, text: str) -> str:
        """kosum-v1-tuned 모델을 사용하여 텍스트를 요약합니다"""
        if not TRANSFORMERS_AVAILABLE or self.kosum_tuned_model is None:
            return self._fallback_summarize(text, 300)
        
        try:
            # 모델이 아직 로드되지 않았으면 로드
            if self.kosum_tuned_model is None:
                self._load_kosum_tuned_model()
            
            if self.kosum_tuned_model is None:
                return self._fallback_summarize(text, 300)
            
            # 텍스트 전처리 강화
            # 1. 불필요한 패턴 제거
            # 해시태그 제거 (#으로 시작하는 단어들)
            text = re.sub(r'#\S+', '', text)
            # 사진 = 연합뉴스 같은 패턴 제거
            text = re.sub(r'사진\s*[=:]\s*[가-힣a-zA-Z\s]+', '', text, flags=re.IGNORECASE)
            text = re.sub(r'그림\s*[=:]\s*[가-힣a-zA-Z\s]+', '', text, flags=re.IGNORECASE)
            text = re.sub(r'표\s*[=:]\s*[가-힣a-zA-Z\s]+', '', text, flags=re.IGNORECASE)
            text = re.sub(r'[/]\s*[가-힣a-zA-Z\s]+\s*제공', '', text, flags=re.IGNORECASE)
            text = re.sub(r'[/]?\s*제공\s*[=:][^\n]*', '', text, flags=re.IGNORECASE)
            text = re.sub(r'[가-힣\s]+(조감도|사진|그림|표|이미지)[\.]?\s*[/]\s*[가-힣a-zA-Z\s]+\s*제공', '', text, flags=re.IGNORECASE)
            
            # 2. 본문만 추출 (첫 문장부터 시작)
            lines = text.split('\n')
            cleaned_lines = []
            found_first_sentence = False
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 해시태그가 포함된 줄 제거
                if re.search(r'#\S+', line):
                    continue
                
                # 사진 = 연합뉴스 같은 패턴이 포함된 줄 제거
                if re.search(r'사진\s*[=:]\s*[가-힣a-zA-Z\s]+', line, re.IGNORECASE):
                    continue
                if re.search(r'그림\s*[=:]\s*[가-힣a-zA-Z\s]+', line, re.IGNORECASE):
                    continue
                if re.search(r'표\s*[=:]\s*[가-힣a-zA-Z\s]+', line, re.IGNORECASE):
                    continue
                
                # 본문 시작 확인 (실제 내용이 있는 문장)
                if not found_first_sentence:
                    # 문장 부호가 있고, 최소 길이가 있는 경우 본문 시작으로 간주
                    if re.search(r'[가-힣]{5,}', line) and ('.' in line or '다' in line or '다.' in line):
                        found_first_sentence = True
                    else:
                        # 캡션이나 제공 정보는 건너뛰기
                        if re.search(r'[/]?\s*제공|조감도|사진\s*제공', line, re.IGNORECASE):
                            continue
                        if len(line) < 20:  # 너무 짧은 줄은 건너뛰기
                            continue
                
                if found_first_sentence:
                    # 기자 정보가 나오면 중단
                    if re.search(r'[가-힣]+\s*[=:]?\s*[가-힣]*\s*기자', line) and len(line) < 50:
                        break
                    cleaned_lines.append(line)
            
            # 본문이 없으면 원본 사용
            if not cleaned_lines:
                cleaned_lines = [line.strip() for line in lines if line.strip() and len(line.strip()) > 20]
            
            text_to_summarize = '\n'.join(cleaned_lines)
            
            # 너무 길면 앞부분 사용 (본문의 핵심 부분)
            max_input_length = 512
            if len(text_to_summarize) > max_input_length:
                # 앞부분만 사용하되, 문장 단위로 자르기
                truncated = text_to_summarize[:max_input_length]
                last_period = max(truncated.rfind('.'), truncated.rfind('다.'), truncated.rfind('다'))
                if last_period > max_input_length * 0.7:
                    text_to_summarize = truncated[:last_period + 1]
                else:
                    text_to_summarize = truncated
            
            # 토크나이징
            inputs = self.kosum_tuned_tokenizer(
                text_to_summarize,
                max_length=max_input_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # 디바이스로 이동
            inputs = {k: v.to(self.kosum_tuned_device) for k, v in inputs.items()}
            
            # 요약 생성 (tuned 모델은 더 나은 품질을 위해 파라미터 조정)
            with torch.no_grad():
                outputs = self.kosum_tuned_model.generate(
                    **inputs,
                    max_length=300,  # tuned 모델은 더 긴 요약 가능
                    min_length=100,  # 최소 길이 조정
                    num_beams=6,     # 5 -> 6으로 증가 (더 나은 품질)
                    early_stopping=True,
                    no_repeat_ngram_size=3,  # 반복 방지
                    length_penalty=1.3,  # 길이 페널티 (더 자연스러운 요약)
                    do_sample=False  # 결정적 생성
                )
            
            # 디코딩
            summary = self.kosum_tuned_tokenizer.decode(outputs[0], skip_special_tokens=True)
            summary = summary.strip()
            
            # 요약 결과에서 불필요한 내용 제거
            summary = self._clean_summary(summary)
            
            # 요약이 완전한 문장으로 끝나도록 처리
            if summary and not summary.endswith(('.', '!', '?', '。', '！', '？', '다')):
                # 마지막 문장 부호 찾기
                last_punct = max(
                    summary.rfind('.'),
                    summary.rfind('!'),
                    summary.rfind('?'),
                    summary.rfind('。'),
                    summary.rfind('！'),
                    summary.rfind('？'),
                    summary.rfind('다.')
                )
                if last_punct > len(summary) * 0.5:  # 중간 이후에 문장 부호가 있으면
                    summary = summary[:last_punct + 1]
                elif summary.endswith('다') and len(summary) > 10:
                    summary = summary  # '다'로 끝나면 그대로 사용
            
            return summary
            
        except Exception as e:
            print(f"kosum-v1-tuned 요약 오류: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_summarize(text, 300)
    
    def summarize_text(self, text: str, max_length: int = 50) -> str:
        """
        선택된 모드에 따라 본문 텍스트를 요약합니다.
        기사 본문만 요약하도록 텍스트를 정제합니다.
        
        Args:
            text: 원본 본문 텍스트
            max_length: 요약 최대 길이 (OpenAI API 및 kosum-v1-fast 사용 시 무시됨)
            
        Returns:
            요약된 텍스트
        """
        if not text or len(text.strip()) == 0:
            return text or ''
        
        # 공백 제거
        text = text.strip()
        
        # 기사 본문만 추출하도록 추가 정제
        text = self._clean_article_text(text)
        
        if not text or len(text.strip()) == 0:
            return ''
        
        # 텍스트가 너무 짧으면 요약하지 않음 (최소 100자 이상)
        if len(text.strip()) < 100:
            # 너무 짧은 경우 앞부분만 반환
            return text[:max_length] if len(text) > max_length else text
        
        # 요약 모드에 따라 분기
        if self.summary_mode == 'openai':
            # OpenAI API를 사용하는 경우
            if self.openai_client:
                try:
                    # 본문이 너무 길면 토큰 제한을 고려하여 앞부분만 사용 (약 4000자)
                    text_to_summarize = text[:4000] if len(text) > 4000 else text
                    
                    response = self.openai_client.chat.completions.create(
                        model="gpt-4o-mini",  # 비용 효율적인 모델 사용
                        messages=[
                            {
                                "role": "system",
                                "content": "당신은 뉴스 기사 요약 전문가입니다. 주어진 뉴스 기사 본문만을 바탕으로 핵심 내용을 3줄 정도로 간결하게 요약해주세요. 요약은 기사의 주요 사실, 인물, 장소, 시간, 이유 등을 포함해야 합니다. 각 줄은 완전한 문장으로 작성해주세요. 다음 내용은 절대 요약에 포함하지 마세요: 기자 정보, 관련 기사, 댓글, 광고, [사진=...], [그림=...], /사진 제공=, 사진 제공=, 인물 이름과 직책만 나열된 캡션(예: '젠슨 황 엔비디아 CEO /사진 제공=엔비디아'). 순수한 기사 본문의 실제 내용만 요약해주세요."
                            },
                            {
                                "role": "user",
                                "content": f"다음 뉴스 기사 본문을 3줄 정도로 요약해주세요. 기사 본문의 실제 내용만 요약하고, 기자 정보, 관련 기사, [사진=...], /사진 제공= 같은 캡션이나 제공 정보는 절대 포함하지 마세요:\n\n{text_to_summarize}"
                            }
                        ],
                        max_tokens=400,  # 더 긴 요약을 위해 토큰 수 증가 (300 -> 400)
                        temperature=0.3
                    )
                    
                    summary = response.choices[0].message.content.strip()
                    
                    # 요약 결과에서 불필요한 내용 제거
                    summary = self._clean_summary(summary)
                    
                    return summary
                    
                except Exception as e:
                    print(f"OpenAI API 요약 오류: {e}")
                    # 오류 발생 시 기본 요약 방식으로 폴백
                    return self._fallback_summarize(text, max_length)
            else:
                # OpenAI 클라이언트가 없으면 기본 요약 방식 사용
                return self._fallback_summarize(text, max_length)
        
        elif self.summary_mode == 'kosum-v1-fast':
            # kosum-v1-fast 모델을 사용하는 경우
            return self._summarize_with_kosum(text)
        
        elif self.summary_mode == 'kosum-v1-tuned':
            # kosum-v1-tuned 모델을 사용하는 경우
            return self._summarize_with_kosum_tuned(text)
        
        else:
            # 알 수 없는 모드면 기본 요약 방식 사용
            print(f"알 수 없는 요약 모드: {self.summary_mode}, 기본 요약 방식 사용")
            return self._fallback_summarize(text, max_length)
    
    def _clean_article_text(self, text: str) -> str:
        """
        기사 본문에서 불필요한 내용을 제거합니다.
        (관련 기사, 댓글, 광고, 기자 정보, 해시태그, 사진 캡션 등)
        """
        if not text:
            return text
        
        # 해시태그 제거 (#으로 시작하는 단어들)
        text = re.sub(r'#\S+', '', text)
        
        # 사진 = 연합뉴스 같은 패턴 제거
        text = re.sub(r'사진\s*[=:]\s*[가-힣a-zA-Z\s]+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'그림\s*[=:]\s*[가-힣a-zA-Z\s]+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'표\s*[=:]\s*[가-힣a-zA-Z\s]+', '', text, flags=re.IGNORECASE)
        
        lines = text.split('\n')
        cleaned_lines = []
        
        # 제거할 패턴들
        skip_patterns = [
            r'^\[사진[=:]',
            r'^\[그림[=:]',
            r'^\[표[=:]',
            r'^\[캡션[=:]',
            r'^\[.*사진.*\]',
            r'^\[.*그림.*\]',
            r'^\[.*표.*\]',
            r'^\[.*캡션.*\]',
            r'^관련\s*기사',
            r'^댓글',
            r'^기자\s*[=:]',
            r'^제보',
            r'^Copyright',
            r'^©',
            r'^무단\s*전재',
            r'^재배포\s*금지',
            r'^기사\s*제보',
            r'^이\s*기사',
            r'^기사\s*내용',
            r'^\[.*기자.*\]',
            r'^.*@.*\.(com|kr|net)',
            r'^.*기자.*=',
            r'^.*특파원.*=',
            r'^.*인턴기자.*=',
            r'^.*기자.*기자',
            r'^.*기자.*특파원',
            r'^.*기자.*인턴',
            r'^.*기자.*=.*기자',
            r'^.*기자.*=.*특파원',
            r'^.*기자.*=.*인턴',
            r'^.*기자.*=.*=',
            r'^.*기자.*기자.*=',
            r'^.*기자.*특파원.*=',
            r'^.*기자.*인턴.*=',
            r'^.*기자.*=.*=.*기자',
            r'^.*기자.*=.*=.*특파원',
            r'^.*기자.*=.*=.*인턴',
            r'^.*기자.*=.*=.*=',
            r'^.*기자.*기자.*=.*=',
            r'^.*기자.*특파원.*=.*=',
            r'^.*기자.*인턴.*=.*=',
            r'^.*기자.*=.*=.*=.*기자',
            r'^.*기자.*=.*=.*=.*특파원',
            r'^.*기자.*=.*=.*=.*인턴',
            r'^.*기자.*=.*=.*=.*=',
            r'^.*기자.*기자.*=.*=.*=',
            r'^.*기자.*특파원.*=.*=.*=',
            r'^.*기자.*인턴.*=.*=.*=',
            r'^.*기자.*=.*=.*=.*=.*기자',
            r'^.*기자.*=.*=.*=.*=.*특파원',
            r'^.*기자.*=.*=.*=.*=.*인턴',
            r'^.*기자.*=.*=.*=.*=.*=',
            r'^.*기자.*기자.*=.*=.*=.*=',
            r'^.*기자.*특파원.*=.*=.*=.*=',
            r'^.*기자.*인턴.*=.*=.*=.*=',
        ]
        
        # 제거할 키워드들 (UI 요소 및 불필요한 내용)
        skip_keywords = [
            '본문 요약',
            '현재위치',
            '지자체',
            '기자명',
            '입력',
            '바로가기',
            '복사하기',
            '본문 글씨',
            '글씨 줄이기',
            '글씨 키우기',
            'SNS',
            '페이스북',
            '트위터',
            'URL복사',
            '기사보내기',
            '공유하기',
            '관련 기사',
            '관련기사',
            '관련사진',
            '관련사진보기',
            '관련기사보기',
            '관련영상',
            '관련영상보기',
            '추천 기사',
            '추천기사',
            '추천기사보기',
            '댓글',
            '댓글보기',
            '좋아요',
            '더보기',
            '전체보기',
            '기자 =',
            '기자=',
            '특파원 =',
            '특파원=',
            '인턴기자 =',
            '인턴기자=',
            'Copyright',
            '©',
            '무단 전재',
            '재배포 금지',
            '기사 제보',
            '이 기사',
            '기사 내용',
            '클릭',
            '보기',
            '더 읽기',
            '전체 읽기',
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 해시태그가 포함된 줄 제거
            if re.search(r'#\S+', line):
                continue
            
            # 사진 = 연합뉴스 같은 패턴이 포함된 줄 제거
            if re.search(r'사진\s*[=:]\s*[가-힣a-zA-Z\s]+', line, re.IGNORECASE):
                continue
            if re.search(r'그림\s*[=:]\s*[가-힣a-zA-Z\s]+', line, re.IGNORECASE):
                continue
            if re.search(r'표\s*[=:]\s*[가-힣a-zA-Z\s]+', line, re.IGNORECASE):
                continue
            
            # 패턴 체크
            skip = False
            for pattern in skip_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    skip = True
                    break
            
            # 키워드 체크
            if not skip:
                for keyword in skip_keywords:
                    if keyword in line:
                        skip = True
                        break
            
            # 기자 정보가 포함된 줄 제거 (기자 이름 패턴)
            if not skip and re.search(r'[가-힣]+\s*[=:]?\s*[가-힣]*\s*기자', line):
                # 하지만 본문에 "기자"라는 단어가 포함된 경우는 제외
                if not re.search(r'기자.*(?:말|보고|전망|분석|설명|밝혀|발표)', line):
                    skip = True
            
            # 이메일 주소가 포함된 줄 제거
            if not skip and re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', line):
                skip = True
            
            # URL이 포함된 줄 제거 (본문이 아닌 링크)
            if not skip and re.search(r'https?://[^\s]+', line) and len(line) < 100:
                skip = True
            
            # "보기", "클릭" 같은 UI 동사로 끝나는 줄 제거
            if not skip and re.search(r'(보기|클릭|읽기|확인|이동|더보기|전체보기|관련보기)$', line, re.IGNORECASE):
                skip = True
            
            # "관련사진보기", "관련기사보기" 같은 UI 요소 제거
            if not skip and re.search(r'관련(사진|기사|영상|뉴스|기사)보기', line, re.IGNORECASE):
                skip = True
            
            # 너무 짧은 줄 제거 (광고나 버튼 텍스트일 가능성)
            if not skip and len(line) < 10 and not re.search(r'[가-힣]{3,}', line):
                skip = True
            
            # 본문이 아닌 UI 요소 패턴 제거 (예: "관련사진보기", "기사 더보기" 등)
            if not skip and re.search(r'^(관련|추천|더|전체|기사|뉴스|사진|영상).*(보기|클릭|읽기|확인|이동)', line, re.IGNORECASE):
                skip = True
            
            # [사진=...], [그림=...], [표=...] 같은 캡션 제거
            if not skip and re.match(r'^\[(사진|그림|표|캡션|포토|이미지)[=:].*\]', line, re.IGNORECASE):
                skip = True
            
            # 대괄호로 둘러싸인 짧은 텍스트 제거 (캡션일 가능성)
            if not skip and re.match(r'^\[.*\]$', line) and len(line) < 50:
                skip = True
            
            # [뉴시스], [연합뉴스] 같은 출처 표시가 포함된 줄 제거 (캡션일 가능성)
            if not skip and re.search(r'^\[(뉴시스|연합뉴스|조선일보|중앙일보|동아일보|한겨레|경향신문|매일경제|한국경제|서울신문|세계일보|문화일보|국민일보|내일신문|헤럴드경제|아시아경제|이데일리|뉴스1|YTN|SBS|KBS|MBC|JTBC|채널A|TV조선|MBN|기자협회|AP|AFP|로이터|로이터통신|Reuters|AP통신)\]\s*', line, re.IGNORECASE):
                skip = True
            
            # (사진=...), (그림=...), (표=...) 같은 캡션 패턴 제거
            if not skip and re.search(r'\(사진\s*[=:]\s*[^)]+\)', line, re.IGNORECASE):
                skip = True
            if not skip and re.search(r'\(그림\s*[=:]\s*[^)]+\)', line, re.IGNORECASE):
                skip = True
            if not skip and re.search(r'\(표\s*[=:]\s*[^)]+\)', line, re.IGNORECASE):
                skip = True
            
            # 출처 + 제목 + (사진=...) 형태의 줄 제거
            # 예: "[뉴시스] 태국에서 체포된 한국인 보이스피싱 조직원들. (사진=더네이션)"
            if not skip and re.search(r'\[.*\]\s*.*\(사진\s*[=:]\s*[^)]+\)', line, re.IGNORECASE):
                skip = True
            
            # /사진 제공=, 사진 제공=, /제공= 패턴 제거
            if not skip and re.search(r'[/]?\s*사진\s*제공\s*[=:]', line, re.IGNORECASE):
                skip = True
            
            # 인물 이름 + 직책 + /사진 제공= 패턴 제거 (예: "젠슨 황 엔비디아 CEO /사진 제공=엔비디아")
            if not skip and re.search(r'[/]?\s*사진\s*제공\s*[=:]', line, re.IGNORECASE):
                # 이 줄 전체를 제거
                skip = True
            
            # 제공=, /제공= 패턴이 포함된 줄 제거
            if not skip and re.search(r'[/]?\s*제공\s*[=:]', line, re.IGNORECASE) and len(line) < 100:
                skip = True
            
            # 인물 이름만 있는 짧은 줄 (캡션일 가능성)
            # 예: "젠슨 황 엔비디아 CEO" 같은 패턴
            if not skip and len(line) < 50:
                # CEO, 대표, 회장 등 직책만 있는 줄 제거
                if re.search(r'\b(CEO|대표|회장|사장|이사|부장|차장|과장|팀장|실장|본부장|그룹장|총괄|책임|담당)\b', line) and not re.search(r'[가-힣]{10,}', line):
                    skip = True
            
            if not skip:
                cleaned_lines.append(line)
        
        # 정제된 텍스트 합치기
        cleaned_text = '\n'.join(cleaned_lines)
        
        # [사진=...], [그림=...] 같은 패턴을 텍스트 내에서도 제거
        cleaned_text = re.sub(r'\[(사진|그림|표|캡션|포토|이미지)[=:][^\]]*\]', '', cleaned_text, flags=re.IGNORECASE)
        
        # (사진=...), (그림=...), (표=...) 같은 캡션 패턴 제거
        cleaned_text = re.sub(r'\(사진\s*[=:]\s*[^)]+\)', '', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r'\(그림\s*[=:]\s*[^)]+\)', '', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r'\(표\s*[=:]\s*[^)]+\)', '', cleaned_text, flags=re.IGNORECASE)
        
        # "관련사진보기", "관련기사보기" 같은 UI 요소 제거
        cleaned_text = re.sub(r'관련(사진|기사|영상|뉴스)보기', '', cleaned_text, flags=re.IGNORECASE)
        
        # "보기", "클릭" 같은 UI 동사로 끝나는 줄 제거
        cleaned_text = re.sub(r'^.*(보기|클릭|읽기|확인|이동|더보기|전체보기|관련보기)$', '', cleaned_text, flags=re.IGNORECASE | re.MULTILINE)
        
        # [뉴시스], [연합뉴스] 같은 출처 표시 제거
        cleaned_text = re.sub(r'\[(뉴시스|연합뉴스|조선일보|중앙일보|동아일보|한겨레|경향신문|매일경제|한국경제|서울신문|세계일보|문화일보|국민일보|내일신문|헤럴드경제|아시아경제|이데일리|뉴스1|YTN|SBS|KBS|MBC|JTBC|채널A|TV조선|MBN|기자협회|AP|AFP|로이터|로이터통신|Reuters|AP통신)\]\s*', '', cleaned_text, flags=re.IGNORECASE)
        
        # 출처 + 제목 + (사진=...) 형태 제거
        cleaned_text = re.sub(r'\[.*\]\s*.*\(사진\s*[=:]\s*[^)]+\)', '', cleaned_text, flags=re.IGNORECASE)
        
        # /사진 제공=, 사진 제공= 패턴 제거
        cleaned_text = re.sub(r'[/]?\s*사진\s*제공\s*[=:][^\n]*', '', cleaned_text, flags=re.IGNORECASE)
        
        # /제공= 패턴 제거 (예: /광주광역시 제공, /엔비디아 제공 등)
        cleaned_text = re.sub(r'[/]\s*[가-힣a-zA-Z\s]+\s*제공', '', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r'[/]?\s*제공\s*[=:][^\n]*', '', cleaned_text, flags=re.IGNORECASE)
        
        # 인물 이름 + 직책 + /사진 제공= 같은 패턴 제거
        # 예: "젠슨 황 엔비디아 CEO /사진 제공=엔비디아" -> "젠슨 황 엔비디아 CEO" 부분도 제거
        cleaned_text = re.sub(r'[가-힣a-zA-Z\s]+\s+(CEO|대표|회장|사장|이사|부장|차장|과장|팀장|실장|본부장|그룹장|총괄|책임|담당)\s*[/]?\s*사진\s*제공\s*[=:][^\n]*', '', cleaned_text, flags=re.IGNORECASE)
        
        # 조감도, 사진 등의 설명 + /... 제공 패턴 제거
        # 예: "광주 운전면허시험장 조성사업 조감도. /광주광역시 제공"
        cleaned_text = re.sub(r'[가-힣\s]+(조감도|사진|그림|표|이미지)[\.]?\s*[/]\s*[가-힣a-zA-Z\s]+\s*제공', '', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r'[가-힣\s]+(조감도|사진|그림|표|이미지)[\.]?\s*[/]\s*[가-힣a-zA-Z\s]+', '', cleaned_text, flags=re.IGNORECASE)
        
        # 연속된 공백 정리 (과도한 줄바꿈 방지)
        cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)  # 3개 이상 연속된 줄바꿈을 2개로 제한
        cleaned_text = re.sub(r'\n\s*\n+', '\n', cleaned_text)  # 2개 연속된 줄바꿈을 1개로
        cleaned_text = re.sub(r' +', ' ', cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        # 본문 시작 부분 찾기 (제목이나 소개 부분 제거)
        # "기자 =", "특파원 =", "인턴기자 =" 같은 패턴이 나오기 전까지만
        lines_after_clean = cleaned_text.split('\n')
        main_content_start = 0
        
        for i, line in enumerate(lines_after_clean):
            # 기자 정보가 나오면 그 전까지가 본문
            if re.search(r'[가-힣]+\s*[=:]?\s*[가-힣]*\s*기자', line) and len(line) < 50:
                main_content_start = i
                break
        
        # 본문만 추출 (기자 정보 이전까지)
        if main_content_start > 0:
            cleaned_text = '\n'.join(lines_after_clean[:main_content_start])
        
        return cleaned_text.strip()
    
    def _clean_summary(self, summary: str) -> str:
        """
        요약 결과에서 불필요한 내용을 제거합니다.
        (캡션, 기자 정보, 제공 정보, 해시태그 등)
        """
        if not summary:
            return summary
        
        # 해시태그 제거 (#으로 시작하는 단어들)
        summary = re.sub(r'#\S+', '', summary)
        
        # 사진 = 연합뉴스 같은 패턴 제거 (괄호 없이)
        summary = re.sub(r'사진\s*[=:]\s*[가-힣a-zA-Z\s]+', '', summary, flags=re.IGNORECASE)
        summary = re.sub(r'그림\s*[=:]\s*[가-힣a-zA-Z\s]+', '', summary, flags=re.IGNORECASE)
        summary = re.sub(r'표\s*[=:]\s*[가-힣a-zA-Z\s]+', '', summary, flags=re.IGNORECASE)
        
        # [뉴시스], [연합뉴스] 같은 출처 표시 제거
        summary = re.sub(r'\[(뉴시스|연합뉴스|조선일보|중앙일보|동아일보|한겨레|경향신문|매일경제|한국경제|서울신문|세계일보|문화일보|국민일보|내일신문|헤럴드경제|아시아경제|이데일리|뉴스1|YTN|SBS|KBS|MBC|JTBC|채널A|TV조선|MBN|기자협회|AP|AFP|로이터|로이터통신|Reuters|AP통신)\]\s*', '', summary, flags=re.IGNORECASE)
        
        # (사진=...), (그림=...), (표=...) 같은 캡션 패턴 제거
        summary = re.sub(r'\(사진\s*[=:]\s*[^)]+\)', '', summary, flags=re.IGNORECASE)
        summary = re.sub(r'\(그림\s*[=:]\s*[^)]+\)', '', summary, flags=re.IGNORECASE)
        summary = re.sub(r'\(표\s*[=:]\s*[^)]+\)', '', summary, flags=re.IGNORECASE)
        
        # 출처 + 제목 + (사진=...) 형태 제거
        summary = re.sub(r'\[.*\]\s*.*\(사진\s*[=:]\s*[^)]+\)', '', summary, flags=re.IGNORECASE)
        
        # /... 제공 패턴 제거 (예: /광주광역시 제공)
        summary = re.sub(r'[/]\s*[가-힣a-zA-Z\s]+\s*제공', '', summary, flags=re.IGNORECASE)
        
        # /사진 제공=, 사진 제공= 패턴 제거
        summary = re.sub(r'[/]?\s*사진\s*제공\s*[=:][^\n]*', '', summary, flags=re.IGNORECASE)
        
        # /제공= 패턴 제거
        summary = re.sub(r'[/]?\s*제공\s*[=:][^\n]*', '', summary, flags=re.IGNORECASE)
        
        # 조감도, 사진 등의 설명 + /... 제공 패턴 제거
        summary = re.sub(r'[가-힣\s]+(조감도|사진|그림|표|이미지)[\.]?\s*[/]\s*[가-힣a-zA-Z\s]+\s*제공', '', summary, flags=re.IGNORECASE)
        summary = re.sub(r'[가-힣\s]+(조감도|사진|그림|표|이미지)[\.]?\s*[/]\s*[가-힣a-zA-Z\s]+', '', summary, flags=re.IGNORECASE)
        
        # [사진=...], [그림=...] 패턴 제거
        summary = re.sub(r'\[(사진|그림|표|캡션|포토|이미지)[=:][^\]]*\]', '', summary, flags=re.IGNORECASE)
        
        # 인물 이름 + 직책 + /사진 제공= 패턴 제거
        summary = re.sub(r'[가-힣a-zA-Z\s]+\s+(CEO|대표|회장|사장|이사|부장|차장|과장|팀장|실장|본부장|그룹장|총괄|책임|담당)\s*[/]?\s*사진\s*제공\s*[=:][^\n]*', '', summary, flags=re.IGNORECASE)
        
        # 기자 정보 제거
        summary = re.sub(r'[가-힣]+\s*[=:]?\s*[가-힣]*\s*기자\s*[=:]', '', summary)
        
        # 연속된 공백 정리
        summary = re.sub(r'\s+', ' ', summary)
        # 과도한 줄바꿈 제거 (연속된 줄바꿈을 공백으로)
        summary = re.sub(r'\n\s*\n+', ' ', summary)
        summary = summary.strip()
        
        # 빈 요약이나 의미 없는 요약 제거
        if len(summary) < 20:
            return ''
        
        return summary
    
    def _fallback_summarize(self, text: str, max_length: int = 300) -> str:
        """
        OpenAI API를 사용할 수 없을 때 사용하는 기본 요약 방식
        완전한 문장으로 끝나도록 처리합니다.
        """
        text = text.strip()
        
        if len(text) <= max_length:
            return text
        
        # 최대 길이까지 자르기
        summary = text[:max_length]
        
        # 완전한 문장으로 끝나도록 처리
        if len(text) > max_length:
            # 문장 부호 찾기 (우선순위: 마침표, 느낌표, 물음표)
            last_punct = max(
                summary.rfind('.'),
                summary.rfind('!'),
                summary.rfind('?'),
                summary.rfind('。'),
                summary.rfind('！'),
                summary.rfind('？')
            )
            
            # 문장 부호가 있으면 그 앞에서 자르기 (최소 50% 이상 위치)
            if last_punct > max_length * 0.5:
                summary = summary[:last_punct + 1]
            else:
                # 문장 부호가 없으면 공백 앞에서 자르기
                last_space = summary.rfind(' ')
                if last_space > max_length * 0.7:  # 70% 이상 위치에 공백이 있으면
                    summary = summary[:last_space]
                    # 공백으로 끝나면 마지막 문장 부호 찾기
                    last_punct_in_summary = max(
                        summary.rfind('.'),
                        summary.rfind('!'),
                        summary.rfind('?'),
                        summary.rfind('。'),
                        summary.rfind('！'),
                        summary.rfind('？')
                    )
                    if last_punct_in_summary > len(summary) * 0.5:
                        summary = summary[:last_punct_in_summary + 1]
                    else:
                        summary += '...'
                else:
                    # 공백도 없으면 그냥 자르고 ... 추가
                    summary += '...'
        
        return summary
    
    def extract_title_from_link(self, link: str) -> Optional[str]:
        """
        링크에서 제목을 추출합니다.
        
        Args:
            link: 네이버 뉴스 기사 링크
            
        Returns:
            기사 제목 또는 None
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
            
            response = requests.get(link, headers=headers, timeout=10, allow_redirects=True)
            response.raise_for_status()
            
            # 인코딩 자동 감지
            if response.encoding is None or response.encoding == 'ISO-8859-1':
                response.encoding = response.apparent_encoding or 'utf-8'
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 제목 선택자 (우선순위 순)
            title_selectors = [
                'meta[property="og:title"]',
                'meta[name="twitter:title"]',
                'title',
                'h1.media_end_head_headline',
                'h1.end_tit',
                '.article_info h3',
                '.article-header h1',
                'h1.article-title',
                'h1',
            ]
            
            for selector in title_selectors:
                try:
                    if selector.startswith('meta'):
                        elem = soup.select_one(selector)
                        if elem and elem.get('content'):
                            title = elem.get('content').strip()
                            if title:
                                return title
                    else:
                        elem = soup.select_one(selector)
                        if elem:
                            title = elem.get_text(strip=True)
                            if title:
                                return title
                except:
                    continue
            
            return None
        except Exception as e:
            print(f"제목 추출 실패 ({link}): {e}")
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
            
            # newspaper3k는 일부 사이트에서 403 에러가 발생하므로 사용하지 않음
            # BeautifulSoup으로 직접 추출
            
            # BeautifulSoup으로 직접 추출 (newspaper3k 실패 시)
            # 네이버 뉴스 및 주요 언론사 특화 선택자 (우선순위 순)
            selectors = [
                # 네이버 뉴스
                '#newsct_article',
                '#newsEndContents',
                '.news_end_body_body',
                '._article_body_contents',
                '#articleBodyContents',
                # 주요 언론사
                '#article-view-content-div',
                '.article-view-content',
                '.article-body',
                '.article_content',
                '#article_content',
                '.article-body-content',
                '#article-body',
                '.article-body-text',
                # 일반적인 선택자
                'article .content',
                'article .body',
                '[id*="article"][id*="body"]',
                '[id*="article"][id*="content"]',
                '[class*="article"][class*="body"]',
                '[class*="article"][class*="content"]',
                'article',
                # 마지막 수단
                '[id*="article"]',
                '[class*="article-body"]',
                '[class*="article-content"]',
            ]
            
            content_elem = None
            for selector in selectors:
                try:
                    if selector.startswith('['):
                        # 속성 선택자 처리
                        if 'id*=' in selector and 'body' in selector:
                            content_elem = soup.find('div', {'id': lambda x: x and 'article' in x.lower() and 'body' in x.lower()})
                        elif 'id*=' in selector and 'content' in selector:
                            content_elem = soup.find('div', {'id': lambda x: x and 'article' in x.lower() and 'content' in x.lower()})
                        elif 'id*=' in selector:
                            content_elem = soup.find('div', {'id': lambda x: x and 'article' in x.lower()})
                        elif 'class*=' in selector and 'body' in selector:
                            content_elem = soup.find('div', {'class': lambda x: x and ('article' in str(x).lower() and 'body' in str(x).lower())})
                        elif 'class*=' in selector and 'content' in selector:
                            content_elem = soup.find('div', {'class': lambda x: x and ('article' in str(x).lower() and 'content' in str(x).lower())})
                        elif 'class*=' in selector:
                            content_elem = soup.find('div', {'class': lambda x: x and ('article' in str(x).lower() or 'body' in str(x).lower())})
                    else:
                        content_elem = soup.select_one(selector)
                    
                    if content_elem:
                        # 본문으로 보이는지 확인 (최소 길이 체크)
                        text_preview = content_elem.get_text(strip=True)
                        if text_preview and len(text_preview) > 200:  # 최소 200자 이상
                            break
                        else:
                            content_elem = None
                except:
                    continue
            
            if content_elem:
                # 불필요한 태그 제거 (스크립트, 스타일 등)
                for tag in content_elem.find_all(['script', 'style', 'iframe', 'noscript', 'svg', 'nav', 'header', 'footer']):
                    try:
                        tag.decompose()
                    except (AttributeError, TypeError):
                        continue
                
                # UI 요소 제거 (버튼, 링크, 메뉴 등)
                ui_keywords = [
                    'button', 'btn', 'menu', 'nav', 'header', 'footer', 
                    'sidebar', 'aside', 'toolbar', 'tool-bar',
                    'share', 'sns', 'social', 'comment', 'reply',
                    'ad', 'advertisement', 'sponsor', 'promo', 'banner',
                    'related', 'recommend', 'recommended', 'more', 'more-news',
                    'current', 'location', 'breadcrumb', 'bread-crumb',
                    'font-size', 'font-size-control', 'text-size',
                    'copy', 'url-copy', 'clipboard',
                    'print', 'email', 'facebook', 'twitter', 'kakao'
                ]
                
                # UI 요소가 포함된 태그 제거
                for tag in content_elem.find_all(['div', 'section', 'aside', 'span', 'a', 'button']):
                    try:
                        classes = tag.get('class', [])
                        tag_id = tag.get('id', '')
                        tag_text = tag.get_text(strip=True)
                        
                        if classes is None:
                            classes = []
                        elif not isinstance(classes, list):
                            classes = [classes] if classes else []
                        
                        if tag_id is None:
                            tag_id = ''
                        
                        classes_str = ' '.join(classes).lower() if classes else ''
                        tag_id_str = str(tag_id).lower()
                        tag_text_lower = tag_text.lower()
                        
                        # UI 키워드가 포함된 태그 제거
                        should_remove = False
                        
                        # 클래스나 ID에 UI 키워드가 있는지 확인
                        for keyword in ui_keywords:
                            if keyword in classes_str or keyword in tag_id_str:
                                should_remove = True
                                break
                        
                        # 텍스트 내용이 UI 요소인지 확인 (짧고 UI 관련 키워드 포함)
                        if not should_remove and len(tag_text) < 50:
                            ui_text_patterns = [
                                '현재위치', '지자체', '기자명', '입력', '바로가기', 
                                '복사하기', '본문 글씨', 'SNS', '페이스북', '트위터',
                                'URL복사', '기사보내기', '공유하기', '댓글', '좋아요',
                                '관련기사', '관련사진', '관련사진보기', '관련기사보기',
                                '관련영상', '관련영상보기', '추천기사', '추천기사보기',
                                '댓글보기', '더보기', '전체보기', '더 읽기', '전체 읽기',
                                '보기', '클릭', '확인', '이동'
                            ]
                            for pattern in ui_text_patterns:
                                if pattern in tag_text:
                                    should_remove = True
                                    break
                        
                        # "관련사진보기", "관련기사보기" 같은 패턴 제거
                        if not should_remove and re.search(r'관련(사진|기사|영상|뉴스)보기', tag_text, re.IGNORECASE):
                            should_remove = True
                        
                        # "보기", "클릭" 같은 UI 동사로 끝나는 짧은 텍스트 제거
                        if not should_remove and len(tag_text) < 30 and re.search(r'(보기|클릭|읽기|확인|이동|더보기|전체보기|관련보기)$', tag_text, re.IGNORECASE):
                            should_remove = True
                        
                        # 날짜 패턴만 있는 짧은 텍스트 (예: "2025.12.11 01:51")
                        if not should_remove and len(tag_text) < 30 and re.match(r'^\d{4}\.\d{2}\.\d{2}', tag_text):
                            should_remove = True
                        
                        if should_remove:
                            tag.decompose()
                    except (AttributeError, TypeError):
                        continue
                
                # 본문 텍스트 추출
                text = content_elem.get_text(separator='\n', strip=True)
                
                # 기사 본문만 추출하도록 정제
                text = self._clean_article_text(text)
                
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
                'view_count': int (조회수, 항상 추출),
                'text': str (본문, include_full_text=True일 때만)
            } 형태의 딕셔너리 리스트
        """
        # 본문 추출 실패 시 대비하여 더 많은 기사를 수집 (최대 2배까지)
        # 실패한 기사를 제외하고도 충분한 수를 확보하기 위함
        items_to_fetch = max_results * 2 if include_full_text else max_results
        
        # sort_by를 sort로 변환 ('date' -> 'date', 'view' -> 'sim' 또는 'date')
        sort_param = 'date' if sort_by == 'date' else 'sim'
        
        items = self.get_all_news(
            query=query,
            max_results=items_to_fetch,
            date_from=date_from,
            date_to=date_to,
            sort=sort_param
        )
        
        results = []
        for item in items:
            # 날짜를 한국어 형식으로 변환
            pub_date = item.get('pubDate', '')
            pub_date_korean = self._format_date_korean(pub_date)
            
            # 제목에서 해시태그 제거
            title = item.get('title', '').replace('<b>', '').replace('</b>', '').strip()
            # 해시태그 제거 (#으로 시작하는 단어들)
            title = re.sub(r'#\S+', '', title).strip()
            # 연속된 공백 정리
            title = re.sub(r'\s+', ' ', title).strip()
            
            result = {
                'title': title,
                'link': item.get('link', ''),
                'description': item.get('description', '').replace('<b>', '').replace('</b>', '').strip(),
                'pubDate': pub_date_korean,  # 한국어 형식으로 변환된 날짜
                'originallink': item.get('originallink', ''),
                'source': self._extract_source_from_link(item.get('originallink', '')),
            }
            
            # 제목이 "..."으로 끝나면 원본 링크에서 제목을 다시 가져오기
            if title.endswith('...') or title.endswith('…'):
                link_to_use = result['originallink'] or result['link']
                if link_to_use:
                    full_title = self.extract_title_from_link(link_to_use)
                    if full_title:
                        # 해시태그 제거 및 정리
                        full_title = full_title.replace('<b>', '').replace('</b>', '').strip()
                        full_title = re.sub(r'#\S+', '', full_title).strip()
                        full_title = re.sub(r'\s+', ' ', full_title).strip()
                        result['title'] = full_title
            
            # 조회수 추출 (항상 추출)
            link_to_use = result['originallink'] or result['link']
            view_count = self.extract_view_count(link_to_use)
            result['view_count'] = view_count if view_count is not None else 0
            if view_count is not None:
                time.sleep(self.delay)  # 조회수 추출 시 추가 대기
            
            if include_full_text:
                # 원본 링크가 있으면 원본 링크 사용, 없으면 네이버 링크 사용
                link_to_use = result['originallink'] or result['link']
                full_text = self.extract_full_text(link_to_use)
                
                if full_text:
                    # 전체 본문 저장 (감정 분석용)
                    result['full_text'] = full_text
                    # 본문을 요약하여 저장 (3줄 요약, 화면 표시용)
                    result['text'] = self.summarize_text(full_text)
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
                    result['text'] = self.summarize_text(description)
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
    
    def _format_date_korean(self, date_str: str) -> str:
        """
        날짜 문자열을 한국어 형식으로 변환합니다.
        예: "Thu, 11 Dec 2025 00:23:00 +0900" -> "2025년 12월 11일 00:23"
        """
        if not date_str:
            return "알 수 없음"
        
        try:
            # 여러 날짜 형식 시도
            date_formats = [
                "%a, %d %b %Y %H:%M:%S %z",  # "Thu, 11 Dec 2025 00:23:00 +0900"
                "%a, %d %b %Y %H:%M:%S",     # "Thu, 11 Dec 2025 00:23:00"
                "%Y-%m-%d %H:%M:%S",         # "2025-12-11 00:23:00"
                "%Y-%m-%d",                  # "2025-12-11"
            ]
            
            parsed_date = None
            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(date_str.strip(), fmt)
                    break
                except ValueError:
                    continue
            
            if parsed_date is None:
                # 마지막 시도: 공백 제거 후 다시 시도
                date_str_clean = date_str.strip()
                # "+0900" 같은 타임존 제거 후 시도
                if '+' in date_str_clean or date_str_clean.endswith('00'):
                    date_str_clean = re.sub(r'\s*[+-]\d{4}$', '', date_str_clean)
                    try:
                        parsed_date = datetime.strptime(date_str_clean, "%a, %d %b %Y %H:%M:%S")
                    except ValueError:
                        pass
            
            if parsed_date is None:
                return date_str
            
            # 한국어 형식으로 변환: "2025년 12월 11일 00:23"
            formatted = parsed_date.strftime("%Y년 %m월 %d일 %H:%M")
            
            # 월 앞의 0 제거 (예: "01월" -> "1월")
            formatted = re.sub(r'0(\d)월', r'\1월', formatted)
            # 일 앞의 0 제거 (예: "01일" -> "1일")
            formatted = re.sub(r'0(\d)일', r'\1일', formatted)
            
            return formatted
        except Exception as e:
            # 파싱 실패 시 원본 반환
            print(f"날짜 파싱 오류: {e}, 원본: {date_str}")
            return date_str
    
    def _is_english_article(self, title: str, description: str = '', text: str = '') -> bool:
        """
        기사가 영어로 작성되었는지 판단합니다.
        
        Args:
            title: 기사 제목
            description: 기사 설명
            text: 기사 본문
            
        Returns:
            영어 기사면 True, 아니면 False
        """
        # 제목, 설명, 본문을 합쳐서 분석
        content = f"{title} {description} {text}".strip()
        
        if not content:
            return False
        
        # 한글이 포함되어 있으면 영어 기사가 아님
        if re.search(r'[가-힣]', content):
            return False
        
        # 영어 문자 비율 계산
        total_chars = len(re.findall(r'[a-zA-Z]', content))
        total_words = len(content.split())
        
        if total_words == 0:
            return False
        
        # 영어 비율이 70% 이상이면 영어 기사로 판단
        english_ratio = total_chars / len(content.replace(' ', '')) if len(content.replace(' ', '')) > 0 else 0
        
        # 제목만 영어이고 설명/본문이 없으면 영어 기사로 판단
        if english_ratio > 0.7 and not re.search(r'[가-힣]', content):
            return True
        
        return False
    
    def get_recent_news(
        self,
        query: str,
        days: int = 1,
        max_results: int = 100,
        sort_by: str = 'date',
        exclude_english: bool = False
    ) -> List[Dict]:
        """
        최근 N일간의 뉴스를 가져옵니다.
        
        Args:
            query: 검색어
            days: 최근 며칠간 (기본 1일)
            max_results: 최대 수집할 기사 수
            sort_by: 정렬 기준 ('date': 날짜순, 'view': 조회수순)
            exclude_english: 영어 뉴스 제외 여부
            
        Returns:
            기사 정보 리스트
        """
        date_to = datetime.now().strftime('%Y%m%d')
        date_from = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
        
        results = self.crawl_news_with_full_text(
            query=query,
            max_results=max_results * 2 if exclude_english else max_results,  # 영어 제외 시 더 많이 수집
            include_full_text=True,
            date_from=date_from,
            date_to=date_to,
            sort_by=sort_by
        )
        
        # 영어 뉴스 제외
        if exclude_english:
            filtered_results = []
            for result in results:
                title = result.get('title', '')
                description = result.get('description', '')
                text = result.get('text', '')
                
                if not self._is_english_article(title, description, text):
                    filtered_results.append(result)
                    
                    # 필요한 수만큼 수집했으면 중단
                    if len(filtered_results) >= max_results:
                        break
            
            return filtered_results
        
        return results


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
        #print(f"본문 길이: {len(result.get('text', ''))}자")
        if result.get('text'):
            print(f"본문 미리보기: {result['text'][:200]}...")
        print(f"링크: {result['link']}")
        print()

