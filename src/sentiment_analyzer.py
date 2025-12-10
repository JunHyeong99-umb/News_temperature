"""
한글 감정 분석 모듈
Hugging Face transformers를 사용한 감정 분석
OpenAI API를 사용한 감정 분석도 지원
"""
import os
import re
import json
from typing import Dict, Optional
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("경고: Pillow가 설치되지 않았습니다. pip install Pillow를 실행하세요.")

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    import torch.nn.functional as F
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("경고: transformers가 설치되지 않았습니다. pip install transformers를 실행하세요.")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("경고: openai가 설치되지 않았습니다. pip install openai를 실행하세요.")

# GPU 사용 가능 여부 확인
def get_device():
    """사용 가능한 디바이스를 반환합니다 (GPU 우선)"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU 사용 가능: {torch.cuda.get_device_name(0)}")
            return 0  # GPU 사용
        else:
            print("GPU를 사용할 수 없습니다. CPU를 사용합니다.")
            return -1  # CPU 사용
    except ImportError:
        print("PyTorch가 설치되지 않았습니다. CPU를 사용합니다.")
        return -1  # CPU 사용


class SentimentAnalyzer:
    """한글 감정 분석 클래스"""
    
    def __init__(self, openai_api_key: Optional[str] = None, use_openai: bool = False):
        """
        감정 분석 파이프라인 초기화
        
        Args:
            openai_api_key: OpenAI API 키 (OpenAI API 사용 시 필요)
            use_openai: OpenAI API 사용 여부 (True면 OpenAI API 사용, False면 로컬 모델 사용)
        """
        self.classifier = None
        self.model = None
        self.tokenizer = None
        self.use_finetuned_model = False
        self.device = None
        self.openai_api_key = openai_api_key
        self.use_openai = use_openai
        self.openai_client = None
        
        # OpenAI 클라이언트 초기화
        if use_openai and openai_api_key and OPENAI_AVAILABLE:
            try:
                self.openai_client = OpenAI(api_key=openai_api_key)
                print("✅ OpenAI API 클라이언트 초기화 완료")
            except Exception as e:
                print(f"❌ OpenAI API 클라이언트 초기화 실패: {e}")
                self.use_openai = False
        elif use_openai and not OPENAI_AVAILABLE:
            print("경고: openai 패키지가 설치되지 않았습니다. 로컬 모델을 사용합니다.")
            self.use_openai = False
        
        if TRANSFORMERS_AVAILABLE:
            # GPU 사용 가능 여부 확인
            device_id = get_device()
            self.device = "cuda" if device_id >= 0 else "cpu"
            
            # 파인튜닝된 모델이 있으면 우선 사용
            if os.path.exists("./sentiment_model") and os.path.isdir("./sentiment_model"):
                try:
                    print("파인튜닝된 뉴스 감정 분석 모델 로드 시도 중...")
                    print(f"디바이스: {'GPU (CUDA)' if device_id >= 0 else 'CPU'}")
                    
                    # 모델과 토크나이저 직접 로드
                    self.tokenizer = AutoTokenizer.from_pretrained("./sentiment_model")
                    self.model = AutoModelForSequenceClassification.from_pretrained("./sentiment_model")
                    
                    # GPU로 이동
                    if device_id >= 0:
                        self.model = self.model.to(self.device)
                        if torch.cuda.is_available():
                            print(f"✅ 파인튜닝된 모델 로드 완료 (GPU: {torch.cuda.get_device_name(0)} 사용 중)")
                        else:
                            print("✅ 파인튜닝된 모델 로드 완료 (CPU 사용 중)")
                    else:
                        print("✅ 파인튜닝된 모델 로드 완료 (CPU 사용 중)")
                    
                    self.model.eval()  # 평가 모드로 설정
                    self.use_finetuned_model = True
                    
                except Exception as e:
                    print(f"❌ 파인튜닝된 모델 로드 실패: {e}")
                    import traceback
                    traceback.print_exc()
                    self.model = None
                    self.tokenizer = None
            
            # 파인튜닝된 모델이 없거나 실패한 경우 pipeline 사용
            if not self.use_finetuned_model:
                # 한글 감정 분석 모델 시도 (우선순위 순)
                models_to_try = [
                    ("matthewburke/korean_sentiment", "한국어 감정 분석 모델"),
                    ("nlptown/bert-base-multilingual-uncased-sentiment", "다국어 감정 분석 모델"),
                ]
                
                for model_name, model_desc in models_to_try:
                    try:
                        print(f"{model_desc} 로드 시도 중... ({model_name})")
                        print(f"디바이스: {'GPU (CUDA)' if device_id >= 0 else 'CPU'}")
                        self.classifier = pipeline(
                            "sentiment-analysis",
                            model=model_name,
                            device=device_id,
                            torch_dtype="auto" if device_id >= 0 else None
                        )
                        # GPU 사용 확인
                        if device_id >= 0:
                            if torch.cuda.is_available():
                                print(f"✅ {model_desc} 로드 완료 (GPU: {torch.cuda.get_device_name(0)} 사용 중)")
                            else:
                                print(f"✅ {model_desc} 로드 완료 (CPU 사용 중)")
                        else:
                            print(f"✅ {model_desc} 로드 완료 (CPU 사용 중)")
                        break
                    except Exception as e:
                        print(f"❌ {model_desc} 로드 실패: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                # 모든 모델 실패 시 기본 모델 시도
                if self.classifier is None:
                    try:
                        print("기본 감정 분석 모델 로드 시도 중...")
                        print(f"디바이스: {'GPU (CUDA)' if device_id >= 0 else 'CPU'}")
                        self.classifier = pipeline(
                            "sentiment-analysis", 
                            device=device_id,
                            torch_dtype="auto" if device_id >= 0 else None
                        )
                        if device_id >= 0:
                            if torch.cuda.is_available():
                                print(f"기본 감정 분석 모델 로드 완료 (GPU: {torch.cuda.get_device_name(0)} 사용 중)")
                            else:
                                print("기본 감정 분석 모델 로드 완료 (CPU 사용 중)")
                        else:
                            print("기본 감정 분석 모델 로드 완료 (CPU 사용 중)")
                    except Exception as e2:
                        print(f"기본 모델 로드도 실패: {e2}")
                        import traceback
                        traceback.print_exc()
    
    def _analyze_with_openai(self, text: str) -> Dict:
        """
        OpenAI API를 사용하여 텍스트의 감정을 분석합니다.
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            {
                'label': '부정적' | '보통' | '긍정적',
                'score': float (0.0 ~ 1.0)
            }
        """
        if not self.openai_client:
            raise ValueError("OpenAI 클라이언트가 초기화되지 않았습니다.")
        
        # 텍스트가 너무 길면 앞부분과 뒷부분을 결합하여 사용
        if len(text) > 3000:
            text_for_analysis = text[:2000] + " " + text[-1000:]
            print(f"[OpenAI 감정 분석] 긴 텍스트 감지: {len(text)}자 -> {len(text_for_analysis)}자로 축약")
        else:
            text_for_analysis = text
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # 비용 효율적인 모델 사용
                messages=[
                    {
                        "role": "system",
                        "content": """당신은 뉴스 기사 감정 분석 전문가입니다. 주어진 뉴스 기사를 분석하여 감정을 평가해주세요.

다음 형식으로 JSON 응답을 해주세요:
{
    "label": "긍정적" 또는 "보통" 또는 "부정적",
    "score": 0.0부터 1.0까지의 숫자 (0.0: 매우 부정적, 0.5: 중립적, 1.0: 매우 긍정적)
}

감정 판단 기준:
- 긍정적: 성장, 발전, 성공, 개선, 혁신, 투자, 협력, 수상, 인정, 긍정적인 전망 등
- 부정적: 감소, 하락, 위기, 문제, 사고, 실패, 손실, 우려, 경고, 부정적인 전망 등
- 보통: 사실 전달 위주, 중립적인 내용, 명확한 감정이 없는 경우

점수 기준:
- 0.0~0.3: 부정적
- 0.4~0.6: 보통
- 0.7~1.0: 긍정적

반드시 유효한 JSON 형식으로만 응답해주세요."""
                    },
                    {
                        "role": "user",
                        "content": f"다음 뉴스 기사를 분석하여 감정을 평가해주세요:\n\n{text_for_analysis}"
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=200
            )
            
            result_text = response.choices[0].message.content.strip()
            result_json = json.loads(result_text)
            
            label = result_json.get('label', '보통')
            score = float(result_json.get('score', 0.5))
            
            # 점수 범위 제한
            score = max(0.0, min(1.0, score))
            
            # 라벨 정규화
            if label not in ['긍정적', '보통', '부정적']:
                # 라벨을 점수 기반으로 변환
                if score >= 0.7:
                    label = '긍정적'
                elif score <= 0.3:
                    label = '부정적'
                else:
                    label = '보통'
            
            print(f"[OpenAI 감정 분석] 라벨: {label}, 점수: {score:.3f}")
            
            return {
                'label': label,
                'score': score
            }
            
        except json.JSONDecodeError as e:
            print(f"OpenAI API 응답 JSON 파싱 오류: {e}")
            print(f"응답 내용: {result_text}")
            # 기본값 반환
            return {
                'label': '보통',
                'score': 0.5
            }
        except Exception as e:
            print(f"OpenAI API 감정 분석 오류: {e}")
            import traceback
            traceback.print_exc()
            # 기본값 반환
            return {
                'label': '보통',
                'score': 0.5
            }
    
    def analyze(self, text: str, article_id: Optional[int] = None) -> Dict:
        """
        텍스트의 감정을 분석합니다.
        
        Args:
            text: 분석할 텍스트 (전체 본문 또는 요약본)
            article_id: 기사 ID (이미지 파일명에 사용, None이면 해시값 사용)
            
        Returns:
            {
                'label': '부정적' | '보통' | '긍정적',
                'score': float (0.0 ~ 1.0),
                'temperature': int (0 ~ 100도),
                'image_path': str (이미지 파일 경로)
            }
        """
        # OpenAI API 사용 시
        if self.use_openai and self.openai_client:
            try:
                openai_result = self._analyze_with_openai(text)
                label = openai_result['label']
                score = openai_result['score']
                
                # 온도 계산: 점수를 0~100도 범위로 변환
                temperature = int(score * 100)
                temperature = max(0, min(100, temperature))
                
                # 이미지 경로 결정
                if label == '긍정적':
                    image_filename = 'temp/3.png'
                elif label == '부정적':
                    image_filename = 'temp/1.png'
                else:
                    image_filename = 'temp/2.png'
                
                print(f"[감정 분석] 최종 결과: {label}, 점수: {score:.3f}, 온도: {temperature}도, 이미지: {image_filename}")
                
                return {
                    'label': label,
                    'score': score,
                    'temperature': temperature,
                    'image_path': image_filename
                }
            except Exception as e:
                print(f"OpenAI API 감정 분석 실패, 로컬 모델로 폴백: {e}")
                # 폴백: 로컬 모델 사용
        
        if not self.classifier and not self.use_finetuned_model:
            # 모델이 없으면 기본값 반환
            return {
                'label': '보통',
                'score': 0.5,
                'temperature': 50,
                'image_path': 'temp/2.png'
            }
        
        # 전체 본문이 너무 길면 앞부분과 뒷부분을 결합하여 사용
        # (앞부분: 주요 내용, 뒷부분: 결론/요약)
        if len(text) > 2000:
            # 앞부분 1500자 + 뒷부분 500자
            text_for_analysis = text[:1500] + " " + text[-500:]
            print(f"[감정 분석] 긴 텍스트 감지: {len(text)}자 -> {len(text_for_analysis)}자로 축약")
        else:
            text_for_analysis = text
        
        try:
            # 파인튜닝된 모델 사용
            if self.use_finetuned_model and self.model and self.tokenizer:
                # 토크나이징
                inputs = self.tokenizer(
                    text_for_analysis,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                
                # GPU로 이동
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 추론
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probabilities = F.softmax(logits, dim=-1)
                
                # 결과 파싱 (0: 부정, 1: 중립, 2: 긍정)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_class].item()
                
                # 각 클래스의 확률
                prob_negative = probabilities[0][0].item()  # 부정 확률
                prob_neutral = probabilities[0][1].item()    # 중립 확률
                prob_positive = probabilities[0][2].item() # 긍정 확률
                
                # 라벨 매핑
                label_map = {0: '부정적', 1: '보통', 2: '긍정적'}
                label = label_map.get(predicted_class, '보통')
                
                # 점수 계산: 확률 분포를 기반으로 0.0~1.0 범위의 점수로 변환
                # 방법: (긍정 확률 - 부정 확률)을 0.5를 중심으로 변환
                # score = 0.5 + (prob_positive - prob_negative) * 0.5
                # 이렇게 하면:
                # - 부정만 높으면: 0.0~0.5 (낮은 점수)
                # - 중립만 높으면: 0.5 근처
                # - 긍정만 높으면: 0.5~1.0 (높은 점수)
                # - 혼합된 경우: 확률 차이에 비례
                score = 0.5 + (prob_positive - prob_negative) * 0.5
                
                # 점수 범위 제한 (0.0~1.0)
                score = max(0.0, min(1.0, score))
                
                # 디버깅: 실제 확률 값 출력
                print(f"[모델 출력] 부정: {prob_negative:.3f}, 중립: {prob_neutral:.3f}, 긍정: {prob_positive:.3f} -> 예측: {label} (신뢰도: {confidence:.3f}, 점수: {score:.3f})")
                
            else:
                # pipeline 사용 (기본 모델)
                result = self.classifier(text_for_analysis)
                
                # 결과 파싱
                if isinstance(result, list) and len(result) > 0:
                    result = result[0]
                
                label = result.get('label', '')
                score = result.get('score', 0.5)
            
            # 디버깅: 모델 출력 확인 (파인튜닝된 모델의 경우 위에서 이미 출력됨)
            if not self.use_finetuned_model:
                print(f"[감정 분석] 텍스트 길이: {len(text)}자 -> 분석용: {len(text_for_analysis)}자")
                print(f"[감정 분석] 원본 라벨: {label}, 원본 점수: {score:.3f}")
            else:
                print(f"[감정 분석] 텍스트 길이: {len(text)}자 -> 분석용: {len(text_for_analysis)}자")
            
            # 파인튜닝된 모델의 경우 직접 사용, 기본 모델의 경우 추가 처리
            if self.use_finetuned_model:
                # 파인튜닝된 모델 출력 + 키워드 기반 보정
                # 전체 본문을 사용하여 키워드 분석 (text_for_analysis 사용)
                text_lower = text_for_analysis.lower()
                
                # 뉴스 긍정 키워드
                positive_keywords = [
                    '완승', '성공', '승리', '발전', '성장', '증가', '개선', '혁신', '확대', '상승',
                    '향상', '도약', '기대', '긍정', '호재', '호조', '확장', '투자', '협력',
                    '파트너십', '기술', '개발', '출시', '수상', '인정', '평가', '우수', '최고',
                    '1위', '선두', '돌파', '기록', '최대', '최고치', '상승세', '호전', '개선세'
                ]
                
                # 뉴스 부정 키워드
                # 강한 부정 키워드 (사망, 사고 등) - 강제 부정 분류
                strong_negative_keywords = [
                    '숨지', '사망', '사고로', '사고로 인해', '사고로 사망', '방치', '유기',
                    '살인', '폭행', '강도', '강간', '성폭행', '학대', '폭력', '테러',
                    '폭발', '화재', '붕괴', '추락', '충돌', '교통사고', '교통사고로',
                    '사고로 숨지', '사고로 사망', '사고로 부상', '사고로 다쳐',
                    '비극', '참사', '재난', '재해', '피해자', '희생자', '부상자'
                ]
                
                # 일반 부정 키워드
                negative_keywords = [
                    '감소', '하락', '위기', '문제', '사고', '부정', '부실', '실패', '폐쇄',
                    '도산', '파산', '손실', '적자', '축소', '감원', '해고', '실업', '불안',
                    '우려', '경고', '위험', '부상', '피해', '비리', '의혹', '논란',
                    '조롱', '비난', '하락세', '악화', '쇠퇴', '후퇴', '실적부진', '부진'
                ]
                
                # 부정적인 문맥 패턴 (직접적인 부정 단어 없이도 부정적 의미를 나타내는 패턴)
                negative_patterns = [
                    # 강한 부정 패턴 (사망, 사고 관련)
                    r'숨지',  # "숨졌다", "숨져"
                    r'사망\s*했다',  # "사망했다"
                    r'방치',  # "방치했다", "방치"
                    r'유기',  # "유기했다"
                    r'차에\s*방치',  # "차에 방치"
                    r'차량에\s*방치',  # "차량에 방치"
                    r'차에\s*남겨',  # "차에 남겨둔다"
                    r'차량에\s*남겨',  # "차량에 남겨둔다"
                    r'사고로\s*숨지',  # "사고로 숨졌다"
                    r'사고로\s*사망',  # "사고로 사망했다"
                    r'사고로\s*인해',  # "사고로 인해"
                    r'교통사고',  # "교통사고"
                    r'교통사고로',  # "교통사고로"
                    # 일반 부정 패턴
                    r'하지\s*못',  # "하지 못했다", "하지 못함"
                    r'못\s*했다',  # "못했다", "못함"
                    r'실패\s*했다',  # "실패했다"
                    r'보다\s*낮',  # "보다 낮다", "보다 낮음"
                    r'보다\s*못',  # "보다 못하다"
                    r'못\s*미친',  # "못 미친다", "못 미침"
                    r'에\s*못\s*미친',  # "에 못 미친다"
                    r'에\s*실패',  # "에 실패했다"
                    r'하지\s*않',  # "하지 않았다" (부정적 맥락에서)
                    r'없\s*었다',  # "없었다"
                    r'없\s*었음',  # "없었음"
                    r'없\s*어',  # "없어"
                    r'부족',  # "부족하다"
                    r'부족\s*했다',  # "부족했다"
                    r'미달',  # "미달했다"
                    r'미달\s*했다',  # "미달했다"
                    r'기대\s*에\s*못\s*미친',  # "기대에 못 미쳤다"
                    r'기대\s*이하',  # "기대 이하"
                    r'예상\s*보다\s*낮',  # "예상보다 낮다"
                    r'예상\s*보다\s*못',  # "예상보다 못하다"
                    r'전년\s*대비\s*감소',  # "전년 대비 감소"
                    r'전년\s*대비\s*하락',  # "전년 대비 하락"
                    r'전년\s*대비\s*줄어',  # "전년 대비 줄어"
                    r'전년\s*대비\s*떨어',  # "전년 대비 떨어졌다"
                    r'전분기\s*대비\s*감소',  # "전분기 대비 감소"
                    r'전분기\s*대비\s*하락',  # "전분기 대비 하락"
                    r'목표\s*에\s*못\s*미친',  # "목표에 못 미쳤다"
                    r'목표\s*이하',  # "목표 이하"
                    r'목표\s*미달',  # "목표 미달"
                    r'기록\s*보다\s*낮',  # "기록보다 낮다"
                    r'기록\s*보다\s*못',  # "기록보다 못하다"
                    r'이전\s*보다\s*나빠',  # "이전보다 나빠졌다"
                    r'이전\s*보다\s*떨어',  # "이전보다 떨어졌다"
                    r'이전\s*보다\s*줄어',  # "이전보다 줄어들었다"
                    r'에도\s*불구하고',  # "~에도 불구하고" (부정적 맥락)
                    r'임에도\s*불구',  # "~임에도 불구하고"
                    r'그럼에도\s*불구',  # "그럼에도 불구하고"
                    r'그러나',  # "그러나" (대조/부정적 맥락)
                    r'하지만',  # "하지만" (대조/부정적 맥락)
                    r'다만',  # "다만" (제한/부정적 맥락)
                    r'아쉽게도',  # "아쉽게도"
                    r'안타깝게도',  # "안타깝게도"
                    r'유감스럽게도',  # "유감스럽게도"
                    r'아쉽',  # "아쉽다"
                    r'안타깝',  # "안타깝다"
                    r'유감',  # "유감이다"
                    r'우려\s*된다',  # "우려된다"
                    r'우려\s*가',  # "우려가 있다"
                    r'걱정\s*된다',  # "걱정된다"
                    r'걱정\s*이',  # "걱정이 있다"
                    r'불안\s*하다',  # "불안하다"
                    r'불안\s*감',  # "불안감"
                    r'위험\s*하다',  # "위험하다"
                    r'위험\s*이',  # "위험이 있다"
                    r'문제\s*가\s*있다',  # "문제가 있다"
                    r'문제\s*가\s*발생',  # "문제가 발생했다"
                    r'문제\s*가\s*나타나',  # "문제가 나타났다"
                    r'어려움',  # "어려움이 있다"
                    r'어려움\s*을\s*겪',  # "어려움을 겪고 있다"
                    r'난관',  # "난관에 봉착했다"
                    r'난관\s*에',  # "난관에"
                    r'장애',  # "장애가 있다"
                    r'장애\s*물',  # "장애물"
                    r'제약',  # "제약이 있다"
                    r'제약\s*이',  # "제약이"
                    r'한계',  # "한계가 있다"
                    r'한계\s*를',  # "한계를 보인다"
                    r'부족\s*하다',  # "부족하다"
                    r'부족\s*한',  # "부족한"
                    r'부족\s*함',  # "부족함"
                    r'미흡',  # "미흡하다"
                    r'미흡\s*하다',  # "미흡하다"
                    r'미흡\s*한',  # "미흡한"
                    r'아쉬움',  # "아쉬움이 있다"
                    r'아쉬움\s*을',  # "아쉬움을 남겼다"
                    r'아쉬운',  # "아쉬운 점"
                    r'아쉬운\s*점',  # "아쉬운 점"
                    r'아쉬운\s*부분',  # "아쉬운 부분"
                ]
                
                # 키워드 기반 점수 보정
                positive_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
                negative_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
                
                # 강한 부정 키워드 감지 (강제 부정 분류)
                strong_negative_count = sum(1 for keyword in strong_negative_keywords if keyword in text_lower)
                
                # 문맥 패턴 기반 부정 감지
                context_negative_count = 0
                for pattern in negative_patterns:
                    if re.search(pattern, text_lower):
                        context_negative_count += 1
                
                # 문맥 패턴이 감지되면 부정 점수 추가
                if context_negative_count > 0:
                    negative_count += context_negative_count
                    # 감지된 패턴 출력 (처음 3개만)
                    detected_patterns = []
                    for pattern in negative_patterns:
                        if re.search(pattern, text_lower):
                            detected_patterns.append(pattern)
                            if len(detected_patterns) >= 3:
                                break
                    if detected_patterns:
                        print(f"[문맥 분석] 부정적 문맥 패턴 {context_negative_count}개 감지: {', '.join(detected_patterns[:3])}")
                
                # 강한 부정 키워드가 있으면 강제로 부정 분류
                if strong_negative_count > 0:
                    print(f"[강한 부정 감지] 강한 부정 키워드 {strong_negative_count}개 감지 - 강제 부정 분류")
                    sentiment = '부정적'
                    image_filename = 'temp/1.png'
                    # 강한 부정 키워드가 있으면 점수를 매우 낮게 설정 (0.0~0.2)
                    adjusted_score = max(0.0, 0.2 - (strong_negative_count * 0.1))
                    temperature = int(adjusted_score * 100)
                    temperature = max(0, min(20, temperature))  # 강한 부정은 최대 20도
                    print(f"[감정 분석] 최종 결과: {sentiment}, 점수: {adjusted_score:.3f}, 온도: {temperature}도, 이미지: {image_filename}")
                    return {
                        'label': sentiment,
                        'score': adjusted_score,
                        'temperature': temperature,
                        'image_path': image_filename
                    }
                
                # 키워드 보정 점수 (-0.4 ~ +0.4)
                # 문맥 패턴은 더 강한 가중치 적용
                keyword_bias = 0.0
                if positive_count > 0:
                    keyword_bias = min(0.4, positive_count * 0.1)  # 최대 +0.4
                if negative_count > 0:
                    # 문맥 패턴이 있으면 더 강한 부정 가중치
                    if context_negative_count > 0:
                        keyword_bias = max(-0.4, -(negative_count * 0.12))  # 문맥 패턴 있으면 더 강하게
                    else:
                        keyword_bias = max(-0.3, -(negative_count * 0.1))  # 최대 -0.3
                
                # 보정된 점수 계산
                adjusted_score = score + keyword_bias
                adjusted_score = max(0.0, min(1.0, adjusted_score))
                
                # 보정된 점수 기반으로 감정 재판단
                if adjusted_score >= 0.65:  # 긍정 기준
                    sentiment = '긍정적'
                    image_filename = 'temp/3.png'
                elif adjusted_score <= 0.35:  # 부정 기준
                    sentiment = '부정적'
                    image_filename = 'temp/1.png'
                else:
                    # 키워드가 강하면 키워드 우선
                    if positive_count >= 2 and positive_count > negative_count:
                        sentiment = '긍정적'
                        image_filename = 'temp/3.png'
                        adjusted_score = min(1.0, 0.65 + (positive_count * 0.1))
                    elif negative_count >= 2 and negative_count > positive_count:
                        sentiment = '부정적'
                        image_filename = 'temp/1.png'
                        adjusted_score = max(0.0, 0.35 - (negative_count * 0.1))
                    else:
                        sentiment = label  # 모델 예측 유지
                        if sentiment == '긍정적':
                            image_filename = 'temp/3.png'
                        elif sentiment == '부정적':
                            image_filename = 'temp/1.png'
                        else:
                            image_filename = 'temp/2.png'
                
                # 온도 계산: 보정된 점수를 0~100도 범위로 변환
                temperature = int(adjusted_score * 100)
                
                # 온도 범위 제한 (0~100도)
                temperature = max(0, min(100, temperature))
                
                # 디버깅 출력
                if positive_count > 0 or negative_count > 0:
                    print(f"[키워드 보정] 긍정: {positive_count}, 부정: {negative_count}, 보정: {keyword_bias:.2f}, 원점수: {score:.3f} -> 보정점수: {adjusted_score:.3f}")
            else:
                # 기본 모델의 경우 기존 로직 사용
                # 뉴스 요약에 특화된 감정 분석
                # 1단계: 모델 출력 기반 초기 감정 판단
                label_lower = str(label).lower()
                
                # 모델별 라벨 매핑
                is_positive_label = any(keyword in label_lower for keyword in [
                    'positive', '긍정', '5 star', '5star', '4 star', '4star'
                ])
                
                is_negative_label = any(keyword in label_lower for keyword in [
                    'negative', '부정', '1 star', '1star', '2 star', '2star'
                ])
                
                is_neutral_label = any(keyword in label_lower for keyword in [
                    'neutral', '보통', '3 star', '3star', '중립'
                ])
                
                # 2단계: 뉴스 도메인 키워드 기반 감정 보정
                # 전체 본문을 사용하여 키워드 분석
                text_lower = text_for_analysis.lower()
                
                # 뉴스 긍정 키워드 (발전, 성장, 증가, 개선, 혁신, 성공, 확대 등)
                positive_keywords = [
                    '발전', '성장', '증가', '개선', '혁신', '성공', '확대', '상승',
                    '향상', '도약', '기대', '긍정', '호재', '호조', '확대', '확장',
                    '투자', '협력', '파트너십', '기술', '혁신', '개발', '출시',
                    '수상', '인정', '평가', '우수', '최고', '1위', '선두'
                ]
                
                # 뉴스 부정 키워드 (감소, 하락, 위기, 문제, 사고, 부정, 부실 등)
                negative_keywords = [
                    '감소', '하락', '위기', '문제', '사고', '부정', '부실', '실패',
                    '폐쇄', '도산', '파산', '손실', '적자', '감소', '축소', '감원',
                    '해고', '실업', '불안', '우려', '경고', '위험', '사고', '사망',
                    '부상', '피해', '손실', '비리', '부정', '비리', '의혹', '논란','조롱', '비난'
                ]
                
                # 키워드 기반 점수 계산
                positive_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
                negative_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
                
                # 키워드 기반 보정 점수 (-1.0 ~ 1.0)
                keyword_bias = 0.0
                if positive_count > 0:
                    keyword_bias = min(0.3, positive_count * 0.1)  # 최대 +0.3
                if negative_count > 0:
                    keyword_bias = max(-0.3, -negative_count * 0.1)  # 최대 -0.3
                
                # 3단계: 최종 감정 판단 (모델 출력 + 키워드 보정)
                # 긍정 비율 조정: 긍정 판단 기준을 더 엄격하게 설정 (0.7 이상)
                if is_positive_label:
                    # 모델이 긍정으로 판단한 경우
                    adjusted_score = min(1.0, score + keyword_bias)
                    if adjusted_score >= 0.7:  # 긍정 기준을 0.5에서 0.7로 상향 조정
                        sentiment = '긍정적'
                        image_filename = 'temp/3.png'
                    elif adjusted_score <= 0.3:  # 부정 기준도 명확히 설정
                        sentiment = '부정적'
                        image_filename = 'temp/1.png'
                    else:
                        sentiment = '보통'
                        image_filename = 'temp/2.png'
                elif is_negative_label:
                    # 모델이 부정으로 판단한 경우
                    adjusted_score = max(0.0, score - abs(keyword_bias))
                    if adjusted_score <= 0.3:
                        sentiment = '부정적'
                        image_filename = 'temp/1.png'
                    elif adjusted_score >= 0.7:
                        sentiment = '긍정적'
                        image_filename = 'temp/3.png'
                    else:
                        sentiment = '보통'
                        image_filename = 'temp/2.png'
                else:
                    # 모델이 중립이거나 알 수 없는 경우
                    # 키워드 기반으로 판단
                    final_score = score + keyword_bias
                    if positive_count > negative_count and positive_count > 2:  # 긍정 키워드가 2개 이상일 때만 긍정 판단
                        sentiment = '긍정적'
                        image_filename = 'temp/3.png'
                        # 키워드 기반 점수 계산 (0.7~1.0 범위)
                        final_score = min(1.0, 0.7 + (min(positive_count, 5) * 0.06))
                    elif negative_count > positive_count and negative_count > 0:
                        sentiment = '부정적'
                        image_filename = 'temp/1.png'
                        # 키워드 기반 점수 계산 (0.0~0.3 범위)
                        final_score = max(0.0, 0.3 - (min(negative_count, 5) * 0.06))
                    else:
                        # 키워드가 없거나 균형인 경우 score 기반 판단
                        if final_score >= 0.7:  # 긍정 기준 상향
                            sentiment = '긍정적'
                            image_filename = 'temp/3.png'
                        elif final_score <= 0.3:  # 부정 기준 하향
                            sentiment = '부정적'
                            image_filename = 'temp/1.png'
                        else:
                            sentiment = '보통'
                            image_filename = 'temp/2.png'
                
                # 점수와 온도를 같게 설정 (score * 100)
                temperature = int(score * 100)
            
            # 온도 범위 제한
            temperature = max(0, min(100, temperature))
            
            # 디버깅 출력 (파인튜닝된 모델과 기본 모델 구분)
            if self.use_finetuned_model:
                # 파인튜닝된 모델의 경우 위에서 이미 키워드 보정 출력이 있음
                print(f"[감정 분석] 최종 결과: {sentiment}, 점수: {score:.3f}, 온도: {temperature}도, 이미지: {image_filename}")
            else:
                print(f"[감정 분석] 키워드 분석 - 긍정: {positive_count}, 부정: {negative_count}, 보정: {keyword_bias:.2f}")
                print(f"[감정 분석] 최종 결과: {sentiment}, 점수: {score:.3f}, 온도: {temperature}도, 이미지: {image_filename}")
            
            return {
                'label': sentiment,
                'score': score,
                'temperature': temperature,
                'image_path': image_filename
            }
            
        except Exception as e:
            print(f"감정 분석 오류: {e}")
            import traceback
            traceback.print_exc()
            # 오류 시 기본값 반환
            return {
                'label': '보통',
                'score': 0.5,
                'temperature': 50,
                'image_path': 'temp/2.png'
            }
    
    def _create_sentiment_image(self, sentiment: str, temperature: int, image_path: str):
        """
        감정 분석 결과를 이미지로 생성합니다.
        
        Args:
            sentiment: 감정 레이블 ('부정적', '보통', '긍정적')
            temperature: 온도 수치 (0-100)
            image_path: 저장할 이미지 경로
        """
        if not PIL_AVAILABLE:
            print("Pillow가 설치되지 않아 이미지를 생성할 수 없습니다.")
            return
        
        # temp 폴더 생성
        os.makedirs('temp', exist_ok=True)
        
        # 이미지 크기
        width, height = 400, 300
        
        # 배경색 결정
        if sentiment == '부정적':
            bg_color = (220, 53, 69)  # 빨간색 계열
        elif sentiment == '긍정적':
            bg_color = (40, 167, 69)  # 초록색 계열
        else:
            bg_color = (255, 193, 7)  # 노란색 계열
        
        # 이미지 생성
        img = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(img)
        
        # 폰트 설정 (시스템 기본 폰트 사용)
        try:
            # Windows
            font_large = ImageFont.truetype("malgun.ttf", 40)
            font_medium = ImageFont.truetype("malgun.ttf", 30)
        except:
            try:
                # Linux
                font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
                font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
            except:
                # 기본 폰트
                font_large = ImageFont.load_default()
                font_medium = ImageFont.load_default()
        
        # 텍스트 그리기
        # 감정 레이블
        text_y = 50
        bbox = draw.textbbox((0, 0), sentiment, font=font_large)
        text_width = bbox[2] - bbox[0]
        text_x = (width - text_width) // 2
        draw.text((text_x, text_y), sentiment, fill=(255, 255, 255), font=font_large)
        
        # 온도 수치
        temp_text = f"{temperature}°C"
        text_y = 150
        bbox = draw.textbbox((0, 0), temp_text, font=font_medium)
        text_width = bbox[2] - bbox[0]
        text_x = (width - text_width) // 2
        draw.text((text_x, text_y), temp_text, fill=(255, 255, 255), font=font_medium)
        
        # 이미지 저장
        img.save(image_path)
        print(f"이미지 저장 완료: {image_path}")


if __name__ == '__main__':
    # 테스트
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze("오늘은 정말 좋은 날입니다!")
    print(result)

