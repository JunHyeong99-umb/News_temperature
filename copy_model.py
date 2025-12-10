"""
모델 복사 스크립트
사용법: python copy_model.py [대상경로]
예시: python copy_model.py ../my_model
     python copy_model.py C:/models/kosum-v1-copy
"""
import shutil
import sys
from pathlib import Path

def copy_model_files(target_dir=None):
    """필요한 모델 파일만 복사"""
    
    # 원본 모델 경로
    source_dir = Path("models/kosum-v1")
    
    if not source_dir.exists():
        print(f"❌ 오류: 모델 디렉토리를 찾을 수 없습니다: {source_dir}")
        return False
    
    # 대상 경로 설정
    if target_dir is None:
        target_dir = Path("models/kosum-v1-copy")
    else:
        target_dir = Path(target_dir)
    
    # 필요한 파일 목록
    required_files = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ]
    
    print(f"원본: {source_dir}")
    print(f"대상: {target_dir}")
    print("-" * 60)
    
    # 디렉토리 생성
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 파일 복사
    copied_files = []
    missing_files = []
    
    for file in required_files:
        source_file = source_dir / file
        target_file = target_dir / file
        
        if source_file.exists():
            try:
                shutil.copy2(source_file, target_file)
                size_mb = source_file.stat().st_size / (1024 * 1024)
                print(f"✓ {file:30s} ({size_mb:.2f} MB)")
                copied_files.append(file)
            except Exception as e:
                print(f"✗ {file:30s} 복사 실패: {e}")
        else:
            print(f"⚠ {file:30s} 없음 (선택사항)")
            missing_files.append(file)
    
    print("-" * 60)
    print(f"\n✅ 복사 완료!")
    print(f"   복사된 파일: {len(copied_files)}개")
    if missing_files:
        print(f"   없는 파일: {len(missing_files)}개 (선택사항)")
    
    # 전체 크기 계산
    total_size = sum((target_dir / f).stat().st_size for f in copied_files if (target_dir / f).exists())
    print(f"   총 크기: {total_size / (1024 * 1024):.2f} MB")
    print(f"\n모델 경로: {target_dir.absolute()}")
    
    return True


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    copy_model_files(target)

