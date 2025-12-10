import os, json, zipfile, argparse, re
from pathlib import Path

# 내부 JSON의 다양성 대응: 가능한 키 후보들
DOC_KEYS = ["article", "document", "orginal_text", "original_text", "text", "content", "body"]
SUM_KEYS = ["abstractive", "summary", "summaries", "target", "headline"]

DOMAIN_MAP = {"법률": "law", "사설": "editorial", "신문기사": "news"}

def first_nonempty(d, keys):
    for k in keys:
        if k in d and d[k]:
            v = d[k]
            if isinstance(v, list):
                v = " ".join([str(x) for x in v if x]).strip()
            return str(v)
    return None

def normalize_text(t: str):
    if not t:
        return None
    t = re.sub(r"\s+", " ", t).strip()
    return t if len(t) > 10 else None

def iter_zip_json(zip_path: Path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        for name in z.namelist():
            if not name.lower().endswith(('.json', '.jsonl')):
                continue
            with z.open(name) as f:
                raw = f.read().decode('utf-8', errors='ignore')
                # jsonl 우선 처리
                if "\n{" in raw.strip():
                    for line in raw.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            yield json.loads(line)
                        except Exception:
                            continue
                else:
                    try:
                        obj = json.loads(raw)
                        if isinstance(obj, list):
                            for o in obj:
                                yield o
                        elif isinstance(obj, dict):
                            # 상위 딕셔너리 아래 리스트가 들어있는 경우
                            candidates = [v for v in obj.values() if isinstance(v, list)]
                            if candidates:
                                for o in candidates[0]:
                                    yield o
                            else:
                                yield obj
                    except Exception:
                        continue

def process_split(split_dir: Path, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_in, total_out = 0, 0
    with out_path.open('w', encoding='utf-8') as w:
        for zname in sorted(split_dir.glob('*.zip')):
            domain_kr = zname.stem.split('_')[0]  # 예: 법률_train_original → 법률
            domain = DOMAIN_MAP.get(domain_kr, domain_kr)
            for obj in iter_zip_json(zname):
                total_in += 1
                doc = first_nonempty(obj, DOC_KEYS)
                summ = first_nonempty(obj, SUM_KEYS)
                doc = normalize_text(doc)
                summ = normalize_text(summ)
                if not doc or not summ:
                    continue
                rec = {"document": doc, "summary": summ, "domain": domain}
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_out += 1
    print(f"{split_dir} → {out_path}: in={total_in}, out={total_out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/ai_hub_raw/문서요약 텍스트')
    ap.add_argument('--outdir', default='data/processed')
    args = ap.parse_args()
    train_dir = Path(args.root) / 'Training'
    valid_dir = Path(args.root) / 'Validation'
    process_split(train_dir, Path(args.outdir) / 'train.jsonl')
    process_split(valid_dir, Path(args.outdir) / 'valid.jsonl')

if __name__ == '__main__':
    main()
