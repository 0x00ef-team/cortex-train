# Training corpus

`code.txt` is built from `bigcode/the-stack-smol-xs` (per-language slices:
Python, Rust, JavaScript, Go, Java, C, C++, TypeScript). The source files are
JSONL with a `content` field; only that field is kept, samples are joined with
blank lines.

Source: https://huggingface.co/datasets/bigcode/the-stack-smol-xs
Please check the upstream dataset card for licensing and attribution.

Raw `*/data.json` files are downloaded locally and ignored by git; `code.txt`
(~10 MB, ~10.2M bytes) is the reproducible training input.

Regenerate with:

```python
import json, pathlib
out = pathlib.Path("datasets/code.txt")
with out.open("w", encoding="utf-8", errors="ignore") as w:
    for p in sorted(pathlib.Path("datasets").glob("*/data.json")):
        for line in p.open(encoding="utf-8", errors="ignore"):
            try: obj = json.loads(line.strip())
            except Exception: continue
            code = obj.get("content")
            if isinstance(code, str) and code.strip():
                w.write(code.strip() + "\n\n")
```
