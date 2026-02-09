from urllib.parse import urlparse
import re
from collections import Counter, defaultdict
from typing import Optional, Dict, Any, List, Tuple

DATASETS_NAME_MAPPING = {
    "medium": 'Medium',
    "quora": 'Quora',
    "reddit": 'Reddit',
    "deepfake": 'Reddit (Deepfake)',
    "fox8": 'Twitter (Fox8-23)',
    "m4": 'Reddit (M4)',
    "discord": 'Discord',
    "gab": 'Gab',
    "telegram": 'Telegram',
    "twitter": 'Twitter',
    "whatsapp": 'WhatsApp',
    "tweepfake": 'Twitter (TweepFake)'
}
    

def normalize_instance(inst):
    if not inst:
        return None
    inst = inst.strip().lower()
    inst = inst.replace("https://", "").replace("http://", "")
    inst = inst.split("/")[0]        # remove path
    inst = inst.split(":")[0]        # remove port
    return inst

def get_instance(obj, track_origin=False):
    """Extract the instance name from a Mastodon-like status."""
    
    def _from_account(acc):
        if not isinstance(acc, dict):
            return None
        
        inst = normalize_instance(acc.get("instance_name"))
        if inst:
            return inst

        acct = acc.get("acct")
        if isinstance(acct, str) and "@" in acct:
            return normalize_instance(acct.split("@", 1)[1])
        
        for k in ("url", "uri"):
            v = acc.get(k)
            if isinstance(v, str):
                p = urlparse(v)
                if p.netloc:
                    return normalize_instance(p.netloc)
        return None

    def _from_status(st):
        if not isinstance(st, dict):
            return None
        
        inst = normalize_instance(st.get("instance_name"))
        if inst:
            return inst
        
        inst = _from_account(st.get("account"))
        if inst:
            return inst
        
        for k in ("uri", "url"):
            v = st.get(k)
            if isinstance(v, str):
                p = urlparse(v)
                if p.netloc:
                    return normalize_instance(p.netloc)
        return None
    
    def _return_unknown():
        unknown_identifier = "unknown.instance"
        print(f"Warning: could not determine instance from object:{obj}, returning: {unknown_identifier}.")
        return unknown_identifier

    if not isinstance(obj, dict):
        return _return_unknown()

    if track_origin and obj.get("reblog"):
        obj = obj.get("reblog")

    inst = normalize_instance(obj.get("instance_name"))
    if inst:
        return inst
    
    inst = _from_status(obj)
    if inst:
        return inst
    
    acc_inst = _from_account(obj.get("account"))
    if acc_inst:
        return acc_inst
    
    return _return_unknown()



# ---------- Patterns / phrases ----------
BAD_PHRASES = [
    "just the rewritten post",
    "please respond with the revised post",
    "here is the completed post",
    "your response here",
    "no response is expected",
    "i've made some minor changes",
    "mastodonsocialassistant",
    "le post réécrit",
    "continuing naturally in",
    "please respond with the completed post",
]

CODE_FENCE_RE = re.compile(r"```")
# catches "```python" or similar even if fences are malformed.
CODE_LANG_HINT_RE = re.compile(r"```[a-zA-Z]+")

# Emoji blocks: this is broad enough for most common emojis.
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF]")


# ---------- Helpers ----------
def token_repeat_stats(text: str) -> Tuple[int, int, float, str]:
    """
    Returns:
      most_common_count: int
      total_tokens: int
      ratio: float (most_common_count / total_tokens)
      most_common_token: str
    """
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    if not tokens:
        return 0, 0, 0.0, ""
    counts = Counter(tokens)
    tok, cnt = counts.most_common(1)[0]
    return cnt, len(tokens), cnt / len(tokens), tok


def emoji_ratio(text: str) -> float:
    """#emoji_chars / #total_chars (safe for unicode)."""
    if not text:
        return 0.0
    emojis = EMOJI_RE.findall(text)
    return len(emojis) / max(len(text), 1)


def has_emoji_spam(text: str, threshold: float = 0.25) -> bool:
    return emoji_ratio(text) > threshold


# ---------- Main detector (typed) ----------
def detect_manifold_type(
    aigt_text: Optional[str],
    *,
    code_fence_min: int = 3,
    repeat_ratio_thresh: float = 0.25,
    repeat_min_repeats: int = 8,
    emoji_ratio_thresh: float = 0.25,
    near_empty_len: int = 8,
) -> Optional[str]:
    """
    Returns:
      manifold_type: str | None
    Types:
      - "meta_placeholder"
      - "code_block_leak"
      - "token_loop"
      - "emoji_spam"
      - "too_short"
    """
    if not aigt_text or not isinstance(aigt_text, str):
        return "too_short"

    raw = aigt_text.strip()
    if len(raw) < near_empty_len:
        return "too_short"

    t = raw.lower()

    # 1) Meta / placeholder / instruction echo
    if any(p in t for p in BAD_PHRASES):
        return "meta_placeholder"

    # 2) Code-block leakage / markdown fence spam
    fence_count = raw.count("```")
    if fence_count >= code_fence_min:
        return "code_block_leak"
    # Catch language fences like ```python even if not repeated much
    if CODE_LANG_HINT_RE.search(raw) and fence_count >= 1:
        return "code_block_leak"

    # 3) Token repetition loop
    most_cnt, total_tok, rr, tok = token_repeat_stats(raw)
    if most_cnt >= repeat_min_repeats and rr >= repeat_ratio_thresh:
        return "token_loop"

    # 4) Emoji spam
    if has_emoji_spam(raw, threshold=emoji_ratio_thresh):
        return "emoji_spam"

    return None


# ---------- Aggregation over records ----------
def process_manifolds(
    records: List[Dict[str, Any]],
    *,
    include_method: bool = False,
    detector_kwargs: Optional[Dict[str, Any]] = None,
    keep_examples: int = 0,
) -> Dict[str, Any]:
    """
    Input record example:
      {
        'text': ...,
        'generation_method': 'polish'|'complete'|'3-iteration paraphrase',
        'llm_model': 'Llama-3-8B-Instruct',
        ...
      }

    Print:
      {
        "counts": { llm_model: { manifold_type: n, ... }, ... },
        "totals": { llm_model: total_records, ... },
        "rates":  { llm_model: { manifold_type: n/total, ... }, ... },
        "by_method": { llm_model: { method: { manifold_type: n }}}  # if include_method
        "examples": { llm_model: { manifold_type: [ {id,text,...}, ... ]}}  # if keep_examples>0
      }
    
    Output:
        safe_records: List of records where no manifold type was detected.
    """
    detector_kwargs = detector_kwargs or {}

    counts = defaultdict(Counter)          # model -> Counter(type->count)
    totals = Counter()                      # model -> total
    by_method = defaultdict(lambda: defaultdict(Counter))  # model -> method -> Counter
    examples = defaultdict(lambda: defaultdict(list))      # model -> type -> list
    
    safe_records = []

    for rec in records:
        model = rec.get("llm_model") or "unknown_model"
        method = rec.get("generation_method") or "unknown_method"
        text = rec.get("text")

        totals[model] += 1
        mtype = detect_manifold_type(text, **detector_kwargs)

        if mtype is not None:
            counts[model][mtype] += 1
            if include_method:
                by_method[model][method][mtype] += 1

            if keep_examples > 0 and len(examples[model][mtype]) < keep_examples:
                examples[model][mtype].append({
                    "original_id": rec.get("original_id"),
                    "generation_method": method,
                    "language": rec.get("language"),
                    "text": text,
                })
        else:
            safe_records.append(rec)

    # rates
    rates = {}
    for model, total in totals.items():
        rates[model] = {}
        if total == 0:
            continue
        for mtype, n in counts[model].items():
            rates[model][mtype] = n / total

    out = {
        "counts": {m: dict(c) for m, c in counts.items()},
        "totals": dict(totals),
        "rates": rates,
    }
    if include_method:
        out["by_method"] = {
            m: {meth: dict(cntr) for meth, cntr in meth_map.items()}
            for m, meth_map in by_method.items()
        }
    if keep_examples > 0:
        out["examples"] = {
            m: {t: lst for t, lst in type_map.items()}
            for m, type_map in examples.items()
        }

    
    # Pretty print
    print(f"TOTAL MANIFOLD AIGTs: {len(records) - len(safe_records)}")
    for model in sorted(totals.keys()):
        print("============================")
        print(f"Model: {model}")
        total = totals[model]
        print(f"  Total records: {total}", f"  Total manifolds: {sum(counts[model].values())}")
        if total == 0:
            continue
        for mtype, n in counts[model].most_common():
            rate = rates[model][mtype]
            print(f"    {mtype}: {n} ({rate:.2%})")
        if include_method and model in by_method:
            print("  Breakdown by method:")
            for method, cntr in by_method[model].items():
                print(f"    Method: {method}")
                for mtype, n in cntr.most_common():
                    rate = n / sum(cntr.values())
                    print(f"      {mtype}: {n} ({rate:.2%})")
        
    return safe_records



