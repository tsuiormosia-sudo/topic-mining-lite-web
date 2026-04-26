import io
import re
import time
from typing import Dict, List, Literal, Tuple

import hashlib
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import unicodedata


st.set_page_config(page_title="Topic Mining Lite (LDA + BERTopic-lite)", layout="wide")


def _safe_action(name: str, fn):
    try:
        return fn()
    except Exception as e:
        st.error(f"{name} 失败：{e}")
        st.stop()


def _reset_all():
    keys = list(st.session_state.keys())
    for k in keys:
        del st.session_state[k]
    st.rerun()


def _decode_best_effort(data: bytes) -> str:
    if not data:
        return ""

    candidates = []
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        candidates.extend(["utf-16"])
    if data.startswith(b"\xef\xbb\xbf"):
        candidates.extend(["utf-8-sig"])
    candidates.extend(["utf-8", "utf-8-sig", "utf-16", "cp1252", "latin1"])

    best_text = None
    best_score = None
    for enc in candidates:
        try:
            txt = data.decode(enc, errors="replace")
        except Exception:
            continue
        rep = txt.count("\ufffd")
        ctrl = sum(1 for ch in txt if unicodedata.category(ch) in {"Cc", "Cf"} and ch not in "\n\t\r")
        weird = txt.count("ï»¿")
        score = rep * 10 + ctrl * 2 + weird * 5
        if best_score is None or score < best_score:
            best_text = txt
            best_score = score
        if score == 0:
            break

    return best_text if best_text is not None else data.decode("utf-8", errors="replace")


def read_csv_smart(uploaded_file):
    try:
        data = uploaded_file.getvalue()
    except Exception:
        data = uploaded_file.read()
    text = _decode_best_effort(data)
    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
        return df, None
    except Exception:
        try:
            df = pd.read_csv(io.StringIO(text))
            return df, None
        except Exception as e:
            return None, f"CSV 读取失败（已尝试自动识别编码/分隔符）：{e}"


def read_uploaded_table(uploaded_file):
    suffix = (uploaded_file.name or "").lower()
    if suffix.endswith(".csv"):
        return read_csv_smart(uploaded_file)
    if suffix.endswith(".xlsx") or suffix.endswith(".xls"):
        return pd.read_excel(uploaded_file), None
    return None, "仅支持 CSV / XLSX / XLS 文件"


def parse_pasted_table(
    raw_text: str,
    delimiter: Literal["auto", "tab", "comma", "semicolon"] = "auto",
    first_row_header: bool = True,
):
    text = (raw_text or "").strip()
    if not text:
        return None, "粘贴内容为空"

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None, "粘贴内容为空"

    first = lines[0]
    sep = None
    if delimiter == "tab":
        sep = "\t"
    elif delimiter == "comma":
        sep = ","
    elif delimiter == "semicolon":
        sep = ";"
    else:
        if "\t" in first:
            sep = "\t"
        elif "," in first:
            sep = ","
        elif ";" in first:
            sep = ";"

    if sep is None:
        values = [ln.strip() for ln in lines if ln.strip()]
        df = pd.DataFrame({"text": values})
        return df, None

    try:
        df = pd.read_csv(
            io.StringIO("\n".join(lines)),
            sep=sep,
            header=0 if first_row_header else None,
            engine="python",
        )
    except Exception as e:
        return None, f"解析失败：{e}"

    if df is None or df.empty:
        return None, "解析成功但表格为空"

    if not first_row_header:
        df.columns = [f"col_{i+1}" for i in range(df.shape[1])]
        if df.shape[1] == 1:
            df = df.rename(columns={"col_1": "text"})

    return df, None

def make_signature(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def normalize_text(text: object) -> str:
    if not isinstance(text, str):
        return ""
    s = unicodedata.normalize("NFKD", text)
    s = "".join(
        ch
        for ch in s
        if not (
            (unicodedata.category(ch) in {"Cf", "Cc"} and ch not in {"\n", "\t", "\r"})
            or unicodedata.category(ch) in {"Mn", "Me"}
        )
    )
    s = s.lower()
    s = re.sub(r"https?://\\S+|www\\.\\S+", " ", s)
    s = re.sub(r"@([\\w_]+)", " ", s)
    s = re.sub(r"#([\\w_]+)", r" \\1 ", s)
    s = re.sub(r"[\\r\\n\\t]+", " ", s)
    s = re.sub(r"[^a-z0-9\\u4e00-\\u9fff ]+", " ", s)
    s = re.sub(r"\\s{2,}", " ", s).strip()
    s = re.sub(r"\\blas\\s+vegas\\b", "las_vegas", s)
    s = re.sub(r"\\bcasa\\s+playa\\b", "casa_playa", s)
    return s


def demojize_if_available(text: str) -> str:
    try:
        import emoji
    except Exception:
        return text

    try:
        s = emoji.demojize(text, delimiters=(" ", " "))
        s = s.replace("_", " ")
        replacements = {
            "party popper": "庆祝",
            "red heart": "爱",
            "blue heart": "爱",
            "green heart": "爱",
            "purple heart": "爱",
            "sparkling heart": "爱",
            "smiling face": "开心",
            "grinning face": "开心",
            "face with tears of joy": "大笑",
            "loudly crying face": "难过",
            "thumbs up": "点赞",
            "clapping hands": "鼓掌",
            "fire": "火爆",
            "hundred points": "满分",
        }
        for k, v in replacements.items():
            s = re.sub(rf"\\b{k}\\b", f" {v} ", s)
        s = re.sub(r"\\s{2,}", " ", s).strip()
        return s
    except Exception:
        return text


def ftfy_available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("ftfy") is not None
    except Exception:
        return False


def fix_mojibake_if_available(text: str) -> str:
    try:
        import ftfy
    except Exception:
        return text

    try:
        return ftfy.fix_text(text)
    except Exception:
        return text


_COMMON_EN_VERBS = {
    "be","am","is","are","was","were","been","being",
    "have","has","had","having",
    "do","does","did","doing",
    "can","could","may","might","must","shall","should","will","would",
    "go","goes","went","gone","going",
    "get","gets","got","getting",
    "make","makes","made","making",
    "say","says","said","saying",
    "see","sees","saw","seen","seeing",
    "know","knows","knew","known","knowing",
    "think","thinks","thought","thinking",
    "take","takes","took","taken","taking",
    "come","comes","came","coming",
    "use","uses","used","using",
    "need","needs","needed","needing",
    "want","wants","wanted","wanting",
    "like","likes","liked","liking",
    "love","loves","loved","loving",
    "try","tries","tried","trying",
    "work","works","worked","working",
    "play","plays","played","playing",
    "stay","stays","stayed","staying",
    "look","looks","looked","looking",
    "feel","feels","felt","feeling",
    "find","finds","found","finding",
    "keep","keeps","kept","keeping",
    "put","puts","putting",
    "let","lets","letting",
}


def keep_nouns_adjs_heuristic(tokens: List[str]) -> List[str]:
    out = []
    for t in tokens:
        if not t:
            continue
        if t in _COMMON_EN_VERBS:
            continue
        if t.endswith("ing") and len(t) >= 6:
            continue
        if t.endswith("ed") and len(t) >= 6:
            continue
        out.append(t)
    return out


def wordfreq_available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("wordfreq") is not None
    except Exception:
        return False


def zipf_en(token: str) -> float | None:
    try:
        from wordfreq import zipf_frequency

        return float(zipf_frequency(token, "en"))
    except Exception:
        return None


@st.cache_resource
def _en_spell_index(n_words: int = 80000) -> Dict[str, Tuple[str, float]]:
    from wordfreq import top_n_list

    idx: Dict[str, Tuple[str, float]] = {}
    for w in top_n_list("en", int(n_words)):
        ww = str(w).lower()
        if not re.fullmatch(r"[a-z]{3,30}", ww):
            continue
        z = zipf_en(ww)
        if z is None:
            continue

        def put(k: str):
            if not k or len(k) < 3:
                return
            prev = idx.get(k)
            if prev is None or float(z) > float(prev[1]):
                idx[k] = (ww, float(z))

        put(ww)
        if len(ww) >= 4:
            put(ww[1:])
            put(ww[:-1])
        if len(ww) >= 5:
            put(ww[:-2])
        for L in range(4, min(10, len(ww)) + 1):
            put(ww[:L])
    return idx


def repair_en_token(token: str, zipf_threshold: float) -> str:
    t = (token or "").strip().lower()
    if not t or not re.fullmatch(r"[a-z]{3,30}", t):
        return token

    z = zipf_en(t)
    if z is not None and float(z) >= float(zipf_threshold):
        return t

    if not wordfreq_available():
        return t

    cand = _en_spell_index().get(t)
    if cand is None:
        return t
    if float(cand[1]) < float(zipf_threshold):
        return t
    return str(cand[0])


def _zh_char_ngrams(s: str, n: int) -> List[str]:
    if not s or n <= 0:
        return []
    chars = [c for c in s if "\\u4e00" <= c <= "\\u9fff"]
    if len(chars) < n:
        return []
    return ["".join(chars[i : i + n]) for i in range(0, len(chars) - n + 1)]


def tokenize(text: str, language: Literal["en", "zh", "mixed"] = "mixed") -> List[str]:
    if not text:
        return []
    tokens: List[str] = []
    if language in {"en", "mixed"}:
        tokens.extend(re.findall(r"[a-z_]{3,35}", text))
        tokens.extend(re.findall(r"\\d{2,}", text))
    if language in {"zh", "mixed"}:
        zh_seqs = re.findall(r"[\\u4e00-\\u9fff]{2,}", text)
        for seq in zh_seqs:
            tokens.extend(_zh_char_ngrams(seq, 2))
            tokens.extend(_zh_char_ngrams(seq, 3))
        tokens.extend(re.findall(r"[\\u4e00-\\u9fff]{1,6}", text))
    return [t for t in tokens if t and t.strip()]


def _detect_language_for_corpus(texts: List[str]) -> Literal["en", "zh", "mixed"]:
    joined = " ".join([t for t in texts[:200] if isinstance(t, str)])
    if not joined:
        return "mixed"
    cjk = sum(1 for ch in joined if "\\u4e00" <= ch <= "\\u9fff")
    latin = sum(1 for ch in joined if "a" <= ch.lower() <= "z")
    if cjk > 0 and latin == 0:
        return "zh"
    if latin > 0 and cjk == 0:
        return "en"
    if cjk >= latin * 0.8:
        return "zh"
    if latin >= cjk * 0.8:
        return "en"
    return "mixed"


def preprocess_texts(
    texts: List[str],
    extra_stopwords: List[str] | None = None,
    language: Literal["auto", "en", "zh", "mixed"] = "auto",
    emoji_to_words: bool = True,
    fix_mojibake: bool = True,
    repair_nonwords: bool = True,
    repair_zipf_threshold: float = 2.5,
    min_token_len: int = 4,
    keep_nouns_adjs_only: bool = False,
) -> Tuple[List[List[str]], List[str], Literal["en", "zh", "mixed"]]:
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        base_sw = set(ENGLISH_STOP_WORDS)
    except Exception:
        base_sw = set()

    zh_sw = {
        "的", "了", "和", "是", "在", "我", "你", "他", "她", "它", "我们", "你们", "他们",
        "这个", "那个", "这些", "那些", "一个", "一种", "这样", "那样", "然后", "但是", "所以",
        "因为", "如果", "就是", "还是", "没有", "不是", "可能", "可以", "感觉", "觉得", "比较",
    }

    lang = _detect_language_for_corpus(texts) if language == "auto" else language
    stop_set = set()
    if lang in {"en", "mixed"}:
        stop_set |= base_sw
    if lang in {"zh", "mixed"}:
        stop_set |= zh_sw
    if extra_stopwords:
        for s in extra_stopwords:
            if not isinstance(s, str):
                continue
            n = normalize_text(s)
            if not n:
                continue
            stop_set.add(n)

    tokens = []
    joined = []
    for t in texts:
        raw = str(t)
        if fix_mojibake:
            raw = fix_mojibake_if_available(raw)
        if emoji_to_words:
            raw = demojize_if_available(raw)
        n = normalize_text(raw)
        toks = tokenize(n, lang)
        if repair_nonwords and lang in {"en", "mixed"} and wordfreq_available():
            toks = [repair_en_token(x, float(repair_zipf_threshold)) for x in toks]
        toks = [x for x in toks if x not in stop_set]
        try:
            min_token_len_int = int(min_token_len)
        except Exception:
            min_token_len_int = 4
        if min_token_len_int > 1:
            filtered = []
            for x in toks:
                if re.fullmatch(r"[a-z_]+", x):
                    if len(x) >= min_token_len_int:
                        filtered.append(x)
                else:
                    filtered.append(x)
            toks = filtered
        if keep_nouns_adjs_only and lang in {"en", "mixed"}:
            toks = keep_nouns_adjs_heuristic(toks)
        tokens.append(toks)
        joined.append(" ".join(toks))
    return tokens, joined, lang


def high_df_tokens(tokenized_docs: List[List[str]], max_doc_freq: float = 0.8) -> Dict[str, int]:
    if not tokenized_docs:
        return {}
    try:
        max_doc_freq = float(max_doc_freq)
    except Exception:
        max_doc_freq = 0.8
    if max_doc_freq <= 0 or max_doc_freq >= 1:
        return {}

    n_docs = len(tokenized_docs)
    df = {}
    for doc in tokenized_docs:
        for tok in set(doc):
            df[tok] = df.get(tok, 0) + 1

    cutoff = int(np.ceil(max_doc_freq * n_docs))
    return {t: c for t, c in df.items() if c >= cutoff}


def drop_high_df_tokens(tokenized_docs: List[List[str]], max_doc_freq: float = 0.8) -> List[List[str]]:
    banned = set(high_df_tokens(tokenized_docs, max_doc_freq=max_doc_freq).keys())
    if not banned:
        return tokenized_docs
    return [[t for t in doc if t not in banned] for doc in tokenized_docs]


def build_analysis_text(df: pd.DataFrame, text_col: str | None) -> pd.DataFrame:
    out = df.copy()
    if text_col and text_col in out.columns:
        out["analysis_text"] = out[text_col].fillna("").astype(str)
        return out

    title_col = None
    for c in ("Title", "title"):
        if c in out.columns:
            title_col = c
            break
    content_col = None
    for c in ("content", "Content", "text", "Text"):
        if c in out.columns:
            content_col = c
            break

    if title_col is None and content_col is None:
        out["analysis_text"] = ""
        return out

    parts = []
    for _, row in out.iterrows():
        t = str(row.get(title_col, "") or "").strip() if title_col else ""
        c = str(row.get(content_col, "") or "").strip() if content_col else ""
        if t and c:
            if c.lower().startswith(t.lower()) or t.lower() == c.lower():
                parts.append(c)
            else:
                parts.append(f"{t} {c}".strip())
        else:
            parts.append((t or c or "").strip())
    out["analysis_text"] = pd.Series(parts, index=out.index).fillna("").astype(str)
    return out


def gensim_available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("gensim") is not None
    except Exception:
        return False


def run_lda(
    tokenized_docs: List[List[str]],
    num_topics: int = 8,
    passes: int = 10,
    iterations: int = 100,
    alpha: str | float = "auto",
    eta: float = 0.01,
    random_state: int = 42,
    no_below: int = 2,
    no_above: float = 0.9,
    keep_n: int = 20000,
    minimum_probability: float = 0.0,
):
    docs = [d for d in tokenized_docs if d]
    if not docs:
        return None, "No valid tokens for LDA"

    k = int(max(2, min(int(num_topics), 60)))
    if gensim_available():
        from gensim import corpora
        from gensim.models import LdaModel

        dictionary = corpora.Dictionary(docs)
        dictionary.filter_extremes(no_below=int(no_below), no_above=float(no_above), keep_n=int(keep_n))
        if len(dictionary) == 0:
            return None, "Empty dictionary after filtering. Try lowering min_df/stopwords."

        corpus = [dictionary.doc2bow(d) for d in docs]
        if not any(len(x) > 0 for x in corpus):
            return None, "Empty corpus after filtering."

        model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            passes=int(max(1, passes)),
            alpha=alpha,
            eta=float(eta),
            random_state=int(random_state),
            iterations=int(max(30, iterations)),
            eval_every=None,
            minimum_probability=float(minimum_probability),
        )
        phi = model.get_topics()
        terms = [dictionary[i] for i in range(int(phi.shape[1]))]
        doc_topic = np.zeros((len(corpus), k), dtype=float)
        for i, bow in enumerate(corpus):
            dist = model.get_document_topics(bow, minimum_probability=0.0)
            for tid, p in dist:
                doc_topic[i, int(tid)] = float(p)

        return {
            "engine": "gensim",
            "model": model,
            "dictionary": dictionary,
            "corpus": corpus,
            "num_topics": k,
            "phi": phi,
            "terms": terms,
            "doc_topic": doc_topic,
        }, None

    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    texts = [" ".join(d) for d in docs]
    vectorizer = CountVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        min_df=int(no_below),
        max_df=float(no_above),
    )
    X = vectorizer.fit_transform(texts)
    if X.shape[1] == 0:
        return None, "Empty vocabulary after filtering. Try lowering min_df/stopwords."

    alpha_val = None
    try:
        alpha_val = float(alpha)
    except Exception:
        alpha_val = None
    eta_val = float(eta)

    model = LatentDirichletAllocation(
        n_components=k,
        random_state=int(random_state),
        learning_method="batch",
        max_iter=int(max(5, iterations)),
        doc_topic_prior=alpha_val,
        topic_word_prior=eta_val,
    )
    doc_topic = model.fit_transform(X)
    components = np.asarray(model.components_, dtype=float)
    phi = components / np.maximum(components.sum(axis=1, keepdims=True), 1e-12)
    terms = list(vectorizer.get_feature_names_out())

    return {
        "engine": "sklearn",
        "model": model,
        "vectorizer": vectorizer,
        "num_topics": k,
        "phi": phi,
        "terms": terms,
        "doc_topic": doc_topic,
    }, None


def lda_topics(
    lda_bundle,
    topn: int = 50,
) -> List[Dict]:
    model = lda_bundle["model"]
    engine = str(lda_bundle.get("engine") or "gensim")
    rows = []
    k = int(lda_bundle["num_topics"])
    if engine == "gensim":
        for tid in range(k):
            words = model.show_topic(tid, topn=int(topn))
            w_list = [w for w, _ in words]
            p_list = [float(p) for _, p in words]
            rows.append({"topic_id": tid, "words": w_list, "weights": p_list or []})
        return rows

    phi = np.asarray(lda_bundle["phi"], dtype=float)
    terms = list(lda_bundle["terms"])
    for tid in range(k):
        weights = phi[tid]
        idx = np.argsort(weights)[::-1][: int(topn)]
        w_list = [str(terms[int(i)]) for i in idx.tolist()]
        p_list = [float(weights[int(i)]) for i in idx.tolist()]
        rows.append({"topic_id": tid, "words": w_list, "weights": p_list or []})
    return rows


def lda_assignments(lda_bundle) -> Tuple[List[int], List[float]]:
    doc_topic = np.asarray(lda_bundle["doc_topic"], dtype=float)
    if doc_topic.size == 0:
        return [], []
    labels = doc_topic.argmax(axis=1).astype(int).tolist()
    scores = doc_topic.max(axis=1).astype(float).tolist()
    return labels, scores


def lda_mds_word_map(
    lda_bundle,
    prob_threshold: float = 0.0035,
    max_words: int = 500,
    random_state: int = 42,
):
    from sklearn.manifold import MDS
    from sklearn.metrics.pairwise import cosine_distances

    phi = np.asarray(lda_bundle["phi"], dtype=float)
    if phi is None or phi.size == 0:
        return None, "Empty topic-word distribution"

    prob_threshold = float(prob_threshold)
    max_words = int(max_words)
    if max_words <= 0:
        return None, "max_words must be > 0"

    vocab_size = int(phi.shape[1])
    terms = list(lda_bundle["terms"])
    if len(terms) != vocab_size:
        return None, "Vocabulary size mismatch"
    phi_t = phi.T

    best_topic = np.argmax(phi_t, axis=1).astype(int)
    best_prob = np.max(phi_t, axis=1)
    keep_idx = np.where(best_prob >= prob_threshold)[0]
    if keep_idx.size == 0:
        return None, f"No words meet threshold {prob_threshold}"

    if keep_idx.size > max_words:
        keep_idx = keep_idx[np.argsort(best_prob[keep_idx])[::-1][:max_words]]

    X = phi_t[keep_idx]
    D = cosine_distances(X)

    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=int(random_state),
        n_init=1,
        max_iter=300,
        normalized_stress="auto",
    )
    coords = mds.fit_transform(D)

    rows = []
    for j, vid in enumerate(keep_idx.tolist()):
        rows.append(
            {
                "word": str(terms[int(vid)]),
                "topic_id": int(best_topic[int(vid)]),
                "prob": float(best_prob[int(vid)]),
                "x": float(coords[j, 0]),
                "y": float(coords[j, 1]),
            }
        )
    return pd.DataFrame(rows), None


def lda_tune_k_alpha_light(
    tokenized_docs: List[List[str]],
    k_values: List[int],
    alpha_values: List[str | float],
    passes: int = 4,
    iterations: int = 80,
    eta: float = 0.01,
    random_state: int = 42,
    sample_size: int = 800,
    coherence: str = "c_v",
    no_below: int = 2,
    no_above: float = 0.9,
    keep_n: int = 20000,
):
    docs = [d for d in tokenized_docs if d]
    if not docs:
        return None, "No valid tokens for tuning"

    if sample_size and len(docs) > int(sample_size):
        rng = np.random.default_rng(int(random_state))
        idx = rng.choice(len(docs), size=int(sample_size), replace=False)
        docs = [docs[int(i)] for i in idx.tolist()]

    k_values = sorted({int(k) for k in k_values if int(k) >= 2})
    if not k_values:
        return None, "k_values empty"

    cleaned_alphas: List[str | float] = []
    for a in alpha_values:
        if gensim_available():
            if isinstance(a, str) and a in {"auto", "symmetric", "asymmetric"}:
                cleaned_alphas.append(a)
                continue
        try:
            cleaned_alphas.append(float(a))
        except Exception:
            continue
    if not cleaned_alphas:
        return None, "alpha_values empty"

    rows = []
    if gensim_available():
        from gensim import corpora
        from gensim.models import CoherenceModel, LdaModel

        dictionary = corpora.Dictionary(docs)
        dictionary.filter_extremes(no_below=int(no_below), no_above=float(no_above), keep_n=int(keep_n))
        if len(dictionary) == 0:
            return None, "Empty dictionary after filtering"

        corpus = [dictionary.doc2bow(d) for d in docs]
        if not any(len(x) > 0 for x in corpus):
            return None, "Empty corpus after filtering"

        for k in k_values:
            for alpha in cleaned_alphas:
                try:
                    model = LdaModel(
                        corpus=corpus,
                        id2word=dictionary,
                        num_topics=int(k),
                        passes=int(max(1, passes)),
                        alpha=alpha,
                        eta=float(eta),
                        random_state=int(random_state),
                        iterations=int(max(30, iterations)),
                        eval_every=None,
                        minimum_probability=0.0,
                    )
                    cm = CoherenceModel(model=model, texts=docs, dictionary=dictionary, coherence=str(coherence))
                    score = float(cm.get_coherence())
                    rows.append({"k": int(k), "alpha": str(alpha), "coherence": score})
                except Exception as e:
                    rows.append({"k": int(k), "alpha": str(alpha), "coherence": None, "error": str(e)})
    else:
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import CountVectorizer

        texts = [" ".join(d) for d in docs]
        vectorizer = CountVectorizer(
            token_pattern=r"(?u)\b\w+\b",
            min_df=int(no_below),
            max_df=float(no_above),
        )
        X = vectorizer.fit_transform(texts)
        if X.shape[1] == 0:
            return None, "Empty vocabulary after filtering"

        n_docs = X.shape[0]
        if n_docs < 10:
            return None, "Need at least 10 docs for tuning"
        split = int(max(5, np.floor(n_docs * 0.8)))
        X_train = X[:split]
        X_test = X[split:]

        for k in k_values:
            for alpha in cleaned_alphas:
                try:
                    model = LatentDirichletAllocation(
                        n_components=int(k),
                        random_state=int(random_state),
                        learning_method="batch",
                        max_iter=int(max(5, iterations)),
                        doc_topic_prior=float(alpha),
                        topic_word_prior=float(eta),
                    )
                    model.fit(X_train)
                    perp = float(model.perplexity(X_test))
                    rows.append({"k": int(k), "alpha": str(alpha), "coherence": -perp})
                except Exception as e:
                    rows.append({"k": int(k), "alpha": str(alpha), "coherence": None, "error": str(e)})

    df = pd.DataFrame(rows)
    ok = df.dropna(subset=["coherence"]).copy()
    if ok.empty:
        return df, "All tuning runs failed"
    best = ok.sort_values("coherence", ascending=False).iloc[0].to_dict()
    return {"grid": df, "best": best}, None


def run_bertopic_lite(
    texts: List[str],
    n_topics: int = 10,
    ngram_min: int = 1,
    ngram_max: int = 2,
    max_features: int = 5000,
    min_df: int = 2,
    max_df: float = 0.95,
    random_state: int = 42,
    topn_words: int = 50,
):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    cleaned = [t for t in texts]
    if not any(s.strip() for s in cleaned):
        return None, "No valid text for BERTopic-lite"

    vectorizer = TfidfVectorizer(
        ngram_range=(int(ngram_min), int(ngram_max)),
        max_features=int(max_features),
        min_df=int(min_df),
        max_df=float(max_df),
    )
    X = vectorizer.fit_transform(cleaned)
    if X.shape[1] == 0:
        return None, "Empty vocabulary after TF-IDF. Try lowering min_df or adjusting preprocessing."

    n_components = int(min(100, max(2, X.shape[1] - 1)))
    svd = TruncatedSVD(n_components=n_components, random_state=int(random_state))
    Xr = svd.fit_transform(X)

    k = int(max(2, min(int(n_topics), len(cleaned))))
    km = KMeans(n_clusters=k, random_state=int(random_state), n_init="auto")
    labels = km.fit_predict(Xr)

    terms = np.array(vectorizer.get_feature_names_out())
    topic_words = {}
    topn = int(max(10, topn_words))
    for tid in range(k):
        mask = labels == tid
        if not np.any(mask):
            topic_words[tid] = []
            continue
        tfidf_sum = np.asarray(X[mask].sum(axis=0)).reshape(-1)
        top_idx = np.argsort(tfidf_sum)[::-1][: min(topn, len(terms))]
        topic_words[tid] = [str(t) for t in terms[top_idx] if str(t).strip()]

    doc_2d = Xr[:, :2] if Xr.shape[1] >= 2 else np.hstack([Xr, np.zeros((Xr.shape[0], 1))])
    return {
        "labels": labels.tolist(),
        "doc_2d": doc_2d.tolist(),
        "topic_words": topic_words,
        "n_topics": k,
    }, None


def build_topic_evolution_sankey(
    df: pd.DataFrame,
    time_col: str,
    topic_col: str,
    top_words_by_topic: Dict[int, List[str]],
    freq: str = "M",
    similarity_threshold: float = 0.35,
    min_docs_per_period: int = 10,
):
    ts = pd.to_datetime(df[time_col], errors="coerce")
    tmp = df.copy()
    tmp["_ts"] = ts
    tmp = tmp.dropna(subset=["_ts"])
    if tmp.empty:
        return None, "No valid timestamps"
    tmp["_period"] = tmp["_ts"].dt.to_period(freq).dt.to_timestamp()
    tmp["_topic"] = tmp[topic_col]

    counts = tmp.groupby(["_period", "_topic"]).size().reset_index(name="count")
    period_totals = counts.groupby("_period")["count"].sum().to_dict()
    periods = sorted([p for p in counts["_period"].unique().tolist() if int(period_totals.get(p, 0)) >= int(min_docs_per_period)])
    if len(periods) < 2:
        return None, "Not enough periods after filtering"
    counts = counts[counts["_period"].isin(periods)]
    by_period = {p: counts[counts["_period"] == p] for p in periods}

    node_id = {}
    labels = []

    def get_node(p, t):
        key = (p, int(t))
        if key in node_id:
            return node_id[key]
        node_id[key] = len(labels)
        labels.append(f"{p.strftime('%Y-%m')} · T{int(t)+1}")
        return node_id[key]

    sources, targets, values = [], [], []
    for i in range(len(periods) - 1):
        p0, p1 = periods[i], periods[i + 1]
        left = by_period[p0]
        right = by_period[p1]
        for _, r0 in left.iterrows():
            t0 = int(r0["_topic"])
            c0 = int(r0["count"])
            w0 = set(top_words_by_topic.get(t0, []) or [])
            n0 = get_node(p0, t0)
            for _, r1 in right.iterrows():
                t1 = int(r1["_topic"])
                c1 = int(r1["count"])
                w1 = set(top_words_by_topic.get(t1, []) or [])
                if not w0 or not w1:
                    sim = 1.0 if t0 == t1 else 0.0
                else:
                    sim = len(w0 & w1) / max(1, len(w0 | w1))
                if float(sim) < float(similarity_threshold):
                    continue
                weight = float(min(c0, c1)) * float(sim)
                if weight <= 0:
                    continue
                n1 = get_node(p1, t1)
                sources.append(n0)
                targets.append(n1)
                values.append(weight)

    return {"labels": labels, "sources": sources, "targets": targets, "values": values}, None


st.title("Topic Mining Lite (LDA + BERTopic-lite)")

with st.sidebar:
    st.header("Data")
    data_mode = st.radio("数据来源", options=["上传文件", "粘贴表格"], index=0)
    uploaded = None
    pasted_text = ""
    paste_delim: Literal["auto", "tab", "comma", "semicolon"] = "auto"
    paste_header = True
    if data_mode == "上传文件":
        uploaded = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx", "xls"])
    else:
        pasted_text = st.text_area("粘贴表格（支持 Excel 直接复制）", value="", height=220)
        paste_header = st.checkbox("首行是表头", value=True)
        paste_delim = st.selectbox("分隔符", options=["auto", "tab", "comma", "semicolon"], index=0)
    if st.button("Reset App", type="secondary"):
        _reset_all()

if data_mode == "上传文件":
    if uploaded is None:
        st.info("先上传一个包含文本列的 CSV / XLSX 文件，或切换到“粘贴表格”。")
        st.stop()
    df_raw, err = _safe_action("读取上传文件", lambda: read_uploaded_table(uploaded))
    data_descriptor = {"mode": "upload", "name": getattr(uploaded, "name", None), "size": getattr(uploaded, "size", None)}
else:
    if not (pasted_text or "").strip():
        st.info("把要分析的表格粘贴进来（支持 Excel 复制），或切换到“上传文件”。")
        st.stop()
    df_raw, err = _safe_action(
        "解析粘贴表格",
        lambda: parse_pasted_table(pasted_text, delimiter=paste_delim, first_row_header=bool(paste_header)),
    )
    data_descriptor = {"mode": "paste", "hash": hashlib.sha1((pasted_text or "").encode("utf-8")).hexdigest()}
if err:
    st.error(err)
    st.stop()
if df_raw is None or df_raw.empty:
    st.error("文件读取成功但内容为空。")
    st.stop()

st.subheader("Data Preview")
st.dataframe(df_raw.head(50), use_container_width=True)

all_cols = df_raw.columns.tolist()

c_cfg, c_run = st.columns([3, 2])
with c_cfg:
    st.subheader("Config")
    text_col = st.selectbox("Text column", options=["(auto)"] + all_cols, index=0)
    time_col = st.selectbox("Time column (optional, for Sankey)", options=["(none)"] + all_cols, index=0)
    language = st.selectbox("Language", options=["auto", "en", "zh", "mixed"], index=0)
    emoji_to_words = st.checkbox("Emoji 转文字", value=True, help="把 😀 🎉 ❤️ 等转成可建模的词（若环境缺少 emoji 库，会自动跳过）。")
    fix_mojibake = st.checkbox("修复英文乱码(ftfy)", value=True, help="修复 â€™ 这类编码混乱导致的乱码（若环境缺少 ftfy 库，会自动跳过）。")
    keep_nouns_adjs_only = st.checkbox("仅保留名词/形容词（近似）", value=True, help="不装 NLP 大库的前提下用启发式过滤动词/助动词。")
    min_token_len = st.slider("英文最小词长", min_value=2, max_value=10, value=3, step=1)
    repair_nonwords = st.checkbox("修复碎词为真实英文", value=True, help="尝试把 eady→ready / igh→high / ope→hope 等恢复成常见英文词（依赖 wordfreq）。")
    extra_sw = st.text_area("Extra stopwords (comma separated)", value="", height=80)
    high_df = st.slider("删除高频词（出现率≥%）", min_value=50, max_value=100, value=80, step=5, help="100 表示关闭该过滤")
    top_words_n = st.slider("每个主题展示高频词数量", min_value=30, max_value=300, value=80, step=10)
    model_kind = st.radio("Model", options=["LDA", "BERTopic-lite"], horizontal=True)

    if model_kind == "LDA":
        lda_k = st.slider("K topics", min_value=2, max_value=30, value=8, step=1, key="lda_k")
        lda_passes = st.slider("passes", min_value=1, max_value=40, value=10, step=1, key="lda_passes")
        lda_iterations = st.slider("iterations", min_value=30, max_value=400, value=120, step=10, key="lda_iterations")
        lda_alpha = st.selectbox("alpha", options=["auto", "symmetric", "asymmetric", "0.1", "0.5", "1.0"], index=0, key="lda_alpha")
        lda_eta = st.number_input("eta (beta)", min_value=0.0001, max_value=1.0, value=0.01, step=0.01, format="%.4f", key="lda_eta")
        lda_no_below = st.slider("min_df (no_below)", min_value=1, max_value=20, value=2, step=1, key="lda_no_below")
        lda_no_above = st.slider("max_df (no_above)", min_value=0.50, max_value=1.00, value=0.90, step=0.05, key="lda_no_above")
        lda_keep_n = st.slider("keep_n", min_value=2000, max_value=60000, value=20000, step=2000, key="lda_keep_n")
    else:
        bt_k = st.slider("K topics", min_value=2, max_value=30, value=10, step=1)
        bt_max_features = st.slider("max_features", min_value=1000, max_value=20000, value=5000, step=500)
        bt_ngram_min = st.selectbox("ngram min", options=[1, 2], index=0)
        bt_ngram_max = st.selectbox("ngram max", options=[1, 2, 3], index=1)
        bt_min_df = st.slider("min_df", min_value=1, max_value=10, value=2, step=1)
        bt_max_df = st.slider("max_df", min_value=0.50, max_value=1.00, value=0.95, step=0.05)

with c_run:
    st.subheader("Run")
    run_btn = st.button("Run Topic Model", type="primary")

df = build_analysis_text(df_raw, None if text_col == "(auto)" else text_col)
texts = df["analysis_text"].fillna("").astype(str).tolist()
stopwords = [s.strip() for s in (extra_sw or "").split(",") if s.strip()]
tokenized, joined, detected_lang = preprocess_texts(
    texts,
    extra_stopwords=stopwords,
    language=language,
    emoji_to_words=bool(emoji_to_words),
    fix_mojibake=bool(fix_mojibake),
    repair_nonwords=bool(repair_nonwords),
    repair_zipf_threshold=2.5,
    min_token_len=int(min_token_len),
    keep_nouns_adjs_only=bool(keep_nouns_adjs_only),
)
max_doc_freq = float(high_df) / 100.0
if high_df < 100:
    banned_df = high_df_tokens(tokenized, max_doc_freq=max_doc_freq)
    tokenized_after = drop_high_df_tokens(tokenized, max_doc_freq=max_doc_freq)
    if not any(tokenized_after):
        st.warning("高频词过滤导致所有文本为空，已自动跳过该过滤。建议把阈值调高到 90–100。")
    else:
        tokenized = tokenized_after
        with st.expander(f"已移除高频词：{len(banned_df)} 个（出现率≥{high_df}%）", expanded=False):
            if banned_df:
                ban_df = pd.DataFrame(sorted(banned_df.items(), key=lambda x: x[1], reverse=True), columns=["token", "doc_count"])
                st.dataframe(ban_df.head(200), use_container_width=True, hide_index=True)
joined = [" ".join(t) for t in tokenized]

valid_mask = [bool(t) for t in tokenized]
if not any(valid_mask):
    st.error("所有文本清洗后都为空，请检查文本列/停用词设置。")
    st.stop()

df_valid = df.loc[[i for i, ok in enumerate(valid_mask) if ok]].copy()
tokenized_valid = [t for t in tokenized if t]
joined_valid = [t for t in joined if t.strip()]

st.caption(f"Detected language: {detected_lang}")
st.caption(f"Valid docs: {len(df_valid)} / {len(df)}")

current_sig = make_signature(
    {
        "data": data_descriptor,
        "text_col": text_col,
        "language": language,
        "emoji_to_words": bool(emoji_to_words),
        "fix_mojibake": bool(fix_mojibake),
        "repair_nonwords": bool(repair_nonwords),
        "keep_nouns_adjs_only": bool(keep_nouns_adjs_only),
        "min_token_len": int(min_token_len),
        "stopwords": stopwords,
        "high_df": int(high_df),
        "model_kind": model_kind,
    }
)

prev_meta = st.session_state.get("topic_meta") or {}
if prev_meta and prev_meta.get("sig") and prev_meta.get("sig") != current_sig:
    st.session_state.pop("topic_result", None)
    st.session_state.pop("topic_meta", None)
    st.session_state.pop("topic_names", None)
    st.session_state.pop("lda_mds_df", None)
    st.session_state.pop("lda_mds_cfg", None)
    st.session_state.pop("sankey", None)
    st.warning("配置或预处理已变更：请重新点击 Run Topic Model。")
    st.stop()

if model_kind == "LDA":
    tune_metric_title = "Coherence (c_v)" if gensim_available() else "(- Perplexity)"
    with st.expander(f"轻量调参：寻找最优 K 与 alpha（{tune_metric_title}）", expanded=False):
        if not gensim_available():
            st.caption("当前环境未安装 gensim，将使用 scikit-learn LDA 以 Perplexity 近似调参（数值越大越好=负 Perplexity）。")
        t_col1, t_col2, t_col3 = st.columns(3)
        with t_col1:
            tune_k_min = st.slider("K min", min_value=2, max_value=30, value=2, step=1, key="tune_k_min")
        with t_col2:
            tune_k_max = st.slider("K max", min_value=2, max_value=30, value=min(15, 30), step=1, key="tune_k_max")
        with t_col3:
            tune_k_step = st.selectbox("K step", options=[1, 2, 3, 4, 5], index=1, key="tune_k_step")
        tune_sample = st.slider("Sample docs（加速）", min_value=100, max_value=2000, value=800, step=100, key="tune_sample")
        tune_passes = st.slider("Tuning passes（加速）", min_value=1, max_value=10, value=4, step=1, key="tune_passes")
        tune_iterations = st.slider("Tuning iterations（加速）", min_value=30, max_value=200, value=80, step=10, key="tune_iterations")
        alpha_options = ["auto", "symmetric", "asymmetric", "0.1", "0.5", "1.0"] if gensim_available() else ["0.05", "0.1", "0.2", "0.5", "1.0"]
        alpha_default = ["auto", "symmetric", "asymmetric"] if gensim_available() else ["0.1", "0.2", "0.5"]
        tune_alpha_candidates = st.multiselect(
            "Alpha candidates",
            options=alpha_options,
            default=alpha_default,
            key="tune_alpha_candidates",
        )
        if st.button("Run tuning", key="run_tuning_btn"):
            if tune_k_max < tune_k_min:
                st.error("K max 必须 >= K min")
            elif not tune_alpha_candidates:
                st.error("至少选择一个 alpha")
            else:
                k_vals = list(range(int(tune_k_min), int(tune_k_max) + 1, int(tune_k_step)))
                with st.spinner("Running lightweight tuning..."):
                    tune_res, tune_err = _safe_action(
                        "调参",
                        lambda: lda_tune_k_alpha_light(
                            tokenized_valid,
                            k_values=k_vals,
                            alpha_values=tune_alpha_candidates,
                            passes=int(tune_passes),
                            iterations=int(tune_iterations),
                            eta=float(st.session_state.get("lda_eta", 0.01)),
                            random_state=42,
                            sample_size=int(tune_sample),
                            coherence="c_v",
                            no_below=int(st.session_state.get("lda_no_below", 2)),
                            no_above=float(st.session_state.get("lda_no_above", 0.9)),
                            keep_n=int(st.session_state.get("lda_keep_n", 20000)),
                        ),
                    )
                if tune_err:
                    st.error(tune_err)
                else:
                    st.session_state.tuning_result = tune_res
        tune_res = st.session_state.get("tuning_result")
        if tune_res:
            grid = tune_res["grid"]
            best = tune_res["best"]
            st.caption(f"Best: K={best.get('k')} alpha={best.get('alpha')} coherence={best.get('coherence')}")
            show = grid.copy().sort_values(["alpha", "k"])
            st.dataframe(show, use_container_width=True, hide_index=True)
            ok = show.dropna(subset=["coherence"])
            if not ok.empty:
                fig_tune = px.line(ok, x="k", y="coherence", color="alpha", markers=True, title=f"{tune_metric_title} by K and alpha")
                st.plotly_chart(fig_tune, use_container_width=True)
            if st.button("Apply best K/alpha", key="apply_best_btn"):
                try:
                    st.session_state["lda_k"] = int(best.get("k") or st.session_state.get("lda_k", 8))
                except Exception:
                    pass
                try:
                    st.session_state["lda_alpha"] = str(best.get("alpha") or st.session_state.get("lda_alpha", "auto"))
                except Exception:
                    pass
                st.success("已应用推荐参数：请重新点击 Run Topic Model。")

result = st.session_state.get("topic_result")
meta = st.session_state.get("topic_meta")

if run_btn or result is None:
    if model_kind == "LDA":
        alpha_val = lda_alpha
        try:
            alpha_val = float(lda_alpha)
        except Exception:
            pass
        with st.spinner("Training LDA..."):
            lda_bundle, lda_err = _safe_action(
                "运行 LDA",
                lambda: run_lda(
                    tokenized_valid,
                    num_topics=int(lda_k),
                    passes=int(lda_passes),
                    iterations=int(lda_iterations),
                    alpha=alpha_val,
                    eta=float(lda_eta),
                    no_below=int(lda_no_below),
                    no_above=float(lda_no_above),
                    keep_n=int(lda_keep_n),
                ),
            )
        if lda_err:
            st.error(lda_err)
            st.stop()
        topics = lda_topics(lda_bundle, topn=int(top_words_n))
        labels, scores = lda_assignments(lda_bundle)
        st.session_state.topic_result = {"kind": "LDA", "topics": topics, "labels": labels, "scores": scores, "lda_bundle": lda_bundle}
        st.session_state.topic_meta = {"k": int(lda_bundle["num_topics"]), "sig": current_sig, "row_index": df_valid.index.tolist()}
    else:
        with st.spinner("Training BERTopic-lite..."):
            bt_res, bt_err = _safe_action(
                "运行 BERTopic-lite",
                lambda: run_bertopic_lite(
                    joined_valid,
                    n_topics=int(bt_k),
                    ngram_min=int(bt_ngram_min),
                    ngram_max=int(max(bt_ngram_min, bt_ngram_max)),
                    max_features=int(bt_max_features),
                    min_df=int(bt_min_df),
                    max_df=float(bt_max_df),
                    topn_words=int(top_words_n),
                ),
            )
        if bt_err:
            st.error(bt_err)
            st.stop()
        st.session_state.topic_result = {"kind": "BERTopic-lite", **bt_res}
        st.session_state.topic_meta = {"k": int(bt_res["n_topics"]), "sig": current_sig, "row_index": df_valid.index.tolist()}

    result = st.session_state.topic_result
    meta = st.session_state.topic_meta
    time.sleep(0.1)

st.markdown("---")
st.subheader("Topics")

topic_names = st.session_state.get("topic_names")
if topic_names is None:
    k = int(meta.get("k", 0) or 0)
    topic_names = {i: f"Topic {i+1}" for i in range(k)}
    st.session_state.topic_names = topic_names

if result["kind"] == "LDA":
    topics_df = pd.DataFrame(
        [{"Topic_ID": r["topic_id"], "Topic": topic_names.get(r["topic_id"], f"Topic {r['topic_id']+1}"), "Words": ", ".join(r["words"][:20])} for r in result["topics"]]
    )
    st.dataframe(topics_df, use_container_width=True, hide_index=True)
    k_all = int(meta.get("k", 0) or 0)
    topic_doc_counts = pd.Series(result["labels"]).value_counts().reindex(range(k_all), fill_value=0).to_dict()
    for r in result["topics"]:
        tid = int(r["topic_id"])
        doc_n = int(topic_doc_counts.get(tid, 0))
        with st.expander(f"Topic {tid+1} 高频词（前 {len(r['words'])}） · 文档数 {doc_n}", expanded=False):
            detail = pd.DataFrame({"word": r["words"], "weight": r["weights"]})
            st.dataframe(detail, use_container_width=True, hide_index=True)

    counts = pd.Series(result["labels"]).value_counts().reindex(range(k_all), fill_value=0).sort_index()
    bar_df = pd.DataFrame({"Topic_ID": counts.index.astype(int), "Count": counts.values})
    bar_df["Topic"] = bar_df["Topic_ID"].map(lambda i: topic_names.get(int(i), f"Topic {int(i)+1}"))
    fig = px.bar(bar_df, x="Count", y="Topic", orientation="h", title="Docs per Topic")
    st.plotly_chart(fig, use_container_width=True)
else:
    topic_words = result.get("topic_words") or {}
    topics_df = pd.DataFrame(
        [{"Topic_ID": int(tid), "Topic": topic_names.get(int(tid), f"Topic {int(tid)+1}"), "Words": ", ".join(words[:20])} for tid, words in topic_words.items()]
    ).sort_values("Topic_ID")
    st.dataframe(topics_df, use_container_width=True, hide_index=True)
    for tid, words in sorted(topic_words.items(), key=lambda x: int(x[0])):
        with st.expander(f"Topic {int(tid)+1} 高频词（前 {len(words)}）", expanded=False):
            st.dataframe(pd.DataFrame({"word": list(words)}), use_container_width=True, hide_index=True)

    k_all = int(meta.get("k", 0) or 0)
    counts = pd.Series(result["labels"]).value_counts().reindex(range(k_all), fill_value=0).sort_index()
    bar_df = pd.DataFrame({"Topic_ID": counts.index.astype(int), "Count": counts.values})
    bar_df["Topic"] = bar_df["Topic_ID"].map(lambda i: topic_names.get(int(i), f"Topic {int(i)+1}"))
    fig = px.bar(bar_df, x="Count", y="Topic", orientation="h", title="Docs per Topic")
    st.plotly_chart(fig, use_container_width=True)

    doc_2d = pd.DataFrame(result["doc_2d"], columns=["x", "y"])
    doc_2d["Topic_ID"] = result["labels"]
    centroids = doc_2d.groupby("Topic_ID")[["x", "y"]].mean().reset_index()
    centroids["Count"] = doc_2d.groupby("Topic_ID").size().values
    centroids["Topic"] = centroids["Topic_ID"].map(lambda i: topic_names.get(int(i), f"Topic {int(i)+1}"))
    fig2 = px.scatter(centroids, x="x", y="y", size="Count", color="Topic", title="Intertopic Distance Map (SVD 2D)")
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Rename Topics")
rename_cols = st.columns(4)
k = int(meta.get("k", 0) or 0)
for i in range(k):
    with rename_cols[i % 4]:
        topic_names[i] = st.text_input(f"T{i+1}", value=topic_names.get(i, f"Topic {i+1}"), key=f"topic_name_{i}")
st.session_state.topic_names = topic_names

st.markdown("---")
st.subheader("Export")

labels = result["labels"]
row_index = (meta or {}).get("row_index") or df_valid.index.tolist()
df_out = df.loc[row_index].copy()
if len(labels) != len(df_out):
    st.session_state.pop("topic_result", None)
    st.session_state.pop("topic_meta", None)
    st.error("主题结果与当前数据长度不一致：请重新点击 Run Topic Model。")
    st.stop()
df_out["Topic_ID"] = labels
df_out["Topic"] = df_out["Topic_ID"].map(lambda i: topic_names.get(int(i), f"Topic {int(i)+1}"))
if result["kind"] == "LDA":
    df_out["Topic_Score"] = result.get("scores") or [None] * len(df_out)

csv_bytes = df_out.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
st.download_button("Download CSV (with topics)", data=csv_bytes, file_name="topics_output.csv", mime="text/csv")

rows = []
if result["kind"] == "LDA":
    for r in result["topics"]:
        tid = int(r["topic_id"])
        for i, (w, wt) in enumerate(zip(r["words"], r["weights"]), start=1):
            rows.append({"Topic_ID": tid, "Rank": i, "Word": w, "Weight": float(wt)})
else:
    for tid, words in (result.get("topic_words") or {}).items():
        tid_int = int(tid)
        for i, w in enumerate(list(words), start=1):
            rows.append({"Topic_ID": tid_int, "Rank": i, "Word": w})
topics_csv = pd.DataFrame(rows).to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
st.download_button("Download Topic Words CSV", data=topics_csv, file_name="topic_words.csv", mime="text/csv")

st.markdown("---")
st.subheader("Multidimensional Scaling (MDS)")

if result["kind"] != "LDA":
    st.info("MDS 可视化仅适用于 LDA。请先选择 Model = LDA 并运行。")
else:
    c_mds1, c_mds2, c_mds3, c_mds4 = st.columns(4)
    with c_mds1:
        prob_th = st.number_input("特征词概率阈值（≥）", min_value=0.0001, max_value=0.2, value=0.0035, step=0.0005, format="%.4f")
    with c_mds2:
        max_words = st.slider("最大显示词数", min_value=100, max_value=1200, value=500, step=50)
    with c_mds3:
        do_labels = st.checkbox("显示文字标签", value=False)
    with c_mds4:
        label_top_n = st.slider("标注词数", min_value=20, max_value=500, value=150, step=10, disabled=not do_labels)

    if st.button("生成 MDS 图", key="run_lda_mds"):
        df_mds, mds_err = _safe_action(
            "生成 MDS 图",
            lambda: lda_mds_word_map(
                result["lda_bundle"],
                prob_threshold=float(prob_th),
                max_words=int(max_words),
                random_state=42,
            ),
        )
        if mds_err:
            st.error(mds_err)
            st.stop()
        st.session_state.lda_mds_df = df_mds
        st.session_state.lda_mds_cfg = {"prob_th": float(prob_th), "max_words": int(max_words), "labels": bool(do_labels), "label_top_n": int(label_top_n)}

    df_mds = st.session_state.get("lda_mds_df")
    if df_mds is not None and not df_mds.empty:
        cfg = st.session_state.get("lda_mds_cfg") or {}
        df_plot = df_mds.copy()
        df_plot["Topic"] = df_plot["topic_id"].map(lambda i: topic_names.get(int(i), f"Topic {int(i)+1}"))
        fig_mds = px.scatter(
            df_plot,
            x="x",
            y="y",
            color="Topic",
            hover_data=["word", "prob", "topic_id"],
            title=f"MDS Word Map (threshold≥{cfg.get('prob_th')}, max_words={cfg.get('max_words')})",
        )

        if cfg.get("labels"):
            n = int(cfg.get("label_top_n") or 150)
            df_labels = df_plot.sort_values("prob", ascending=False).head(n)
            fig_mds.add_trace(
                go.Scatter(
                    x=df_labels["x"],
                    y=df_labels["y"],
                    mode="text",
                    text=df_labels["word"],
                    textposition="top center",
                    textfont=dict(size=11),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        st.plotly_chart(fig_mds, use_container_width=True)

        mds_csv = df_plot.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("Download MDS CSV", data=mds_csv, file_name="lda_mds_words.csv", mime="text/csv")

st.markdown("---")
st.subheader("Sankey (Topic Evolution)")

if time_col != "(none)" and time_col in df_out.columns:
    freq = st.selectbox("Time grain", options=["M", "Q", "Y"], index=0)
    sim_th = st.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
    min_docs = st.slider("Min docs per period", min_value=1, max_value=200, value=10, step=1)
    if st.button("Build Sankey"):
        if result["kind"] == "LDA":
            lda_words = {r["topic_id"]: r["words"] for r in result["topics"]}
            sank, sank_err = _safe_action(
                "生成演化桑基图",
                lambda: build_topic_evolution_sankey(
                    df_out,
                    time_col=time_col,
                    topic_col="Topic_ID",
                    top_words_by_topic=lda_words,
                    freq=freq,
                    similarity_threshold=float(sim_th),
                    min_docs_per_period=int(min_docs),
                ),
            )
        else:
            bt_words = {int(tid): words for tid, words in (result.get("topic_words") or {}).items()}
            sank, sank_err = _safe_action(
                "生成演化桑基图",
                lambda: build_topic_evolution_sankey(
                    df_out,
                    time_col=time_col,
                    topic_col="Topic_ID",
                    top_words_by_topic=bt_words,
                    freq=freq,
                    similarity_threshold=float(sim_th),
                    min_docs_per_period=int(min_docs),
                ),
            )
        if sank_err:
            st.error(sank_err)
            st.stop()
        st.session_state.sankey = sank

    sank = st.session_state.get("sankey")
    if sank:
        fig_s = go.Figure(
            data=[
                go.Sankey(
                    node=dict(label=sank["labels"], pad=12, thickness=12),
                    link=dict(source=sank["sources"], target=sank["targets"], value=sank["values"]),
                )
            ]
        )
        fig_s.update_layout(height=650)
        st.plotly_chart(fig_s, use_container_width=True)
else:
    st.info("选择一个时间列后可生成演化桑基图。")
