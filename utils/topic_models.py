from __future__ import annotations

import os
import re
import threading as _threading
from typing import Dict, List, Tuple, Literal, Set, Callable, Optional

import numpy as np
import pandas as pd


def persist_base_dir() -> str:
    configured = os.getenv("SOCIALSCRAPER_PERSIST_DIR", "").strip()
    if configured:
        base = configured
    elif os.path.isdir("/data") and os.access("/data", os.W_OK):
        base = "/data/socialscraperapp"
    else:
        base = os.path.join(os.path.expanduser("~"), ".cache", "socialscraperapp")
    try:
        os.makedirs(base, exist_ok=True)
    except Exception:
        pass
    return base


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    s = text.lower()
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"@\w+", " ", s)
    s = re.sub(r"#\w+", " ", s)
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", " ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def _zh_char_ngrams(s: str, n: int) -> List[str]:
    if not s or n <= 0:
        return []
    chars = [c for c in s if "\u4e00" <= c <= "\u9fff"]
    if len(chars) < n:
        return []
    return ["".join(chars[i : i + n]) for i in range(0, len(chars) - n + 1)]


def tokenize(text: str, language: Literal["en", "zh", "mixed"] = "mixed") -> List[str]:
    if not text:
        return []
    tokens: List[str] = []
    if language in {"en", "mixed"}:
        tokens.extend(re.findall(r"[a-z]{3,25}", text))
        tokens.extend(re.findall(r"\d{2,}", text))
    if language in {"zh", "mixed"}:
        zh_seqs = re.findall(r"[\u4e00-\u9fff]{2,}", text)
        for seq in zh_seqs:
            tokens.extend(_zh_char_ngrams(seq, 2))
            tokens.extend(_zh_char_ngrams(seq, 3))
        tokens.extend(re.findall(r"[\u4e00-\u9fff]{1,6}", text))
    return [t for t in tokens if t and t.strip()]


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


def _detect_language_for_corpus(texts: List[str]) -> Literal["en", "zh", "mixed"]:
    joined = " ".join([t for t in texts[:200] if isinstance(t, str)])
    if not joined:
        return "mixed"
    cjk = sum(1 for ch in joined if "\u4e00" <= ch <= "\u9fff")
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
        n = normalize_text(t)
        toks = [x for x in tokenize(n, lang) if x not in stop_set]
        tokens.append(toks)
        joined.append(" ".join(toks))
    return tokens, joined, lang


def drop_high_df_tokens(tokenized_docs: List[List[str]], max_doc_freq: float = 0.8) -> List[List[str]]:
    if not tokenized_docs:
        return tokenized_docs
    try:
        max_doc_freq = float(max_doc_freq)
    except Exception:
        max_doc_freq = 0.8
    if max_doc_freq <= 0 or max_doc_freq >= 1:
        return tokenized_docs

    n_docs = len(tokenized_docs)
    df = {}
    for doc in tokenized_docs:
        for tok in set(doc):
            df[tok] = df.get(tok, 0) + 1

    cutoff = int(np.ceil(max_doc_freq * n_docs))
    banned = {t for t, c in df.items() if c >= cutoff}
    if not banned:
        return tokenized_docs
    return [[t for t in doc if t not in banned] for doc in tokenized_docs]


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


def run_lda(
    tokenized_docs: List[List[str]],
    num_topics: int = 8,
    passes: int = 10,
    alpha: str | float = "auto",
    eta: float = 0.01,
    random_state: int = 42,
):
    from gensim import corpora
    from gensim.models import LdaModel

    docs = [d for d in tokenized_docs if d]
    if not docs:
        return None, "No valid tokens for LDA"

    dictionary = corpora.Dictionary(docs)
    dictionary.filter_extremes(no_below=2, no_above=0.9, keep_n=10000)
    if len(dictionary) == 0:
        return None, "Empty dictionary after filtering. Try lowering min_df/stopwords."

    corpus = [dictionary.doc2bow(d) for d in docs]
    if not any(len(x) > 0 for x in corpus):
        return None, "Empty corpus after filtering."

    k = int(max(2, min(int(num_topics), 50)))
    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=k,
        passes=int(max(1, passes)),
        alpha=alpha,
        eta=float(eta),
        random_state=int(random_state),
        iterations=100,
        eval_every=None,
        minimum_probability=0.0,
    )
    return {"model": model, "dictionary": dictionary, "corpus": corpus, "num_topics": k}, None


def lda_topics(lda_bundle, topn: int = 200) -> List[Dict]:
    model = lda_bundle["model"]
    rows = []
    for tid in range(int(lda_bundle["num_topics"])):
        words = model.show_topic(tid, topn=int(topn))
        rows.append({"topic_id": tid, "words": [w for w, _ in words], "weights": [float(p) for _, p in words]})
    return rows


def lda_assignments(lda_bundle) -> List[int]:
    model = lda_bundle["model"]
    corpus = lda_bundle["corpus"]
    labels = []
    for bow in corpus:
        dist = model.get_document_topics(bow, minimum_probability=0.0)
        best = max(dist, key=lambda x: x[1])[0] if dist else 0
        labels.append(int(best))
    return labels


def lda_mds_word_map(
    lda_bundle,
    prob_threshold: float = 0.0035,
    max_words: int = 500,
    random_state: int = 42,
):
    try:
        from sklearn.manifold import MDS
        from sklearn.metrics.pairwise import cosine_distances
    except Exception as e:
        return None, f"Missing scikit-learn for MDS: {e}"

    model = lda_bundle["model"]
    dictionary = lda_bundle["dictionary"]
    if model is None or dictionary is None:
        return None, "LDA model/dictionary missing"

    try:
        phi = model.get_topics()
    except Exception:
        return None, "Unable to extract topic-word distribution"

    if phi is None or phi.size == 0:
        return None, "Empty topic-word distribution"

    prob_threshold = float(prob_threshold)
    max_words = int(max_words)
    if max_words <= 0:
        return None, "max_words must be > 0"

    vocab_size = int(phi.shape[1])
    terms = [dictionary[i] for i in range(vocab_size)]
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
    passes: int = 5,
    eta: float = 0.01,
    random_state: int = 42,
    sample_size: int = 800,
    coherence: str = "c_v",
):
    from gensim import corpora
    from gensim.models import CoherenceModel, LdaModel

    docs = [d for d in tokenized_docs if d]
    if not docs:
        return None, "No valid tokens for tuning"

    if sample_size and len(docs) > int(sample_size):
        rng = np.random.default_rng(int(random_state))
        idx = rng.choice(len(docs), size=int(sample_size), replace=False)
        docs = [docs[int(i)] for i in idx.tolist()]

    dictionary = corpora.Dictionary(docs)
    dictionary.filter_extremes(no_below=2, no_above=0.9, keep_n=20000)
    if len(dictionary) == 0:
        return None, "Empty dictionary after filtering"

    corpus = [dictionary.doc2bow(d) for d in docs]
    if not any(len(x) > 0 for x in corpus):
        return None, "Empty corpus after filtering"

    k_values = sorted({int(k) for k in k_values if int(k) >= 2})
    if not k_values:
        return None, "k_values empty"

    cleaned_alphas: List[str | float] = []
    for a in alpha_values:
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
                    iterations=80,
                    eval_every=None,
                    minimum_probability=0.0,
                )
                cm = CoherenceModel(model=model, texts=docs, dictionary=dictionary, coherence=str(coherence))
                score = float(cm.get_coherence())
                rows.append({"k": int(k), "alpha": str(alpha), "coherence": score})
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
    topn_words: int = 200,
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


# ============================================================================
#  ACADEMIC-GRADE PREPROCESSING PIPELINE (Multi-Lingual: es / en / zh)
#  严格对应论文 "5.2. Text pre-processing" 的 8 个步骤（已扩展至三语）：
#    a) Tokenization + 长度过滤 (西/英 <4 or >25; 中文 <2 or >12 删除)
#    b) 移除 mentions/hashtags/标点/数字/emoticons/URLs
#    c) 小写 + 重音统一 (西语保 ñ; 中文繁简可选)
#    d) NLTK 官方停用词 + 语料专属停用词 (星期/月份/品牌/店名...)
#    e) POS 标注 → 仅保留 NOUN (名词) + ADJ (形容词) + PROPN (专有名词)
#    f) 词形还原 (lemmatization) → 字典形式 (spaCy; 中文无屈折则 pass-through)
#    g) DF 过滤: >60% 文档出现 或 <10 文档出现 的 token 删除
#    h) N-grams 识别 (NLTK + gensim Phrases) → 高频搭配当实体 (如 valentine's_day / 情人节)
#
#  实际执行顺序 (论文罗列与工程顺序不同, 按可运行的学术顺序):
#    b → c → [分词+POS+lemma] → a → d → e → f → h → g
# ============================================================================


Lang = Literal["es", "en", "zh"]


# ---------------------------------------------------------------------------
# (d) Corpus-specific 停用词 (三语)
# ---------------------------------------------------------------------------

# --- 西班牙语 ---
_CORPUS_SPECIFIC_ES_STOPWORDS: Set[str] = {
    # 星期
    "lunes", "martes", "miércoles", "miercoles", "jueves", "viernes", "sábado", "sabado", "domingo",
    # 月份
    "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre",
    "setiembre", "octubre", "noviembre", "diciembre",
    # 常见时间词
    "hoy", "mañana", "manana", "ayer", "tarde", "noches", "noche", "día", "dia", "dias", "días",
    "semanas", "semana", "meses", "mes", "años", "anos", "año", "ano", "fin", "semana", "findesemana",
    # 常见酒店/餐饮占位词
    "hotel", "hoteles", "hostal", "hostales", "restaurante", "restaurantes", "bar", "bares",
    "café", "cafe", "cafetería", "cafeteria", "habitación", "habitacion", "habitaciones",
    "servicio", "servicios", "personal", "cliente", "clientes", "visita", "visitas",
    "experiencia", "experiencias", "reserva", "reservas", "recepción", "recepcion",
    "marca", "marcas", "cadena", "cadenas",
    # 评价副词
    "muy", "mucho", "muchos", "muchas", "bastante", "poco", "poca", "pocos", "pocas",
    "demasiado", "demasiada", "nada", "casi", "apenas", "solo", "sólo", "solamente",
    "siempre", "nunca", "jamás", "jamas", "todavía", "todavia", "aún", "aun",
    "también", "tambien", "además", "ademas",
}

# --- 英语 ---
_CORPUS_SPECIFIC_EN_STOPWORDS: Set[str] = {
    # 星期 (小写形式)
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    # 月份
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    # 时间词
    "today", "tonight", "tomorrow", "yesterday", "morning", "afternoon", "evening", "night", "day", "days",
    "week", "weeks", "weekend", "weekends", "month", "months", "year", "years",
    # 酒店/餐饮通用 (占位词, 品牌通过 extra_stopwords 填具体名)
    "hotel", "hostel", "inn", "lodge", "resort", "motel", "restaurant", "bar", "cafe", "café",
    "bistro", "diner", "cafeteria", "room", "rooms", "suite", "suites",
    "service", "staff", "guest", "guests", "experience", "booking", "reservation",
    "brand", "chain", "location",
    # 评价副词/无信息量
    "very", "really", "quite", "extremely", "highly", "totally", "absolutely", "completely",
    "just", "only", "almost", "nearly", "always", "never", "often", "sometimes", "usually",
    "still", "yet", "also", "too", "even",
    # 常见美式/英式填充词
    "thing", "things", "bit", "lot", "lots", "kind", "way", "anyway", "anyways",
    # ----- 英文 YouTube Vlog / 字幕场景专有补充 -----
    # 1. YouTube 动作类 + 平台填充（已在 corpus 里，防重复加）
    "video", "videos", "channel", "subscribe", "notification", "notifications",
    "comment", "comments", "like", "likes", "share", "shares", "description", "link", "links",
    "bio", "caption", "captions", "tag", "tags", "algorithm", "algorithms",
    "view", "views", "subscriber", "subscribers", "vlog", "vlogs", "vlogger", "vloggers",
    "youtuber", "youtubers", "youtube",
    # 2. 口语感叹/语气助词（英文口语字幕高频纯语气）
    "bro", "dude", "guys", "guy", "mate", "man", "gosh", "golly", "bless", "heck",
    "omg", "ohh", "ahh", "ahhhh", "ohhhh", "woah", "whoa", "wow", "oops", "yikes",
    "uh", "um", "umm", "ah", "eh", "er", "erm", "hmm", "hm", "mm", "mmm",
    "yep", "nope", "nah", "yeah", "yah", "ya", "yea", "naw",
    "literally", "basically", "apparently", "honestly", "frankly", "obviously",
    "eventually", "actually", "especially", "specifically",
    "bloody", "hell", "hecking", "freaking", "fricking", "sucks", "awesome",
    # 3. 字幕常见噪声（[Music]/[Applause]/>> 残留 token 或单独成词的大写符号）
    "music", "applause", "laughter", "cheering", "crowd", "silence",
    # 4. 伪词（修复失败时的退化兜底，保证 erience 等彻底不出现在主题词表）
    "erience", "erienced", "eriences", "eriencing",
    "periment", "periments", "perimented", "perimenting",
    "citement", "pression", "pressions", "planation", "planations",
    "cellent", "cutive", "cutives", "ercise", "ercises",
    "ample", "amples", "chnology", "chnologies", "mperature", "mperatures",
    "utomation", "rtificial", "nformation", "eception", "eceptionist",
    "ospitality", "ospital", "ustom", "ustomer", "ustomers",
    "xperience", "xperienced", "xperiences", "xperiment", "xperiments",
    "xcitement", "xpression", "xpressions", "xplanation", "xplanations",
    "xcellent", "xecutive", "xecutives", "xercise", "xample", "xamples",
    "echnology", "echnologies", "emperature", "emperatures",
    "perience", "perienced", "periencing", "formation", "tificial",
    "spitality", "stomer", "stomers",
    # 5. 常见无意义 n-gram（字幕重复句）
    "people_people", "thank_thank", "much_much", "time_time", "day_day",
    "good_good", "new_new", "first_first", "way_way", "right_right",
}

# --- 中文 ---
_CORPUS_SPECIFIC_ZH_STOPWORDS: Set[str] = {
    # 星期
    "周一", "周二", "周三", "周四", "周五", "周六", "周日", "星期日", "星期一", "星期二", "星期三",
    "星期四", "星期五", "星期六", "星期天", "周末", "礼拜一", "礼拜二", "礼拜三", "礼拜四",
    "礼拜五", "礼拜六", "礼拜天", "礼拜日",
    # 月份
    "一月", "二月", "三月", "四月", "五月", "六月", "七月", "八月", "九月", "十月", "十一月", "十二月",
    "1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月",
    # 时间
    "今天", "明天", "昨天", "前天", "后天", "早上", "上午", "中午", "下午", "晚上", "夜里",
    "白天", "夜晚", "这周", "上周", "下周", "这个月", "上个月", "下个月", "今年", "去年", "明年",
    "春节", "新年", "元旦", "清明", "五一", "端午", "中秋", "国庆", "圣诞", "情人节", "元宵节",
    # 酒店/餐饮通用
    "酒店", "饭店", "宾馆", "旅馆", "民宿", "旅店", "旅店", "餐厅", "饭馆", "餐馆", "咖啡厅",
    "咖啡馆", "咖啡店", "奶茶店", "酒吧", "房间", "客房", "套房", "大床房", "双床房",
    "服务", "服务员", "前台", "工作人员", "员工", "客户", "客人", "顾客",
    "体验", "感受", "预订", "预定", "订房", "预约", "位置", "地点", "品牌", "连锁",
    # 中文评价/语气助词/连接词 (补充 NLTK 没覆盖的)
    "非常", "特别", "相当", "比较", "十分", "格外", "异常", "极度", "真是", "真的", "确实",
    "简直", "完全", "几乎", "差不多", "大概", "大约", "左右", "一些", "一下", "一点",
    "有点儿", "还是", "还有", "就是", "只是", "不过", "但是", "而且", "并且", "然后",
    "因为", "所以", "如果", "虽然", "但是", "而且", "其实", "结果", "当然", "肯定",
    "本来", "一直", "总是", "经常", "偶尔", "从来", "永远", "马上", "立刻",
    "觉得", "感觉", "以为", "认为", "知道", "看到", "发现", "听说",
}


# ---------------------------------------------------------------------------
# (d) NLTK 官方停用词 (三语懒加载)
# ---------------------------------------------------------------------------

_NLTK_SW_CACHE: Dict[str, Set[str]] = {}


def _nltk_stopwords(lang: Lang) -> Set[str]:
    """加载 NLTK stopwords: es/en/zh."""
    if lang in _NLTK_SW_CACHE:
        return set(_NLTK_SW_CACHE[lang])
    try:
        from nltk.corpus import stopwords
        if lang == "es":
            s = set(stopwords.words("spanish"))
        elif lang == "en":
            s = set(stopwords.words("english"))
        elif lang == "zh":
            try:
                s = set(stopwords.words("chinese"))
            except LookupError:
                s = set()
        else:
            s = set()
    except Exception:
        s = set()
    _NLTK_SW_CACHE[lang] = s
    return set(s)


def _corpus_specific_stopwords(lang: Lang) -> Set[str]:
    """按语言返回内置 corpus-specific 停用词 (论文 d 步)."""
    if lang == "es":
        return set(_CORPUS_SPECIFIC_ES_STOPWORDS)
    if lang == "en":
        return set(_CORPUS_SPECIFIC_EN_STOPWORDS)
    if lang == "zh":
        return set(_CORPUS_SPECIFIC_ZH_STOPWORDS)
    return set()


# ---------------------------------------------------------------------------
# 论文 b + c 通用清洗函数 (三语分发)
# ---------------------------------------------------------------------------

def _emoji_remove_or_replace(text: str) -> str:
    """论文 b 步: 移除 emoji / emoticons。能装 emoji 库就彻底删，不行就正则覆盖常见符号。"""
    try:
        import emoji
        s = emoji.replace_emoji(text, replace=" ")
    except Exception:
        s = text
    # Western emoticons
    s = re.sub(r"[:;=8xX][\-~^oO]?[DPp)(\]\[\\/|3O0*]", " ", s)
    s = re.sub(r"<3+", " ", s)
    s = re.sub(r"\^\^", " ", s)
    # 中日韩颜文字兜底
    s = re.sub(r"[（(]?[>﹡\*＠#・.。~\-_^・][_\-~\.]?[TtOo0\^><DdpPqQ_\-~][)）]?", " ", s)
    return s


def _strip_accents_generic(text: str, preserve_enie: bool = False) -> str:
    """字母语言通用: 去重音. preserve_enie=True (西语) 时保 ñ."""
    import unicodedata
    s = text
    if preserve_enie:
        PLACEHOLDER = "\ue000"
        s = s.replace("ñ", PLACEHOLDER).replace("Ñ", PLACEHOLDER)
    nfkd = unicodedata.normalize("NFKD", s)
    out = []
    for ch in nfkd:
        cat = unicodedata.category(ch)
        if cat in {"Mn", "Me", "Mc"}:
            continue
        out.append(ch)
    s = "".join(out)
    if preserve_enie:
        s = s.replace("\ue000", "ñ")
    return s


def _zh_traditional_to_simplified(text: str) -> str:
    """繁简转换. 没装 OpenCC 时用小型常见映射表兜底 (不保证全覆盖)."""
    try:
        import opencc
        cc = opencc.OpenCC("t2s")
        return cc.convert(text)
    except Exception:
        t2s_map = {
            "這": "这", "那": "那", "邊": "边", "還": "还", "說": "说", "點": "点",
            "裡": "里", "裏": "里", "個": "个", "們": "们", "麼": "么", "對": "对",
            "會": "会", "來": "来", "去": "去", "過": "过", "時": "时", "長": "长",
            "開": "开", "門": "门", "間": "间", "間": "间", "關": "关", "係": "系",
            "員": "员", "動": "动", "電": "电", "視": "视", "燈": "灯", "風": "风",
            "機": "机", "東": "东", "買": "买", "賣": "卖", "讀": "读", "覺": "觉",
            "錯": "错", "錢": "钱", "銀": "银", "飲": "饮", "館": "馆", "務": "务",
            "衛": "卫", "態": "态", "體": "体", "驗": "验", "燒": "烧", "餅": "饼",
            "雞": "鸡", "鴨": "鸭", "魚": "鱼", "蝦": "虾", "貝": "贝", "葉": "叶",
            "醬": "酱", "麵": "面", "鹽": "盐", "豐": "丰", "富": "富",
            "舒": "舒", "適": "适", "牀": "床", "鋪": "铺",
            "飯": "饭", "廳": "厅", "後": "后", "菜": "菜", "單": "单",
            "雙": "双", "層": "层", "潔": "洁", "淨": "净", "氣": "气",
            "溫": "温", "熱": "热", "親": "亲", "切": "切", "響": "响",
            "受": "受", "歡": "欢", "迎": "迎", "認": "认", "為": "为",
            "難": "难", "簡": "简", "單": "单", "準": "准", "備": "备",
            "價": "价", "錢": "钱", "值": "值", "評": "评", "價": "价",
            "論": "论", "確": "确", "實": "实", "現": "现", "點": "点",
            "訂": "订", "房": "房", "間": "间", "床": "床", "鋪": "铺",
            "潔": "洁", "淨": "净", "寬": "宽", "敞": "敞", "安": "安",
            "靜": "静", "吵": "吵", "雜": "杂", "亂": "乱", "舊": "旧",
            "新": "新", "裝": "装", "修": "修", "潢": "潢", "設": "设",
            "計": "计", "設": "设", "備": "备", "配": "配", "套": "套",
            "齊": "齐", "全": "全", "專": "专", "業": "业", "用": "用",
            "心": "心", "耐": "耐", "禮": "礼", "貌": "貌", "主": "主",
            "動": "动", "熱": "热", "情": "情", "客": "客", "氣": "气",
            "氛": "氛", "環": "环", "境": "境", "地": "地", "點": "点",
            "位": "位", "置": "置", "優": "优", "越": "越", "便": "便",
            "利": "利", "交": "交", "通": "通", "購": "购", "物": "物",
            "飲": "饮", "食": "食", "娛": "娱", "樂": "乐", "設": "设",
            "施": "施", "浴": "浴", "室": "室", "淋": "淋", "浴": "浴",
            "花": "花", "灑": "洒", "熱": "热", "水": "水", "壓": "压",
            "力": "力", "穩": "稳", "定": "定", "水": "水", "龍": "龙",
            "頭": "头", "壞": "坏", "損": "损", "壞": "坏", "漏": "漏",
            "水": "水", "堵": "堵", "塞": "塞", "臭": "臭", "味": "味",
            "異": "异", "味": "味", "毛": "毛", "髮": "发", "污": "污",
            "漬": "渍", "灰": "灰", "塵": "尘", "蜘": "蜘", "蛛": "蛛",
            "網": "网", "蟑": "蟑", "螂": "螂", "螞": "蚂", "蟻": "蚁",
            "蚊": "蚊", "子": "子", "蒼": "苍", "蠅": "蝇", "害": "害",
            "蟲": "虫", "滅": "灭", "鼠": "鼠", "騷": "骚", "擾": "扰",
            "噪": "噪", "聲": "声", "音": "音", "打": "打", "擾": "扰",
            "睡": "睡", "眠": "眠", "夢": "梦", "遊": "游", "泳": "泳",
            "池": "池", "健": "健", "身": "身", "房": "房", "桑": "桑",
            "拿": "拿", "按": "按", "摩": "摩", "SPA": "SPA", "兒": "儿",
            "童": "童", "遊": "游", "戲": "戏", "區": "区", "托": "托",
            "兒": "儿", "嬰": "婴", "服": "服", "務": "务", "商": "商",
            "務": "务", "中": "中", "心": "心", "會": "会", "議": "议",
            "室": "室", "宴": "宴", "會": "会", "廳": "厅", "婚": "婚",
            "禮": "礼", "宴": "宴", "請": "请", "酒": "酒", "會": "会",
            "展": "展", "覽": "览", "廳": "厅", "新": "新", "聞": "闻",
            "發": "发", "佈": "布", "會": "会", "签": "签", "到": "到",
            "處": "处", "寄": "寄", "存": "存", "行": "行", "李": "李",
            "服": "服", "務": "务", "洗": "洗", "衣": "衣", "服": "服",
            "務": "务", "熨": "熨", "燙": "烫", "服": "服", "務": "务",
            "快": "快", "遞": "递", "服": "服", "務": "务", "接": "接",
            "送": "送", "機": "机", "服": "服", "務": "务", "泊": "泊",
            "車": "车", "場": "场", "停": "停", "車": "车", "位": "位",
            "電": "电", "梯": "梯", "昇": "升", "降": "降", "機": "机",
            "空": "空", "調": "调", "冷": "冷", "氣": "气", "暖": "暖",
            "氣": "气", "獨": "独", "立": "立", "控": "控", "制": "制",
            "係": "系", "統": "统", "電": "电", "視": "视", "機": "机",
            "頻": "频", "道": "道", "節": "节", "目": "目", "有": "有",
            "線": "线", "電": "电", "視": "视", "衛": "卫", "星": "星",
            "電": "电", "視": "视", "網": "网", "絡": "络", "宽": "宽",
            "帶": "带", "無": "无", "線": "线", "網": "网", "絡": "络",
            "WiFi": "WiFi", "信": "信", "號": "号", "強": "强", "穩": "稳",
            "密": "密", "碼": "码", "登": "登", "錄": "录", "冰": "冰",
            "箱": "箱", "雪": "雪", "櫃": "柜", "保": "保", "險": "险",
            "箱": "箱", "收": "收", "費": "费", "小": "小", "吧": "吧",
            "膠": "胶", "囊": "囊", "咖": "咖", "啡": "啡", "機": "机",
            "電": "电", "熱": "热", "水": "水", "壺": "壶", "茶": "茶",
            "包": "包", "免": "免", "費": "费", "礦": "矿", "泉": "泉",
            "水": "水", "罐": "罐", "裝": "装", "飲": "饮", "料": "料",
            "餅": "饼", "乾": "干", "點": "点", "心": "心", "酒": "酒",
            "類": "类", "啤": "啤", "酒": "酒", "紅": "红", "酒": "酒",
            "白": "白", "酒": "酒", "香": "香", "檳": "槟", "小": "小",
            "食": "食", "零": "零", "食": "食", "煙": "烟", "灰": "灰",
            "缸": "缸", "打": "打", "火": "火", "機": "机", "滅": "灭",
            "煙": "烟", "器": "器", "防": "防", "煙": "烟", "報": "报",
            "警": "警", "器": "器", "門": "门", "鎖": "锁", "鑰": "钥",
            "匙": "匙", "房": "房", "卡": "卡", "門": "门", "禁": "禁",
            "系": "系", "統": "统", "安": "安", "全": "全", "監": "监",
            "控": "控", "系": "系", "統": "统", "24": "24", "小": "小",
            "時": "时", "保": "保", "安": "安", "巡": "巡", "邏": "逻",
            "消": "消", "防": "防", "安": "安", "全": "全", "出": "出",
            "口": "口", "樓": "楼", "梯": "梯", "避": "避", "難": "难",
            "所": "所", "應": "应", "急": "急", "照": "照", "明": "明",
            "燈": "灯", "廣": "广", "播": "播", "系": "系", "統": "统",
            "員": "员", "工": "工", "服": "服", "務": "务", "態": "态",
            "度": "度", "热": "热", "忱": "忱", "友": "友", "善": "善",
            "親": "亲", "切": "切", "主": "主", "動": "动", "耐": "耐",
            "心": "心", "細": "细", "緻": "致", "周": "周", "到": "到",
            "禮": "礼", "貌": "貌", "專": "专", "業": "业", "熟": "熟",
            "練": "练", "有": "有", "素": "素", "養": "养", "英": "英",
            "語": "语", "流": "流", "利": "利", "溝": "沟", "通": "通",
            "無": "无", "障": "障", "礙": "碍", "中": "中", "文": "文",
            "粵": "粤", "語": "语", "日": "日", "語": "语", "韓": "韩",
            "語": "语", "其": "其", "他": "他", "外": "外", "語": "语",
            "解": "解", "決": "决", "問": "问", "題": "题", "迅": "迅",
            "速": "速", "及": "及", "時": "时", "有": "有", "效": "效",
            "投": "投", "訴": "诉", "處": "处", "理": "理", "及": "及",
            "時": "时", "反": "反", "饋": "馈", "改": "改", "善": "善",
            "建": "建", "議": "议", "意": "意", "見": "见", "採": "采",
            "納": "纳", "總": "总", "體": "体", "評": "评", "價": "价",
            "滿": "满", "意": "意", "度": "度", "高": "高", "超": "超",
            "出": "出", "預": "预", "期": "期", "值": "值", "得": "得",
            "推": "推", "薦": "荐", "再": "再", "次": "次", "回": "回",
            "訪": "访", "入": "入", "住": "住", "選": "选", "擇": "择",
            "首": "首", "選": "选", "不": "不", "會": "会", "後": "后",
            "悔": "悔", "遺": "遗", "憾": "憾", "失": "失", "望": "望",
            "再": "再", "也": "也", "不": "不", "會": "会", "來": "来",
            "絕": "绝", "對": "对", "不": "不", "推": "推", "薦": "荐",
            "損": "损", "害": "害", "投": "投", "訴": "诉", "索": "索",
            "賠": "赔", "償": "偿", "退": "退", "款": "款", "賠": "赔",
            "付": "付", "爭": "争", "議": "议", "糾": "纠", "紛": "纷",
            "處": "处", "理": "理", "法": "法", "律": "律", "途": "途",
            "徑": "径", "消": "消", "費": "费", "者": "者", "權": "权",
            "益": "益", "保": "保", "護": "护", "维": "维", "權": "权",
        }
        return "".join(t2s_map.get(ch, ch) for ch in text)


# ---------------------------------------------------------------------------
# YouTube 字幕断行修复 (e Xperience / e Xcitement 合并)
# 根因：YouTube 自动字幕在屏幕宽度不足时会用换行 + 连词省略号等把单词从 1~2 字母处截断，
#       导致 experience / excitement / technology / expression 洗成两个 token：
#       "e xperience"、"t echnology"、"e xcitement"、"e xcellent"、"e xpression"。
# ---------------------------------------------------------------------------
def _repair_youtube_hyphenation(text: str, lang: Lang) -> str:
    """在 (b+c) 清洗结尾再跑一次：解决 YouTube 字幕断行导致的 e xperience 类词首字母截断。"""
    # 最外层兜底: 任何异常返回原始字符串, 不影响其他文档预处理
    try:
        if lang not in {"en", "es"}:
            return text if isinstance(text, str) else ""
        if not isinstance(text, str):
            return ""
        text = text.strip()
        if not text:
            return text

        if lang == "en":
            en_merges = [
                (r"\be\s+(xperience\w*)\b",          r"e\1"),
                (r"\be\s+(xperiment\w*)\b",          r"e\1"),
                (r"\be\s+(xcitement\w*)\b",          r"e\1"),
                (r"\be\s+(xpression\w*)\b",          r"e\1"),
                (r"\be\s+(xplanation\w*)\b",         r"e\1"),
                (r"\be\s+(xcellent\w*)\b",           r"e\1"),
                (r"\be\s+(xecutive\w*)\b",           r"e\1"),
                (r"\be\s+(xercise\w*)\b",            r"e\1"),
                (r"\be\s+(xample\w*)\b",             r"e\1"),
                (r"\bt\s+(echnology\w*)\b",          r"t\1"),
                (r"\bt\s+(emperature\w*)\b",         r"t\1"),
                (r"\bt\s+(ravel\w*)\b",              r"t\1"),
                (r"\bt\s+(our\w*)\b",                r"t\1"),
                (r"\bs\s+(ervice\w*)\b",             r"s\1"),
                (r"\bs\s+(ystem\w*)\b",              r"s\1"),
                (r"\ba\s+(utomation\w*)\b",          r"a\1"),
                (r"\ba\s+(rtificial\w*)\b",          r"a\1"),
                (r"\bc\s+(ustomer\w*)\b",            r"c\1"),
                (r"\bc\s+(heck\w*)\b",               r"c\1"),
                (r"\bc\s+(ompan\w*)\b",              r"c\1"),
                (r"\bi\s+(nformation\w*)\b",         r"i\1"),
                (r"\br\s+(obot\w*)\b",               r"r\1"),
                (r"\br\s+(eception\w*)\b",           r"r\1"),
                (r"\bex\s+(perience\w*)\b",          r"ex\1"),
                (r"\bex\s+(periment\w*)\b",          r"ex\1"),
                (r"\bex\s+(citement\w*)\b",          r"ex\1"),
                (r"\bex\s+(pression\w*)\b",          r"ex\1"),
                (r"\bex\s+(planation\w*)\b",         r"ex\1"),
                (r"\bex\s+(cellent\w*)\b",           r"ex\1"),
                (r"\bex\s+(ecutive\w*)\b",           r"ex\1"),
                (r"\bex\s+(ercise\w*)\b",            r"ex\1"),
                (r"\bex\s+(ample\w*)\b",             r"ex\1"),
                (r"\bte\s+(chnology\w*)\b",          r"te\1"),
                (r"\bte\s+(mperature\w*)\b",         r"te\1"),
                (r"\bin\s+(formation\w*)\b",         r"in\1"),
                (r"\bar\s+(tificial\w*)\b",          r"ar\1"),
                (r"\bcus\s+(tomer\w*)\b",            r"cus\1"),
                (r"\bser\s+(vice\w*)\b",             r"ser\1"),
                (r"\bsys\s+(tem\w*)\b",              r"sys\1"),
                (r"\bho\s+(spitality\w*)\b",         r"ho\1"),
                (r"\bho\s+(tel\w*)\b",               r"ho\1"),
            ]
            for pat, repl in en_merges:
                try:
                    text = re.sub(pat, repl, text)
                except Exception:
                    pass
            en_standalone_fix = {
                "erience": "experience", "erienced": "experienced", "eriences": "experiences",
                "eriencing": "experiencing",
                "periment": "experiment", "periments": "experiments", "perimented": "experimented",
                "perimenting": "experimenting",
                "citement": "excitement",
                "pression": "expression", "pressions": "expressions",
                "planation": "explanation", "planations": "explanations",
                "cellent": "excellent",
                "cutive": "executive", "cutives": "executives",
                "ercise": "exercise", "ercises": "exercises",
                "ample": "example", "amples": "examples",
                "chnology": "technology", "chnologies": "technologies", "chnological": "technological",
                "mperature": "temperature", "mperatures": "temperatures",
                "utomation": "automation",
                "rtificial": "artificial",
                "nformation": "information",
                "eception": "reception", "eceptionist": "receptionist",
                "ospitality": "hospitality",
                "xperience": "experience", "xperienced": "experienced", "xperiences": "experiences",
                "xperiment": "experiment", "xperiments": "experiments",
                "xcitement": "excitement",
                "xpression": "expression", "xpressions": "expressions",
                "xplanation": "explanation", "xplanations": "explanations",
                "xcellent": "excellent",
                "xecutive": "executive", "xecutives": "executives",
                "xercise": "exercise",
                "xample": "example",
                "echnology": "technology", "echnologies": "technologies",
                "emperature": "temperature",
            }
            def _en_patch(m):
                try:
                    w = m.group(0)
                    if not isinstance(w, str):
                        return w if isinstance(w, str) else ""
                    low = w.lower()
                    return en_standalone_fix.get(low, w)
                except Exception:
                    return m.group(0) if isinstance(m.group(0), str) else ""
            try:
                text = re.sub(r"\b[a-z]{5,}\b", _en_patch, text)
            except Exception:
                pass

        else:  # es
            es_merges = [
                (r"\be\s+(xperiencia\w*)\b", r"e\1"),
                (r"\bte\s+(cnolog\w*)\b", r"te\1"),
                (r"\bi\s+(nformaci[oó]n\w*)\b", r"i\1"),
                (r"\be\s+(xcelente\w*)\b", r"e\1"),
            ]
            for pat, repl in es_merges:
                try:
                    text = re.sub(pat, repl, text)
                except Exception:
                    pass
            es_standalone_fix = {
                "xperiencia": "experiencia", "xperiencias": "experiencias",
                "xperiment": "experimento", "xperimentos": "experimentos",
                "cnologia": "tecnologia", "cnologias": "tecnologias",
                "cnología": "tecnología", "cnologías": "tecnologías",
                "xcelente": "excelente",
            }
            def _es_patch(m):
                try:
                    w = m.group(0)
                    if not isinstance(w, str):
                        return w if isinstance(w, str) else ""
                    low = w.lower()
                    return es_standalone_fix.get(low, w)
                except Exception:
                    return m.group(0) if isinstance(m.group(0), str) else ""
            try:
                text = re.sub(r"\b[a-zñáéíóúü]{5,}\b", _es_patch, text)
            except Exception:
                pass

        return text if isinstance(text, str) else str(text)
    except Exception:
        return text if isinstance(text, str) else (str(text) if text is not None else "")


def preprocess_clean_for_spacy_multilang(text: str, lang: Lang, *, zh_t2s: bool = True) -> str:
    """
    论文步骤 b + c (多语言版).

    b) URL / mention / hashtag / emoji / emoticon / 数字 / 标点 清理
    c) 小写 (英/西) + 去重音 (英/西, 西语保 ñ) + 繁简转换 (中文)
    """
    if not isinstance(text, str):
        return ""
    s = text

    # (b.1) URL / www
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    # (b.2) mentions + hashtags (任何语言都去掉)
    s = re.sub(r"@[\w_]+", " ", s)
    s = re.sub(r"#\S+", " ", s)
    # (b.3) emoji + emoticons
    s = _emoji_remove_or_replace(s)
    # (b.4) 纯数字 (电话号码/日期/价格等全部去掉; 中文数字后面用停用词过滤)
    s = re.sub(r"\d+", " ", s)

    if lang in {"es", "en"}:
        # (c) 拉丁语言统一: 小写 + 重音 (西语保 ñ)
        s = s.lower()
        s = _strip_accents_generic(s, preserve_enie=(lang == "es"))
        # (b.5) 标点 & 非允许字符: 英文字母 a-z (西语额外 ñ) + 空格
        allowed = r"a-zñ" if lang == "es" else r"a-z"
        s = re.sub(r"[^" + allowed + r"\s]+", " ", s)
    elif lang == "zh":
        # (c) 繁简转换 (默认开启) + 去掉非 CJK / 中文标点
        if zh_t2s:
            s = _zh_traditional_to_simplified(s)
        # 保留: CJK Unified Ideographs (4e00-9fff) + CJK Ext A + 空格
        # 中文标点/符号/拉丁字母/数字 → 全部删掉 (论文 b 步要求去标点; 混入的英文用停用词不现实,所以直接删)
        s = re.sub(r"[^\u4e00-\u9fff\u3400-\u4dbf\s]+", " ", s)
    # 多余空格压缩 (中文连续汉字间一般无空格, 这个正则对中文安全)
    s = re.sub(r"\s{2,}", " ", s).strip()
    # 最后: YouTube 字幕断行修复 (e Xperience -> experience 等)
    # 必须在 b+c 全部跑完之后再做, 因为它依赖 clean whitespace 后的单字母 token
    if lang in {"en", "es"}:
        s = _repair_youtube_hyphenation(s, lang)
    return s


# ---------------------------------------------------------------------------
# (e + f) spaCy 管道 (三语懒加载) + 中文 POS: 优先 jieba 兜底
# ---------------------------------------------------------------------------

_SPACY_NLP_CACHE: Dict[str, object] = {}
_SPACY_LOCK = _threading.Lock()


def _nltk_pos_tag_en(sentence: str) -> List[Tuple[str, str, str]]:
    """
    英文 POS 兜底 (Streamlit Community Cloud 冷启动环境 spaCy en_core_web_sm 未安装时启用).
    NLTK Penn Treebank POS tag -> Universal POS (NOUN/PROPN/ADJ/VERB/ADV) 粗略映射.
    Lemma 兜底: WordNetLemmatizer + WordNet POS 映射, 失败退化为小写原词.
    """
    try:
        import nltk
    except Exception as e:
        raise RuntimeError("英文 NLTK POS 兜底依赖 nltk, 但未安装. pip install nltk") from e
    _ensure_nltk_resources()
    try:
        from nltk import pos_tag, word_tokenize
        from nltk.stem import WordNetLemmatizer
    except Exception as e:
        raise RuntimeError(f"NLTK 模块缺失: {e}") from e
    lem = WordNetLemmatizer()

    def _wn_pos(p: str) -> str:
        h = p[0].upper() if p else ""
        return {"J": "a", "V": "v", "N": "n", "R": "r"}.get(h, "n")

    def _upos(p: str) -> str:
        p2 = (p or "").upper()
        if p2.startswith("NNP"):
            return "PROPN"
        if p2.startswith("NN"):
            return "NOUN"
        if p2.startswith("JJ"):
            return "ADJ"
        if p2.startswith("VB"):
            return "VERB"
        if p2.startswith("RB"):
            return "ADV"
        return "X"

    try:
        toks = word_tokenize(sentence)
        tagged = pos_tag(toks)
    except Exception:
        toks = sentence.split()
        tagged = [(t, "NN") for t in toks]
    out: List[Tuple[str, str, str]] = []
    for w, tag in tagged:
        try:
            lemma = lem.lemmatize(w.lower(), _wn_pos(tag)) or (w.lower() if w else "")
        except Exception:
            lemma = w.lower() if isinstance(w, str) else ""
        out.append((w, _upos(tag), lemma))
    return out


_NLTK_RES_LOCK = _threading.Lock()
_NLTK_RES_OK = False
_NLTK_DOWNLOAD_DIR_LOCK = _threading.Lock()
_NLTK_DOWNLOAD_DIR_READY = None


def _pick_nltk_download_dir() -> str:
    """Streamlit Community Cloud 沙盒禁止写 ~/nltk_data -> 按优先级挑选一个可写目录, 并写入 os.environ['NLTK_DATA'] + nltk.path."""
    global _NLTK_DOWNLOAD_DIR_READY
    if _NLTK_DOWNLOAD_DIR_READY:
        return _NLTK_DOWNLOAD_DIR_READY
    with _NLTK_DOWNLOAD_DIR_LOCK:
        if _NLTK_DOWNLOAD_DIR_READY:
            return _NLTK_DOWNLOAD_DIR_READY
        candidates = []
        env = os.environ.get("NLTK_DATA", "").strip()
        if env:
            candidates.append(env)
        candidates.extend([
            "/tmp/nltk_data_cc",
            os.path.join(os.path.expanduser("~"), "nltk_data"),
            os.path.join(os.getcwd(), "nltk_data"),
        ])
        picked = None
        for p in candidates:
            try:
                os.makedirs(p, exist_ok=True)
                test = os.path.join(p, ".write_test")
                with open(test, "w", encoding="utf-8") as f:
                    f.write("ok")
                try:
                    os.remove(test)
                except Exception:
                    pass
                picked = p
                break
            except Exception:
                continue
        if picked is None:
            picked = candidates[-1]
            try:
                os.makedirs(picked, exist_ok=True)
            except Exception:
                pass
        import nltk as _nltk_mod
        if picked not in list(_nltk_mod.data.path):
            _nltk_mod.data.path.insert(0, picked)
        os.environ["NLTK_DATA"] = picked
        _NLTK_DOWNLOAD_DIR_READY = picked
        return picked


def _ensure_nltk_resources():
    global _NLTK_RES_OK
    if _NLTK_RES_OK:
        return
    with _NLTK_RES_LOCK:
        if _NLTK_RES_OK:
            return
        import nltk
        download_dir = _pick_nltk_download_dir()
        needed = [
            ("punkt", "tokenizers/punkt"),
            ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
            ("wordnet", "corpora/wordnet"),
            ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
            ("omw-1.4", "corpora/omw-1.4"),
        ]
        for res_name, data_path in needed:
            try:
                nltk.data.find(data_path)
                continue
            except Exception:
                pass
            try:
                nltk.download(res_name, download_dir=download_dir, quiet=True)
            except Exception:
                # Try alternate URL mirror (github raw)
                try:
                    nltk.download(res_name, download_dir=download_dir, quiet=True,
                                  raise_on_error=True)
                except Exception:
                    pass
        _NLTK_RES_OK = True


def _get_spacy_nlp(lang: Lang):
    """按语言加载 spaCy 模型。Streamlit 冷启动缺模型时, en/es 用 NLTK/jieba 替代, 调用方会退回此 fallback。"""
    if lang in _SPACY_NLP_CACHE:
        return _SPACY_NLP_CACHE[lang]
    with _SPACY_LOCK:
        if lang in _SPACY_NLP_CACHE:
            return _SPACY_NLP_CACHE[lang]
        model_pkg_map = {
            "es": ("es_core_news_sm", "es_core_news_sm"),
            "en": ("en_core_web_sm", "en_core_web_sm"),
            "zh": ("zh_core_web_sm", "zh_core_web_sm"),
        }
        pkg, mname = model_pkg_map[lang]
        try:
            import importlib
            mod = importlib.import_module(pkg)
            nlp = mod.load(disable=["ner", "parser"])
        except Exception:
            try:
                import spacy
                nlp = spacy.load(mname, disable=["ner", "parser"])
            except Exception:
                raise RuntimeError(
                    f"spaCy {lang} 模型 {mname} 未安装。请先运行:\n"
                    f"  python -m spacy download {mname}\n"
                    f"(Streamlit Community Cloud 环境将自动退回 NLTK/jieba POS 兜底)"
                )
        try:
            nlp.max_length = 3_000_000
        except Exception:
            pass
        _SPACY_NLP_CACHE[lang] = nlp
        return nlp


def _jieba_pos_tag_cn(sentence: str) -> List[Tuple[str, str]]:
    """
    中文 POS 兜底: jieba.posseg. 我们把 jieba POS tag 映射到 UD Universal POS
    (和 spaCy 的 PROPN/NOUN/ADJ 对齐).
      - jieba 'n'/'nr'/'ns'/'nt'/'nz' → NOUN/PROPN
      - jieba 'a'/'ad'/'an' → ADJ
      - 其它 → X
    """
    try:
        import jieba.posseg as pseg
    except Exception as e:
        raise RuntimeError("中文 POS 依赖 jieba, 但 jieba 未安装。pip install jieba") from e
    out: List[Tuple[str, str]] = []
    for w, f in pseg.cut(sentence):
        ww = w.strip()
        if not ww:
            continue
        head = f[0].lower() if f else ""
        if head == "n":
            # nr=人名 ns=地名 nt=机构 nz=其它专名 → 都当 PROPN/NOUN
            if f.lower() in {"nr", "ns", "nt", "nz"}:
                out.append((ww, "PROPN"))
            else:
                out.append((ww, "NOUN"))
        elif head == "a":
            out.append((ww, "ADJ"))
        else:
            out.append((ww, "X"))
    return out


def _ngrams_gensim_apply(
    docs: List[List[str]],
    min_count: int = 5,
    threshold: float = 10.0,
    scoring: str = "default",
    bigram_only: bool = False,
) -> Tuple[List[List[str]], Dict]:
    """
    论文 h 步: n-grams 识别。用 gensim Phrases 发现高频搭配, 再应用到语料。
    (NLTK ngrams 仅能枚举所有可能, 无法自动筛选"需要当实体"的那些;
    实际学术界论文里 "NLTK n-grams" 通常搭配 PMI/频率阈值做筛选, 这里用最常见的实现方式)

    scoring 说明:
      - "default": gensim 原始 (log-count based) 打分, threshold 推荐 5~20 (默认)
      - "npmi"   : 归一化 PMI, 范围 [-1, 1], threshold 推荐 0.3~0.95
    返回: (合并后的 docs, 报告 dict)
    """
    report: Dict = {"found_bigrams": 0, "found_trigrams": 0}
    if not docs or not any(docs):
        return docs, report

    try:
        from gensim.models import Phrases
    except Exception:
        # gensim 不可用时兜底: NLTK 里取 freq 高的 bigram 硬拼接
        return _ngrams_nltk_fallback(docs, min_count=min_count), report

    non_empty = [d for d in docs if d]
    if not non_empty:
        return docs, report

    sc = str(scoring).lower()
    th_bg = float(threshold)
    # npmi 阈值必须在 [-1, 1]
    if sc == "npmi":
        th_bg = max(-1.0, min(1.0, th_bg))
    th_tg = th_bg * 0.6 if sc != "default" else max(3.0, th_bg * 0.6)

    # Bigram
    bigram = Phrases(
        non_empty,
        min_count=int(min_count),
        threshold=th_bg,
        scoring=sc,
        connector_words=frozenset(_es_stopwords_nltk() | {"san", "santa", "reyes", "dia", "nochevieja"}),
    )
    found_bg = 0
    try:
        phrases = bigram.export_phrases()
        found_bg = len(phrases)
    except Exception:
        pass
    report["found_bigrams"] = found_bg
    docs_bg = [bigram[d] for d in docs]

    if bigram_only:
        return docs_bg, report

    # Trigram (在 bigram 基础上再跑一轮)
    trigram = Phrases(
        [d for d in docs_bg if d],
        min_count=max(2, int(min_count) // 2),
        threshold=th_tg,
        scoring=sc,
    )
    try:
        found_tg = len(trigram.export_phrases())
    except Exception:
        found_tg = 0
    report["found_trigrams"] = found_tg
    docs_tg = [trigram[d] for d in docs_bg]
    return docs_tg, report


def _ngrams_nltk_fallback(docs: List[List[str]], min_count: int = 5) -> List[List[str]]:
    """gensim 不可用时的 NLTK n-grams 兜底实现。"""
    from collections import Counter
    from nltk import bigrams as _nltk_bigrams

    counter: Counter = Counter()
    for d in docs:
        if not d:
            continue
        counter.update(_nltk_bigrams(d))
    keep = {bg for bg, c in counter.items() if c >= int(min_count)}
    if not keep:
        return docs
    merged = []
    for d in docs:
        if not d:
            merged.append([])
            continue
        i = 0
        out = []
        L = len(d)
        while i < L:
            if i + 1 < L and (d[i], d[i + 1]) in keep:
                out.append(f"{d[i]}_{d[i+1]}")
                i += 2
            else:
                out.append(d[i])
                i += 1
        merged.append(out)
    return merged


def preprocess_texts_academic(
    texts: List[str],
    *,
    extra_stopwords: Optional[List[str]] = None,
    language: Literal["auto", "es", "en", "zh", "mixed"] = "es",
    # (a) token length filter: 西/英默认 4~25; 中文默认 2~12 (字符级)
    min_token_len: Optional[int] = None,
    max_token_len: Optional[int] = None,
    # (e) POS filter
    pos_keep: Optional[Set[str]] = None,  # 默认 {"NOUN", "PROPN", "ADJ"}
    # (g) DF filter defaults = 论文推荐: 60% / 10 docs
    df_no_above: Optional[float] = 0.60,
    df_no_below: Optional[int] = 10,
    # (h) n-grams
    enable_ngrams: bool = True,
    ngram_min_count: int = 5,
    ngram_threshold: float = 10.0,
    ngram_bigram_only: bool = False,
    # 中文专属
    zh_use_spacy_not_jieba: bool = True,   # True=优先 spaCy zh; False=jieba
    zh_t2s: bool = True,
    # 性能
    spacy_batch_size: int = 200,
    spacy_n_process: Optional[int] = None,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> Tuple[List[List[str]], List[str], Dict]:
    """
    学术级文本预处理。严格对应论文 5.2 (a)-(h)，已扩展三语 (es / en / zh)。

    Parameters
    ----------
    language : Literal["auto","es","en","zh","mixed"]
        - "es" / "en" / "zh": 强指定语言 (推荐)
        - "auto" : 用字符启发式检测 (CJK vs Latin)
        - "mixed": 如果中英/中西混合, 按段落单检测后可能不准，建议强指定
    min_token_len / max_token_len : int | None
        None 时用语言默认值: 西/英 4~25, 中文 2~12
    """
    # ------------------------------------------------------------------
    # 0. 语言标准化 + 检测 (auto / mixed 用字符启发式)
    # ------------------------------------------------------------------
    lang_raw = str(language).lower()
    if lang_raw in {"auto", "mixed"}:
        # 前 200 条样本检测 CJK 占比
        cjk_cnt = lat_cnt = 0
        import unicodedata as _ud
        for t in texts[:200]:
            if not isinstance(t, str):
                continue
            for ch in t:
                cp = ord(ch)
                if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
                    cjk_cnt += 1
                elif ch.isalpha():
                    lat_cnt += 1
        if cjk_cnt > 0 and cjk_cnt >= lat_cnt:
            lang_norm: Lang = "zh"
        else:
            # 区分英 vs 西: 看是否有西语特有字符 (ñ / ¿ / ¡ / 重音 áéíóúü)
            es_hint = False
            for t in texts[:200]:
                if not isinstance(t, str):
                    continue
                for ch in t.lower():
                    if ch in {"ñ", "¿", "¡", "á", "é", "í", "ó", "ú", "ü"}:
                        es_hint = True
                        break
                if es_hint:
                    break
            lang_norm = "es" if es_hint else "en"
    elif lang_raw == "es":
        lang_norm = "es"
    elif lang_raw == "en":
        lang_norm = "en"
    elif lang_raw == "zh":
        lang_norm = "zh"
    else:
        lang_norm = "en"  # 兜底

    # token 长度默认
    if min_token_len is None:
        min_token_len = 2 if lang_norm == "zh" else 4
    if max_token_len is None:
        max_token_len = 12 if lang_norm == "zh" else 25
    if pos_keep is None:
        pos_keep = {"NOUN", "PROPN", "ADJ"}

    report: Dict = {
        "language_detected": lang_norm,
        "language_requested": language,
        "input_docs": int(len(texts)),
        "step_a_len_filter": {"kept_tokens": 0, "dropped_too_short": 0, "dropped_too_long": 0,
                              "min": int(min_token_len), "max": int(max_token_len)},
        "step_b_cleaned_chars": 0,
        "step_c_enie_preserved": None,
        "step_d_stopwords_removed_tokens": 0,
        "step_e_pos_kept": {p: 0 for p in sorted(pos_keep)},
        "step_e_pos_dropped": 0,
        "step_f_lemmatized": 0,
        "step_g_df_filter": {"banned_high": 0, "banned_low": 0, "kept_vocab": 0},
        "step_h_ngrams": {"found_bigrams": 0, "found_trigrams": 0},
        "output_valid_docs": 0,
        "output_non_empty_docs": 0,
        "output_avg_tokens": 0.0,
        "output_total_tokens": 0,
        "output_vocab_size": 0,
    }

    def _p(p: float, msg: str):
        if progress_cb is not None:
            try:
                progress_cb(float(max(0.0, min(1.0, p))), str(msg))
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 1) 加载 NLP 管道 + 停用词集合
    # ------------------------------------------------------------------
    _p(0.01, f"加载 {lang_norm} NLP 管道与停用词")
    nltk_sw = _nltk_stopwords(lang_norm)
    corpus_sw = _corpus_specific_stopwords(lang_norm)
    extra_sw: Set[str] = set()
    if extra_stopwords:
        for w in extra_stopwords:
            if not isinstance(w, str):
                continue
            ww = w.strip()
            if not ww:
                continue
            # 用清洗后的形式加停用词（保持统一）
            ww_norm = preprocess_clean_for_spacy_multilang(ww, lang_norm, zh_t2s=zh_t2s)
            if ww_norm:
                extra_sw.add(ww_norm)
    full_stopwords: Set[str] = nltk_sw | corpus_sw | extra_sw

    # ------------------------------------------------------------------
    # 2) (b) + (c) 粗清洗 (URL/mention/hashtag/emoji/num/标点 + 小写/重音/繁简)
    # ------------------------------------------------------------------
    _p(0.05, f"[{lang_norm}] 步骤 b/c: 清理 / 小写 / 重音统一 / 繁简统一")
    cleaned_texts: List[str] = []
    total_chars_before = 0
    total_chars_after = 0
    for t in texts:
        if not isinstance(t, str):
            t = ""
        total_chars_before += len(t)
        c = preprocess_clean_for_spacy_multilang(t, lang_norm, zh_t2s=zh_t2s)
        total_chars_after += len(c)
        cleaned_texts.append(c)
    report["step_b_cleaned_chars"] = {"before": total_chars_before, "after": total_chars_after}
    # 西语特殊: 标记 ñ 保留启用
    if lang_norm == "es":
        report["step_c_enie_preserved"] = True
    if lang_norm == "zh":
        report["step_c"] = {"zh_t2s": bool(zh_t2s)}

    # ------------------------------------------------------------------
    # 3) 分词 + POS + lemma (按语言分发)
    # ------------------------------------------------------------------
    _p(0.15, f"[{lang_norm}] 批量分词 + POS + lemma (N={len(cleaned_texts)})")
    N = len(cleaned_texts)
    all_tokens_raw: List[List[Tuple[str, str, str]]] = []  # (text, pos, lemma)

    if lang_norm == "zh" and not bool(zh_use_spacy_not_jieba):
        # 中文走 jieba 路径 (中文 lemma = text 本身)
        processed = 0
        for s in cleaned_texts:
            if not s:
                all_tokens_raw.append([])
                processed += 1
                continue
            tagged = _jieba_pos_tag_cn(s)
            all_tokens_raw.append([(w, pos, w) for w, pos in tagged])
            processed += 1
            cur = 0.15 + 0.30 * (processed / max(1, N))
            if (processed & 31) == 0:
                _p(cur, f"jieba: {processed}/{N}")
    else:
        # spaCy 路径 (es/en, 中文也可以走 spaCy zh); Streamlit CC 冷启动缺模型时 en 自动回退 NLTK/jieba
        nlp = None
        fallback_nltk = False
        try:
            nlp = _get_spacy_nlp(lang_norm)
        except RuntimeError:
            if lang_norm == "en":
                fallback_nltk = True
                _p(0.155, "spaCy en_core_web_sm not installed; falling back to NLTK POS (Streamlit CC cold start).")
            elif lang_norm == "zh":
                fallback_nltk = False
                for s in cleaned_texts:
                    if not isinstance(s, str) or not s.strip():
                        all_tokens_raw.append([])
                        continue
                    tagged = _jieba_pos_tag_cn(s)
                    all_tokens_raw.append([(w, pos, w) for w, pos in tagged])
                _p(0.45, "spaCy zh_core_web_sm not installed; fell back to jieba POS.")
                # jump past spaCy block below by resetting cleaned_texts empty marker
                lang_norm_cast_done_zh_skip_spacy = True
            else:
                raise
        total_chars = sum(len(s) for s in cleaned_texts)
        if (
            ('lang_norm_cast_done_zh_skip_spacy' in locals() and lang_norm_cast_done_zh_skip_spacy)
        ):
            pass  # zh fallback handled above
        elif fallback_nltk and lang_norm == "en":
            processed = 0
            last_progress = 0.15
            N0 = max(1, N)
            for s in cleaned_texts:
                if not isinstance(s, str) or not s.strip():
                    all_tokens_raw.append([])
                else:
                    all_tokens_raw.append(_nltk_pos_tag_en(s))
                processed += 1
                cur = 0.15 + 0.30 * (processed / N0)
                if cur - last_progress >= 0.04:
                    _p(cur, f"NLTK-en (fallback): {processed}/{N}")
                    last_progress = cur
        elif total_chars <= 0:
            for _ in cleaned_texts:
                all_tokens_raw.append([])
        else:
            assert nlp is not None
            n_process = 1
            if spacy_n_process is not None and int(spacy_n_process) > 1:
                n_process = int(spacy_n_process)
            bs = max(10, int(spacy_batch_size))
            try:
                docs_iter = nlp.pipe(
                    cleaned_texts,
                    batch_size=bs,
                    n_process=n_process,
                    disable=["ner", "parser"],
                )
            except TypeError:
                docs_iter = nlp.pipe(cleaned_texts, batch_size=bs)

            processed = 0
            last_progress = 0.15
            for doc in docs_iter:
                per_doc: List[Tuple[str, str, str]] = []
                for tok in doc:
                    per_doc.append((tok.text, tok.pos_, tok.lemma_))
                all_tokens_raw.append(per_doc)
                processed += 1
                cur = 0.15 + 0.30 * (processed / max(1, N))
                if cur - last_progress >= 0.04:
                    _p(cur, f"spaCy-{lang_norm}: {processed}/{N}")
                    last_progress = cur

    # ------------------------------------------------------------------
    # 4) (a) 长度 + (e) POS + (f) lemma + (d) 停用词 合并过滤
    # ------------------------------------------------------------------
    _p(0.50, "步骤 a/d/e/f: 长度 / POS / 停用词 / Lemma")
    min_len = int(min_token_len)
    max_len = int(max_token_len)
    lemmatized_docs: List[List[str]] = []
    total_step_f = 0

    # 不同语言 lemma 再清洗规则
    if lang_norm == "zh":
        def _clean_lemma(lem: str) -> str:
            lem = lem.strip() if isinstance(lem, str) else ""
            # 只保留中文汉字
            return re.sub(r"[^\u4e00-\u9fff\u3400-\u4dbf]", "", lem)
    elif lang_norm == "es":
        # YouTube 断词独立词修正表 (lemma_clean 后如果是残缺的常见西语词根, 补前缀)
        _ES_STANDALONE_LEMMA_FIX = {
            "xperiencia": "experiencia", "xperiencias": "experiencias",
            "xperimento": "experimento", "xperimentos": "experimentos",
            "cnologia": "tecnologia", "cnologias": "tecnologias",
            "xcelente": "excelente",
            "nformacion": "informacion",
            "nformación": "información",
            "rvicio": "servicio", "rvicios": "servicios",
            "ospitalidad": "hospitalidad",
            "abitacion": "habitacion", "abitaciones": "habitaciones",
            "ecepción": "recepcion", "ecepciones": "recepciones",
        }
        def _clean_lemma(lem: str) -> str:
            lem = lem.lower().strip() if isinstance(lem, str) else ""
            # 注意：spaCy lemmatizer 可能还原出带重音的词典形 (habitación)
            # 而我们在步骤 c 已约定去重音 + 保 ñ，所以这里再执行一次
            lem = _strip_accents_generic(lem, preserve_enie=True)
            res = re.sub(r"[^a-zñ]", "", lem)
            # lemma_clean 兜底: 常见 YouTube 断词. 虽然 step_bc 已合并,
            # 但 spaCy/lemmatizer 偶尔会产生二次截断, 所以最后再检查一次
            if res in _ES_STANDALONE_LEMMA_FIX:
                res = _ES_STANDALONE_LEMMA_FIX[res]
            return res
    else:  # en
        # YouTube 断词独立词修正表 (lemma_clean 后如果是残缺的常见词根, 补前缀)
        _EN_STANDALONE_LEMMA_FIX = {
            # 1 字母截断残留
            "erience": "experience", "erienced": "experienced",
            "eriences": "experiences", "eriencing": "experiencing",
            "periment": "experiment", "periments": "experiments",
            "perimented": "experimented", "perimenting": "experimenting",
            "citement": "excitement", "cited": "excited", "citing": "exciting",
            "pression": "expression", "pressions": "expressions",
            "pressed": "expressed", "pressing": "expressing",
            "planation": "explanation", "planations": "explanations",
            "plained": "explained", "plaining": "explaining",
            "cellent": "excellent", "cellence": "excellence",
            "cutive": "executive", "cutives": "executives",
            "cuted": "executed", "cuting": "executing",
            "ercise": "exercise", "ercises": "exercises",
            "ercised": "exercised", "ercising": "exercising",
            "ample": "example", "amples": "examples",
            "amined": "examined", "amining": "examining",
            "amination": "examination", "aminations": "examinations",
            "chnology": "technology", "chnologies": "technologies",
            "chnological": "technological",
            "mperature": "temperature", "mperatures": "temperatures",
            "utomation": "automation", "utomated": "automated",
            "rtificial": "artificial",
            "nformation": "information", "nformed": "informed",
            "nteresting": "interesting", "nterest": "interest",
            "nteraction": "interaction", "nteractions": "interactions",
            "eception": "reception", "eceptionist": "receptionist",
            "ospitality": "hospitality", "ospital": "hospital",
            "ustom": "custom", "ustomer": "customer", "ustomers": "customers",
            # 2 字母截断残留 (xperience / echnology / omatic)
            "xperience": "experience", "xperienced": "experienced",
            "xperiences": "experiences", "xperiencing": "experiencing",
            "xperiment": "experiment", "xperiments": "experiments",
            "xcitement": "excitement", "xcited": "excited", "xciting": "exciting",
            "xpression": "expression", "xpressions": "expressions",
            "xpressed": "expressed", "xpressing": "expressing",
            "xplanation": "explanation", "xplanations": "explanations",
            "xplained": "explained", "xplaining": "explaining",
            "xcellent": "excellent", "xcellence": "excellence",
            "xecutive": "executive", "xecutives": "executives",
            "xercise": "exercise", "xercises": "exercises",
            "xample": "example", "xamples": "examples",
            "echnology": "technology", "echnologies": "technologies",
            "emperature": "temperature", "emperatures": "temperatures",
            "ompanion": "companion", "ompanions": "companions",
            "omfortable": "comfortable", "omfort": "comfort",
            # 3 字母处截断常见残留
            "perience": "experience", "periences": "experiences",
            "perienced": "experienced", "periencing": "experiencing",
            "chnology": "technology",
            "formation": "information",
            "tificial": "artificial",
            "spitality": "hospitality",
            "stomer": "customer", "stomers": "customers",
        }
        def _clean_lemma(lem: str) -> str:
            lem = lem.lower().strip() if isinstance(lem, str) else ""
            # 英文罕见但也做重音保护，防止 loanword (café → cafe) 被误裁掉
            lem = _strip_accents_generic(lem, preserve_enie=False)
            res = re.sub(r"[^a-z]", "", lem)
            if res in _EN_STANDALONE_LEMMA_FIX:
                res = _EN_STANDALONE_LEMMA_FIX[res]
            return res

    for per_doc in all_tokens_raw:
        out: List[str] = []
        for text, pos, lemma in per_doc:
            # 先取 lemma, 再标准化；中文 zh_core_web_sm 无 lemmatizer，lemma='' 时回退到 tok.text
            if isinstance(lemma, str) and lemma.strip() != "":
                lemma_stripped = lemma
            elif isinstance(text, str) and text.strip() != "":
                lemma_stripped = text
            else:
                continue
            lemma_clean = _clean_lemma(lemma_stripped)
            L = len(lemma_clean)
            if L < min_len:
                report["step_a_len_filter"]["dropped_too_short"] += 1
                continue
            if L > max_len:
                report["step_a_len_filter"]["dropped_too_long"] += 1
                continue
            # POS 过滤
            if pos not in pos_keep:
                report["step_e_pos_dropped"] += 1
                continue
            report["step_e_pos_kept"][pos] = report["step_e_pos_kept"].get(pos, 0) + 1
            # 停用词
            if lemma_clean in full_stopwords:
                report["step_d_stopwords_removed_tokens"] += 1
                continue
            out.append(lemma_clean)
            total_step_f += 1
        report["step_a_len_filter"]["kept_tokens"] += len(out)
        lemmatized_docs.append(out)
    report["step_f_lemmatized"] = total_step_f

    # ------------------------------------------------------------------
    # (h) n-grams 合并 (语言无关; connector_words 用各自语言停用词)
    # ------------------------------------------------------------------
    if enable_ngrams:
        _p(0.75, "步骤 h: n-grams 搭配发现与合并 (gensim Phrases)")
        lemmatized_docs, ng_report = _ngrams_gensim_apply_multilang(
            lemmatized_docs,
            min_count=int(ngram_min_count),
            threshold=float(ngram_threshold),
            scoring="default",
            bigram_only=bool(ngram_bigram_only),
            lang=lang_norm,
        )
        report["step_h_ngrams"] = ng_report

    # ------------------------------------------------------------------
    # (g) DF 过滤 (语言无关)
    # ------------------------------------------------------------------
    if df_no_above is not None or df_no_below is not None:
        _p(0.88, f"步骤 g: DF 过滤 (no_above={df_no_above}, no_below={df_no_below})")
        no_above = float(df_no_above) if df_no_above is not None else 1.0
        no_below = int(df_no_below) if df_no_below is not None else 0
        n_docs = len([d for d in lemmatized_docs if d])
        df_counter: Dict[str, int] = {}
        for d in lemmatized_docs:
            for tok in set(d):
                df_counter[tok] = df_counter.get(tok, 0) + 1
        high_cut = int(np.ceil(no_above * n_docs)) if 0 < no_above < 1 else int(1e18)
        low_cut = int(no_below) if no_below > 1 else 0
        banned_high: Set[str] = set()
        banned_low: Set[str] = set()
        for tok, c in df_counter.items():
            if 0 < no_above < 1 and c >= high_cut:
                banned_high.add(tok)
            if low_cut > 0 and c < low_cut:
                banned_low.add(tok)
        banned = banned_high | banned_low
        if banned:
            filtered = []
            for d in lemmatized_docs:
                filtered.append([t for t in d if t not in banned])
            lemmatized_docs = filtered
        report["step_g_df_filter"] = {
            "banned_high": len(banned_high),
            "banned_low": len(banned_low),
            "dropped_above": len(banned_high),
            "dropped_below": len(banned_low),
            "no_above_threshold_pct": round(no_above * 100, 2) if 0 < no_above < 1 else None,
            "no_above_cutoff_docs": high_cut if 0 < no_above < 1 else None,
            "no_below_cutoff_docs": low_cut if low_cut > 0 else None,
            "corpus_docs_for_df": n_docs,
            "vocab_before_filter": len(df_counter),
            "vocab_total_dropped": len(banned),
            "kept_vocab": len(df_counter) - len(banned),
        }
    else:
        report["step_g_df_filter"] = {"banned_high": 0, "banned_low": 0,
                                       "dropped_above": 0, "dropped_below": 0,
                                       "kept_vocab": len(report.get("step_h_ngrams", {}) or {}),
                                       "note": "DF 过滤已禁用"}

    # ------------------------------------------------------------------
    # 输出统计
    # ------------------------------------------------------------------
    _p(0.96, "汇总输出统计")
    valid_docs = len(lemmatized_docs)
    non_empty = 0
    total_toks = 0
    vocab: Set[str] = set()
    for d in lemmatized_docs:
        if d:
            non_empty += 1
            total_toks += len(d)
            vocab.update(d)
    report["output_valid_docs"] = valid_docs
    report["output_non_empty_docs"] = non_empty
    report["output_total_tokens"] = total_toks
    report["output_avg_tokens"] = round(float(total_toks) / float(max(1, non_empty)), 2)
    report["output_vocab_size"] = len(vocab)

    joined = [" ".join(d) for d in lemmatized_docs]
    _p(1.0, f"预处理完成 ({lang_norm})")
    return lemmatized_docs, joined, report


def _ngrams_gensim_apply_multilang(
    docs: List[List[str]],
    min_count: int = 5,
    threshold: float = 10.0,
    scoring: str = "default",
    bigram_only: bool = False,
    lang: Lang = "en",
) -> Tuple[List[List[str]], Dict]:
    """n-grams 的多语言版本: connector_words 用对应语言停用词, 识别 san_valentin / valentines_day 等."""
    report: Dict = {"found_bigrams": 0, "found_trigrams": 0}
    if not docs or not any(docs):
        return docs, report
    try:
        from gensim.models import Phrases
    except Exception:
        return _ngrams_nltk_fallback(docs, min_count=min_count), report

    non_empty = [d for d in docs if d]
    if not non_empty:
        return docs, report

    sc = str(scoring).lower()
    th_bg = float(threshold)
    if sc == "npmi":
        th_bg = max(-1.0, min(1.0, th_bg))
    th_tg = th_bg * 0.6 if sc != "default" else max(3.0, th_bg * 0.6)

    # connector_words: 按语言加专属停用词 (San / Santa / de / of 等)
    lang_extra: Dict[str, Set[str]] = {
        "es": {"san", "santa", "reyes", "dia", "nochevieja", "día", "del", "al", "con"},
        "en": {"valentines", "st", "saint", "new", "year", "day", "of", "and"},
        "zh": set(),  # 中文一般 2 字以上固定搭配, connector 机制作用有限
    }
    connectors = _nltk_stopwords(lang) | _corpus_specific_stopwords(lang) | lang_extra.get(lang, set())

    bigram = Phrases(
        non_empty,
        min_count=int(min_count),
        threshold=th_bg,
        scoring=sc,
        connector_words=frozenset(connectors),
    )
    try:
        report["found_bigrams"] = len(bigram.export_phrases())
    except Exception:
        pass
    docs_bg = [bigram[d] for d in docs]
    if bigram_only:
        return docs_bg, report

    trigram = Phrases(
        [d for d in docs_bg if d],
        min_count=max(2, int(min_count) // 2),
        threshold=th_tg,
        scoring=sc,
    )
    try:
        report["found_trigrams"] = len(trigram.export_phrases())
    except Exception:
        pass
    return [trigram[d] for d in docs_bg], report
