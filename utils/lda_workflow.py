"""
End-to-end LDA workflow (academic preprocessing + sklearn LDA K-sweep + topic naming + visualisation output writer).
"""
from __future__ import annotations

import io, os, re, sys, time, json, traceback
import numpy as np
import pandas as pd
from typing import Optional

# -------- safe imports: return friendly error if missing --------
def _m(pkg, how='pip install', extra=''):
    return f"Missing package `{pkg}` — please run `{how} {pkg}` {extra}".strip()

try:
    from utils.topic_models import (
        _detect_language_for_corpus as detect_lang_auto,
        preprocess_texts_academic,
    )
    HAS_ACADEMIC = True
    ACADEMIC_ERR = None
except Exception as e:
    try:
        from topic_models import (
            _detect_language_for_corpus as detect_lang_auto,
            preprocess_texts_academic,
        )
        HAS_ACADEMIC = True
        ACADEMIC_ERR = None
    except Exception as e2:
        HAS_ACADEMIC = False
        ACADEMIC_ERR = f"utils.topic_models import failed: {e} ; fallback topic_models failed: {e2}"

# Predefined presets (mimic the previous module's LOOSE_FOR_YOUTUBE / PAPER_DEFAULT preset dicts so app code can pass them by string)
LOOSE_FOR_YOUTUBE_PRESET = {
    'mode': 'loose_for_youtube',
    'language': 'auto',
    'min_token_len': 3,
    'max_token_len': 25,
    'pos_keep': {'NOUN', 'PROPN', 'ADJ', 'VERB'},
    'df_no_above': 0.80,
    'df_no_below': 5,
    'enable_ngrams': True,
    'ngram_min_count': 5,
    'ngram_threshold': 10.0,
    'ngram_bigram_only': False,
    'extra_stopwords': [],
}
PAPER_DEFAULT_PRESET = {
    'mode': 'paper_default',
    'language': 'auto',
    'min_token_len': 4,
    'max_token_len': 25,
    'pos_keep': {'NOUN', 'PROPN', 'ADJ'},
    'df_no_above': 0.60,
    'df_no_below': 10,
    'enable_ngrams': True,
    'ngram_min_count': 5,
    'ngram_threshold': 10.0,
    'ngram_bigram_only': False,
    'extra_stopwords': [],
}
_PRESETS_BY_NAME = {'paper_default': PAPER_DEFAULT_PRESET, 'loose_for_youtube': LOOSE_FOR_YOUTUBE_PRESET, 'loose': LOOSE_FOR_YOUTUBE_PRESET}

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams['font.family'] = 'DejaVu Sans'
    rcParams['axes.unicode_minus'] = False
    rcParams['axes.grid'] = True
    rcParams['grid.alpha'] = 0.25
    rcParams['axes.axisbelow'] = True
    rcParams['figure.dpi'] = 130
    HAS_MPL = True
except Exception as e:
    HAS_MPL = False

try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    HAS_SKLEARN = True
except Exception as e:
    HAS_SKLEARN = False

try:
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel
    HAS_GENSIM = True
except Exception as e:
    HAS_GENSIM = False

try:
    import pyLDAvis
    HAS_PYLVIS = True
except Exception as e:
    HAS_PYLVIS = False

try:
    from sklearn.manifold import TSNE
    import sklearn
    HAS_TSNE = True
except Exception as e:
    HAS_TSNE = False

# =========================================================================
# 1. Loader: auto .xls (xlrd) / .xlsx (openpyxl) / csv / parquet
# =========================================================================
def load_any_table(file_path_or_bytesio, original_name_hint=None, sheet_hint=None):
    """
    Returns (DataFrame or None, error_string or None).
    Accepts either a path (str) or a BytesIO (from streamlit uploader) + original_name_hint to sniff extension.
    For Excel (.xls/.xlsx): returns the *largest* non-empty sheet by nrows*ncols (user's xls has a tiny 1143×256 summary and a usable 1061×2 Sheet1).
    """
    if not HAS_SKLEARN:
        return None, _m('scikit-learn')
    if file_path_or_bytesio is None:
        return None, "No file provided"

    # determine name & ext
    name = None
    if isinstance(file_path_or_bytesio, str):
        name = file_path_or_bytesio
        fp = file_path_or_bytesio
    else:
        # file-like
        name = str(getattr(file_path_or_bytesio, 'name', None) or original_name_hint or '').lower()
        fp = file_path_or_bytesio
    ext = os.path.splitext(name)[1].lower()

    try:
        engine = None
        sheets = ['Sheet1']
        if ext in ('.xls',):
            try:
                import xlrd  # noqa: F401
            except Exception as e:
                return None, f"{_m('xlrd==2.0.1')} — required for legacy .xls binary files (openpyxl cannot read .xls). detail: {e}"
            engine = 'xlrd'
            _xls = pd.ExcelFile(fp, engine=engine)
            sheets = list(_xls.sheet_names)
            del _xls
        elif ext in ('.xlsx', '.xlsm'):
            engine = 'openpyxl'
            _xls = pd.ExcelFile(fp, engine=engine)
            sheets = list(_xls.sheet_names)
            del _xls
        elif ext == '.parquet':
            return pd.read_parquet(fp), None
        elif ext == '.csv':
            return pd.read_csv(fp), None
        else:
            try:
                return pd.read_csv(fp), None
            except Exception as e:
                return None, f"Unsupported extension '{ext}' (expected .xls/.xlsx/.csv/.parquet). Try convert to CSV first. detail: {e}"

        if sheet_hint and sheet_hint in sheets:
            return pd.read_excel(fp, engine=engine, sheet_name=sheet_hint), None
        best_sheet, best_score = None, -1
        for sh in sheets:
            try:
                _hdr = pd.read_excel(fp, engine=engine, sheet_name=sh, nrows=0)
                _n_rows = len(pd.read_excel(fp, engine=engine, sheet_name=sh, usecols=[0]).index)
                score = max(0, int(_n_rows)) * max(1, len(_hdr.columns))
            except Exception:
                score = -1
            if score > best_score:
                best_score = score
                best_sheet = sh
        if best_sheet is None:
            return None, f"No non-empty sheet found. Available sheets: {sheets}"
        return pd.read_excel(fp, engine=engine, sheet_name=best_sheet), None
    except Exception as e:
        return None, f"load_any_table failed: {e}"

# =========================================================================
# 2. Preprocess wrapper (8-step academic pipeline)
# =========================================================================
def _find_text_col(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    prefs = ['text', 'analysis_text', 'raw_text', 'content', 'Content', 'Comment_Content', 'replies_content',
             'transcript', 'caption', 'sentence', 'comment', 'body', 'title', 'Title', 'text_en']
    cols_lower = {str(c).lower(): c for c in df.columns}
    for p in prefs:
        if p in df.columns:
            return p
        if p.lower() in cols_lower:
            return cols_lower[p.lower()]
    # fall back to the longest string column
    best_c, best_len = None, -1
    for c in df.columns:
        try:
            s = df[c].astype(str)
            s = s.replace('nan', '').replace('None', '')
            L = s.str.len().fillna(0).sum()
            if int(L) > best_len:
                best_len = int(L)
                best_c = c
        except Exception:
            continue
    return best_c

def preprocess_for_lda(df, mode='loose_for_youtube', extra_stopwords=None, lang='auto',
                       text_col_hint=None, progress_cb=None):
    if not HAS_ACADEMIC:
        return None, None, ACADEMIC_ERR
    if df is None or df.empty:
        return None, None, "Input df is empty"
    text_col = text_col_hint or _find_text_col(df)
    if text_col is None:
        return None, None, "Cannot detect a text column. Explicitly pass text_col_hint."
    texts = df[text_col].fillna('').astype(str).tolist()
    if progress_cb is not None:
        progress_cb(0.01, f"Detected text column: {text_col} ({len(texts)} rows)")
    try:
        if isinstance(mode, dict):
            preset = dict(mode)
        elif mode in _PRESETS_BY_NAME:
            preset = dict(_PRESETS_BY_NAME[mode])
        else:
            preset = dict(PAPER_DEFAULT_PRESET)
        preset['extra_stopwords'] = list(set(list(preset.get('extra_stopwords', [])) + list(extra_stopwords or [])))
        # Map preset dict -> keyword args of `preprocess_texts_academic`
        kw = {}
        key_map = {
            'language': 'language',
            'min_token_len': 'min_token_len',
            'max_token_len': 'max_token_len',
            'pos_keep': 'pos_keep',
            'df_no_above': 'df_no_above',
            'df_no_below': 'df_no_below',
            'enable_ngrams': 'enable_ngrams',
            'ngram_min_count': 'ngram_min_count',
            'ngram_threshold': 'ngram_threshold',
            'ngram_bigram_only': 'ngram_bigram_only',
            'zh_use_spacy_not_jieba': 'zh_use_spacy_not_jieba',
            'zh_t2s': 'zh_t2s',
            'spacy_batch_size': 'spacy_batch_size',
            'spacy_n_process': 'spacy_n_process',
        }
        for k, kw_key in key_map.items():
            if k in preset and preset[k] is not None:
                kw[kw_key] = preset[k]
        if lang not in (None, 'auto'):
            kw['language'] = lang
        elif 'language' not in kw:
            kw['language'] = 'auto'
        tokens_list, clean_texts, report = preprocess_texts_academic(
            texts,
            extra_stopwords=list(preset['extra_stopwords']) if preset.get('extra_stopwords') else None,
            progress_cb=progress_cb,
            **{k: v for k, v in kw.items() if k != 'extra_stopwords'},
        )
        out_df = df.copy()
        out_df['__text_col_used'] = text_col
        out_df['text_preprocessed_academic'] = list(clean_texts)
        out_df['text_preprocessed_academic_tokens'] = [' '.join(t) if t else '' for t in tokens_list]
        out_df['text_preprocessed_academic_token_count'] = [int(len(t)) for t in tokens_list]
        report['_meta_text_col'] = text_col
        report['_meta_n_rows'] = int(len(df))
        report['_meta_nonempty_docs_after_8step'] = int(sum(1 for t in tokens_list if len(t) > 0))
        return out_df, report, None
    except Exception as e:
        return None, None, f"preprocess_for_lda failed: {e}\n{traceback.format_exc()}"

# =========================================================================
# 3. Coherence helpers (gensim C_V + UMass on top of sklearn LDA)
# =========================================================================
def _tokens_from_df(preprocessed_df):
    s = preprocessed_df.get('text_preprocessed_academic_tokens', pd.Series([], dtype=str))
    def _spl(x):
        if x is None:
            return []
        if isinstance(x, list):
            return [str(a) for a in x]
        ss = str(x).strip()
        return ss.split() if ss else []
    return s.map(_spl).tolist()

def _coherence_scores(tokens_train, bows_train, gensim_dict, topic_top_words, coherence_processes=1):
    """topic_top_words: list[list[str]] length K, each entry top salient words for topic i."""
    out = {'c_v': float('nan'), 'u_mass': float('nan')}
    if not HAS_GENSIM or not tokens_train:
        return out
    try:
        cm = CoherenceModel(topics=topic_top_words, texts=tokens_train, dictionary=gensim_dict,
                            coherence='c_v', processes=coherence_processes, window_size=110)
        out['c_v'] = float(cm.get_coherence())
    except Exception:
        pass
    try:
        cm = CoherenceModel(topics=topic_top_words, texts=tokens_train, dictionary=gensim_dict,
                            corpus=bows_train, coherence='u_mass', processes=coherence_processes)
        out['u_mass'] = float(cm.get_coherence())
    except Exception:
        pass
    return out

# =========================================================================
# 4. LDA K sweep (sklearn)
# =========================================================================
def run_lda_k_sweep(preprocessed_df, k_list=None, k_defaults=(3,4,5,6,7,8,9,10,12,15),
                    random_state=42, coherence_processes=1, progress_cb=None):
    """
    Returns (search_df: pd.DataFrame, best_row: dict, turning: dict, err).
    Uses *only sklearn LDA* — avoids gensim LdaMulticore macOS spawn RuntimeError.
    """
    if not HAS_SKLEARN:
        return pd.DataFrame(), {}, {}, _m('scikit-learn')
    if preprocessed_df is None or preprocessed_df.empty:
        return pd.DataFrame(), {}, {}, "Preprocessed df is empty"
    tokens_all = _tokens_from_df(preprocessed_df)
    mask_nonempty = np.array([len(t) > 0 for t in tokens_all], dtype=bool)
    if int(mask_nonempty.sum()) < 6:
        return pd.DataFrame(), {}, {}, "Not enough non-empty docs (<6) after preprocessing to run LDA."
    tokens_train = [t for t, ok in zip(tokens_all, mask_nonempty) if ok]
    try:
        gensim_dct = Dictionary(tokens_train) if HAS_GENSIM else None
    except Exception:
        gensim_dct = None
    vocab_sk = {w: i for i, w in enumerate(gensim_dct.token2id.keys())} if gensim_dct is not None else None
    if vocab_sk is None or not vocab_sk:
        return pd.DataFrame(), {}, {}, "Empty vocab after preprocessing (all words removed)."
    texts_joined = [' '.join(t) for t in tokens_train]
    vec = CountVectorizer(vocabulary=vocab_sk, token_pattern=r"(?u)\b\w+\b")
    try:
        tf = vec.fit_transform(texts_joined)
    except Exception as e:
        return pd.DataFrame(), {}, {}, f"CountVectorizer failed: {e}"
    feat = np.array(vec.get_feature_names_out())
    # TF sum / doc lengths for perplexity (already handled internally by sklearn.perplexity)
    N = tf.shape[0]
    V = len(feat)
    K_LIST = list(k_list) if k_list is not None else list(k_defaults)
    K_LIST = sorted({int(k) for k in K_LIST if int(k) >= 2 and int(k) <= max(60, int(N**0.5)*3)})
    rows = []
    for idx, K in enumerate(K_LIST):
        t0 = time.time()
        if progress_cb is not None:
            progress_cb((idx+0.05)/len(K_LIST), f"K sweep: training K={K} ({idx+1}/{len(K_LIST)})")
        try:
            lda = LatentDirichletAllocation(n_components=int(K), learning_method='batch',
                                            max_iter=50, learning_offset=50.0,
                                            random_state=random_state, n_jobs=1, evaluate_every=-1)
            theta = lda.fit_transform(tf)
            t = time.time() - t0
            perp = float(lda.perplexity(tf))
            loglik = float(lda.score(tf))
            top_idx = np.argsort(-lda.components_, axis=1)[:, :25]
            top_words = [[feat[j] for j in row] for row in top_idx]
            # Build gensim-compatible BoW: [(token_id, count_int), ...] per doc (COO sparse expansion)
            bows = []
            for i in range(N):
                r = tf[i].tocoo()
                bows.append(list(zip(r.col.astype(int).tolist(), r.data.astype(int).tolist())))
            coh = _coherence_scores(tokens_train, bows, gensim_dct, [w[:15] for w in top_words],
                                    coherence_processes=coherence_processes)
            rows.append({'K': int(K), 'perplexity': perp, 'log_likelihood': loglik,
                         'coherence_cv': coh['c_v'], 'coherence_umass': coh['u_mass'],
                         'train_time_s': round(t, 3), 'N_docs_trained': int(N), 'V_vocab': int(V)})
        except Exception as e:
            rows.append({'K': int(K), 'perplexity': float('nan'), 'log_likelihood': float('nan'),
                         'coherence_cv': float('nan'), 'coherence_umass': float('nan'),
                         'train_time_s': 0.0, 'N_docs_trained': int(N), 'V_vocab': int(V),
                         'error': str(e)})
        if progress_cb is not None:
            progress_cb((idx+1)/len(K_LIST), f"K sweep: K={K} done. Perplexity={rows[-1].get('perplexity', float('nan')):.0f}, C_V={rows[-1].get('coherence_cv', float('nan')):.3f}")
    search_df = pd.DataFrame(rows)
    # Turning point (kneedle-like)
    turning = {}
    try:
        valid = search_df.dropna(subset=['coherence_cv', 'perplexity']).sort_values('K')
        if len(valid) >= 4:
            K_arr = valid['K'].values.astype(int)
            CV_arr = valid['coherence_cv'].values.astype(float)
            PP_arr = valid['perplexity'].values.astype(float)
            def _kneedle(xs, ys, higher_better=True):
                ys2 = ys if higher_better else -ys
                x0, xn = xs[0], xs[-1]
                y0, yn = ys2[0], ys2[-1]
                dx = xn - x0 if xn > x0 else 1
                m = (yn - y0) / dx
                baseline = y0 + m * (xs - x0)
                dists = ys2 - baseline
                idx = int(np.argmax(dists))
                return int(xs[idx])
            turning['cv_knee_K'] = int(_kneedle(K_arr, CV_arr, True))
            turning['perplexity_elbow_K'] = int(_kneedle(K_arr, PP_arr, False))
            turning['cv_best_K'] = int(K_arr[int(np.argmax(CV_arr))])
            turning['perplexity_best_K'] = int(K_arr[int(np.argmin(PP_arr))])
            dCV = np.diff(CV_arr)
            first_big = int(np.argmax(dCV)) if len(dCV) else 0
            turning['first_big_cv_jump_at_pair'] = (int(K_arr[first_big]), int(K_arr[first_big+1])) if len(dCV) else (None, None)
            candidates = sorted({turning['cv_knee_K'], turning['perplexity_elbow_K'], turning['cv_best_K']})
            candidates_in_range = [c for c in candidates if 4 <= c <= 14] or candidates
            turning['recommended_K_composite'] = int(np.median(candidates_in_range)) if candidates_in_range else int(turning['cv_best_K'])
    except Exception as e:
        turning['_error'] = str(e)
    best_row = {}
    try:
        cvdf = search_df.dropna(subset=['coherence_cv'])
        if not cvdf.empty:
            best_row = cvdf.sort_values('coherence_cv', ascending=False).iloc[0].to_dict()
            for kkk in list(best_row.keys()):
                if isinstance(best_row[kkk], (np.floating,)):
                    best_row[kkk] = float(best_row[kkk])
                elif isinstance(best_row[kkk], (np.integer,)):
                    best_row[kkk] = int(best_row[kkk])
    except Exception:
        pass
    return search_df, best_row, turning, None

# =========================================================================
# 5. Train LDA K=X, assign θ, run heuristic naming
# =========================================================================
def _abbreviate(s, n=32):
    s = str(s)
    return s if len(s) <= n else s[:n-1] + "…"

def _heuristic_name_topic(topic_id, top15_words, repr_snippets):
    """Rule-based EN topic labelling from top-15 words + top-2 repr text snippets."""
    W = [str(w).lower() for w in top15_words]
    S = set(W)
    txt_all = " ".join(repr_snippets).lower()
    # language clues
    CLUES = [
        ('Tagalog (Filipino)', {'ayan','ang','tak','yung','dito','ano','kami','tayo','ito','mga','lang','po','sa','ko','ng','na'}),
        ('Spanish', {'que','los','las','nos','del','por','con','para','como','una','mas','pero','muy','esto','esta','este','estos','ser','estar'}),
        ('Italian', {'che','non','una','musica','con','qui','pero','era','hai','due','del','il','lo','anche','sono','cosa'}),
        ('Afrikaans', {'die','nie','dit','daar','het','yan','vir','kan','van','baie','dis','hulle','ons','jy','ek','is','was','sal'}),
        ('Malay/Indonesian', {'yang','dan','kita','kamu','dia','ini','itu','ya','saya','untuk','dengan','pada','dari','juga','lebih','sudah'}),
    ]
    lang = []
    for lab, s in CLUES:
        h = sum(1 for w in W if w in s)
        if h >= 3:
            lang.append((lab, h))
    if lang:
        top_lang = sorted(lang, key=lambda x: -x[1])[0][0]
        safe = top_lang.replace(' ','')
        safe = ''.join(ch if ch.isalnum() else '_' for ch in safe)
        prefix_top = f"T{topic_id:02d}_MULTI_{safe}"
    else:
        prefix_top = f"T{topic_id:02d}_EN"
    # semantic keywords
    def has(*ws):
        return any((w in S) for w in ws)
    suffix = "Mixed_Travel_Commentary"
    if has('robot','robotic','humanoid') and has('human','future','machine','job'):
        suffix = "Robot_vs_Human_Service_Jobs_Future"
    elif has('technology','hospitality','industry','solution','communication','system','market'):
        suffix = "B2B_HospitalityTech_SalesPitch_Solution"
    elif has('customer','operation','question','datum','model','system','instructor','hotel'):
        suffix = "HotelSchool_Lecture_CRM_Ops_Training"
    elif has('book','launch','future','industry','business','talent','course','development'):
        suffix = "BookLaunch_Panel_AI_Future_of_Work"
    elif has('tokyo','japan','japanese','station','toilet','tokyo tower','yen'):
        suffix = "Tokyo_Japan_Travel_Vlog"
    elif has('stay','modern','traveler','perfect','convenience','comfort','trip','seamless','luxury'):
        suffix = "HennNa_Chain_Structured_Hotel_Review"
    elif has('breakfast','amenity','facility','option','review','great','free','convenient'):
        suffix = "Hotel_Room_Amenities_Breakfast_Review"
    elif has('friend','coffee','order','kagoshima','name','welcome','ready','bed','correct'):
        suffix = "Kagoshima_RobotFrontDesk_Dialogue_ServiceRants"
    elif has('universal','orlando','aventura','food','staycation','cool','stuff','amazing','little'):
        suffix = "Orlando_Universal_Staycation_Lifestyle_Vlog"
    elif has('news','cigarette','tax','las vegas','panel','project','criminologist','door','big'):
        suffix = "NewsPanel_CigaretteTax_LasVegasAI_Hotel"
    elif has('podcast','revenate','seo','llm','direct booking','cmo','website','drive','traffic'):
        suffix = "Podcast_Revenate_SEO_for_LLM_DirectBooking"
    # wrap
    short = f"{prefix_top}_{suffix}"
    descr_parts = [f"Top-10 words: {', '.join(W[:10])}."]
    if lang:
        descr_parts.append(f"Detected multilingual function-word signal: {lang}.")
    descr_parts.append(f"Auto-suffix label={suffix}. Review top-words + representative docs below to refine name.")
    return short, " ".join(descr_parts)

def train_lda_and_assign(preprocessed_df, K, random_state=42, topn_words=25, auto_name=True,
                         coherence_processes=1, progress_cb=None):
    if not HAS_SKLEARN:
        return None, None, None, None, None, _m('scikit-learn')
    if preprocessed_df is None or preprocessed_df.empty:
        return None, None, None, None, None, "Empty preprocessed_df"
    K = int(K)
    if K < 2:
        return None, None, None, None, None, "K must be >=2"
    tokens_all = _tokens_from_df(preprocessed_df)
    N = len(tokens_all)
    mask_nonempty = np.array([len(t) > 0 for t in tokens_all], dtype=bool)
    if int(mask_nonempty.sum()) < max(6, K):
        return None, None, None, None, None, f"Not enough non-empty docs ({int(mask_nonempty.sum())}) to train K={K}."
    tokens_train = [t for t, ok in zip(tokens_all, mask_nonempty) if ok]
    train_rows = np.where(mask_nonempty)[0]
    try:
        gensim_dct = Dictionary(tokens_train) if HAS_GENSIM else None
    except Exception:
        gensim_dct = None
    vocab_sk = {w: i for i, w in enumerate(gensim_dct.token2id.keys())} if gensim_dct is not None else None
    if not vocab_sk:
        return None, None, None, None, None, "Empty vocab."
    texts_joined = [' '.join(t) for t in tokens_train]
    vec = CountVectorizer(vocabulary=vocab_sk, token_pattern=r"(?u)\b\w+\b")
    tf = vec.fit_transform(texts_joined)
    feat = np.array(vec.get_feature_names_out())
    V = len(feat)
    if progress_cb is not None:
        progress_cb(0.05, f"sklearn LDA K={K} max_iter=80 batch ({int(mask_nonempty.sum())} docs, V={V})")
    try:
        lda = LatentDirichletAllocation(n_components=int(K), learning_method='batch', max_iter=80,
                                        learning_offset=50.0, random_state=random_state,
                                        n_jobs=1, evaluate_every=-1)
        theta_train = lda.fit_transform(tf)
        perp = float(lda.perplexity(tf))
        loglik = float(lda.score(tf))
        top_idx = np.argsort(-lda.components_, axis=1)[:, :topn_words]
        top_words = [[feat[j] for j in row] for row in top_idx]
        top_probs = [[float(lda.components_[i, j] / lda.components_[i].sum()) for j in row] for i, row in enumerate(top_idx)]
        if progress_cb is not None:
            progress_cb(0.6, f"LDA fit done. Perplexity={perp:.1f}. Naming topics...")
        # All-theta for all N docs (fill 0 for empty)
        theta_all = np.zeros((N, K), dtype=np.float64)
        theta_all[train_rows] = theta_train
        best_id = np.full(N, -1, dtype=int)
        best_p = np.full(N, np.nan, dtype=np.float64)
        for i in train_rows:
            kk = int(np.argmax(theta_all[i]))
            best_id[i] = kk
            best_p[i] = float(theta_all[i, kk])
        counts = np.bincount(best_id[best_id >= 0], minlength=K)
    except Exception as e:
        return None, None, None, None, None, None, None, f"train_lda_and_assign fit failed: {e}\n{traceback.format_exc()}"
    # Repr docs + naming
    repr_docs_per_topic = []
    topic_name_short = []
    topic_name_long = []
    text_col_in_original = preprocessed_df.get('__text_col_used', pd.Series([None]*N)).iloc[0] if '__text_col_used' in preprocessed_df.columns else None
    original_texts_series = None
    if text_col_in_original and text_col_in_original in preprocessed_df.columns:
        original_texts_series = preprocessed_df[text_col_in_original]
    else:
        # fall back any text-like
        for c in ['text','analysis_text','raw_text','content','Content']:
            if c in preprocessed_df.columns:
                original_texts_series = preprocessed_df[c]; break
    for k in range(K):
        order = np.argsort(-theta_train[:, k])[:2]
        entries = []
        snippets = []
        for train_i in order:
            orig_i = int(train_rows[train_i])
            prob = float(theta_train[train_i, k])
            if original_texts_series is not None:
                try:
                    txt = str(original_texts_series.iloc[orig_i])
                except Exception:
                    txt = ''
            else:
                txt = ' '.join(tokens_train[train_i])[:200]
            txt = re.sub(r'\s+', ' ', txt).strip()[:260]
            snippets.append(txt)
            entries.append({'orig_df_position': int(orig_i),
                            'theta_prob': prob,
                            'text_snippet_260': txt,
                            'token_count': int(preprocessed_df['text_preprocessed_academic_token_count'].iloc[orig_i])})
        repr_docs_per_topic.append(entries)
        if auto_name:
            sname, lname = _heuristic_name_topic(k, top_words[k][:15], snippets)
        else:
            sname, lname = f"T{k:02d}_Topic_{k:02d}", f"Topic #{k:02d} (user-naming required). Top words: {', '.join(top_words[k][:10])}"
        topic_name_short.append(sname)
        topic_name_long.append(lname)
    # Coherence on final model
    coh = {'c_v': float('nan'), 'u_mass': float('nan')}
    if HAS_GENSIM and gensim_dct is not None:
        try:
            bows = []
            for i in range(tf.shape[0]):
                r = tf[i].tocoo()
                bows.append(list(zip(r.col.tolist(), r.data.astype(int).tolist())))
            coh = _coherence_scores(tokens_train, bows, gensim_dct, [w[:15] for w in top_words],
                                    coherence_processes=coherence_processes)
        except Exception:
            pass
    # Assemble outputs
    assignments_df = preprocessed_df.copy()
    assignments_df['lda_best_topic_id'] = best_id
    assignments_df['lda_best_topic_name'] = [(topic_name_short[int(k)] if k >= 0 else 'N/A (empty preprocessing)') for k in best_id]
    assignments_df['lda_best_topic_prob_theta'] = best_p
    assignments_df['lda_best_topic_description'] = [(topic_name_long[int(k)] if k >= 0 else '') for k in best_id]
    assignments_df['lda_best_topic_top3_words'] = [(', '.join(top_words[int(k)][:3]) if k >= 0 else '') for k in best_id]
    for k in range(K):
        assignments_df[f'lda_topic_{k:02d}_theta'] = theta_all[:, k]

    N_TRAIN = int(mask_nonempty.sum())
    rows_topic_dict = []
    topic_dict_obj = {}
    for k in range(K):
        entry = {
            'topic_id': k,
            'topic_short_name': topic_name_short[k],
            'topic_description': topic_name_long[k],
            'num_docs_assigned': int(counts[k]),
            'pct_training_docs': round(counts[k] / N_TRAIN * 100, 2) if N_TRAIN > 0 else 0.0,
            'top_15_words': ', '.join(top_words[k][:15]),
            'top_15_word_probs': ', '.join(f'{p:.4f}' for p in top_probs[k][:15]),
            'repr_doc_1_theta': round(repr_docs_per_topic[k][0]['theta_prob'], 4),
            'repr_doc_1_snippet': repr_docs_per_topic[k][0]['text_snippet_260'],
            'repr_doc_2_theta': round(repr_docs_per_topic[k][1]['theta_prob'], 4),
            'repr_doc_2_snippet': repr_docs_per_topic[k][1]['text_snippet_260'],
        }
        rows_topic_dict.append(entry)
        topic_dict_obj[k] = dict(entry, top_words_full=top_words[k], top_probs_full=top_probs[k],
                                 repr_docs_full=repr_docs_per_topic[k])
    topic_dict_df = pd.DataFrame(rows_topic_dict)
    stats_row = {
        'K': K,
        'N_docs_total': int(N),
        'N_docs_nonempty_trained': N_TRAIN,
        'vocab_size_V': int(V),
        'perplexity': round(perp, 3),
        'log_likelihood': round(loglik, 2),
        'coherence_cv': round(coh['c_v'], 5) if isinstance(coh['c_v'], float) and np.isfinite(coh['c_v']) else None,
        'coherence_umass': round(coh['u_mass'], 5) if isinstance(coh['u_mass'], float) and np.isfinite(coh['u_mass']) else None,
        'random_state': int(random_state),
    }
    model_stats_df = pd.DataFrame({'metric': list(stats_row.keys()), 'value': list(stats_row.values())})
    extra = {
        'model_stats': stats_row,
        'doc_counts_per_topic': [int(counts[k]) for k in range(K)],
        'topic_dict_json_ready': {str(k): v for k, v in topic_dict_obj.items()},
    }
    return lda, vec, gensim_dct, assignments_df, topic_dict_df, model_stats_df, extra, None

# =========================================================================
# 6. Writer: Excel (3 sheets) + CSV topic dict + JSON + plots (bar/pie/salient)
# =========================================================================
def write_all_outputs(out_dir, preprocessed_df=None, assignments_df=None, topic_dict_df=None,
                      k_sweep_df=None, turning=None, lda=None, vec=None, model_stats_df=None,
                      extra=None):
    paths = {}
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    def _p(base, **kw):
        params = {'ts': ts}
        params.update(kw)
        return os.path.join(out_dir, base.format(**params))

    # (A) Excel: 3 sheets
    try:
        if assignments_df is not None:
            K = int(extra['model_stats']['K']) if extra and 'model_stats' in extra else 15
            xlsx_path = _p('lda_K{K}_outputs_{ts}.xlsx', K=K)
            with pd.ExcelWriter(xlsx_path, engine='openpyxl') as w:
                assignments_df.to_excel(w, sheet_name=f'DocAssignments_K{K}', index=False)
                if topic_dict_df is not None:
                    topic_dict_df.to_excel(w, sheet_name=f'TopicDictionary_K{K}', index=False)
                if model_stats_df is not None:
                    model_stats_df.to_excel(w, sheet_name=f'ModelStats_K{K}', index=False)
                if k_sweep_df is not None and not k_sweep_df.empty:
                    k_sweep_df.to_excel(w, sheet_name=f'KSweep_Tuning', index=False)
            paths['xlsx'] = xlsx_path
    except Exception as e:
        paths['xlsx_error'] = str(e)

    # (B) Topic dict CSV
    try:
        if topic_dict_df is not None:
            csv_p = _p('lda_topic_dictionary_{ts}.csv')
            topic_dict_df.to_csv(csv_p, index=False)
            paths['csv_topic_dict'] = csv_p
    except Exception as e:
        paths['csv_error'] = str(e)

    # (C) JSON (topic_dict + turning + stats)
    try:
        blob = {}
        if extra and 'topic_dict_json_ready' in extra:
            blob['topic_dictionary'] = extra['topic_dict_json_ready']
        if extra and 'model_stats' in extra:
            blob['model_stats'] = extra['model_stats']
        if extra and 'doc_counts_per_topic' in extra:
            blob['doc_counts_per_topic'] = extra['doc_counts_per_topic']
        if turning:
            blob['turning_point_analysis'] = turning
        if k_sweep_df is not None and not k_sweep_df.empty:
            blob['k_sweep'] = k_sweep_df.to_dict(orient='records')
        if blob:
            jp = _p('lda_run_outputs_{ts}.json')
            with open(jp, 'w') as f:
                json.dump(blob, f, indent=2, ensure_ascii=False, default=str)
            paths['json'] = jp
    except Exception as e:
        paths['json_error'] = str(e)

    # (D) Plots (minimal: doc-count bar + salient words per topic)
    if HAS_MPL and topic_dict_df is not None and lda is not None and vec is not None:
        try:
            K = int(lda.n_components)
            counts = np.array([int(topic_dict_df.iloc[k]['num_docs_assigned']) for k in range(K)])
            names = [_abbreviate(topic_dict_df.iloc[k]['topic_short_name'], 30) for k in range(K)]
            order = np.argsort(-counts)
            # Doc count bar
            fig, ax = plt.subplots(figsize=(14, 6.2))
            colors = plt.get_cmap('tab20').colors
            ys = counts[order]
            xs = np.arange(K)
            ax.bar(xs, ys, color=[colors[i % len(colors)] for i in order], edgecolor='white')
            for i, (x, y) in enumerate(zip(xs, ys)):
                ax.text(x, y + max(ys) * 0.01, f"{y}", ha='center', fontsize=9, fontweight='bold')
            ax.set_xticks(xs)
            ax.set_xticklabels([names[i] for i in order], rotation=40, ha='right', fontsize=9)
            ax.set_title(r'Document Count per Topic ($\arg\max \theta$, hard assignment)'
                         f'\n$K={K}$, total non-empty docs trained $N={int(counts.sum())}$')
            ax.set_ylabel('Number of documents')
            ax.set_xlabel('Topic (sorted by count, desc)')
            fig.tight_layout()
            p = _p('lda_K{K}_doc_count_bar_{ts}.png', K=K)
            fig.savefig(p, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            paths['png_doc_count_bar'] = p

            # Salient Top-5 words per topic
            comps = lda.components_ / lda.components_.sum(axis=1, keepdims=True)
            feat = np.array(vec.get_feature_names_out())
            nrows = int(np.ceil(K / 5))
            ncols = min(5, K)
            fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4.1 * nrows))
            axes = np.array(axes).flatten()
            for k in range(K):
                ax = axes[k]
                top = np.argsort(-comps[k])[:5][::-1]
                probs = comps[k][top]
                words = feat[top]
                ax.barh(range(5), probs, color=colors[k % len(colors)], edgecolor='white')
                ax.set_yticks(range(5))
                ax.set_yticklabels(words, fontsize=9, fontweight='bold')
                ax.set_title(f"T{k:02d} {_abbreviate(names[k], 26)}\n"
                             f"n={counts[k]} ({counts[k]/counts.sum():.0%})", fontsize=8.5)
                for i, (pr, wr) in enumerate(zip(probs, words)):
                    ax.text(pr + max(probs)*0.02, i, f"{pr:.3f}", va='center', fontsize=7.5)
            for kk in range(K, len(axes)):
                axes[kk].set_visible(False)
            fig.suptitle(r'Top-5 Most Probable Words per Topic, $\phi(w \mid k)$'
                         f'\n$K={K}$', fontsize=13, fontweight='bold', y=1.0)
            fig.tight_layout()
            p = _p('lda_K{K}_top5_salient_words_{ts}.png', K=K)
            fig.savefig(p, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            paths['png_top5_salient'] = p
        except Exception as e:
            paths['plot_error'] = f"{e}\n{traceback.format_exc()}"
    return paths
