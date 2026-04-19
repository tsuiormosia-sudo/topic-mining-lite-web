import io
import re
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


st.set_page_config(page_title="LDA / BERTopic Lite", layout="wide")

if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None
if "uploaded_file_signature" not in st.session_state:
    st.session_state["uploaded_file_signature"] = None


def clean_text(text):
    text = str(text or "").strip().lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_uploaded_file(uploaded_file):
    suffix = uploaded_file.name.lower()
    if suffix.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if suffix.endswith(".xlsx") or suffix.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("仅支持 CSV / XLSX / XLS 文件")


def top_words_from_component(feature_names, component, n_words=10):
    indices = component.argsort()[-n_words:][::-1]
    return ", ".join(feature_names[i] for i in indices)


def run_lda(documents, n_topics):
    vectorizer = CountVectorizer(
        stop_words="english",
        min_df=2,
        max_df=0.95,
    )
    X = vectorizer.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method="batch",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = vectorizer.get_feature_names_out()

    topics_df = pd.DataFrame(
        {
            "topic_id": list(range(n_topics)),
            "keywords": [
                top_words_from_component(feature_names, lda.components_[i])
                for i in range(n_topics)
            ],
        }
    )
    assignments = pd.DataFrame(
        {
            "document_id": list(range(len(documents))),
            "dominant_topic": doc_topic.argmax(axis=1),
            "topic_score": doc_topic.max(axis=1),
            "text": documents,
        }
    )
    return topics_df, assignments


def safe_svd_components(matrix):
    max_components = min(matrix.shape[0] - 1, matrix.shape[1] - 1)
    return max(1, min(50, max_components))


def build_topic_name_map(topics_df):
    topic_names = {}
    for _, row in topics_df.iterrows():
        topic_names[row["topic_id"]] = row.get("topic_name", f"Topic {row['topic_id']}")
    return topic_names


def get_topic_keywords_df(topic_model, topic_id, top_n=10):
    topic_words = topic_model.get_topic(topic_id) or []
    rows = [
        {"word": word, "score": score}
        for word, score in topic_words[:top_n]
        if word and word.strip()
    ]
    return pd.DataFrame(rows)


def run_bertopic(
    documents,
    min_topic_size,
    reduction_mode,
    topic_upper_bound,
    reduce_outliers_flag,
):
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
    )
    tfidf = vectorizer.fit_transform(documents)

    n_components = safe_svd_components(tfidf)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    embeddings = svd.fit_transform(tfidf)

    umap_model = UMAP(
        n_neighbors=min(15, max(2, len(documents) - 1)),
        n_components=min(5, embeddings.shape[1]) if embeddings.shape[1] > 1 else 1,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=False,
    )
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=False,
        verbose=False,
    )

    topics, _ = topic_model.fit_transform(documents, embeddings=embeddings)
    info_df = topic_model.get_topic_info()
    initial_topics_count = int((info_df["Topic"] != -1).sum())
    initial_outliers = int((pd.Series(topics) == -1).sum())

    if reduction_mode == "自动":
        topic_model.reduce_topics(documents, nr_topics="auto")
    elif reduction_mode == "按上限压缩" and initial_topics_count > topic_upper_bound:
        topic_model.reduce_topics(documents, nr_topics=topic_upper_bound)

    if reduce_outliers_flag and -1 in set(topic_model.topics_):
        new_topics = topic_model.reduce_outliers(
            documents,
            topic_model.topics_,
            strategy="c-tf-idf",
        )
        topic_model.update_topics(documents, topics=new_topics)

    final_topics = topic_model.topics_
    info_df = topic_model.get_topic_info()
    info_df = info_df.rename(columns={"Topic": "topic_id", "Name": "topic_name", "Count": "count"})
    info_df["is_outlier"] = info_df["topic_id"] == -1
    info_df = info_df[["topic_id", "topic_name", "count", "is_outlier"]]
    topic_name_map = build_topic_name_map(info_df)

    assignments = pd.DataFrame(
        {
            "document_id": list(range(len(documents))),
            "dominant_topic": final_topics,
            "topic_name": [topic_name_map.get(topic, f"Topic {topic}") for topic in final_topics],
            "text": documents,
        }
    )
    metrics = {
        "initial_topics_count": initial_topics_count,
        "final_topics_count": int((info_df["topic_id"] != -1).sum()),
        "initial_outliers": initial_outliers,
        "final_outliers": int((assignments["dominant_topic"] == -1).sum()),
    }
    return topic_model, info_df, assignments, metrics


def to_download_bytes(df):
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    return buffer.getvalue()


st.title("LDA / BERTopic Lite")
st.caption("轻量版本地主题分析工具：上传文本文件，直接跑 LDA 或 BERTopic。")

uploaded_file = st.file_uploader("上传 CSV / XLSX 文件", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    current_signature = f"{uploaded_file.name}:{uploaded_file.size}"
    if st.session_state["uploaded_file_signature"] != current_signature:
        st.session_state["uploaded_file_signature"] = current_signature
        st.session_state["analysis_result"] = None

    try:
        raw_df = read_uploaded_file(uploaded_file)
    except Exception as exc:
        st.error(f"读取文件失败: {exc}")
        st.stop()

    st.subheader("原始数据预览")
    st.dataframe(raw_df.head(20), use_container_width=True)

    text_column = st.selectbox("选择文本列", options=list(raw_df.columns))
    model_type = st.radio("选择模型", options=["LDA", "BERTopic"], horizontal=True)

    left, middle, right = st.columns(3)
    with left:
        lda_topics = st.slider("LDA 主题数", min_value=2, max_value=20, value=5)
    with middle:
        min_topic_size = st.slider("BERTopic 最小主题大小", min_value=2, max_value=50, value=5)
    with right:
        bertopic_topic_upper_bound = st.slider("BERTopic 主题数上限", min_value=2, max_value=30, value=8)

    bertopic_reduction_mode = "不压缩"
    bertopic_reduce_outliers = False
    if model_type == "BERTopic":
        extra_left, extra_right = st.columns(2)
        with extra_left:
            bertopic_reduction_mode = st.selectbox(
                "主题压缩方式",
                options=["不压缩", "自动", "按上限压缩"],
                help="自动或按上限压缩会在训练后做主题压缩。",
            )
        with extra_right:
            bertopic_reduce_outliers = st.checkbox(
                "尝试处理离群点 (-1)",
                value=True,
                help="将部分离群文本重新归入最接近的主题。",
            )

    if st.button("开始分析", type="primary"):
        texts = raw_df[text_column].fillna("").astype(str).map(clean_text)
        texts = texts[texts != ""]

        if len(texts) < 5:
            st.error("至少需要 5 条非空文本。")
            st.stop()

        st.write(f"有效文本数: {len(texts)}")
        with st.spinner(f"正在运行 {model_type} ..."):
            if model_type == "LDA":
                topics_df, assignments_df = run_lda(texts.tolist(), lda_topics)
                topic_model = None
                bertopic_metrics = None
            else:
                topic_model, topics_df, assignments_df, bertopic_metrics = run_bertopic(
                    texts.tolist(),
                    min_topic_size,
                    bertopic_reduction_mode,
                    bertopic_topic_upper_bound,
                    bertopic_reduce_outliers,
                )

        st.session_state["analysis_result"] = {
            "model_type": model_type,
            "topic_model": topic_model,
            "topics_df": topics_df,
            "assignments_df": assignments_df,
            "bertopic_metrics": bertopic_metrics,
        }

    analysis_result = st.session_state.get("analysis_result")
    if analysis_result:
        model_type = analysis_result["model_type"]
        topic_model = analysis_result["topic_model"]
        topics_df = analysis_result["topics_df"]
        assignments_df = analysis_result["assignments_df"]
        bertopic_metrics = analysis_result["bertopic_metrics"]

        st.subheader("主题结果")
        st.dataframe(topics_df, use_container_width=True)

        if model_type == "BERTopic" and bertopic_metrics:
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            metric_col1.metric("初始主题数", bertopic_metrics["initial_topics_count"])
            metric_col2.metric("最终主题数", bertopic_metrics["final_topics_count"])
            metric_col3.metric("初始离群点", bertopic_metrics["initial_outliers"])
            metric_col4.metric("最终离群点", bertopic_metrics["final_outliers"])

        st.subheader("文档主题分配")
        st.dataframe(assignments_df.head(100), use_container_width=True)

        topic_counter = Counter(assignments_df["dominant_topic"])
        chart_df = pd.DataFrame(
            {"topic": list(topic_counter.keys()), "count": list(topic_counter.values())}
        ).sort_values("count", ascending=False)
        st.subheader("主题分布")
        st.plotly_chart(
            px.bar(chart_df, x="topic", y="count", title="主题大小分布"),
            use_container_width=True,
        )

        if model_type == "BERTopic":
            selectable_topics = topics_df["topic_id"].tolist()
            selected_topic = st.selectbox(
                "查看单个主题细节",
                options=selectable_topics,
                format_func=lambda topic_id: (
                    f"Topic {topic_id} | "
                    + str(
                        topics_df.loc[topics_df["topic_id"] == topic_id, "topic_name"].iloc[0]
                    )
                ),
            )
            keyword_df = get_topic_keywords_df(topic_model, selected_topic)

            detail_left, detail_right = st.columns([1, 2])
            with detail_left:
                st.markdown("**主题关键词**")
                st.dataframe(keyword_df, use_container_width=True)
            with detail_right:
                if not keyword_df.empty:
                    st.plotly_chart(
                        px.bar(
                            keyword_df.sort_values("score", ascending=True),
                            x="score",
                            y="word",
                            orientation="h",
                            title=f"Topic {selected_topic} 关键词权重",
                        ),
                        use_container_width=True,
                    )

            st.markdown("**该主题下的文本样本**")
            topic_examples = assignments_df[assignments_df["dominant_topic"] == selected_topic].head(20)
            st.dataframe(topic_examples, use_container_width=True)

        st.download_button(
            "下载主题结果 Excel",
            data=to_download_bytes(topics_df),
            file_name=f"{model_type.lower()}_topics.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.download_button(
            "下载文档主题分配 Excel",
            data=to_download_bytes(assignments_df),
            file_name=f"{model_type.lower()}_assignments.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
else:
    st.info("先上传一个包含文本列的 CSV / XLSX 文件。")
