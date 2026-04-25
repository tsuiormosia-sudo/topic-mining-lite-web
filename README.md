# Topic Mining Lite Web

一个给学生直接使用的轻量版主题分析工具，支持上传 `CSV / XLSX` 文件后运行：

- `LDA`
- `BERTopic`

## 主要功能

- 上传文件并选择文本列
- 运行 `LDA` 或 `BERTopic`
- BERTopic 支持：
  - 主题压缩方式：`不压缩 / 自动 / 按上限压缩`
  - 离群点处理 `(-1)`
  - 主题大小分布图
  - 单个主题关键词查看
  - 主题下文本样本查看
- 导出主题结果和文档主题分配结果

## 本地运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 推荐上线方式

这个项目是 `Streamlit` 应用，不适合用 `GitHub Pages` 直接部署。

最简单的方式是：

1. 把这个目录推到一个公开 GitHub 仓库
2. 打开 [Streamlit Community Cloud](https://share.streamlit.io/)
3. 选择这个仓库
4. 主文件填写 `app.py`
5. 点击 Deploy

部署成功后，通过网页链接访问。

## 目录

- `app.py`：主应用
- `requirements.txt`：依赖
- `.streamlit/config.toml`：Streamlit 配置
- `runtime.txt`：Python 版本

## 说明

- 输入文件至少需要包含一列文本数据
- 适合教学、探索式分析和轻量演示
- 如果数据量非常大，BERTopic 运行时间会明显增加
