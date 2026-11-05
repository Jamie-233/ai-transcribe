# 声迹 AI 转写台

把音频转成文字，结合上下文纠正语音识别的错误，再按话题分部分并逐部分总结。

**在线使用**：https://jamie-233.github.io/ai-transcribe/
（转写和纠错总结都可以完全在浏览器里运行，不需要 API Key，音频不会上传；需要支持 WebGPU 的 Chrome / Edge）

## 流程

1. **转写**：faster-whisper（Whisper 的开源高速实现，本地运行）或 OpenAI Whisper API，边识别边显示。
2. **上下文纠错**：把转写分批交给大模型，每批附带前文末尾几句和已确认的修正表，找出同音字、术语、专有名词、外语词音译等识别错误。只改识别错误，不润色。每处修正都带理由和置信度。
3. **分部分**：按话题和设定时长（默认约 5 分钟）切分。
4. **总结**：每部分给出标题、2–4 句总结和要点。

界面里每处修正都是橙色标记，可以逐条采纳、还原或改写；低置信度的有虚线框，建议点“听这段”核对。

## 运行本地版

```bash
pip install -r requirements.txt
cp .env.example .env      # 填 OPENAI_API_KEY，或改成 Ollama / DeepSeek
python app.py             # 打开 http://127.0.0.1:8000
```

- 第一次转写会自动下载 Whisper 模型（small 约 500 MB）。
- 不需要单独安装 ffmpeg，faster-whisper 自带解码。
- 完全离线：本地 faster-whisper + Ollama（`ollama pull qwen2.5:7b`），在 `.env` 里设置 `LLM_BASE_URL` 和 `LLM_MODEL`。
- “背景 / 术语表”会同时作为 Whisper 的 initial prompt 和纠错提示，填课程名、人名、专业词能明显减少错误。

## 三种使用方式

同一个 `index.html`，会自动识别运行环境：

| | 网页版（GitHub Pages） | 本地版（python app.py） | claude.ai Artifact |
|---|---|---|---|
| 音频转写 | 浏览器内 Whisper（transformers.js，支持 WebGPU） | faster-whisper / OpenAI API | 导入转写文件 |
| 纠错与总结 | 浏览器内 Qwen2.5（WebLLM，免 Key）或 OpenAI 兼容接口 | 你配置的大模型 | Claude |
| 回放、校对、导出 | 支持 | 支持 | 支持 |

也可以直接导入 Whisper 输出的 JSON（`verbose_json` 或 `whisper` 命令行的 `.json`）、SRT、VTT 或纯文本。

网页版说明：
- 转写模型第一次下载约 80 MB（base），纠错模型 Qwen2.5 3B 约 2 GB，之后都从浏览器缓存加载。
- 浏览器内大模型需要 WebGPU（新版 Chrome / Edge），显存建议 4 GB 以上；7B 需要约 6 GB。
- 小模型纠错效果不如 API 或本地 14B 级模型，重要内容建议用“对比”视图逐条核对。
- 选择“API 接口”时，Key 只保存在你自己的浏览器里；也可以填本地 Ollama（需设置 `OLLAMA_ORIGINS=https://jamie-233.github.io`）。

## 接口

- `GET /api/health`：引擎和模型状态
- `POST /api/transcribe`：表单上传 `file`，可选 `engine`(local|openai)、`language`、`prompt`；返回 NDJSON 流
- `POST /api/llm`：`{"prompt": "..."}`，返回模型给出的 JSON

## 导出

Markdown 报告（部分总结 + 修正后全文 + 修正记录）、SRT 字幕、纯文本、JSON（可重新导入继续校对）。
