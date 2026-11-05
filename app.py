"""
声迹 AI 转写台 · 本地服务

- 语音转写：faster-whisper（本地，开源）或 OpenAI Whisper API
- 上下文纠错与分部分总结：任意 OpenAI 兼容的大模型（OpenAI / DeepSeek / 通义 / Ollama 本地模型等）
- 前端界面：index.html（与在线版是同一个页面，在本地会自动启用“开始转写”）

运行：python app.py   然后打开 http://127.0.0.1:8000
"""
import json
import os
import re
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

HERE = Path(__file__).parent

# ---------- 配置（环境变量，可写在 .env 里） ----------
def _load_dotenv():
    p = HERE / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

_load_dotenv()

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")          # tiny / base / small / medium / large-v3 / turbo
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")         # auto / cpu / cuda
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE_TYPE", "default")  # cpu 推荐 int8，GPU 推荐 float16
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_STT_MODEL = os.getenv("OPENAI_STT_MODEL", "whisper-1")

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")                 # 留空 = OpenAI 官方；Ollama: http://localhost:11434/v1
LLM_API_KEY = os.getenv("LLM_API_KEY", "") or OPENAI_API_KEY or ("ollama" if LLM_BASE_URL else "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

try:
    import faster_whisper  # noqa: F401
    HAS_LOCAL = True
except Exception:
    HAS_LOCAL = False

app = FastAPI(title="声迹 AI 转写台")
_model = None


def get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
    return _model


# ---------- 页面 ----------
SKELETON = ('<!doctype html><html lang="zh"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
            '<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top);padding-bottom:env(safe-area-inset-bottom)}'
            'body{margin:0;font:14px system-ui,sans-serif}img{max-width:100%}[hidden]{display:none!important}</style>'
            '</head><body>{body}</body></html>')


@app.get("/", response_class=HTMLResponse)
def index():
    html = (HERE / "index.html").read_text(encoding="utf-8")
    return html if html.lstrip().lower().startswith("<!doctype") else SKELETON.replace("{body}", html)


@app.get("/api/health")
def health():
    return {
        "app": "ai-transcribe",
        "engines": {"local": HAS_LOCAL, "openai": bool(OPENAI_API_KEY)},
        "whisper_model": WHISPER_MODEL,
        "llm": {"configured": bool(LLM_API_KEY), "model": LLM_MODEL, "base_url": LLM_BASE_URL or "https://api.openai.com/v1"},
    }


# ---------- 转写（NDJSON 流式返回，前端边识别边显示） ----------
def _line(obj):
    return json.dumps(obj, ensure_ascii=False) + "\n"


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...), engine: str = Form("local"),
                     language: str = Form("auto"), prompt: str = Form("")):
    if engine == "openai" and not OPENAI_API_KEY:
        raise HTTPException(400, "未设置 OPENAI_API_KEY，无法使用 OpenAI Whisper API")
    if engine != "openai" and not HAS_LOCAL:
        raise HTTPException(400, "未安装 faster-whisper：pip install faster-whisper")

    suffix = Path(file.filename or "audio").suffix or ".bin"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    while chunk := await file.read(1 << 20):
        tmp.write(chunk)
    tmp.close()
    lang = None if language in ("", "auto") else language

    def gen():
        try:
            if engine == "openai":
                from openai import OpenAI
                client = OpenAI(api_key=OPENAI_API_KEY)
                with open(tmp.name, "rb") as f:
                    kw = dict(model=OPENAI_STT_MODEL, file=f, response_format="verbose_json",
                              timestamp_granularities=["segment"])
                    if lang:
                        kw["language"] = lang
                    if prompt:
                        kw["prompt"] = prompt[:800]
                    r = client.audio.transcriptions.create(**kw)
                yield _line({"type": "info", "language": getattr(r, "language", None), "duration": getattr(r, "duration", None)})
                for i, s in enumerate(getattr(r, "segments", None) or []):
                    g = (lambda k: s.get(k) if isinstance(s, dict) else getattr(s, k))
                    yield _line({"type": "segment", "id": i, "start": round(float(g("start")), 2),
                                 "end": round(float(g("end")), 2), "text": str(g("text")).strip()})
            else:
                model = get_model()
                segments, info = model.transcribe(tmp.name, language=lang, beam_size=5, vad_filter=True,
                                                  initial_prompt=prompt or None)
                yield _line({"type": "info", "language": info.language, "duration": round(info.duration, 2)})
                for i, s in enumerate(segments):
                    text = s.text.strip()
                    if text:
                        yield _line({"type": "segment", "id": i, "start": round(s.start, 2), "end": round(s.end, 2), "text": text})
            yield _line({"type": "done"})
        except Exception as e:  # 让前端看到原因
            yield _line({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    return StreamingResponse(gen(), media_type="application/x-ndjson")


# ---------- 纠错 / 总结：调用大模型，返回 JSON ----------
class LLMReq(BaseModel):
    prompt: str


def parse_json(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    i, j = text.find("{"), text.rfind("}")
    if i >= 0 and j > i:
        return json.loads(text[i:j + 1])
    raise ValueError("模型没有返回 JSON")


@app.post("/api/llm")
def llm(req: LLMReq):
    if not LLM_API_KEY:
        raise HTTPException(400, "未配置纠错模型：设置 OPENAI_API_KEY，或 LLM_BASE_URL + LLM_MODEL")
    from openai import OpenAI
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL or None)
    messages = [
        {"role": "system", "content": "你是严谨的语音转写校对与摘要助手。只输出 JSON。"},
        {"role": "user", "content": req.prompt},
    ]
    try:
        try:
            r = client.chat.completions.create(model=LLM_MODEL, messages=messages, temperature=0.2,
                                               response_format={"type": "json_object"})
        except Exception:
            r = client.chat.completions.create(model=LLM_MODEL, messages=messages, temperature=0.2)
        return parse_json(r.choices[0].message.content or "")
    except Exception as e:
        raise HTTPException(502, f"大模型调用失败：{e}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    print(f"\n  声迹 AI 转写台 → http://127.0.0.1:{port}\n")
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=port)
