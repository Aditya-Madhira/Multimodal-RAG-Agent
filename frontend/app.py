"""
ShopBot — Multimodal Product Search UI
Streamlit chat interface powered by the ADK product search agent.

Run from the project root:
    streamlit run frontend/app.py
"""
import asyncio
import io
import json
import os
import sys
import uuid
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

if "messages"       not in st.session_state:
    st.session_state.messages = []
if "session_id"     not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "pending_image"     not in st.session_state:
    st.session_state.pending_image = None   # (bytes, filename) or None
if "pending_image_url"  not in st.session_state:
    st.session_state.pending_image_url = None  # str URL or None
if "uploader_key"   not in st.session_state:
    st.session_state.uploader_key = 0       # bump to reset uploader widget
if "show_mic_recording" not in st.session_state:
    st.session_state.show_mic_recording = False
if "mic_input_key" not in st.session_state:
    st.session_state.mic_input_key = 0
if "mic_used" not in st.session_state:
    st.session_state.mic_used = False
# ── Path & env setup (must happen before any backend imports) ─────────────────
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_FRONTEND_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _FRONTEND_DIR.parent
_BACKEND_DIR  = _PROJECT_ROOT / "backend"
_AGENTS_DIR   = _BACKEND_DIR / "agents"
_IMAGES_DIR   = _PROJECT_ROOT   # product images live here (ps5.jpg etc.)

for _p in [str(_AGENTS_DIR), str(_BACKEND_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv
load_dotenv(str(_AGENTS_DIR / ".env"))

os.environ.setdefault("PYTHONUTF8", "1")

_USING_GEMINI = bool(os.getenv("GOOGLE_API_KEY"))
if not _USING_GEMINI:
    _ollama_base  = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    _ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
    os.environ["OPENAI_API_BASE"] = f"{_ollama_base}/v1"
    os.environ.setdefault("OPENAI_API_KEY", "ollama")

# ── ADK imports ───────────────────────────────────────────────────────────────
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Blob, Content, Part
from agent import product_search_agent
from ingest_data import ingest_products

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ShopBot — Multimodal Search",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ────────────────────────────────────────────────────────────────
APP_NAME = "shopbot_ui"
USER_ID  = "streamlit_user"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- global ---------- */
[data-testid="stAppViewContainer"] { background: #0f0f11; }
[data-testid="stSidebar"]          { background: #16161a; border-right: 1px solid #2a2a35; }
[data-testid="stChatMessage"]      { background: transparent; }

/* ---------- product card ---------- */
.product-card {
    background: #1e1e26;
    border: 1px solid #2e2e3e;
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 10px;
    transition: border-color 0.2s;
}
.product-card:hover { border-color: #6c63ff; }

.product-name  { font-size: 1rem; font-weight: 700; color: #e8e8f0; margin-bottom: 4px; }
.product-price { font-size: 1.15rem; font-weight: 800; color: #7cfc9f; margin-bottom: 6px; }
.product-brand { font-size: 0.8rem; color: #888; margin-bottom: 8px; }
.product-desc  { font-size: 0.82rem; color: #aaa; line-height: 1.4; }

/* ---------- badges ---------- */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 6px;
    margin-bottom: 8px;
}
.badge-text   { background: #1a3a5c; color: #5eb6ff; border: 1px solid #2a5080; }
.badge-image  { background: #3a1a5c; color: #c07aff; border: 1px solid #5a2a80; }
.badge-web    { background: #1a3a2a; color: #5effa0; border: 1px solid #2a6040; }

.sim-bar-wrap { background: #2a2a35; border-radius: 4px; height: 5px; margin-top: 6px; }
.sim-bar      { background: #6c63ff; height: 5px; border-radius: 4px; }

/* ---------- upload preview ---------- */
.upload-preview {
    border: 2px dashed #3a3a50;
    border-radius: 10px;
    padding: 12px;
    margin-bottom: 8px;
    text-align: center;
    color: #888;
    font-size: 0.85rem;
}

/* ---------- sidebar labels ---------- */
.sidebar-section {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #555;
    margin: 18px 0 8px;
}
.status-dot-ok  { color: #5effa0; }
.status-dot-err { color: #ff6b6b; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Agent runner helpers
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Initialising ShopBot agent…")
def _init_runner():
    """Create the ADK Runner once per Streamlit process (cached across reruns)."""
    svc = InMemorySessionService()
    runner = Runner(agent=product_search_agent, app_name=APP_NAME, session_service=svc)
    return runner, svc


def _mime_for_image(filename: str) -> str:
    """Return IANA mime type for common image extensions."""
    suf = (Path(filename).suffix or "").lower()
    return {"": "image/jpeg", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}.get(suf, "image/jpeg")


async def _arun_agent(runner, svc, session_id: str, query: str, image_bytes: bytes | None = None, image_name: str | None = None):
    """Run one agent turn. If image_bytes is set, the user message includes the image so the LLM sees it."""
    existing = await svc.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
    if existing is None:
        await svc.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)

    parts = [Part(text=query)]
    if image_bytes:
        mime = _mime_for_image(image_name or "image.jpg")
        parts.append(Part(inline_data=Blob(data=image_bytes, mime_type=mime)))
    msg = Content(role="user", parts=parts)
    final_text = ""
    products   = []
    search_mode = None
    tool_name   = None
    seen_present = False  # when True, we use only what the agent passed to present_products

    async for event in runner.run_async(
        user_id=USER_ID, session_id=session_id, new_message=msg
    ):
        if not (event.content and event.content.parts):
            continue
        for part in event.content.parts:
            if hasattr(part, "function_response") and part.function_response:
                resp = part.function_response.response or {}
                name = part.function_response.name
                if name == "present_products" and isinstance(resp, dict):
                    products     = resp.get("products") or []
                    search_mode  = resp.get("search_mode", "text")
                    tool_name    = name
                    seen_present = True
                elif name in ("catalog_search", "rag_search", "image_search") and isinstance(resp, dict) and resp.get("products") and not seen_present:
                    products    = resp["products"]
                    search_mode = resp.get("search_mode", "text")
                    tool_name   = name
            if event.is_final_response() and hasattr(part, "text") and part.text:
                final_text = part.text

    return final_text, products, search_mode, tool_name


def run_agent(runner, svc, session_id: str, query: str, image_bytes: bytes | None = None, image_name: str | None = None):
    """Synchronous Streamlit-safe wrapper around the async agent."""
    return asyncio.run(_arun_agent(runner, svc, session_id, query, image_bytes=image_bytes, image_name=image_name))


# ══════════════════════════════════════════════════════════════════════════════
#  Display helpers
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_product_image(image_url: str):
    """Return a PIL Image for a product, or None if not found."""
    if not image_url:
        return None
    p = _IMAGES_DIR / image_url
    if p.exists():
        try:
            return Image.open(p)
        except Exception:
            return None
    return None


def _badge(label: str, kind: str) -> str:
    return f'<span class="badge badge-{kind}">{label}</span>'


def _render_product_cards(products: list, search_mode: str):
    """Render a grid of product cards. Only call with products the agent chose to show (via present_products)."""
    if not products:
        return

    mode_label = {"text": "Text search", "image": "Image search", "hybrid": "Hybrid search", "web": "Web"}.get(
        search_mode, search_mode or "search"
    )
    mode_kind = {"text": "text", "image": "image", "hybrid": "text"}.get(search_mode, "web")
    st.markdown(f'{_badge(mode_label, mode_kind)}', unsafe_allow_html=True)

    cols_per_row = min(len(products), 3)
    for row_start in range(0, len(products), cols_per_row):
        row_products = products[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, p in zip(cols, row_products):
            with col:
                image_url = p.get("image_url") or ""
                if image_url.startswith(("http://", "https://")):
                    st.image(image_url, use_container_width=True)
                else:
                    img = _resolve_product_image(image_url)
                    if img:
                        st.image(img, use_container_width=True)
                    else:
                        st.markdown(
                            '<div class="upload-preview">No image</div>',
                            unsafe_allow_html=True,
                        )
                modality = p.get("matched_modality", "")
                mod_badge = _badge(modality, "image" if modality == "image" else "text")
                price_val = p.get("price")
                price_str = f"${price_val:.2f}" if price_val is not None else "N/A"
                name_str = p.get("name") or "—"
                desc_str = (p.get("description") or "")[:120]
                if desc_str:
                    desc_str += "…"
                brand_cat = " · ".join(filter(None, [p.get("brand"), p.get("category")])) or "—"
                st.markdown(
                    f"""
                    <div class="product-card">
                      <div class="product-name">{name_str}</div>
                      <div class="product-price">{price_str}</div>
                      <div class="product-brand">{brand_cat}</div>
                      <div class="product-desc">{desc_str}</div>
                      {mod_badge}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def _render_message(msg: dict):
    """Re-render a single stored message (for chat history replay)."""
    role = msg["role"]
    with st.chat_message(role, avatar="🧑" if role == "user" else "🛍️"):
        if msg.get("image_bytes"):
            st.image(msg["image_bytes"], caption="Uploaded image", width=260)
        if msg.get("content"):
            st.markdown(msg["content"])
        if msg.get("products"):
            _render_product_cards(msg["products"], msg.get("search_mode"))


# ══════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("# 🛍️ ShopBot")
    st.markdown(
        "<span style='color:#888;font-size:0.85rem'>"
        "Multimodal product search — text, image & web"
        "</span>",
        unsafe_allow_html=True,
    )

    st.divider()

    # Backend status
    st.markdown('<div class="sidebar-section">Backend</div>', unsafe_allow_html=True)
    if _USING_GEMINI:
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        st.markdown(
            f'<span class="status-dot-ok">●</span> Gemini cloud · `{model_name}`',
            unsafe_allow_html=True,
        )
    else:
        model_name = os.getenv("OLLAMA_MODEL", "llama3.2")
        is_cloud   = "cloud" in model_name
        label      = "Ollama cloud" if is_cloud else "Ollama local"
        st.markdown(
            f'<span class="status-dot-ok">●</span> {label} · `{model_name}`',
            unsafe_allow_html=True,
        )

    st.divider()

    # Filters
    st.markdown('<div class="sidebar-section">Search Filters</div>', unsafe_allow_html=True)
    st.caption("Applied automatically to catalog searches.")

    category_options = ["All", "gaming", "electronics", "accessories", "audio", "wearables"]
    selected_category = st.selectbox(
        "Category", category_options, index=0, label_visibility="collapsed"
    )
    filter_category = None if selected_category == "All" else selected_category

    use_price_filter = st.toggle("Limit max price", value=False)
    filter_max_price = None
    if use_price_filter:
        filter_max_price = st.slider("Max price ($)", 50, 2000, 500, step=25)
        st.caption(f"Results ≤ ${filter_max_price}")

    st.divider()

    # How-to
    st.markdown('<div class="sidebar-section">How to use</div>', unsafe_allow_html=True)
    st.markdown("""
<span style='font-size:0.83rem;color:#aaa'>
<b>Text search</b> — describe what you want<br>
<code style='font-size:0.78rem'>Find a gaming console under $300</code><br><br>
<b>Image search</b> — upload a photo below the chat, then ask<br>
<code style='font-size:0.78rem'>Find products like this image</code><br><br>
<b>Web info</b> — ask for reviews or comparisons<br>
<code style='font-size:0.78rem'>Is the PS5 worth it in 2026?</code>
</span>
""", unsafe_allow_html=True)

    st.divider()

    # Add data to catalog
    st.markdown('<div class="sidebar-section">Add data to catalog</div>', unsafe_allow_html=True)
    with st.expander("What JSON do I need?", expanded=False):
        st.markdown("""
**Required (at least one for search):**  
You need something to embed — use **name** and/or **description**.  
**product_id** / **id** is optional; one is auto-generated if missing.

**Optional fields (all can be omitted):**
| Field | Aliases | Example |
|-------|---------|---------|
| **product_id** | id | `"prod_001"` |
| **name** | title | `"PlayStation 5"` |
| **description** | desc | `"Gaming console…"` |
| **price** | — | `499.99` |
| **category** | — | `"gaming"` |
| **brand** | — | `"Sony"` |
| **image_url** | image, image_path, img | URL or relative path |
| **color** | — | `"White and black"` |
| **design** | — | `"Curved design"` |

**Image:** Use **image_url** with an `https://` link to fetch from the web, or a path relative to the project (e.g. `ps5.jpg`). Missing fields are stored as null.
        """)
        st.code("""[
  {"name": "Product", "description": "...", "price": 99.99, "image_url": "https://example.com/img.jpg"},
  {"name": "Other", "category": "gaming", "brand": "Acme"}
]""", language="json")
    json_input = st.text_area(
        "Product JSON",
        height=140,
        placeholder='Paste an object or array of product objects (see "What JSON do I need?" above).',
        label_visibility="collapsed",
    )
    if st.button("Ingest into vector DB", type="primary", use_container_width=True, key="ingest_btn"):
        if not json_input.strip():
            st.warning("Paste some JSON first.")
        else:
            try:
                data = json.loads(json_input)
                if isinstance(data, dict):
                    data = [data]
                if not isinstance(data, list):
                    st.error("JSON must be an object or array of objects.")
                else:
                    with st.spinner("Ingesting… (encoding text and fetching images)…"):
                        ingest_products(
                            products=data,
                            base_path=str(_PROJECT_ROOT),
                            chroma_path=str(_BACKEND_DIR / "chroma_db"),
                        )
                    st.success(f"Ingested {len(data)} product(s) into the catalog.")
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")

    st.divider()

    # Clear chat
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages        = []
        st.session_state.session_id      = str(uuid.uuid4())
        st.session_state.pending_image    = None
        st.session_state.pending_image_url = None
        st.session_state.uploader_key    += 1
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  Main chat area
# ══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown(
    "<h2 style='margin-bottom:0;color:#e8e8f0'>ShopBot</h2>"
    "<p style='color:#666;margin-top:2px;font-size:0.9rem'>"
    "Ask anything · Upload an image · Search the catalog or the web"
    "</p>",
    unsafe_allow_html=True,
)

# ── Replay stored messages ─────────────────────────────────────────────────────
for msg in st.session_state.messages:
    _render_message(msg)

# ── Image: upload or paste URL (above the text input) ─────────────────────────
_has_image = st.session_state.pending_image or st.session_state.pending_image_url
with st.expander(
    "📎 Attach an image" + (" — 1 image ready" if _has_image else ""),
    expanded=bool(_has_image),
):
    uploaded = st.file_uploader(
        "Drag & drop or browse (jpg, png, webp)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
        key=f"uploader_{st.session_state.uploader_key}",
        label_visibility="collapsed",
    )
    if uploaded:
        img_bytes = uploaded.read()
        st.session_state.pending_image = (img_bytes, uploaded.name)
        st.session_state.pending_image_url = None
        st.image(img_bytes, caption=uploaded.name, width=220)
        st.success("Image attached — type your message below and hit Enter.")

    st.caption("Or paste an image URL (we'll fetch it for search)")
    image_url_input = st.text_input(
        "Image URL",
        value=st.session_state.pending_image_url or "",
        placeholder="https://example.com/product.jpg",
        label_visibility="collapsed",
        key="image_url_input",
    )
    if image_url_input and image_url_input.strip():
        st.session_state.pending_image_url = image_url_input.strip()
        if not st.session_state.pending_image:
            st.success("URL saved — type your message and hit Enter to search with this image.")
    else:
        st.session_state.pending_image_url = None

    if _has_image and st.button("✕ Remove image / URL", key="remove_img"):
        st.session_state.pending_image = None
        st.session_state.pending_image_url = None
        st.session_state.uploader_key += 1
        st.rerun()

# ── Whisper transcription helper (local, no API needed) ───────────────────────
@st.cache_resource(show_spinner=False)
def _get_whisper_model():
    """Load Whisper tiny model once and cache it across all sessions."""
    import whisper, torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model("small", device=device)

def _transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe WAV audio bytes using local Whisper — no ffmpeg required.

    Reads the WAV bytes directly into a float32 numpy array and passes it to
    Whisper, which accepts numpy arrays and skips its ffmpeg-based loader.
    """
    import io
    import wave
    import numpy as np

    model = _get_whisper_model()

    # Decode WAV header and raw PCM frames (no ffmpeg needed)
    with wave.open(io.BytesIO(audio_bytes)) as wf:
        n_channels  = wf.getnchannels()
        sample_width = wf.getsampwidth()   # bytes per sample (1=8-bit, 2=16-bit, 4=32-bit)
        sample_rate  = wf.getframerate()
        raw_frames   = wf.readframes(wf.getnframes())

    # Map sample width → signed integer dtype
    _dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sample_width, np.int16)
    audio = np.frombuffer(raw_frames, dtype=_dtype).astype(np.float32)
    audio /= float(np.iinfo(_dtype).max)   # normalise to [-1.0, 1.0]

    # Stereo / multi-channel → mono
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    # Resample to 16 kHz (Whisper's required sample rate)
    if sample_rate != 16000:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(sample_rate), 16000)
        audio = resample_poly(audio, 16000 // g, sample_rate // g).astype(np.float32)

    # Pass numpy array directly — Whisper skips its ffmpeg loader for arrays
    result = model.transcribe(audio, word_timestamps=False)
    return result["text"].strip()

# ── Chat input (pinned to bottom by Streamlit) ────────────────────────────────
prompt = st.chat_input("Message ShopBot…")

# ── Voice input — mic → local Whisper STT ─────────────────────────────────────
with st.expander("🎤 Voice input", expanded=False):
    st.caption("Record a query — transcribed locally with Whisper small (no internet needed)")
    audio_value = st.audio_input(
        "Record your query",
        key=f"mic_audio_{st.session_state.mic_input_key}",
    )
    if audio_value is not None and not prompt:
        with st.spinner("Transcribing with Whisper…"):
            try:
                transcribed = _transcribe_audio(audio_value.getvalue())
                if transcribed:
                    st.info(f'🎤 Transcribed: *"{transcribed}"*')
                    prompt = transcribed
                    st.session_state.mic_used = True
            except Exception as exc:
                st.error(f"Transcription error: {exc}")
    if audio_value is not None:
        if st.button("🗑️ Clear recording", key="clear_mic"):
            st.session_state.mic_input_key += 1
            st.rerun()

if prompt:
    # ── Resolve image: from file upload or from URL ─────────────────────────
    image_bytes  = None
    image_name   = None
    image_path   = None

    if st.session_state.pending_image:
        image_bytes, image_name = st.session_state.pending_image
        tmp_dir = _FRONTEND_DIR / ".tmp_uploads"
        tmp_dir.mkdir(exist_ok=True)
        suffix = Path(image_name).suffix or ".jpg"
        image_path = str(tmp_dir / f"{uuid.uuid4()}{suffix}")
        with open(image_path, "wb") as f:
            f.write(image_bytes)
    elif st.session_state.pending_image_url:
        url = st.session_state.pending_image_url.strip()
        try:
            tmp_dir = _FRONTEND_DIR / ".tmp_uploads"
            tmp_dir.mkdir(exist_ok=True)
            if url.startswith("data:"):
                # Base64 data URL — decode inline, e.g. data:image/jpeg;base64,/9j/...
                import base64
                header, b64data = url.split(",", 1)
                mime = header.split(":")[1].split(";")[0]  # e.g. "image/jpeg"
                ext = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/gif": ".gif"}.get(mime, ".jpg")
                image_bytes = base64.b64decode(b64data)
                image_name = f"image{ext}"
            elif url.startswith(("http://", "https://")):
                r = requests.get(url, timeout=15)
                r.raise_for_status()
                image_bytes = r.content
                image_name = url.split("/")[-1].split("?")[0] or "image.jpg"
            else:
                st.error("Unsupported image URL. Paste an https:// link or a base64 data: URL.")
                st.stop()
            suffix = Path(image_name).suffix or ".jpg"
            image_path = str(tmp_dir / f"{uuid.uuid4()}{suffix}")
            with open(image_path, "wb") as f:
                f.write(image_bytes)
        except requests.RequestException as e:
            st.error(f"Could not load image from URL: {e}. Check the link and try again.")
            st.stop()
        except Exception as e:
            st.error(f"Error using image: {e}. Try a different link or upload a file directly.")
            st.stop()

    runner, svc = _init_runner()

    # Build the query — embed filter hints and image path if present
    query_parts = [prompt]
    if filter_category:
        query_parts.append(f"(filter by category: {filter_category})")
    if filter_max_price:
        query_parts.append(f"(max price: ${filter_max_price})")
    if image_path:
        query_parts.append(
            f"The user has uploaded an image. Use catalog_search with image_path='{image_path}'"
            f"{' and also pass query= with their text description for dual search' if prompt.strip() else ''}."
        )
    full_query = " ".join(query_parts)

    # ── Render user bubble ──────────────────────────────────────────────────
    user_msg = {
        "role": "user",
        "content": prompt,
        "image_bytes": image_bytes,
        "products": None,
        "search_mode": None,
    }
    st.session_state.messages.append(user_msg)
    with st.chat_message("user", avatar="🧑"):
        if image_bytes:
            st.image(image_bytes, caption=image_name, width=260)
        st.markdown(prompt)

    # ── Clear pending image / URL now that message is submitted ─────────────
    st.session_state.pending_image = None
    st.session_state.pending_image_url = None
    st.session_state.uploader_key += 1

    # ── Run agent ───────────────────────────────────────────────────────────
    with st.chat_message("assistant", avatar="🛍️"):
        with st.spinner("Searching…"):
            try:
                final_text, products, search_mode, tool_name = run_agent(
                    runner, svc, st.session_state.session_id, full_query,
                    image_bytes=image_bytes, image_name=image_name,
                )
            except Exception as exc:
                final_text  = f"Sorry, something went wrong: {exc}"
                products    = []
                search_mode = None
                tool_name   = None

        if products:
            _render_product_cards(products, search_mode)

        if final_text:
            st.markdown(final_text)

    # ── Store assistant message ─────────────────────────────────────────────
    st.session_state.messages.append({
        "role": "assistant",
        "content": final_text,
        "products": products,
        "search_mode": search_mode,
        "image_bytes": None,
    })

    # ── Auto-clear mic recording after message is sent ──────────────────────
    if st.session_state.mic_used:
        st.session_state.mic_used = False
        st.session_state.mic_input_key += 1
