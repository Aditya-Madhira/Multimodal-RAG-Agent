"""
Run a single test query against the product search agent,
explicitly instructing it to use the internet_search tool.

Quick start (cloud — free, no local GPU needed):
    1. Get a free API key: https://aistudio.google.com/apikey
    2. Create backend/agents/.env with:
           GOOGLE_API_KEY=your_key_here
           GOOGLE_GENAI_USE_VERTEXAI=FALSE
    3. python run_agent.py

Quick start (local Ollama):
    1. ollama serve
    2. ollama pull llama3.2          # needs ~2 GB RAM
    3. python run_agent.py
"""
# Fix for Windows Unicode issues (emojis, special chars in LLM responses)
import os
import sys
os.environ.setdefault("PYTHONUTF8", "1")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import asyncio

import requests
from dotenv import load_dotenv

load_dotenv()

QUERY = (
    "I have an image of a gaming console at 'ps5.jpg'. "
    "Use catalog_search to find products that look like it."
)


def preflight_check() -> bool:
    """Verify the selected backend is ready before importing the agent."""
    google_key = os.getenv("GOOGLE_API_KEY")

    if google_key:
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        print(f"Backend : Gemini cloud  ({model})")
        print("Status  : API key found — ready.")
        return True

    # ── Ollama path ──
    base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    is_cloud = "cloud" in model
    backend_label = "Ollama cloud (proxied via local Ollama)" if is_cloud else "Ollama local"
    print(f"Backend : {backend_label}  ({model}  @  {base})")

    try:
        resp = requests.get(f"{base}/api/tags", timeout=5)
        resp.raise_for_status()
    except Exception as exc:
        print(f"\n  ERROR: Cannot reach Ollama — {exc}")
        print("  → Start Ollama with:  ollama serve")
        print("\n  Or switch to free Gemini cloud by adding to .env:")
        print("      GOOGLE_API_KEY=<your key from https://aistudio.google.com/apikey>")
        print("      GOOGLE_GENAI_USE_VERTEXAI=FALSE")
        return False

    available = {m["name"].split(":")[0] for m in resp.json().get("models", [])}
    model_base = model.split(":")[0]

    if model_base not in available:
        full_names = [m["name"] for m in resp.json().get("models", [])]
        print(f"\n  ERROR: Model '{model}' not found in Ollama.")
        print(f"  Available: {full_names}")
        print(f"\n  Pull a tool-capable model:")
        print(f"      ollama pull llama3.2      # ~2 GB")
        print(f"      ollama pull qwen2.5:7b    # ~4.5 GB")
        print(f"\n  Or switch to free Gemini cloud (no local RAM needed):")
        print(f"      Add GOOGLE_API_KEY=<key> to .env")
        return False

    print(f"Status  : model found — ready.")
    return True


if not preflight_check():
    sys.exit(1)

# ── Import after preflight so env vars are in place ──
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from agent import product_search_agent

APP_NAME = "product_search_app"
USER_ID = "user_1"
SESSION_ID = "session_1"


async def main():
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )

    runner = Runner(
        agent=product_search_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    print(f"\nQuery: {QUERY}")
    print("-" * 60)

    user_message = Content(role="user", parts=[Part(text=QUERY)])

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=user_message,
    ):
        if not (event.content and event.content.parts):
            continue
        for part in event.content.parts:
            if hasattr(part, "function_call") and part.function_call:
                print(f"\n[Tool call]   {part.function_call.name}({part.function_call.args})")
            if hasattr(part, "function_response") and part.function_response:
                print(f"[Tool result] {part.function_response.response}")
            if hasattr(part, "text") and part.text:
                label = "[Final answer]" if event.is_final_response() else "[Agent]"
                print(f"\n{label}\n{part.text}")


if __name__ == "__main__":
    asyncio.run(main())
