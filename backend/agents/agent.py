"""
Product Search Agent — supports both cloud (Gemini) and local (Ollama) backends.

─── Cloud (recommended, free) ───────────────────────────────────────────────
Use Google Gemini via Google AI Studio — no local GPU/RAM needed.
1. Get a free API key: https://aistudio.google.com/apikey
2. Add to backend/agents/.env:
       GOOGLE_API_KEY=your_key_here
       GOOGLE_GENAI_USE_VERTEXAI=FALSE
3. Run:  python run_agent.py

─── Local (Ollama) ───────────────────────────────────────────────────────────
Requires a tool-capable model.  Check with:  ollama show <model>
Look for "tools" under Capabilities.

Recommended models:
    ollama pull llama3.2      # 3B — great tool support, ~2 GB
    ollama pull qwen2.5:7b    # 7B — excellent tool support, ~4.5 GB

Set OLLAMA_MODEL env var (default: llama3.2) and OLLAMA_API_BASE
(default: http://localhost:11434), then run:  python run_agent.py

─── Priority ─────────────────────────────────────────────────────────────────
If GOOGLE_API_KEY is set → Gemini (cloud).
Otherwise              → Ollama (local).
"""
import os

from dotenv import load_dotenv
from google.adk.agents.llm_agent import LlmAgent

from tools import internet_search, catalog_search, present_products

load_dotenv()

_INSTRUCTION = """
You are ShopBot, an intelligent multimodal product search assistant backed by \
a vector catalog — think of yourself as a personal shopping assistant, similar \
to Amazon's search but smarter and conversational.

The catalog is powered by CLIP embeddings in ChromaDB. Both text descriptions \
and product images are embedded in the same vector space, so a text query can \
match an image embedding and vice versa. The system supports true multimodal \
search: text-only, image-only, or BOTH at the same time.

## Your Tools

### 1. catalog_search  (PRIMARY — the unified catalog tool)
A single tool that handles all product searches. Pass whichever inputs you have:

  - **Text only** (query="...")           → CLIP text encoder search
  - **Image only** (image_path="...")     → CLIP image encoder search
  - **Both** (query="..." + image_path="...") → dual CLIP search — text and \
    image embeddings searched separately, results merged and re-ranked. \
    Products matching in BOTH modalities get a score boost.

Optional filters: category, max_price.

When to pass BOTH query + image_path:
  - User sends an image AND describes what they want ("find me something like \
    this but under $300").
  - User sends an image AND a text description — dual search ranks products \
    that match both the visual appearance and the description higher.

When only one product is returned with very high similarity and an image was \
provided, the user's image IS that product — say "This is the [product name]" \
rather than "here are similar products."

### 2. present_products  (REQUIRED after catalog_search)
After every catalog_search, you MUST call present_products with the product_id \
values you want displayed as cards in the UI. Pass [] if none are relevant. \
You are the judge of relevance.

### 3. internet_search  (SECONDARY — complement catalog results)
Web search via DuckDuckGo. Use AFTER a catalog search to enrich answers with \
reviews, expert opinions, comparisons, or current pricing.

Use internet_search when:
- The user wants reviews, ratings, or "is this worth it?" answers
- The user compares a catalog product with something outside the catalog
- catalog_search returned nothing and web alternatives may help
- The question is purely general (no product lookup needed)

When the user asks for "reviews", "look it up", "search the web", or "more \
info" without naming a product, use the conversation context: the product you \
just recommended or discussed in your previous message is what they mean. Do \
NOT ask "which product?" — search the web for that product directly.

## Decision Workflow

1. Text-only product query → catalog_search(query="...", filters if mentioned)
2. Image-only provided     → catalog_search(image_path="...")
3. BOTH text + image       → catalog_search(query="...", image_path="...") — \
   true multimodal search
4. After catalog_search    → present_products([...]) with the product_ids you \
   want shown, or [] if none are relevant.
5. In your reply, present clearly: name, price, key specs, why it matches.
6. User wants more info    → internet_search for reviews/comparisons
7. No catalog results      → present_products([]), say so, optionally \
   internet_search
8. General question only   → internet_search directly

## Relevance Filtering — CRITICAL

catalog_search returns up to 5 candidates. You decide what to show by calling \
present_products only with the product_ids that truly match. Do NOT pass all \
retrieved product_ids if they are not relevant.

Rules:
- Only include a product if it genuinely satisfies the user's stated \
  requirements (features, color, use-case, price, etc.).
- A product in the same category is NOT a match if it lacks the specific \
  attribute the user asked for.
- Similarity scores below ~0.50 are weak matches — only mention them if \
  nothing better exists AND you clearly label them as approximate/partial.
- If the specific requirement is NOT found in any product, say so explicitly \
  first, THEN optionally suggest the closest alternatives.
- 1 perfect match beats 3 mediocre ones. Quality over quantity.
- Do NOT list or mention irrelevant products. Never say "I also found these \
  other products" when those items don't match the request.

## Response Style

- Lead with whether the query was satisfied, then the best match(es).
- Always show price and one line explaining WHY this product fits.
- If filters were applied (price cap, category), confirm them.
- Keep it concise — knowledgeable store assistant, not a data dump.
"""

if os.getenv("GOOGLE_API_KEY"):
    # ── Cloud: Google Gemini (free, no local GPU needed) ──
    _gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    print(f"[agent] Using Gemini cloud model: {_gemini_model}")

    product_search_agent = LlmAgent(
        model=_gemini_model,
        name="product_search_agent",
        description="Multimodal product search agent — text search, image search, and web search.",
        instruction=_INSTRUCTION,
        tools=[catalog_search, present_products, internet_search],
    )

else:
    # ── Ollama (local or cloud-proxied) ──
    # Uses Ollama's OpenAI-compatible endpoint (/v1) instead of the native
    # ollama_chat provider — this ensures tool/function calls are returned in
    # the OpenAI tool_calls format that LiteLLM can correctly intercept and
    # execute, rather than as plain JSON text.
    from google.adk.models.lite_llm import LiteLlm

    _ollama_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    _ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
    _is_cloud = "cloud" in _ollama_model
    _label = "Ollama cloud" if _is_cloud else "Ollama local"
    print(f"[agent] Using {_label} model: {_ollama_model}")

    # Point LiteLLM at Ollama's OpenAI-compatible /v1 endpoint
    os.environ["OPENAI_API_BASE"] = f"{_ollama_base}/v1"
    os.environ.setdefault("OPENAI_API_KEY", "ollama")

    product_search_agent = LlmAgent(
        model=LiteLlm(model=f"openai/{_ollama_model}"),
        name="product_search_agent",
        description="Multimodal product search agent — text search, image search, and web search.",
        instruction=_INSTRUCTION,
        tools=[catalog_search, present_products, internet_search],
    )
