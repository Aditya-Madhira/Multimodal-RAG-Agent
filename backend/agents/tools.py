"""
Tools for the Product Search Agent.

ADK automatically builds each tool's schema from the function name,
docstring (including Args: section), and type hints — so keep them precise.

Search modes supported:
  - catalog_search  : unified — text, image, or both → CLIP → vector search + metadata filters
  - internet_search : web search via DuckDuckGo
"""
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional

from langchain_community.tools import DuckDuckGoSearchRun

# Make backend/ importable from backend/agents/
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# ChromaDB lives at backend/chroma_db/
_CHROMA_DB_PATH = str(Path(_BACKEND_DIR) / "chroma_db")

# Project root — where product images (ps5.jpg, switch.webp, xbox.png) live
_PROJECT_ROOT = str(Path(_BACKEND_DIR).parent)

# Lazy singleton — CLIP model loads once on first tool call
_retriever = None

# Last search results and mode — used by present_products so the agent can choose what to show
_last_search_results: list = []
_last_search_mode: str = "text"


def _get_retriever():
    """Return the shared HybridRetriever, initializing it on first use."""
    global _retriever
    if _retriever is None:
        from retriever import HybridRetriever
        _retriever = HybridRetriever(chroma_db_path=_CHROMA_DB_PATH)
    return _retriever


def _resolve_image_path(image_path: str) -> str:
    """
    Resolve image_path to an absolute path.
    Relative paths are resolved from the project root (where product images live).
    """
    p = Path(image_path)
    if p.is_absolute():
        return str(p)
    resolved = Path(_PROJECT_ROOT) / p
    if resolved.exists():
        return str(resolved)
    # Fallback: try relative to backend/
    fallback = Path(_BACKEND_DIR) / p
    return str(fallback)


def _format_products(results) -> list:
    """Convert ProductResult objects to plain dicts for the agent (dynamic schema — null for missing)."""
    out = []
    for p in results:
        out.append({
            "product_id": p.product_id,
            "name": p.name,
            "description": p.description,
            "price": round(p.price, 4) if p.price is not None else None,
            "category": p.category,
            "brand": p.brand,
            "image_url": p.image_url,
            "color": getattr(p, "color", None),
            "design": getattr(p, "design", None),
            "similarity_score": round(p.similarity_score, 4),
            "matched_modality": p.matched_modality,
        })
    return out


# ── Tools ────────────────────────────────────────────────────────────────────

def internet_search(query: str) -> Dict:
    """
    Search the internet for general information, current events, or anything
    not in the product catalog — e.g. reviews, news, expert comparisons,
    pricing trends, or general tech questions.

    Args:
        query: The search query string.

    Returns:
        Dictionary with keys 'status', 'query', and 'results' (a text summary
        of the top web results) or 'error' on failure.
    """
    try:
        results = DuckDuckGoSearchRun().run(query)
        return {"status": "success", "query": query, "results": results}
    except Exception as exc:
        return {"status": "error", "query": query, "error": str(exc)}


def catalog_search(
    query: Optional[str] = None,
    image_path: Optional[str] = None,
    category: Optional[str] = None,
    max_price: Optional[float] = None,
) -> Dict:
    """
    Search the product catalog. Supports three modes — pass whichever you have:
      - text only   (query)       → CLIP text encoder search
      - image only  (image_path)  → CLIP image encoder search
      - both        (query + image_path) → dual CLIP search, merged & re-ranked

    Results are ranked by semantic similarity. When both text and image are
    provided, products matching in BOTH modalities get a score boost.

    Args:
        query: Natural-language description of what the user wants, e.g.
               "gaming console with 4K and ray tracing". Pass None when
               searching by image only.
        image_path: Absolute or project-relative path to the query image.
                    Supported formats: jpg, jpeg, png, webp. Pass None when
                    searching by text only.
        category: Optional metadata filter by product category, e.g. "gaming".
                  Pass None to search across all categories.
        max_price: Optional metadata filter — maximum price in USD.
                   Pass None for no price limit.

    Returns:
        Dictionary with 'status', 'search_mode' (text/image/hybrid), 'count',
        and 'products' (list of matched products with product_id, name,
        description, price, brand, image_url, similarity_score,
        matched_modality).
    """
    global _last_search_results, _last_search_mode
    if not query and not image_path:
        return {"status": "error", "error": "Provide at least query or image_path."}

    try:
        retriever = _get_retriever()
        filters = {}
        if category:
            filters["category"] = category
        if max_price:
            filters["price"] = {"$lte": max_price}

        resolved_image = _resolve_image_path(image_path) if image_path else None

        results = retriever.retrieve_products(
            query_text=query or None,
            query_image_path=resolved_image,
            top_k=5,
            filters=filters if filters else None,
        )
        products = _format_products(results)

        if query and image_path:
            mode = "hybrid"
        elif image_path:
            mode = "image"
        else:
            mode = "text"

        _last_search_results = products
        _last_search_mode = mode
        return {
            "status": "success",
            "query": query,
            "image_path": resolved_image,
            "search_mode": mode,
            "count": len(products),
            "products": products,
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def present_products(product_ids: List[str]) -> Dict:
    """
    Call this after catalog_search to choose which products to show the user.
    Only the products you list here will be displayed as cards in the UI. Use an empty
    list if none of the search results are relevant to the user's query.

    Args:
        product_ids: List of product_id strings from the most recent catalog_search to display.
                     Use the exact product_id values returned by catalog_search.
                     Pass [] to show no product cards (e.g. when nothing matched).

    Returns:
        Dictionary with 'products' (the subset to display) and 'search_mode'.
    """
    global _last_search_results, _last_search_mode
    ids_set = set(product_ids) if product_ids else set()
    chosen = [p for p in _last_search_results if p.get("product_id") in ids_set]
    return {
        "products": chosen,
        "search_mode": _last_search_mode,
    }
