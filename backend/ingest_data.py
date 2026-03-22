import base64
import json
import os
import sys
import uuid
from io import BytesIO
from pathlib import Path

import chromadb
import requests
from PIL import Image

from embedding_service import EmbeddingService

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Dynamic schema: map common aliases to standard keys. Missing → omit in Chroma (no None).
_KEY_ALIASES = {
    "id": "product_id",
    "title": "name",
    "desc": "description",
    "img": "image_url",
    "image": "image_url",
    "image_path": "image_url",
}
_STANDARD_KEYS = ("product_id", "name", "description", "price", "category", "brand", "image_url", "color", "design")


def normalize_product(product: dict, base_path: str = "."):
    """
    Normalize a flexible product dict. Returns (product_id, metadata_dict, text_for_embedding, image_path_or_None).
    metadata may contain None; Chroma caller must omit those keys.
    """
    raw = {}
    for k, v in product.items():
        key = _KEY_ALIASES.get(k, k)
        if key in _STANDARD_KEYS:
            raw[key] = v
    product_id = raw.get("product_id") or raw.get("id") or str(uuid.uuid4())
    if isinstance(product_id, (int, float)):
        product_id = str(product_id)
    name = raw.get("name") or raw.get("title")
    description = raw.get("description") or raw.get("desc")
    price = raw.get("price")
    if price is not None and not isinstance(price, (int, float)):
        try:
            price = float(price)
        except (TypeError, ValueError):
            price = None
    category = raw.get("category")
    brand = raw.get("brand")
    image_url = raw.get("image_url") or raw.get("image_path") or raw.get("image") or raw.get("img")
    color = raw.get("color")
    design = raw.get("design")
    metadata = {
        "product_id": product_id,
        "name": name,
        "description": description,
        "price": price,
        "category": category,
        "brand": brand,
        "image_url": image_url,
        "color": color,
        "design": design,
    }
    # Enrich embeddable text with color/design so queries like "black and white" or "red and blue" match
    parts = [description or name or product_id]
    if color:
        parts.append(f"Colors: {color}.")
    if design:
        parts.append(f"Design: {design}.")
    text_for_embedding = " ".join(parts).strip() or product_id
    image_path = None
    if image_url:
        if image_url.startswith(("http://", "https://", "data:")):
            # Pass URLs and base64 data URLs through as-is; ingest_products will handle them
            image_path = image_url
        else:
            p = Path(image_url)
            image_path = str(Path(base_path) / image_url) if not p.is_absolute() else image_url
    return product_id, metadata, text_for_embedding, image_path


def _metadata_for_chroma(metadata: dict, modality: str) -> dict:
    """Omit None so ChromaDB doesn't get invalid metadata."""
    out = {**metadata, "modality": modality}
    return {k: v for k, v in out.items() if v is not None}


def ingest_products(products=None, base_path="..", chroma_path="./chroma_db", clear_first=False):
    """
    Ingest products into ChromaDB. Dynamic schema: missing fields are omitted
    (Chroma gets only present keys; retriever treats missing as null).
    products: list of dicts with flexible keys; if None, load from products.json.
    base_path: directory to resolve relative image paths.
    clear_first: if True, delete the existing 'products' collection and reingest from scratch.
    """
    import numpy as np

    if products is None:
        with open("products.json", "r") as f:
            products = json.load(f)

    print("Initializing embedding service...")
    embedder = EmbeddingService()
    print("Initializing ChromaDB client...")
    client = chromadb.PersistentClient(path=chroma_path)

    if clear_first:
        try:
            client.delete_collection("products")
            print("Deleted existing 'products' collection.")
        except Exception:
            pass
        collection = client.create_collection(
            name="products",
            metadata={"description": "E-commerce product embeddings"},
        )
        print("Created new 'products' collection.")
    else:
        try:
            collection = client.get_collection("products")
            print("Using existing 'products' collection")
        except Exception:
            collection = client.create_collection(
                name="products",
                metadata={"description": "E-commerce product embeddings"},
            )
            print("Created new 'products' collection")

    print(f"\nProcessing {len(products)} products (dynamic schema)...")

    for product in products:
        product_id, metadata, text_for_embedding, image_path = normalize_product(product, base_path)
        name = metadata.get("name") or product_id
        meta_clean = _metadata_for_chroma(metadata, "text")

        print(f"\n--- Processing: {name} ---")

        if text_for_embedding:
            text_embedding = embedder.encode_text(text_for_embedding)
            collection.add(
                ids=[f"{product_id}_text"],
                embeddings=[text_embedding.tolist()],
                documents=[text_for_embedding[:50000]],
                metadatas=[{**meta_clean, "modality": "text"}],
            )
            print(f"  Stored text entry: {product_id}_text")

        image = None
        saved_image_path = None  # local path if we save the image to disk
        if image_path:
            if image_path.startswith("data:"):
                # Base64 data URL — decode inline bytes, save locally so UI can display it
                try:
                    header, b64data = image_path.split(",", 1)
                    mime = header.split(":")[1].split(";")[0]
                    ext = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/gif": ".gif"}.get(mime, ".jpg")
                    img_bytes = base64.b64decode(b64data)
                    image = Image.open(BytesIO(img_bytes)).convert("RGB")
                    # Save decoded image to base_path so UI can find it later
                    images_dir = Path(base_path) / "product_images"
                    images_dir.mkdir(exist_ok=True)
                    local_filename = f"product_images/{product_id}{ext}"
                    local_abs = Path(base_path) / local_filename
                    image.save(str(local_abs))
                    saved_image_path = local_filename  # relative path stored in metadata
                    print(f"  Decoded base64 image → saved as {local_filename}")
                except Exception as e:
                    print(f"  Skipped image (base64 decode failed): {e}")
            elif image_path.startswith(("http://", "https://")):
                try:
                    r = requests.get(image_path, timeout=15)
                    r.raise_for_status()
                    image = Image.open(BytesIO(r.content)).convert("RGB")
                except Exception as e:
                    print(f"  Skipped image (fetch failed): {e}")
            elif os.path.isfile(image_path):
                image = Image.open(image_path).convert("RGB")
            else:
                print(f"  Skipped image (file not found): {image_path}")

        if image is not None:
            # If we saved the image locally, update the metadata image_url to the local path
            if saved_image_path:
                metadata["image_url"] = saved_image_path
                meta_clean = _metadata_for_chroma(metadata, "text")  # rebuild with updated url

            image_embedding = embedder.encode_image(image)
            collection.add(
                ids=[f"{product_id}_image"],
                embeddings=[image_embedding.tolist()],
                documents=[metadata.get("image_url") or image_path or ""],
                metadatas=[{**meta_clean, "modality": "image"}],
            )
            print(f"  Stored image entry: {product_id}_image")

    print(f"\nDone. Collection count: {collection.count()}")
    return collection


if __name__ == "__main__":
    import numpy as np
    clear_first = "--clear" in sys.argv
    if clear_first:
        print("Clear flag set: will delete existing collection and reingest.\n")
    ingest_products(clear_first=clear_first)
