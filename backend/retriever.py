import chromadb
from embedding_service import EmbeddingService
from PIL import Image
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import os


@dataclass
class ProductResult:
    """Deduplicated product result with metadata. All fields except product_id and similarity/modality are optional (dynamic schema)."""
    product_id: str
    similarity_score: float
    matched_modality: str  # "text", "image", or "both"
    name: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None
    category: Optional[str] = None
    brand: Optional[str] = None
    image_url: Optional[str] = None
    color: Optional[str] = None
    design: Optional[str] = None


class HybridRetriever:
    """Hybrid retriever combining vector search with metadata filtering."""
    
    def __init__(self, chroma_db_path: str = "./chroma_db"):
        """Initialize retriever with embedding service and ChromaDB client."""
        print("Initializing HybridRetriever...")
        self.embedder = EmbeddingService()
        self.client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = self.client.get_collection("products")
        print(f"Connected to ChromaDB. Collection has {self.collection.count()} entries.")
    
    def retrieve_products(
        self,
        query_text: Optional[str] = None,
        query_image_path: Optional[str] = None,
        query_image_bytes: Optional[bytes] = None,
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[ProductResult]:
        """
        Retrieve products using hybrid search (vector + metadata filtering).
        When BOTH text and image are provided, runs two separate CLIP searches
        (text encoder + image encoder), merges and re-ranks the results.

        Args:
            query_text: Text query string
            query_image_path: Path to query image file
            query_image_bytes: Raw image bytes
            top_k: Number of results to return (after deduplication)
            filters: Metadata filters, e.g. {"category": "gaming", "price": {"$lte": 300}}

        Returns:
            List of ProductResult objects, deduplicated by product_id
        """
        if not query_text and not query_image_path and not query_image_bytes:
            raise ValueError("Must provide either query_text, query_image_path, or query_image_bytes")

        has_text = bool(query_text)
        has_image = bool(query_image_path or query_image_bytes)

        if has_text and has_image:
            return self._dual_search(query_text, query_image_path, query_image_bytes, top_k, filters)

        return self._single_search(query_text, query_image_path, query_image_bytes, top_k, filters)

    # ── single-modality search (text OR image) ───────────────────────────────

    def _single_search(self, query_text, query_image_path, query_image_bytes, top_k, filters):
        if query_text:
            print(f"Encoding text query: '{query_text}'")
            query_embedding = self.embedder.encode_text(query_text)
        elif query_image_path:
            print(f"Encoding image query: '{query_image_path}'")
            image = Image.open(query_image_path).convert("RGB")
            query_embedding = self.embedder.encode_image(image)
        else:
            print("Encoding image query from bytes")
            from io import BytesIO
            image = Image.open(BytesIO(query_image_bytes)).convert("RGB")
            query_embedding = self.embedder.encode_image(image)

        query_params = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": top_k * 3,
        }
        if filters:
            query_params["where"] = filters
            print(f"Applying filters: {filters}")

        results = self.collection.query(**query_params)
        deduplicated = self._deduplicate_results(results)
        products = self._to_product_results(deduplicated[:top_k])
        print(f"Single-mode search retrieved {len(products)} unique products")
        return products

    # ── dual-modality search (text AND image) ────────────────────────────────

    _DUAL_BOOST = 1.05  # products matching in BOTH modalities get a 5% score boost

    def _dual_search(self, query_text, query_image_path, query_image_bytes, top_k, filters):
        """Run text search and image search separately, merge and re-rank."""
        print(f"Dual-mode search: text='{query_text}' + image")

        text_embedding = self.embedder.encode_text(query_text)

        if query_image_path:
            image = Image.open(query_image_path).convert("RGB")
        else:
            from io import BytesIO
            image = Image.open(BytesIO(query_image_bytes)).convert("RGB")
        image_embedding = self.embedder.encode_image(image)

        query_kwargs = {"n_results": top_k * 3}
        if filters:
            query_kwargs["where"] = filters
            print(f"Applying filters: {filters}")

        text_results = self.collection.query(query_embeddings=[text_embedding.tolist()], **query_kwargs)
        image_results = self.collection.query(query_embeddings=[image_embedding.tolist()], **query_kwargs)

        text_deduped = self._deduplicate_results(text_results)
        image_deduped = self._deduplicate_results(image_results)

        merged: Dict[str, dict] = {}
        for item in text_deduped:
            pid = item['product_id']
            merged[pid] = {**item, '_text_score': item['similarity'], '_image_score': 0.0, 'modality': 'text'}

        for item in image_deduped:
            pid = item['product_id']
            if pid in merged:
                merged[pid]['_image_score'] = item['similarity']
                merged[pid]['modality'] = 'both'
                merged[pid]['similarity'] = max(merged[pid]['_text_score'], item['similarity'])
            else:
                merged[pid] = {**item, '_text_score': 0.0, '_image_score': item['similarity'], 'modality': 'image'}

        for item in merged.values():
            if item['modality'] == 'both':
                item['similarity'] = min(item['similarity'] * self._DUAL_BOOST, 1.0)

        ranked = sorted(merged.values(), key=lambda x: x['similarity'], reverse=True)[:top_k]
        products = self._to_product_results(ranked)
        print(f"Dual-mode search retrieved {len(products)} unique products "
              f"({sum(1 for p in products if p.matched_modality == 'both')} matched both modalities)")
        return products
    
    def _to_product_results(self, items: List[Dict]) -> List[ProductResult]:
        """Convert raw deduped dicts to ProductResult objects."""
        return [
            ProductResult(
                product_id=item['product_id'],
                similarity_score=item['similarity'],
                matched_modality=item.get('modality', 'text'),
                name=item.get('name'),
                description=item.get('description'),
                price=item.get('price'),
                category=item.get('category'),
                brand=item.get('brand'),
                image_url=item.get('image_url'),
                color=item.get('color'),
                design=item.get('design'),
            )
            for item in items
        ]

    def _deduplicate_results(self, results: Dict) -> List[Dict]:
        """
        Deduplicate results by product_id, keeping the highest similarity entry.
        
        Args:
            results: ChromaDB query results
        
        Returns:
            List of deduplicated result dictionaries
        """
        seen = {}
        
        for doc_id, distance, metadata in zip(
            results['ids'][0],
            results['distances'][0],
            results['metadatas'][0]
        ):
            # product_id required for dedup; fallback to doc_id without _text/_image suffix
            product_id = metadata.get('product_id') or str(doc_id).rsplit('_', 1)[0]
            similarity = 1 - distance  # Convert distance to similarity

            # Keep the entry with highest similarity for each product_id (dynamic schema: use .get)
            if product_id not in seen or similarity > seen[product_id]['similarity']:
                seen[product_id] = {
                    'product_id': product_id,
                    'name': metadata.get('name'),
                    'description': metadata.get('description'),
                    'price': metadata.get('price'),
                    'category': metadata.get('category'),
                    'brand': metadata.get('brand'),
                    'image_url': metadata.get('image_url'),
                    'color': metadata.get('color'),
                    'design': metadata.get('design'),
                    'similarity': similarity,
                    'modality': metadata.get('modality', 'text'),
                }
        
        # Sort by similarity (highest first)
        deduplicated = sorted(seen.values(), key=lambda x: x['similarity'], reverse=True)
        return deduplicated
    
    def retrieve_by_text(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
        max_price: Optional[float] = None
    ) -> List[ProductResult]:
        """
        Convenience method for text-based retrieval with common filters.
        
        Args:
            query: Text query string
            top_k: Number of results
            category: Filter by category
            max_price: Filter by maximum price
        
        Returns:
            List of ProductResult objects
        """
        filters = {}
        if category:
            filters["category"] = category
        if max_price:
            filters["price"] = {"$lte": max_price}
        
        return self.retrieve_products(
            query_text=query,
            top_k=top_k,
            filters=filters if filters else None
        )
    
    def retrieve_by_image(
        self,
        image_path: str,
        top_k: int = 5,
        category: Optional[str] = None
    ) -> List[ProductResult]:
        """
        Convenience method for image-based retrieval.
        
        Args:
            image_path: Path to query image
            top_k: Number of results
            category: Filter by category
        
        Returns:
            List of ProductResult objects
        """
        filters = {"category": category} if category else None
        
        return self.retrieve_products(
            query_image_path=image_path,
            top_k=top_k,
            filters=filters
        )


def main():
    """Test the hybrid retriever."""
    retriever = HybridRetriever()
    
    print("\n" + "="*70)
    print("TEST 1: Text query with cross-modal retrieval")
    print("="*70)
    results = retriever.retrieve_by_text("gaming console with ray tracing", top_k=3)
    for i, product in enumerate(results, 1):
        print(f"\n{i}. {product.name}")
        print(f"   Price: ${product.price}")
        print(f"   Matched via: {product.matched_modality}")
        print(f"   Similarity: {product.similarity_score:.4f}")
    
    print("\n" + "="*70)
    print("TEST 2: Text query with price filter (under $300)")
    print("="*70)
    results = retriever.retrieve_by_text("gaming console", top_k=3, max_price=300)
    for i, product in enumerate(results, 1):
        print(f"\n{i}. {product.name}")
        print(f"   Price: ${product.price}")
        print(f"   Matched via: {product.matched_modality}")
        print(f"   Similarity: {product.similarity_score:.4f}")
    
    print("\n" + "="*70)
    print("TEST 3: Image query")
    print("="*70)
    results = retriever.retrieve_by_image("../ps5.jpg", top_k=3)
    for i, product in enumerate(results, 1):
        print(f"\n{i}. {product.name}")
        print(f"   Price: ${product.price}")
        print(f"   Matched via: {product.matched_modality}")
        print(f"   Similarity: {product.similarity_score:.4f}")
    
    print("\n" + "="*70)
    print("TEST 4: Hybrid search - text query + category filter")
    print("="*70)
    results = retriever.retrieve_by_text("portable device", top_k=3, category="gaming")
    for i, product in enumerate(results, 1):
        print(f"\n{i}. {product.name}")
        print(f"   Category: {product.category}")
        print(f"   Price: ${product.price}")
        print(f"   Matched via: {product.matched_modality}")
        print(f"   Similarity: {product.similarity_score:.4f}")


if __name__ == "__main__":
    main()
