# Multimodal RAG Agent

A Multimodal RAG (Retrieval-Augmented Generation) AI Assistant designed to search an e-commerce product catalog using text, audio, and images.

The core AI Agent is built using the **Google GenAI SDK**, allowing it to dynamically route multimodal user queries and retrieve the most relevant products from a vector database.

## Features
- **Multimodal Search**: Query the catalog via text, voice commands, or image uploads.
- **Google SDK Agent**: An intelligent backend built with Google's framework that handles reasoning and tool calling.
- **RAG Architecture**: Augments the LLM with real context fetched from a database for accurate product recommendations.

## Tech Stack
- **Backend:** Python, Google GenAI SDK, Vector Embeddings
- **Frontend:** Streamlit 
- **Data:** JSON-based product catalog 
