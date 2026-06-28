# Medical Chatbot System

## Architecture Overview
The Medical Chatbot System is an AI-powered conversational agent designed to retrieve and provide context-aware medical information. It leverages a Retrieval-Augmented Generation (RAG) architecture using LangChain, OpenAI's GPT-4o-mini, Pinecone Vector Database, and a Flask backend to serve real-time user queries.

## Prerequisites
Before you begin, ensure you have the following installed:
- **Python**: Version 3.11+
- **pip**: Python package installer
- **Git**: To clone the repository
- API keys for **OpenAI** and **Pinecone**.

## Local Development Setup

Follow these steps to set up the project locally:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/abdelrhmanHasaan/Medical-system.git
   cd Medical-system
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Linux/MacOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   Copy the example environment variables file and fill in your credentials.
   ```bash
   cp .env.example .env
   ```
   Open the `.env` file and set your API keys:
   ```ini
   PINECONE_API_KEY="your_pinecone_api_key_here"
   OPENAI_API_KEY="your_openai_api_key_here"
   ```

5. **Index the Data (First Time Setup):**
   Place your medical PDFs inside the `./data/` directory (create it if it doesn't exist). Then, run the indexing script to parse the documents, create text chunks, and store the embeddings in Pinecone.
   ```bash
   python store_index.py
   ```

6. **Run the Application:**
   Start the local Flask development server.
   ```bash
   python app.py
   ```
   The chatbot will be accessible at `http://localhost:8080/`.

## Production Deployment Guide

For a production environment, you should use a robust WSGI server (like Gunicorn) instead of the built-in Flask development server.

1. **Environment Setup:** Ensure all dependencies from `requirements.txt` are installed in the production environment.
2. **Environment Variables:** Inject the `PINECONE_API_KEY` and `OPENAI_API_KEY` into your production environment securely (e.g., using AWS Secrets Manager, GitHub Secrets, or a secure `.env` file).
3. **Build & Run:** Use Gunicorn to start the application with multiple workers for concurrency.
   ```bash
   # Install Gunicorn if not present
   pip install gunicorn

   # Run the application with Gunicorn
   gunicorn -w 4 -b 0.0.0.0:8080 app:app
   ```
4. **Proxy:** It is recommended to place the application behind a reverse proxy like NGINX or an Application Load Balancer to handle SSL termination and static asset delivery.
