# DeepWiki-Open

![DeepWiki Banner](screenshots/Deepwiki.png)

**DeepWiki** is my own implementation attempt of DeepWiki, automatically creates beautiful, interactive wikis for any GitHub, GitLab, or BitBucket repository! Just enter a repo name, and DeepWiki will:

1. Analyze the code structure
2. Generate comprehensive documentation
3. Create visual diagrams to explain how everything works
4. Organize it all into an easy-to-navigate wiki

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://buymeacoffee.com/sheing)

[![Twitter/X](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://x.com/sashimikun_void)
[![Discord](https://img.shields.io/badge/Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.com/invite/VQMBGR8u5v)

## ✨ Features

- **Instant Documentation**: Turn any GitHub, GitLab or BitBucket repo into a wiki in seconds
- **Private Repository Support**: Securely access private repositories with personal access tokens
- **Local Repository Analysis**: Analyze repositories directly from your local filesystem
- **Interactive Mermaid Diagrams**: Automatically creates visual diagrams for architecture, workflows, and data flows
- **LLM-Powered Auto-Fix**: Automatically detects and fixes Mermaid diagram syntax errors using AI when diagrams fail to render
- **Smart Wiki Structure**: Intelligently organizes documentation based on repo type and contents
- **Context-Aware Generation**: Creates documentation tailored to the specific technologies in your repo
- **Self-Contained**: All generated documentation is stored locally for privacy and performance
- **Responsive Design**: Clean, modern UI that works on desktop and mobile devices
- **Repository Snapshots**: Get a quick overview of the technologies, languages, and services used in the project
- **Progressive Content Generation**: Watch the documentation generate in real-time, with visual progress tracking
- **Ask Feature**: Chat with your repository using RAG-powered AI to get accurate answers

## 🏃 Running the Project

### Running Locally (Development)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AsyncFuncAI/deepwiki-open.git
   cd deepwiki-open
   ```
2. **Run the setup script:**
   ```bash
   bash scripts/setup_dev.sh
   ```
   This will create a virtual environment, install dependencies, and check for API keys.
3. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```
4. **Start the backend API:**
   ```bash
   python -m api.main
   ```
5. **Start the frontend (in a new terminal):**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
6. **Access the app:**
   Open [http://localhost:3000](http://localhost:3000) in your browser.

### Running with Docker Compose

1. **Edit the .env file with your API keys** (see example above).
2. **Build and start the app:**
   ```bash
   docker-compose up --build
   ```
   - Backend API: [http://localhost:8001](http://localhost:8001)
   - Frontend: [http://localhost:3000](http://localhost:3000)

   **Note:** Docker volumes are used for persistence:
   - `wiki-data` (for generated wikis)
   - `wiki-data/chromadb` (for vector storage)

### Running with Docker (Manual)

1. **Build the Docker image:**
   ```bash
   docker build -t deepwiki-open .
   ```
2. **Run the container:**
   ```bash
   docker run -p 8001:8001 -p 3000:3000 \
     -e GOOGLE_API_KEY=your_google_api_key \
     -e OPENAI_API_KEY=your_openai_api_key \
     -v $(pwd)/wiki-data:/app/wiki-data \
     -v $(pwd)/wiki-data/chromadb:/app/wiki-data/chromadb \
     deepwiki-open
   ```

### Using Utility Scripts

All utility and CLI scripts are in the `/scripts` directory. For example:
```bash
python scripts/list_collections.py
bash scripts/curl_test_customs_chat.sh
```

## 🔍 How the RAG Pipeline Works

DeepWiki uses a modular, graph-based RAG pipeline powered by LangGraph:

1. **Document Loading:** Clones the repo or scans a local directory, excluding irrelevant files.
2. **Text Splitting:** Chunks documents by file type for optimal embedding.
3. **Embedding:** Generates embeddings using OpenAI or Ollama.
4. **Vector Storage:** Stores vectors and metadata in ChromaDB (persisted in Docker volume).
5. **Retrieval:** Finds relevant code/docs for your query.
6. **Generation:** Uses Gemini (or OpenAI) to answer based on retrieved docs.
7. **Memory:** Tracks chat history for context in multi-turn conversations.

There are two main pipeline modes:
- **Full Pipeline (Indexing + Retrieval):** Used when a repo is first added or reindexed. Runs all steps above.
- **Retrieval-Only Mode:** Used for chat/questions after indexing. Only runs retrieval, generation, and memory nodes.

You can test the pipeline directly:
```bash
# Test with a GitHub repo
python -m api.test_langgraph --repo https://github.com/username/repository

# Test with a local directory
python -m api.test_langgraph --local /path/to/local/directory

# Additional options:
#   --ollama   Use Ollama models (requires Ollama running locally)
#   --top-k N  Set number of docs to retrieve (default: 5)
```

## 🧩 Collection Name Resolution

When resolving ChromaDB collections, DeepWiki will (soon) support prefix-based matching to handle cases where collection names have unique suffixes. For now, ensure you use the full collection name as listed by the utility scripts.

## 🧩 LangGraph Backend (Current Implementation)

DeepWiki now uses a modern, graph-based RAG backend powered by [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain]. This replaces the legacy adalflow pipeline and brings several improvements:

- **Flexible Graph Architecture:** Modular, node-based RAG pipeline for indexing and chat.
- **Enhanced Repository Support:** Works with both Git repositories and local directories.
- **Persistent Vector Storage:** Uses ChromaDB for efficient, persistent vector storage (see Docker volumes).
- **Improved File Filtering:** Smarter exclusion of irrelevant files (e.g., node_modules, .git).
- **Multi-Repository Queries:** (Coming soon) Support for querying across multiple repositories.
- **Better Configuration:** Uses Pydantic models for flexible pipeline config.

> **Note:**
> This documentation and codebase reflect a complete rewrite of DeepWiki. All previous documentation, features, and architecture have been superseded by this new implementation.

## 🛠️ Project Structure

```
deepwiki/
├── api/                  # Backend API server
│   ├── main.py           # API entry point
│   ├── api.py            # FastAPI implementation
│   ├── rag.py            # Retrieval Augmented Generation
│   ├── data_pipeline.py  # Data processing utilities
│   └── requirements.txt  # Python dependencies
│
├── src/                  # Frontend Next.js app
│   ├── app/              # Next.js app directory
│   │   └── page.tsx      # Main application page
│   └── components/       # React components
│       └── Mermaid.tsx   # Mermaid diagram renderer
│
├── public/               # Static assets
├── package.json          # JavaScript dependencies
└── .env                  # Environment variables (create this)
├── scripts/              # Utility and CLI scripts
│   ├── list_collections.py
│   ├── curl_test_customs_chat.sh
│   ├── setup_dev.sh
│   ├── verify_embeddings.py
│   ├── generate_embeddings.py
│   ├── test_langgraph.py
│   └── test_api.py
```

_All utility and CLI scripts are now in the `/scripts` directory._

## 🛠️ Advanced Setup

### Environment Variables

| Variable | Description | Required | Note |
|----------|-------------|----------|------|
| `GOOGLE_API_KEY` | Google Gemini API key for AI generation | Yes |
| `OPENAI_API_KEY` | OpenAI API key for embeddings | Yes |
| `PORT` | Port for the API server (default: 8001) | No | If you host API and frontend on the same machine, make sure change port of `NEXT_PUBLIC_SERVER_BASE_URL` accordingly |
| `NEXT_PUBLIC_SERVER_BASE_URL` | Base URL for the API server (default: http://localhost:8001) | No |

### Docker Setup

You can use Docker to run DeepWiki:

```bash
# Pull the image from GitHub Container Registry
docker pull ghcr.io/asyncfuncai/deepwiki-open:latest

# Run the container with environment variables
docker run -p 8001:8001 -p 3000:3000 \
  -e GOOGLE_API_KEY=your_google_api_key \
  -e OPENAI_API_KEY=your_openai_api_key \
  -v ~/.adalflow:/root/.adalflow \
  ghcr.io/asyncfuncai/deepwiki-open:latest
```

Or use the provided `docker-compose.yml` file:

```bash
# Edit the .env file with your API keys first
docker-compose up
```

#### Using a .env file with Docker

You can also mount a .env file to the container:

```bash
# Create a .env file with your API keys
echo "GOOGLE_API_KEY=your_google_api_key" > .env
echo "OPENAI_API_KEY=your_openai_api_key" >> .env

# Run the container with the .env file mounted
docker run -p 8001:8001 -p 3000:3000 \
  -v $(pwd)/.env:/app/.env \
  -v ~/.adalflow:/root/.adalflow \
  ghcr.io/asyncfuncai/deepwiki-open:latest
```

#### Building the Docker image locally

If you want to build the Docker image locally:

```bash
# Clone the repository
git clone https://github.com/AsyncFuncAI/deepwiki-open.git
cd deepwiki-open

# Build the Docker image
docker build -t deepwiki-open .

# Run the container
docker run -p 8001:8001 -p 3000:3000 \
  -e GOOGLE_API_KEY=your_google_api_key \
  -e OPENAI_API_KEY=your_openai_api_key \
  deepwiki-open
```

### API Server Details

The API server provides:
- Repository cloning and indexing
- RAG (Retrieval Augmented Generation)
- Streaming chat completions

For more details, see the [API README](./api/README.md).

## 🤖 Ask & DeepResearch Features

### Ask Feature

The Ask feature allows you to chat with your repository using Retrieval Augmented Generation (RAG):

- **Context-Aware Responses**: Get accurate answers based on the actual code in your repository
- **RAG-Powered**: The system retrieves relevant code snippets to provide grounded responses
- **Real-Time Streaming**: See responses as they're generated for a more interactive experience
- **Conversation History**: The system maintains context between questions for more coherent interactions

## ❓ Troubleshooting

### API Key Issues
- **"Missing environment variables"**: Make sure your `.env` file is in the project root and contains both API keys
- **"API key not valid"**: Check that you've copied the full key correctly with no extra spaces

### Connection Problems
- **"Cannot connect to API server"**: Make sure the API server is running on port 8001
- **"CORS error"**: The API is configured to allow all origins, but if you're having issues, try running both frontend and backend on the same machine

### Generation Issues
- **"Error generating wiki"**: For very large repositories, try a smaller one first
- **"Invalid repository format"**: Make sure you're using a valid GitHub, GitLab or Bitbucket URL format
- **"Could not fetch repository structure"**: For private repositories, ensure you've entered a valid personal access token with appropriate permissions
- **"Diagram rendering error"**: The app will automatically try to fix broken diagrams

### Common Solutions
1. **Restart both servers**: Sometimes a simple restart fixes most issues
2. **Check console logs**: Open browser developer tools to see any JavaScript errors
3. **Check API logs**: Look at the terminal where the API is running for Python errors

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Open issues for bugs or feature requests
- Submit pull requests to improve the code
- Share your feedback and ideas

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=AsyncFuncAI/deepwiki-open&type=Date)](https://star-history.com/#AsyncFuncAI/deepwiki-open&Date)

## Useful Features

### Direct Collection Name Access

For repositories with special characters in their names, you can directly specify the collection name to bypass automatic resolution:

```
https://your-deepwiki-instance.com/?repo=repository_id&collection=collection_name
```

To find the correct collection name for a repository, use the utility script:

```bash
./check_collections.py repository_id
```

See `docs/CHAT_API_GUIDE.md` for more details.
