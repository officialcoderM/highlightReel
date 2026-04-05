AI Internship Project


Day 1: Virtual Environment Setup
- Created project folder `ai-internship-project`
- Created virtual environment with `python3 -m venv env`
- Activated environment


Day 2: OpenAI API – First Chat Completion
- Installed `openai` and `python-dotenv`.
- Created `.env` file to store API key securely.
- Wrote a Python script that:
  - Loads environment variables with `load_dotenv()`.
  - Retrieves the API key with `os.getenv()`.
  - Sends a chat completion request to `gpt-3.5-turbo` with system and user messages.
  - Prints the assistant’s response.
- Learned about:
  - Environment variables and why they’re used for secrets.
  - The structure of a chat completion call: `model`, `messages`, `temperature`, `max_tokens`.
  - How `temperature` controls randomness (low = deterministic, high = creative).
  - `max_tokens` limits output length.


Day 3: Hugging Face Local Model Text Generation
- Loaded and ran open-source language models locally on CPU without any API keys
- Tested multiple models including GPT-2, DialoGPT-small, and GPT-Neo-125m to compare output quality
- Used tokenizers to convert text to numbers the model understands and back to readable text
- Implemented generation parameters including temperature, top-k, top-p, and repetition penalty to control creativity and coherence
- Moved models to appropriate devices using PyTorch for computation
- Learned about:
  - Tokenization and why text must be converted to numbers for models to process
  - Causal language modeling and how models predict the next token based only on previous tokens
  - How temperature, top-k, and top-p work together to control output randomness
  - Why repetition penalty prevents models from getting stuck in loops
  - The relationship between model size and output quality


Day 4: RAG (Retrieval-Augmented Generation) Pipeline
- Built a complete RAG system that answers questions from documents with source citations
- Loaded documents from a text file containing ski resort information
- Split documents into 500-character chunks with 50-character overlap to preserve context across boundaries
- Used sentence transformers (all-MiniLM-L6-v2) to convert text chunks into 384-dimension embedding vectors
- Stored embeddings in ChromaDB vector database for efficient similarity search
- Retrieved top-3 most relevant chunks for each user question using semantic search
- Integrated Microsoft Phi-2 language model to generate answers grounded only in retrieved context
- Created prompts that force the model to answer only from provided context, reducing hallucinations
- Displayed source citations showing which chunks were used to generate each answer
- Learned about:
  - Semantic search and how it finds meaning rather than exact keywords
  - Embeddings and why similar text produces similar vectors
  - Vector databases and how they enable fast similarity search at scale
  - Chunking strategies and why overlap prevents information loss at boundaries
  - Grounding and why RAG reduces hallucinations compared to raw LLM generation


  