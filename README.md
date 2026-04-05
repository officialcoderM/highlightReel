# AI Internship Project

## Day 1: Virtual Environment Setup
- Created project folder `ai-internship-project`
- Created virtual environment with `python3 -m venv env`
- Activated environment

## Day 2: OpenAI API – First Chat Completion

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
  - `max_tokens` limits output length.# highlightReel
