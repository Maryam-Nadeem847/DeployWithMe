import os
from pathlib import Path

from dotenv import load_dotenv

# Prefer .env; also allow .env.example in local experiments.
load_dotenv()
load_dotenv(".env.example")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")

# Groq exposes an OpenAI-compatible API
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

DEFAULT_CONTAINER_PYTHON = os.getenv("DEPLOY_AGENT_CONTAINER_PYTHON", "3.11")
BUILD_ROOT = Path(os.getenv("DEPLOY_AGENT_BUILD_ROOT", Path.cwd() / ".deploy_builds")).resolve()
