import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY", "sk-XXXX")
DEFAULT_MODEL = "gpt-4o"
FALLBACK_MODEL = "gpt-3.5-turbo"
