"""Configure the OpenAI-compatible endpoint using LLM_* environment variables."""

import os

import openai

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:20128/v1").rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
MODEL_NAME = os.getenv("LLM_MODEL", "cx/gpt-6-luna")

openai.api_type = "open_ai"
openai.api_base = LLM_BASE_URL
openai.api_version = None
openai.api_key = LLM_API_KEY
