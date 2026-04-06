from __future__ import annotations

import json
import logging
import re
from typing import Any

from deployment_agent import config

logger = logging.getLogger(__name__)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def complete_json(system: str, user: str) -> dict[str, Any]:
    """Ask the primary model for a JSON object; fall back to Groq on 429/503."""
    raw = _complete_raw(system, user)
    parsed = _extract_json_object(raw)
    if parsed is None:
        raise ValueError(f"Model did not return valid JSON. Raw:\n{raw[:2000]}")
    return parsed


def complete_text(system: str, user: str) -> str:
    return _complete_raw(system, user).strip()


def _complete_raw(system: str, user: str) -> str:
    if config.GEMINI_API_KEY:
        try:
            return _gemini_generate(system, user)
        except Exception as e:
            err = str(e).lower()
            if "429" in err or "503" in err or "resource exhausted" in err or "unavailable" in err:
                logger.warning("Gemini unavailable (%s); trying Groq.", e)
            else:
                raise
    if not config.GROQ_API_KEY:
        raise RuntimeError(
            "No working LLM: set GEMINI_API_KEY and/or GROQ_API_KEY for healing / ambiguous detection."
        )
    return _groq_generate(system, user)


def _gemini_generate(system: str, user: str) -> str:
    from google import genai

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    # google-genai: contents can be a string or structured parts
    prompt = f"{system}\n\n{user}"
    resp = client.models.generate_content(model=config.GEMINI_MODEL, contents=prompt)
    text = getattr(resp, "text", None)
    if text:
        return text
    if resp.candidates:
        parts = resp.candidates[0].content.parts
        return "".join(getattr(p, "text", "") for p in parts)
    raise RuntimeError("Empty Gemini response")


def _groq_generate(system: str, user: str) -> str:
    from groq import Groq

    client = Groq(api_key=config.GROQ_API_KEY)
    chat = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    choice = chat.choices[0].message.content
    if not choice:
        raise RuntimeError("Empty Groq response")
    return choice
