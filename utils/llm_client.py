"""
Provider-agnostic LLM client (OpenAI-compatible transport).
===========================================================

Call sites ask for a *role*, not a model:

    complete(prompt, role="research")                 # web-capable, verbose, cited
    complete_json(prompt, schema, role="judgment")    # strict card JSON

Config maps role -> (provider, model), so swapping models is a settings change
and no call site moves. That matters because the two paths of the valuation
pipeline want genuinely different models: the research pass is high-volume and
tolerant (one call per name, cheap model, web access), while the judgment pass
is low-volume and must adhere to a JSON schema.

Why this replaces the Gemini binding
------------------------------------
ai_report_engine.py hard-bound `genai.GenerativeModel('gemini-1.5-pro')`, keyed
off a password box in the sidebar (main.py). That is a per-session,
manually-typed credential — workable for one ad-hoc report, unusable for a batch
across ~1000 names, and it silently reduces to "feature unavailable" whenever
the box is empty. An OpenAI-compatible base_url + key from config works for
OpenCode's local server, and for anything else that speaks the same protocol.

Structured output
-----------------
`complete_json` asks for a json_schema response when the model supports it and
falls back to json_object plus local validation when it does not. Adherence
varies by model, so validation is always performed locally regardless of what
the server claims — a malformed card is rejected here rather than becoming a
plausible-looking verdict downstream.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

DEFAULT_TIMEOUT = 180
DEFAULT_RETRIES = 3

# Sensible fallbacks; override in config. Roles, not hard-coded models.
DEFAULT_ROLES = {
    "research": {"provider": "opencode", "model": "deepseek-v4-flash"},
    "judgment": {"provider": "opencode", "model": "gpt-5.6-luna"},
}
DEFAULT_BASE_URLS = {"opencode": "http://localhost:4096/v1"}


class LLMNotConfigured(RuntimeError):
    pass


class LLMSchemaError(ValueError):
    pass


# ---------------------------------------------------------------------------
def _secrets() -> Dict:
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "llm" in st.secrets:
            return dict(st.secrets["llm"])
    except Exception:
        pass
    return {}


@dataclass
class Endpoint:
    provider: str
    model: str
    base_url: str
    api_key: str


def resolve(role: str = "judgment", model: Optional[str] = None,
            provider: Optional[str] = None) -> Endpoint:
    """Role -> concrete endpoint, with env overriding config overriding defaults."""
    cfg = _secrets()
    roles = dict(DEFAULT_ROLES)
    for k, v in (cfg.get("roles") or {}).items():
        roles[k] = dict(v)

    spec = dict(roles.get(role) or roles["judgment"])
    if provider:
        spec["provider"] = provider
    if model:
        spec["model"] = model

    spec["model"] = (os.environ.get(f"LLM_MODEL_{role.upper()}")
                     or spec.get("model") or "")
    prov = spec.get("provider") or "opencode"

    providers = cfg.get("providers") or {}
    pcfg = dict(providers.get(prov) or {})
    base_url = (os.environ.get("LLM_BASE_URL") or pcfg.get("base_url")
                or DEFAULT_BASE_URLS.get(prov, ""))
    api_key = (os.environ.get("LLM_API_KEY") or pcfg.get("api_key")
               # local servers frequently ignore the key but the SDK demands one
               or "not-needed")

    if not base_url:
        raise LLMNotConfigured(
            f"no base_url for provider '{prov}'. Set LLM_BASE_URL, or "
            f"[llm.providers.{prov}] base_url in Streamlit secrets.")
    if not spec["model"]:
        raise LLMNotConfigured(f"no model configured for role '{role}'.")
    return Endpoint(prov, spec["model"], base_url.rstrip("/"), api_key)


def is_configured(role: str = "judgment") -> bool:
    try:
        resolve(role)
        return True
    except LLMNotConfigured:
        return False


def status(role: str = "judgment") -> str:
    try:
        e = resolve(role)
        return f"{role}: {e.model} via {e.provider} ({e.base_url})"
    except LLMNotConfigured as exc:
        return f"{role}: not configured — {exc}"


def _client(ep: Endpoint):
    try:
        from openai import OpenAI
    except ImportError as exc:                       # pragma: no cover
        raise LLMNotConfigured(
            "the `openai` package is required for the OpenAI-compatible "
            "transport — add it to requirements.txt") from exc
    return OpenAI(base_url=ep.base_url, api_key=ep.api_key, timeout=DEFAULT_TIMEOUT)


# ---------------------------------------------------------------------------
def complete(prompt: str, role: str = "judgment", *, system: Optional[str] = None,
             model: Optional[str] = None, provider: Optional[str] = None,
             temperature: float = 0.2, max_tokens: Optional[int] = None,
             retries: int = DEFAULT_RETRIES) -> str:
    """Plain text completion."""
    ep = resolve(role, model, provider)
    client = _client(ep)
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]

    last = None
    for attempt in range(1, retries + 1):
        try:
            kw: Dict[str, Any] = dict(model=ep.model, messages=msgs,
                                      temperature=temperature)
            if max_tokens:
                kw["max_tokens"] = max_tokens
            r = client.chat.completions.create(**kw)
            return (r.choices[0].message.content or "").strip()
        except Exception as exc:
            last = exc
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM call failed after {retries} attempts: {last}")


def _extract_json(text: str) -> str:
    """Models wrap JSON in prose or fences more often than they should."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
        t = t.strip()
    if not t.startswith(("{", "[")):
        for op, cl in (("{", "}"), ("[", "]")):
            i, j = t.find(op), t.rfind(cl)
            if i != -1 and j > i:
                return t[i:j + 1]
    return t


def _validate(obj: Any, schema: Dict) -> None:
    try:
        import jsonschema
    except ImportError:
        return                                   # validation is best-effort
    try:
        jsonschema.validate(obj, schema)
    except Exception as exc:
        raise LLMSchemaError(str(exc)) from exc


def complete_json(prompt: str, schema: Dict, role: str = "judgment", *,
                  system: Optional[str] = None, model: Optional[str] = None,
                  provider: Optional[str] = None, temperature: float = 0.0,
                  retries: int = DEFAULT_RETRIES) -> Dict:
    """
    Structured completion. Tries the json_schema response format, falls back to
    json_object, and validates locally either way — a server that accepts the
    schema is not evidence that the model honoured it.
    """
    ep = resolve(role, model, provider)
    client = _client(ep)
    sys_msg = (system or "") + (
        "\n\nReturn a single JSON object matching the required schema. "
        "No prose, no code fences, no trailing commentary.")
    msgs = [{"role": "system", "content": sys_msg.strip()},
            {"role": "user", "content": prompt}]

    formats = [
        {"type": "json_schema",
         "json_schema": {"name": "card", "strict": True, "schema": schema}},
        {"type": "json_object"},
        None,
    ]

    last: Optional[Exception] = None
    for fmt in formats:
        for attempt in range(1, retries + 1):
            try:
                kw: Dict[str, Any] = dict(model=ep.model, messages=msgs,
                                          temperature=temperature)
                if fmt:
                    kw["response_format"] = fmt
                r = client.chat.completions.create(**kw)
                raw = (r.choices[0].message.content or "").strip()
                obj = json.loads(_extract_json(raw))
                _validate(obj, schema)
                return obj
            except LLMSchemaError as exc:
                # The model answered but broke the contract — re-ask with the
                # validator's complaint rather than silently accepting it.
                last = exc
                msgs = msgs[:2] + [
                    {"role": "assistant", "content": raw[:4000]},
                    {"role": "user",
                     "content": f"That response failed schema validation: {exc}. "
                                f"Return corrected JSON only."}]
            except Exception as exc:
                last = exc
                msg = str(exc).lower()
                if any(s in msg for s in ("response_format", "json_schema",
                                          "not supported", "unsupported")):
                    break                        # try the next, looser format
                if attempt < retries:
                    time.sleep(2 ** attempt)
    raise RuntimeError(f"structured LLM call failed: {last}")
