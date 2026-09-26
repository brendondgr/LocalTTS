"""AI-Enhance: an LLM inserts Breeze vocal-event tags such as (sigh) and suggests a delivery.

Breeze reads events written in parentheses; Kokoro does not, so Kokoro requests have the
known tags stripped (see strip_tags). The LLM is any OpenAI-compatible chat endpoint,
by default the DashLLM relay on this machine (LOCALTTS_LLM_URL / LOCALTTS_LLM_MODEL).
"""

from __future__ import annotations

import asyncio
import json
import re
import urllib.error
import urllib.request

from .config import settings

MAX_ENHANCE_CHARS = 6000

# Vocal events Breeze documents; AI-Enhance uses only these. tag -> description.
TAGS: dict[str, str] = {
    "laugh": "a laugh",
    "sigh": "a sigh",
    "cough": "a cough",
    "clears throat": "clearing the throat",
}
# Not documented. Breeze does not read these out as words (checked with Whisper), but whether
# each one makes the sound was not verified by ear, so the UI lists them as "try it".
EXPERIMENTAL: dict[str, str] = {
    "chuckle": "a short chuckle",
    "giggle": "a giggle",
    "gasp": "a gasp",
    "groan": "a groan",
    "sniff": "a sniff",
    "breath": "a breath",
    "hmm": "a thoughtful hmm",
    "cry": "crying",
}
KNOWN = {**TAGS, **EXPERIMENTAL}
# Spellings an LLM tends to produce -> the tag Breeze knows.
SYNONYMS = {"laughs": "laugh", "laughing": "laugh", "sighs": "sigh", "sighing": "sigh",
            "coughs": "cough", "coughing": "cough", "clear throat": "clears throat",
            "clearing throat": "clears throat", "throat clear": "clears throat",
            "chuckles": "chuckle", "giggles": "giggle", "gasps": "gasp", "groans": "groan",
            "sniffs": "sniff", "breathes": "breath", "cries": "cry", "crying": "cry"}

_PAREN = re.compile(r"[(\[]\s*([A-Za-z][A-Za-z ]{0,30}?)\s*[)\]]")


def _canon(word: str) -> str | None:
    w = " ".join(word.lower().split())
    w = SYNONYMS.get(w, w)
    return w if w in KNOWN else None


def strip_tags(text: str) -> str:
    """Drop known vocal-event tags (for engines that would read them aloud)."""
    out = _PAREN.sub(lambda m: " " if _canon(m.group(1)) else m.group(0), text)
    return re.sub(r"[ \t]{2,}", " ", out).strip()


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", strip_tags(text).lower())


def _clean(original: str, text: str) -> tuple[str, int]:
    """Normalise tag spelling to (tag), drop invented parentheticals. -> (text, tags in it)"""
    kept = {m.group(0) for m in _PAREN.finditer(original)}
    count = 0

    def fix(m: re.Match) -> str:
        nonlocal count
        if m.group(0) in kept:
            return m.group(0)
        tag = _canon(m.group(1))
        if tag:
            count += 1
            return f"({tag})"
        return " "  # a tag Breeze does not know would be spoken as a word

    out = _PAREN.sub(fix, text)
    return re.sub(r"[ \t]{2,}", " ", out).strip(), count


STYLES = {
    "subtle": "Add about one tag for every two or three sentences (at least one for any text "
              "with a clear emotion), choosing the spots where it matters most.",
    "expressive": "Be expressive: add a tag to most sentences where a person would plausibly make "
                  "that sound, but never two tags in a row.",
}


def _prompt(style: str) -> str:
    tags = "\n".join(f"- ({t}): {d}" for t, d in TAGS.items())
    return f"""You make text more expressive for a text-to-speech voice that performs vocal events.

The ONLY vocal-event tags that exist are these, written exactly like this, in parentheses:
{tags}

Rules:
1. Insert tags where the speaker would naturally make that sound: a sigh before resignation or
   relief, a laugh after or before something funny or delightful, clears throat before a formal
   or awkward statement, a cough for hesitation or embarrassment. {STYLES.get(style, STYLES["subtle"])}
2. Put each tag at the start of a sentence or right after a comma or full stop, never inside a phrase.
3. Keep every original word and punctuation mark exactly as given, in order. Only insert tags.
4. Write "delivery": one short direction for the voice actor about emotion, tone and pace,
   at most 15 words.

Reply with JSON only, no commentary: {{"text": "<the text with tags>", "delivery": "<direction>"}}

Examples:
Input: Well, that didn't go the way I planned. Anyway, let's try again tomorrow.
Output: {{"text": "(sigh) Well, that didn't go the way I planned. Anyway, let's try again tomorrow.", "delivery": "Resigned but lightly amused, unhurried."}}
Input: You wore that to the interview? That's amazing. Did they say anything?
Output: {{"text": "You wore that to the interview? (laugh) That's amazing. Did they say anything?", "delivery": "Delighted disbelief, playful and quick."}}
Input: Right. Before we start, I need to apologise for last week.
Output: {{"text": "(clears throat) Right. Before we start, I need to apologise for last week.", "delivery": "Awkward and sincere, a little slow."}}"""


class EnhanceError(RuntimeError):
    pass


def _post(path: str, body: dict | None = None, timeout: float | None = None) -> dict:
    headers = {"content-type": "application/json"}
    if settings.llm_api_key:
        headers["authorization"] = f"Bearer {settings.llm_api_key}"
    req = urllib.request.Request(settings.llm_url + path, headers=headers,
                                 data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=timeout or settings.llm_timeout_s) as r:
            return json.load(r)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        try:
            detail = json.loads(detail)["error"]["message"]
        except Exception:
            pass
        raise EnhanceError(f"LLM endpoint returned {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise EnhanceError(f"LLM endpoint {settings.llm_url} is unreachable ({exc})") from exc


def _parse(content: str) -> dict:
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in the reply")
    data = json.loads(content[start:end + 1])
    if not isinstance(data.get("text"), str):
        raise ValueError("reply has no text")
    return data


async def models() -> dict:
    data = await asyncio.to_thread(_post, "/models", None, 5)
    out = []
    for m in data.get("data", []):
        health = (m.get("relay") or {}).get("health")
        out.append({"id": m["id"], "healthy": health != "failed"})
    return {"url": settings.llm_url, "default": settings.llm_model, "models": out}


async def enhance(text: str, style: str = "subtle", model: str | None = None,
                  direction: str | None = None) -> dict:
    model = model or settings.llm_model
    user = text if not direction else f"(Current direction for the speaker: {direction})\n\n{text}"
    messages = [{"role": "system", "content": _prompt(style)}, {"role": "user", "content": user}]
    want, warning, result = _words(text), None, None
    for attempt in range(2):
        body = {"model": model, "messages": messages, "temperature": 0.4,
                "max_tokens": min(8192, 400 + len(text))}
        reply = await asyncio.to_thread(_post, "/chat/completions", body)
        content = reply["choices"][0]["message"].get("content") or ""
        try:
            data = _parse(content)
        except (ValueError, json.JSONDecodeError):
            messages += [{"role": "assistant", "content": content},
                         {"role": "user", "content": 'Reply with the JSON object only: {"text": ..., "delivery": ...}'}]
            continue
        out, count = _clean(text, data["text"])
        result = {"text": out, "delivery": str(data.get("delivery") or "").strip(), "tags_added": count,
                  "model": reply.get("model", model)}
        if _words(out) == want:
            return result
        warning = "the AI changed some words; review the text before generating"
        messages += [{"role": "assistant", "content": content},
                     {"role": "user", "content": "You changed the original words. Keep every word exactly "
                                                 "as given and only insert tags. Try again, JSON only."}]
    if result is None:
        raise EnhanceError("the LLM did not return usable JSON; try again or pick another model")
    return {**result, "warning": warning}
