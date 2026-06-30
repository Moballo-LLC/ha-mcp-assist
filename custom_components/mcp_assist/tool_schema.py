"""Provider-neutral MCP tool schema helpers."""

from __future__ import annotations

import json
import re
from typing import Any

ADAPTIVE_TOOL_CATALOG_NAME = "list_available_tools"
ADAPTIVE_TOOL_SCHEMA_NAME = "load_tool_schemas"
ADAPTIVE_META_TOOL_NAMES = frozenset(
    {
        ADAPTIVE_TOOL_CATALOG_NAME,
        ADAPTIVE_TOOL_SCHEMA_NAME,
    }
)
ADAPTIVE_QUERY_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "again",
        "also",
        "and",
        "any",
        "are",
        "as",
        "at",
        "can",
        "check",
        "could",
        "did",
        "does",
        "for",
        "from",
        "get",
        "go",
        "has",
        "have",
        "how",
        "in",
        "is",
        "into",
        "it",
        "let",
        "look",
        "many",
        "me",
        "much",
        "my",
        "of",
        "on",
        "please",
        "read",
        "show",
        "summarise",
        "summarize",
        "summary",
        "that",
        "the",
        "there",
        "this",
        "today",
        "turn",
        "use",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "with",
        "would",
        "you",
    }
)
ADAPTIVE_QUERY_ALIASES: dict[str, tuple[str, ...]] = {
    # Weather and forecasts
    "météo": ("weather", "forecast"),
    "tiempo": ("weather", "forecast"),
    "clima": ("weather", "forecast"),
    "pronóstico": ("weather", "forecast"),
    "meteo": ("weather", "forecast"),
    "wetter": ("weather", "forecast"),
    "weer": ("weather", "forecast"),
    "pogoda": ("weather", "forecast"),
    "previsão": ("weather", "forecast"),
    "previsao": ("weather", "forecast"),
    "väder": ("weather", "forecast"),
    "vær": ("weather", "forecast"),
    "vejr": ("weather", "forecast"),
    "sää": ("weather", "forecast"),
    "počasí": ("weather", "forecast"),
    "καιρός": ("weather", "forecast"),
    "hava": ("weather", "forecast"),
    "panahon": ("weather", "forecast"),
    "طقس": ("weather", "forecast"),
    "मौसम": ("weather", "forecast"),
    "погода": ("weather", "forecast"),
    "天气": ("weather", "forecast"),
    "天氣": ("weather", "forecast"),
    "天気": ("weather", "forecast"),
    "날씨": ("weather", "forecast"),
    # Calendar, agenda, and tasks
    "calendrier": ("calendar", "event"),
    "calendario": ("calendar", "event"),
    "kalender": ("calendar", "event"),
    "kalendarz": ("calendar", "event"),
    "calendário": ("calendar", "event"),
    "календарь": ("calendar", "event"),
    "日历": ("calendar", "event"),
    "日程": ("calendar", "event"),
    "カレンダー": ("calendar", "event"),
    "일정": ("calendar", "event"),
    "ημερολόγιο": ("calendar", "event"),
    "takvim": ("calendar", "event"),
    "kalendář": ("calendar", "event"),
    "kalenteri": ("calendar", "event"),
    "تقويم": ("calendar", "event"),
    "कैलेंडर": ("calendar", "event"),
    "agenda": ("calendar", "event"),
    "evento": ("calendar", "event"),
    "événement": ("calendar", "event"),
    "termin": ("calendar", "event"),
    "tarea": ("todo", "task"),
    "tâche": ("todo", "task"),
    "aufgabe": ("todo", "task"),
    "задача": ("todo", "task"),
    "任务": ("todo", "task"),
    "タスク": ("todo", "task"),
    "할일": ("todo", "task"),
    # Memory
    "mémoire": ("memory", "remember"),
    "memoria": ("memory", "remember"),
    "pamięć": ("memory", "remember"),
    "memória": ("memory", "remember"),
    "память": ("memory", "remember"),
    "记忆": ("memory", "remember"),
    "記憶": ("memory", "remember"),
    "メモリ": ("memory", "remember"),
    "기억": ("memory", "remember"),
    "μνήμη": ("memory", "remember"),
    "hafıza": ("memory", "remember"),
    "alaala": ("memory", "remember"),
    "ذاكرة": ("memory", "remember"),
    "याद": ("memory", "remember"),
    # Search and web reading
    "buscar": ("search", "web"),
    "rechercher": ("search", "web"),
    "suchen": ("search", "web"),
    "cercare": ("search", "web"),
    "procurar": ("search", "web"),
    "zoeken": ("search", "web"),
    "szukaj": ("search", "web"),
    "pesquisar": ("search", "web"),
    "поиск": ("search", "web"),
    "искать": ("search", "web"),
    "搜索": ("search", "web"),
    "搜尋": ("search", "web"),
    "検索": ("search", "web"),
    "검색": ("search", "web"),
    "sök": ("search", "web"),
    "søk": ("search", "web"),
    "arama": ("search", "web"),
    "بحث": ("search", "web"),
    "खोज": ("search", "web"),
    # Music
    "música": ("music", "media"),
    "musique": ("music", "media"),
    "musik": ("music", "media"),
    "musica": ("music", "media"),
    "muziek": ("music", "media"),
    "muzyka": ("music", "media"),
    "музыка": ("music", "media"),
    "音乐": ("music", "media"),
    "音樂": ("music", "media"),
    "音楽": ("music", "media"),
    "음악": ("music", "media"),
    "müzik": ("music", "media"),
    "musika": ("music", "media"),
    "موسيقى": ("music", "media"),
    "संगीत": ("music", "media"),
    # History and recorder
    "historia": ("history", "recorder"),
    "historique": ("history", "recorder"),
    "verlauf": ("history", "recorder"),
    "chronologie": ("history", "recorder"),
    "cronologia": ("history", "recorder"),
    "geschiedenis": ("history", "recorder"),
    "histórico": ("history", "recorder"),
    "история": ("history", "recorder"),
    "历史": ("history", "recorder"),
    "履歴": ("history", "recorder"),
    "기록": ("history", "recorder"),
    "historik": ("history", "recorder"),
    "historie": ("history", "recorder"),
    "geçmiş": ("history", "recorder"),
    "سجل": ("history", "recorder"),
    "इतिहास": ("history", "recorder"),
    "count": ("history", "recorder"),
    "times": ("count", "history", "recorder"),
    "opened": ("open", "access", "count", "history", "recorder"),
    "opening": ("open", "access", "history", "recorder"),
    "closed": ("close", "access", "count", "history", "recorder"),
    "closing": ("close", "access", "history", "recorder"),
    "locked": ("lock", "access", "count", "history", "recorder"),
    "unlocked": ("unlock", "access", "count", "history", "recorder"),
    # Images
    "imagen": ("image", "vision"),
    "bild": ("image", "vision"),
    "immagine": ("image", "vision"),
    "afbeelding": ("image", "vision"),
    "obraz": ("image", "vision"),
    "imagem": ("image", "vision"),
    "изображение": ("image", "vision"),
    "图片": ("image", "vision"),
    "圖片": ("image", "vision"),
    "画像": ("image", "vision"),
    "이미지": ("image", "vision"),
    "görsel": ("image", "vision"),
    "larawan": ("image", "vision"),
    "صورة": ("image", "vision"),
    "छवि": ("image", "vision"),
    # Math and conversion
    "calcular": ("calculator", "calculate"),
    "calculadora": ("calculator", "calculate"),
    "berechnen": ("calculator", "calculate"),
    "rechnen": ("calculator", "calculate"),
    "calculer": ("calculator", "calculate"),
    "calcolare": ("calculator", "calculate"),
    "oblicz": ("calculator", "calculate"),
    "converter": ("convert", "unit"),
    "convertir": ("convert", "unit"),
    "umrechnen": ("convert", "unit"),
    "unidade": ("convert", "unit"),
    "unidad": ("convert", "unit"),
    "unité": ("convert", "unit"),
    "einheit": ("convert", "unit"),
    "przelicz": ("convert", "unit"),
    "转换": ("convert", "unit"),
    "轉換": ("convert", "unit"),
    "変換": ("convert", "unit"),
    "변환": ("convert", "unit"),
    "dönüştür": ("convert", "unit"),
    "تحويل": ("convert", "unit"),
}


def compact_text(text: str, *, max_len: int = 160) -> str:
    """Compact instructional text for lower token usage."""
    normalized = " ".join(str(text).split()).strip()
    if not normalized:
        return ""

    for separator in (". ", "\n", "; "):
        if separator in normalized:
            normalized = normalized.split(separator, 1)[0].strip()
            break

    if len(normalized) <= max_len:
        return normalized

    truncated = normalized[: max_len - 1].rstrip()
    last_space = truncated.rfind(" ")
    if last_space > 40:
        truncated = truncated[:last_space]
    return truncated.rstrip(" ,;:.") + "."


def compact_schema_for_llm(schema: Any, *, keep_description: bool = False) -> Any:
    """Strip nonessential JSON-schema verbosity before sending tools to the LLM."""
    if isinstance(schema, list):
        compacted_list = [
            compact_schema_for_llm(item, keep_description=keep_description)
            for item in schema
        ]
        return [item for item in compacted_list if item not in (None, {}, [])]

    if not isinstance(schema, dict):
        return schema

    compacted: dict[str, Any] = {}

    for key, value in schema.items():
        if key in {"$schema", "title", "default", "examples", "example"}:
            continue

        if key == "description":
            if keep_description:
                compact_description = compact_text(str(value), max_len=120)
                if compact_description:
                    compacted[key] = compact_description
            continue

        if key == "properties":
            properties: dict[str, Any] = {}
            for prop_name, prop_schema in value.items():
                compact_prop = compact_schema_for_llm(prop_schema)
                if compact_prop:
                    properties[prop_name] = compact_prop
            if properties:
                compacted[key] = properties
            continue

        if key == "required":
            if value:
                compacted[key] = value
            continue

        if key == "additionalProperties":
            continue

        compact_value = compact_schema_for_llm(
            value, keep_description=keep_description
        )
        if compact_value not in (None, {}, []):
            compacted[key] = compact_value

    return compacted


def build_tool_routing_summary(routing_hints: Any) -> str:
    """Build a compact description suffix from optional routing hints."""
    if not isinstance(routing_hints, dict):
        return ""

    preferred_when = str(routing_hints.get("preferred_when") or "").strip()
    if preferred_when:
        compact_preferred = compact_text(preferred_when, max_len=90)
        if compact_preferred:
            return f"Use for: {compact_preferred}"

    example_queries = routing_hints.get("example_queries")
    if isinstance(example_queries, list):
        cleaned_examples = [
            str(item).strip() for item in example_queries if str(item).strip()
        ][:1]
        if cleaned_examples:
            compact_example = compact_text(cleaned_examples[0], max_len=80)
            if compact_example:
                return f"Example: {compact_example}"

    keywords = routing_hints.get("keywords")
    if isinstance(keywords, list):
        cleaned_keywords = [
            str(item).strip() for item in keywords if str(item).strip()
        ][:3]
        if cleaned_keywords:
            return f"Keywords: {', '.join(cleaned_keywords)}"

    return ""


def tool_definition_name(tool: dict[str, Any]) -> str:
    """Return the MCP tool name for a raw tool definition."""
    return str(tool.get("name") or "")


def normalize_adaptive_query_terms(query: str) -> list[str]:
    """Return useful search terms for adaptive tool matching."""
    normalized_query = str(query or "").casefold()
    terms = []
    if re.search(r"https?://", normalized_query):
        terms.extend(["url", "webpage", "web"])
        normalized_query = re.sub(r"https?://\S+", " ", normalized_query)
    for term in re.findall(r"\w+", normalized_query, flags=re.UNICODE):
        if len(term) < 2 or term in ADAPTIVE_QUERY_STOPWORDS:
            continue
        if term not in terms:
            terms.append(term)
    for alias, expanded_terms in ADAPTIVE_QUERY_ALIASES.items():
        if alias in normalized_query:
            for term in expanded_terms:
                if term not in terms:
                    terms.append(term)
    return terms


def _adaptive_text_terms(text: str) -> set[str]:
    """Return normalized match terms for tool metadata text."""
    normalized = re.sub(r"[_\-/]+", " ", str(text or "").casefold())
    terms: set[str] = set()
    for term in re.findall(r"\w+", normalized, flags=re.UNICODE):
        if len(term) < 2:
            continue
        terms.add(term)
        if term.endswith("s") and len(term) > 3:
            terms.add(term[:-1])
    return terms


def _routing_hint_text(tool: dict[str, Any], *keys: str) -> str:
    routing_hints = tool.get("routingHints")
    if not isinstance(routing_hints, dict):
        return ""

    parts: list[str] = []
    for key in keys:
        value = routing_hints.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts).casefold()


def score_adaptive_tool_match(
    tool: dict[str, Any],
    query: str,
    *,
    base_tool_names: frozenset[str] = frozenset(),
) -> int:
    """Score how well a raw tool definition matches an adaptive query."""
    normalized_query = " ".join(str(query or "").split()).casefold()
    terms = normalize_adaptive_query_terms(normalized_query)
    if not normalized_query and not terms:
        return 0

    name = tool_definition_name(tool).casefold()
    llm_description = str(
        tool.get("llmDescription") or tool.get("llm_description") or ""
    ).casefold()
    description = str(tool.get("description") or "").casefold()
    keyword_text = _routing_hint_text(tool, "keywords")
    routing_text = _routing_hint_text(tool, "preferred_when", "example_queries")
    name_terms = _adaptive_text_terms(name)
    keyword_terms = _adaptive_text_terms(keyword_text)
    routing_terms = _adaptive_text_terms(routing_text)
    llm_description_terms = _adaptive_text_terms(llm_description)
    description_terms = _adaptive_text_terms(description)

    score = 0
    if normalized_query and normalized_query == name:
        score += 100
    if terms and all(term in name_terms for term in terms):
        score += 40

    for term in terms:
        if term in name_terms:
            score += 24
        if term in keyword_terms:
            score += 18
        if term in routing_terms:
            score += 14
        if term in llm_description_terms:
            score += 12
        if term in description_terms:
            score += 6

    if name not in base_tool_names and score > 0:
        score += 1
    return score


def match_adaptive_tool_definitions(
    tools: list[dict[str, Any]],
    *,
    query: str = "",
    tool_names: list[str] | None = None,
    limit: int = 20,
    base_tool_names: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Return matching raw tool definitions for adaptive discovery/loading."""
    visible_tools = [
        tool
        for tool in tools
        if tool_definition_name(tool) not in ADAPTIVE_META_TOOL_NAMES
    ]
    by_name = {tool_definition_name(tool): tool for tool in visible_tools}

    if tool_names:
        matches: list[dict[str, Any]] = []
        for name in tool_names:
            tool = by_name.get(name)
            if tool and tool not in matches:
                matches.append(tool)
        return matches[:limit]

    query = " ".join(str(query or "").split()).casefold()
    if not query:
        return [
            tool
            for tool in visible_tools
            if tool_definition_name(tool) not in base_tool_names
        ][:limit]

    scored: list[tuple[int, str, dict[str, Any]]] = []
    for tool in visible_tools:
        score = score_adaptive_tool_match(
            tool,
            query,
            base_tool_names=base_tool_names,
        )
        if score > 0:
            scored.append((score, tool_definition_name(tool), tool))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [tool for _score, _name, tool in scored[:limit]]


def convert_mcp_tools_to_llm_tools(
    tools: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Convert MCP tools to a compact provider-neutral function schema."""
    llm_tools = []

    for tool in tools:
        parameters = compact_schema_for_llm(
            tool.get("inputSchema", {}), keep_description=False
        )
        if not parameters:
            parameters = {"type": "object", "properties": {}}
        elif parameters.get("type") == "object" and "properties" not in parameters:
            parameters["properties"] = {}

        llm_description = compact_text(
            str(tool.get("llmDescription") or tool.get("llm_description") or ""),
            max_len=120,
        )
        base_description = compact_text(
            llm_description or tool.get("description", ""),
            max_len=140,
        )
        description_parts = [base_description.rstrip(" .")]
        routing_summary = build_tool_routing_summary(tool.get("routingHints"))
        if routing_summary:
            description_parts.append(routing_summary)

        llm_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": compact_text(
                        " | ".join(part for part in description_parts if part),
                        max_len=220,
                    ),
                    "parameters": parameters,
                },
            }
        )

    return llm_tools


def build_adaptive_meta_tools() -> list[dict[str, Any]]:
    """Return tiny meta tools for on-demand tool discovery in adaptive mode."""
    return [
        {
            "type": "function",
            "function": {
                "name": ADAPTIVE_TOOL_CATALOG_NAME,
                "description": (
                    "Search optional, built-in, and custom MCP tools before loading "
                    "their full schemas."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": ADAPTIVE_TOOL_SCHEMA_NAME,
                "description": (
                    "Load full schemas for specific optional/custom tools so they can "
                    "be called on the next turn."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tool_names": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 8},
                    },
                },
            },
        },
    ]


def build_adaptive_llm_tools(
    tools: list[dict[str, Any]],
    *,
    base_tool_names: frozenset[str],
    loaded_tool_names: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Return the LLM-facing tool surface for adaptive context mode."""
    selected_tool_names = set(base_tool_names) | set(loaded_tool_names)
    selected_tools = [
        tool
        for tool in tools
        if str(tool.get("name") or "") in selected_tool_names
    ]
    return [
        *convert_mcp_tools_to_llm_tools(selected_tools),
        *build_adaptive_meta_tools(),
    ]


def json_size_bytes(value: Any) -> int:
    """Return UTF-8 JSON size without exposing the JSON itself."""
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)
    return len(text.encode("utf-8"))


def estimate_tokens_from_bytes(byte_count: int) -> int:
    """Return a rough conservative token estimate for compact JSON payloads."""
    return (byte_count + 3) // 4
