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
DESCRIPTION_WORTHY_SCHEMA_PROPERTIES = frozenset(
    {
        "action",
        "event",
        "mode",
        "operation",
        "period",
        "state",
        "target_state",
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
ADAPTIVE_NON_PLURAL_S_TERMS = frozenset(
    {
        "access",
        "analysis",
        "analytics",
        "downstairs",
        "news",
        "series",
        "species",
        "status",
        "upstairs",
    }
)
ADAPTIVE_NON_PLURAL_S_SUFFIXES = ("ss", "us", "is", "ics", "ness", "stairs")
ADAPTIVE_PRESENT_HISTORY_ACTIONS = frozenset({"close", "open"})
ADAPTIVE_PRESENT_HISTORY_CONTEXT_TOKENS = frozenset(
    {
        "ago",
        "count",
        "counts",
        "did",
        "duration",
        "ever",
        "history",
        "how",
        "last",
        "long",
        "many",
        "often",
        "recorded",
        "time",
        "times",
        "today",
        "was",
        "were",
        "when",
        "yesterday",
    }
)
ADAPTIVE_PRESENT_HISTORY_TERMS = ("access", "history", "recorder")
ADAPTIVE_ENTITY_ID_DOMAINS = frozenset(
    {
        "alarm_control_panel",
        "automation",
        "binary_sensor",
        "button",
        "calendar",
        "camera",
        "climate",
        "conversation",
        "cover",
        "device_tracker",
        "event",
        "fan",
        "humidifier",
        "input_boolean",
        "input_button",
        "input_datetime",
        "input_number",
        "input_select",
        "input_text",
        "lawn_mower",
        "light",
        "lock",
        "media_player",
        "notify",
        "number",
        "person",
        "remote",
        "scene",
        "script",
        "select",
        "sensor",
        "siren",
        "stt",
        "sun",
        "switch",
        "text",
        "timer",
        "todo",
        "tts",
        "update",
        "vacuum",
        "valve",
        "water_heater",
        "weather",
        "zone",
    }
)
ADAPTIVE_GENERIC_ENTITY_QUERY_TERMS = ADAPTIVE_ENTITY_ID_DOMAINS | frozenset(
    {
        "entity",
        "entities",
    }
)
ADAPTIVE_ENTITY_ID_REFERENCE_RE = re.compile(
    r"(?<![@\w])(?P<host>(?:"
    + "|".join(
        re.escape(domain)
        for domain in sorted(ADAPTIVE_ENTITY_ID_DOMAINS, key=len, reverse=True)
    )
    + r")\.[a-z0-9_]+)(?=$|[^a-z0-9_])",
    flags=re.IGNORECASE,
)
ADAPTIVE_URL_ACTION_TERMS = frozenset(
    {"browse", "fetch", "read", "summarise", "summarize", "visit"}
)
ADAPTIVE_EXPLICIT_URL_INTENT_RE = re.compile(
    r"https?://\S+"
    r"|(?<!@)\bwww\.(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}"
    r"(?=$|[^a-z0-9_-])"
    r"(?::\d{2,5})?(?:/[^\s]*)?",
    flags=re.IGNORECASE,
)
ADAPTIVE_BARE_DOMAIN_INTENT_RE = re.compile(
    r"(?<![@.])\b(?P<host>(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z]{2,63})"
    r"(?=$|[^a-z0-9_-])"
    r"(?::\d{2,5})?(?:/[^\s]*)?",
    flags=re.IGNORECASE,
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


def compact_schema_for_llm(
    schema: Any,
    *,
    keep_description: bool = False,
    property_name: str | None = None,
) -> Any:
    """Strip nonessential JSON-schema verbosity before sending tools to the LLM."""
    if isinstance(schema, list):
        compacted_list = [
            compact_schema_for_llm(
                item,
                keep_description=keep_description,
                property_name=property_name,
            )
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
            if (
                keep_description
                or str(property_name or "").casefold()
                in DESCRIPTION_WORTHY_SCHEMA_PROPERTIES
            ):
                compact_description = compact_text(str(value), max_len=120)
                if compact_description:
                    compacted[key] = compact_description
            continue

        if key == "properties":
            properties: dict[str, Any] = {}
            for prop_name, prop_schema in value.items():
                compact_prop = compact_schema_for_llm(
                    prop_schema,
                    property_name=str(prop_name),
                )
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
            value,
            keep_description=keep_description,
            property_name=property_name,
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


def _has_adaptive_url_action_context(text: str, match: re.Match[str]) -> bool:
    """Return true when a bare-domain match is preceded by a URL-reading verb."""
    preceding_terms = re.findall(r"\w+", text[: match.start()], flags=re.UNICODE)[-3:]
    return any(term in ADAPTIVE_URL_ACTION_TERMS for term in preceding_terms)


def _is_adaptive_entity_id_like_host(
    host: str,
    *,
    text: str = "",
    match: re.Match[str] | None = None,
) -> bool:
    """Return true for dotted Home Assistant entity IDs that resemble bare domains."""
    labels = str(host or "").casefold().split(".")
    if len(labels) != 2 or labels[0] not in ADAPTIVE_ENTITY_ID_DOMAINS:
        return False
    return not (match is not None and _has_adaptive_url_action_context(text, match))


def _has_adaptive_url_intent(text: str) -> bool:
    """Return true when text includes an explicit URL or a non-entity bare domain."""
    if ADAPTIVE_EXPLICIT_URL_INTENT_RE.search(text):
        return True
    return any(
        not _is_adaptive_entity_id_like_host(
            match.group("host"),
            text=text,
            match=match,
        )
        for match in ADAPTIVE_BARE_DOMAIN_INTENT_RE.finditer(text)
    )


def _strip_adaptive_url_intents(text: str) -> str:
    """Remove URL-like spans already represented by url/web terms."""
    stripped = ADAPTIVE_EXPLICIT_URL_INTENT_RE.sub(" ", text)

    def replace_bare_domain(match: re.Match[str]) -> str:
        if _is_adaptive_entity_id_like_host(
            match.group("host"),
            text=stripped,
            match=match,
        ):
            return match.group(0)
        return " "

    return ADAPTIVE_BARE_DOMAIN_INTENT_RE.sub(replace_bare_domain, stripped)


def _has_adaptive_entity_reference(text: str) -> bool:
    """Return true when text contains an HA entity-id-shaped reference."""
    return bool(_adaptive_entity_reference_domains(text))


def _adaptive_entity_reference_domains(text: str) -> set[str]:
    """Return HA entity domains referenced by entity-id-shaped text spans."""
    domains: set[str] = set()
    for match in ADAPTIVE_BARE_DOMAIN_INTENT_RE.finditer(text):
        host = match.group("host")
        if _is_adaptive_entity_id_like_host(
            host,
            text=text,
            match=match,
        ):
            domains.add(host.split(".", 1)[0].casefold())
    for match in ADAPTIVE_ENTITY_ID_REFERENCE_RE.finditer(text):
        host = match.group("host")
        if _is_adaptive_entity_id_like_host(
            host,
            text=text,
            match=match,
        ):
            domains.add(host.split(".", 1)[0].casefold())
    return domains


def normalize_adaptive_query_terms(query: str) -> list[str]:
    """Return useful search terms for adaptive tool matching."""
    normalized_query = str(query or "").casefold()
    terms = []
    if _has_adaptive_url_intent(normalized_query):
        terms.extend(["url", "webpage", "web"])
        normalized_query = _strip_adaptive_url_intents(normalized_query)
    for term in re.findall(r"\w+", normalized_query, flags=re.UNICODE):
        if len(term) < 2 or term in ADAPTIVE_QUERY_STOPWORDS:
            continue
        if term not in terms:
            terms.append(term)
        if singular := _adaptive_singular_text_term(term):
            if singular not in terms:
                terms.append(singular)
    normalized_tokens = set(re.findall(r"\w+", normalized_query, flags=re.UNICODE))
    for alias, expanded_terms in ADAPTIVE_QUERY_ALIASES.items():
        matches_alias = (
            alias in normalized_tokens
            if re.fullmatch(r"[a-z0-9_]+", alias)
            else alias in normalized_query
        )
        if matches_alias:
            for term in expanded_terms:
                if term not in terms:
                    terms.append(term)
    if (
        normalized_tokens & ADAPTIVE_PRESENT_HISTORY_ACTIONS
        and normalized_tokens & ADAPTIVE_PRESENT_HISTORY_CONTEXT_TOKENS
    ):
        for term in ADAPTIVE_PRESENT_HISTORY_TERMS:
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
        if singular := _adaptive_singular_text_term(term):
            terms.add(singular)
    return terms


def _adaptive_singular_text_term(term: str) -> str | None:
    """Return a conservative singular form for plural metadata terms."""
    if len(term) <= 3 or not term.endswith("s"):
        return None
    if (
        term in ADAPTIVE_NON_PLURAL_S_TERMS
        or term.endswith(ADAPTIVE_NON_PLURAL_S_SUFFIXES)
    ):
        return None
    if term.endswith("ies") and len(term) > 4:
        return f"{term[:-3]}y"
    return term[:-1]


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

    matched_terms: set[str] = set()
    matched_name_terms: set[str] = set()
    for term in terms:
        if term in name_terms:
            score += 24
            matched_terms.add(term)
            matched_name_terms.add(term)
        if term in keyword_terms:
            score += 18
            matched_terms.add(term)
        if term in routing_terms:
            score += 14
            matched_terms.add(term)
        if term in llm_description_terms:
            score += 12
            matched_terms.add(term)
        if term in description_terms:
            score += 6
            matched_terms.add(term)

    if name not in base_tool_names and score > 0:
        entity_domains = _adaptive_entity_reference_domains(normalized_query)
        if (
            matched_terms
            and matched_terms <= ADAPTIVE_GENERIC_ENTITY_QUERY_TERMS
            and entity_domains
            and not (matched_name_terms & entity_domains)
        ):
            return 0
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
