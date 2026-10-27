"""Lossless setting splitting and frozen content normalization for preparation."""
import json
import re
import unicodedata


def split_settings(text):
    """Parse complete top-level key=value sequences, otherwise preserve one literal.

    Return a list rather than a dictionary: duplicate keys/order/unknown keys
    carry information. Quotes and JSON nesting protect embedded semicolons.
    No heuristic extraction from an unparseable remainder is allowed.
    """
    import re
    pieces, start, quote, escaped, stack = [], 0, None, False, []
    for index, char in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in '\"\'':
            quote = char
        elif char in "[{":
            stack.append(char)
        elif char in "]}":
            if not stack or stack.pop() != {']':'[', '}':'{'}[char]:
                return None
        elif char == ';' and not stack:
            pieces.append(text[start:index])
            start = index + 1
    if quote or stack:
        return None
    pieces.append(text[start:])
    if not pieces[-1] and text.endswith(';'):
        pieces.pop()
    settings = []
    for piece in pieces:
        match = re.fullmatch(r'\s*([A-Za-z_][A-Za-z0-9_.-]*)=(.*)', piece, re.DOTALL)
        if not match:
            return None
        settings.append((match.group(1), match.group(2)))
    return settings or None


def normalized_content(content):
    """Conservative task identity across JSON/chat wrappers; never read graders.

    Preserve context/options in dictionaries. For chronological chat lists use
    the last user task. This is an exact normalized-content screen, not a claim
    to recognize translations or every paraphrase.
    """
    text = str(content)
    try:
        value = json.loads(text)
    except ValueError:
        value = None
    if isinstance(value, dict) and isinstance(value.get("messages"), list):
        value = value["messages"]
    if isinstance(value, list) and all(isinstance(v, dict) for v in value):
        users = [v.get("content", "") for v in value if v.get("role") == "user"]
        if users:
            text = str(users[-1])
    elif isinstance(value, dict):
        text = json.dumps(value, sort_keys=True, ensure_ascii=False)
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+", text))
