"""Experimental readable rendering of already-projected competition fields.

No enrichment, summarization, field selection, or benchmark lookup. JSON strings
are rendered structurally; all their values (including false/zero/null) remain.
"""

import json


def readable(value, depth=0, *, parse_json=True):
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            parsed = None
        if parse_json and isinstance(parsed, (dict, list)):
            return readable(parsed, depth, parse_json=False)
        return value
    if isinstance(value, (dict, list)):
        if not value:
            return json.dumps(value)
        entries = value.items() if isinstance(value, dict) else enumerate(value, 1)
        return "\n".join(f"{'  ' * depth}{key}: {readable(child, depth + 1, parse_json=parse_json)}"
                         for key, child in entries)
    return json.dumps(value, ensure_ascii=False)


def render_text(state):
    sections = []
    if state["target_system"]:
        sections.append("Target AI system\n" + readable(state["target_system"]))
    sections.append("Target task\n" + readable(state["target_item"]))
    for index, example in enumerate(state.get("revealed_examples", []), 1):
        sections.append(f"Revealed example {index}\n" + readable(example))
    return "\n\n".join(sections)


# Only explicit settings containers are URL-decoded; task text and arbitrary
# literal strings must never be interpreted as encoded instructions/configuration.
_CONFIG_KEYS = {"agent_configuration", "tool_configuration", "configuration", "config", "settings"}
_GROUPS = (
    ("Recorded model identifiers and checkpoints", {
        "model_identifier", "source_model", "source_model_name", "source_model_identifier",
        "source_model_label", "reported_model", "reported_model_id", "base_model", "checkpoint"}),
    ("Harness and agent settings", _CONFIG_KEYS | {"serving_environment", "inference_api", "sandbox"}),
    ("Reasoning and generation settings", {"reasoning_effort", "temperature", "top_p", "seed",
        "max_tokens", "max_new_tokens", "max_output_tokens", "max_input_tokens", "reported_task_generation_size"}),
    ("Prompting and few-shot settings", {"n_shots", "shot_size", "method", "prompting", "prompt_style",
        "prompt_condition", "prompting_condition", "system_prompt", "api_system_prompt", "chat_template"}),
)


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


def setting_value(key, value):
    from urllib.parse import unquote
    candidate = value
    decoded = False
    if key in _CONFIG_KEYS and value.lstrip().lower().startswith(('%7b', '%5b')):
        import re
        if re.search(r'%(?![0-9a-fA-F]{2})', value):
            return value
        try:
            candidate = unquote(value, encoding='utf-8', errors='strict')
        except UnicodeError:
            return value
        decoded = True
    try:
        # Duplicate JSON keys cannot be represented by a dict without data loss.
        def unique(pairs):
            result = {}
            for name, child in pairs:
                if name in result:
                    raise ValueError('duplicate JSON key')
                result[name] = child
            return result
        parsed = json.loads(candidate, object_pairs_hook=unique,
                            parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite')))
    except (ValueError, TypeError, RecursionError):
        return value
    if not isinstance(parsed, (dict, list)):
        return value
    # Compact structural JSON preserves leaf strings and their escaping literally.
    result = json.dumps(parsed, ensure_ascii=False, separators=(',', ':'))
    return ('[URL-decoded JSON] ' if decoded else '') + result


def render_context(state):
    """context_v1: organize supplied fields, with no identity/metadata enrichment."""
    system = state['target_system']
    lines = []
    for key in ('normalized_name', 'provider', 'harness', 'harness_version', 'reasoning_effort'):
        if system.get(key):
            lines.append(f'{key}: {system[key]}')
    extra = system.get('subject_features_extra')
    if extra:
        entries = split_settings(extra)
        if entries is None:
            lines.append('subject_features_extra (unparsed literal): ' + extra)
        else:
            lines.append('subject_features_extra (recorded settings):')
            remaining = list(entries)
            for title, keys in _GROUPS:
                selected = [(key, value) for key, value in remaining if key in keys]
                remaining = [(key, value) for key, value in remaining if key not in keys]
                if selected:
                    lines.append('  ' + title + ':')
                    lines.extend(f'    {key}: {setting_value(key, value)}' for key, value in selected)
            if remaining:
                lines.append('  Other recorded settings:')
                lines.extend(f'    {key}: {setting_value(key, value)}' for key, value in remaining)
    for key in ('release_date', 'access_date'):
        if system.get(key):
            lines.append(f'{key}: {system[key]}')
    sections = ['Target AI system\n' + '\n'.join(lines)] if lines else []
    # Keep the stable item labels used by the exact native token visibility audit.
    # Task contents remain literal: quoted JSON, code, and encoded strings are data.
    sections.append('Target task\n' + '\n'.join(f'{key}: {value}' for key, value in state['target_item'].items()))
    for index, example in enumerate(state.get('revealed_examples', []), 1):
        sections.append(f'Revealed example {index}\n' + '\n'.join(
            f'{key}: {value}' for key, value in example['item'].items()) + f"\noutcome: {example['outcome']}")
    return '\n\n'.join(sections)
