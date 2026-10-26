"""Structured features for the CatBoost base predictor.

Every column is a transparent measurement of the twelve published competition
fields, of their absence, or - in the declared ``system_profile`` group - of a
documented research-profile fact attached to a supplied system identifier. No
benchmark name, reference answer, grading rule or outcome is read here.
Anonymous benchmark IDs are never a feature; they exist only to scope revealed
evidence elsewhere. Profile values are computed in ``system_profile_features``
from external catalogs, so that group is research-only.

Task text is reduced to counts, proportions and structural flags. It is not
tokenized into learned vocabulary, embedded, clipped or annotated with skills,
difficulty or language identity. The feature catalog below is the declared
contract: name, ablation group, kind and definition for each column.

Standard library only, so feature construction is testable without CatBoost.
"""

from datetime import date
import math
import re

from .competition_fields import SUBJECT_FIELDS, ITEM_FIELDS, checked_fields

VERSION = "catboost_structured_v1"

SYSTEM_GROUPS = ("system_attributes", "system_missingness")
TASK_GROUPS = ("task_structure", "item_supplied")
IDENTITY_GROUP = "system_identity"
# Documented research-profile facts. Values are computed in
# ``system_profile_features`` because they need the external catalogs; the
# columns are declared here so this module stays standard-library only and the
# catalog remains the single contract. External sources are research-only.
PROFILE_GROUP = "system_profile"
# The canonical system name is a diagnostic ablation only. The competition is a
# benchmark-level cold start with subject coverage that may differ between pools,
# so a name column can memorize training systems without transferring.
DEFAULT_GROUPS = (*SYSTEM_GROUPS, *TASK_GROUPS)

# name -> (group, kind, definition). Keep this the single source of truth.
CATALOG = {
    "provider": ("system_attributes", "categorical", "Supplied provider string, or __missing__."),
    "harness": ("system_attributes", "categorical", "Supplied harness string, or __missing__."),
    "size_class": ("system_attributes", "categorical", "Narrow parameter-size class parsed from the supplied name."),
    "release_year_class": ("system_attributes", "categorical", "ISO release year as a category, or __missing__."),
    "release_year_scaled": ("system_attributes", "numeric", "(ISO release year - 2000) / 20; 0 when unknown."),
    "release_date_known": ("system_attributes", "numeric", "1 when release_date parses as an ISO date."),
    "access_date_known": ("system_attributes", "numeric", "1 when access_date parses as an ISO date."),
    "nominal_parameters_known": ("system_attributes", "numeric", "1 when the name contains exactly one plain N B/M size label."),
    "nominal_parameters_log": ("system_attributes", "numeric", "log1p(nominal parameters) / 30, without inventing MoE totals."),
    "nominal_parameters_is_moe": ("system_attributes", "numeric", "1 when the name shows an AxB expert label."),
    "extra_setting_count": ("system_attributes", "numeric", "Count of key=value settings in subject_features_extra."),
    "extra_setting_characters_log": ("system_attributes", "numeric", "log1p(characters of subject_features_extra)."),
    "harness_version_characters_log": ("system_attributes", "numeric", "log1p(characters of harness_version)."),
    "reasoning_effort_characters_log": ("system_attributes", "numeric", "log1p(characters of reasoning_effort)."),
    "missing_normalized_name": ("system_missingness", "numeric", "1 when the supplied system name is empty."),
    "missing_provider": ("system_missingness", "numeric", "1 when provider is empty."),
    "missing_release_date": ("system_missingness", "numeric", "1 when release_date is empty."),
    "missing_access_date": ("system_missingness", "numeric", "1 when access_date is empty."),
    "missing_harness": ("system_missingness", "numeric", "1 when harness is empty."),
    "missing_harness_version": ("system_missingness", "numeric", "1 when harness_version is empty."),
    "missing_reasoning_effort": ("system_missingness", "numeric", "1 when reasoning_effort is empty."),
    "missing_subject_features_extra": ("system_missingness", "numeric", "1 when subject_features_extra is empty."),
    "task_characters_log": ("task_structure", "numeric", "log1p(characters of item_content), no clipping."),
    "task_words_log": ("task_structure", "numeric", "log1p(whitespace-separated words)."),
    "task_lines_log": ("task_structure", "numeric", "log1p(newline count + 1)."),
    "task_paragraphs_log": ("task_structure", "numeric", "log1p(blank-line separated paragraphs)."),
    "task_sentences_log": ("task_structure", "numeric", "log1p(terminal punctuation groups + 1)."),
    "task_mean_word_characters": ("task_structure", "numeric", "Mean word length, 0 for empty text."),
    "task_digit_fraction": ("task_structure", "numeric", "Digits / characters."),
    "task_letter_fraction": ("task_structure", "numeric", "Unicode letters / characters."),
    "task_uppercase_fraction": ("task_structure", "numeric", "Uppercase letters / characters."),
    "task_nonascii_fraction": ("task_structure", "numeric", "Non-ASCII characters / characters."),
    "task_punctuation_fraction": ("task_structure", "numeric", "Punctuation characters / characters."),
    "task_has_code_fence": ("task_structure", "numeric", "1 when item_content contains a ``` fence."),
    "task_backtick_count_log": ("task_structure", "numeric", "log1p(backtick count)."),
    "task_has_math_marker": ("task_structure", "numeric", "1 when the text contains a dollar sign, backslash-parenthesis or backslash-begin marker."),
    "task_has_table_pipe": ("task_structure", "numeric", "1 when item_content contains a pipe character."),
    "task_option_marker_count": ("task_structure", "numeric", "Count of line-leading A-H option labels."),
    "task_question_marks": ("task_structure", "numeric", "Count of question marks."),
    "task_bullet_count": ("task_structure", "numeric", "Count of line-leading - or * bullets."),
    "task_numbered_item_count": ("task_structure", "numeric", "Count of line-leading 1. / 1) items."),
    "task_numeric_span_count": ("task_structure", "numeric", "Count of decimal-number spans."),
    "task_math_operator_count": ("task_structure", "numeric", "Count of + - * / = < > operator characters."),
    "task_ends_with_question": ("task_structure", "numeric", "1 when the stripped text ends in ?."),
    "task_has_cjk": ("task_structure", "numeric", "1 when CJK ideographs appear."),
    "task_has_cyrillic": ("task_structure", "numeric", "1 when Cyrillic characters appear."),
    "task_has_arabic": ("task_structure", "numeric", "1 when Arabic characters appear."),
    "item_features_characters_log": ("item_supplied", "numeric", "log1p(characters of item_features)."),
    "item_features_present": ("item_supplied", "numeric", "1 when item_features is nonempty."),
    "interactors_characters_log": ("item_supplied", "numeric", "log1p(characters of interactors)."),
    "interactors_present": ("item_supplied", "numeric", "1 when interactors is nonempty."),
    "profile_model_found": (PROFILE_GROUP, "numeric", "1 when the supplied subject resolves to a documented model profile."),
    "profile_harness_found": (PROFILE_GROUP, "numeric", "1 when the supplied harness resolves to a documented harness profile."),
    "profile_documented_numeric_facts": (PROFILE_GROUP, "numeric", "Count of documented numeric architecture facts, 0 to 8."),
    "profile_total_parameters_log": (PROFILE_GROUP, "numeric", "log1p(documented total parameters / 1e9); 0 when unknown."),
    "profile_active_parameters_log": (PROFILE_GROUP, "numeric", "log1p(documented active parameters per token / 1e9); 0 when unknown."),
    "profile_active_parameter_share": (PROFILE_GROUP, "numeric", "Active / total parameters; 0 when either is unknown."),
    "profile_layers": (PROFILE_GROUP, "numeric", "Documented transformer layer count; 0 when unknown."),
    "profile_hidden_size_log": (PROFILE_GROUP, "numeric", "log1p(documented hidden size); 0 when unknown."),
    "profile_attention_heads_log": (PROFILE_GROUP, "numeric", "log1p(documented attention heads); 0 when unknown."),
    "profile_key_value_head_share": (PROFILE_GROUP, "numeric", "Key/value heads / attention heads; 0 when either is unknown."),
    "profile_experts_total_log": (PROFILE_GROUP, "numeric", "log1p(documented expert count); 0 when unknown."),
    "profile_family": (PROFILE_GROUP, "categorical", "Documented architecture family, or __missing__."),
    "profile_model_type": (PROFILE_GROUP, "categorical", "Documented model type, or __missing__."),
    "profile_parameterization": (PROFILE_GROUP, "categorical", "Documented parameterization, or __missing__."),
    "profile_input_modalities": (PROFILE_GROUP, "categorical", "Documented input modalities, or __missing__."),
    "profile_output_modalities": (PROFILE_GROUP, "categorical", "Documented output modalities, or __missing__."),
    "profile_workflow": (PROFILE_GROUP, "categorical", "Documented harness workflows, or __missing__."),
    "system_identity": (IDENTITY_GROUP, "categorical", "Supplied normalized_name; diagnostic ablation only."),
}

PROFILE_COLUMNS = tuple(name for name, (group, _, _) in CATALOG.items() if group == PROFILE_GROUP)

_WORD = re.compile(r"\w+", re.UNICODE)
_SENTENCE = re.compile(r"[.!?]+")
_BULLET = re.compile(r"(?m)^\s*[-*]\s+")
_NUMBERED = re.compile(r"(?m)^\s*\d+[.)]\s+")
_OPTION = re.compile(r"(?m)^\s*\(?[A-Ha-h][.):\]]\s+")
_SIZE = re.compile(r"(?<![\w.])[+-]?\d+(?:\.\d+)?\s*[bBmM](?!\w)")
_MOE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?\s*[xX]\s*\d+(?:\.\d+)?\s*[bB](?!\w)")
_SETTING = re.compile(r"[A-Za-z_][\w.-]*\s*=")


def feature_names(groups=DEFAULT_GROUPS):
    """Declared columns for the requested ablation groups, in catalog order."""
    unknown = set(groups) - {spec[0] for spec in CATALOG.values()}
    if unknown:
        raise ValueError("Unknown feature group: " + ", ".join(sorted(unknown)))
    return [name for name, (group, _, _) in CATALOG.items() if group in groups]


def categorical_names(groups=DEFAULT_GROUPS):
    return [name for name in feature_names(groups) if CATALOG[name][1] == "categorical"]


def _fraction(count, length):
    return float(count) / length if length else 0.0


def _parse_year(value):
    try:
        return date.fromisoformat(value).year
    except ValueError:
        return None


def _size_measurement(name):
    """Return (size_class, parameters, is_moe) from the supplied name only.

    A single plain label such as ``8B`` is read as a nominal size. An expert
    label such as ``8x7B`` is flagged but never multiplied into a fabricated
    total or active count. Several plain labels are ambiguous and stay unknown.
    """
    if _MOE.search(name):
        return "moe_label", 0.0, 1.0
    matches = _SIZE.findall(name)
    if len(matches) != 1:
        return "unknown", 0.0, 0.0
    nominal = matches[0].replace(" ", "")
    try:
        number = float(nominal[:-1]) * (1e9 if nominal[-1].casefold() == "b" else 1e6)
    except ValueError:
        return "unknown", 0.0, 0.0
    if not math.isfinite(number) or number <= 0:
        return "unknown", 0.0, 0.0
    if number < 2e9:
        label = "lt_2b"
    elif number < 6e9:
        label = "2_to_6b"
    elif number < 1.2e10:
        label = "6_to_12b"
    elif number < 2e10:
        label = "12_to_20b"
    elif number < 5e10:
        label = "20_to_50b"
    elif number < 1e11:
        label = "50_to_100b"
    else:
        label = "gte_100b"
    return label, number, 0.0


def system_feature_record(subject):
    """Measurements of one supplied subject dictionary."""
    if not isinstance(subject, dict) or not set(subject) <= set(SUBJECT_FIELDS) or any(
            type(value) is not str for value in subject.values()):
        raise ValueError("Expected supplied system fields")
    subject = {key: subject.get(key, "") for key in SUBJECT_FIELDS}
    release_year = _parse_year(subject["release_date"])
    size_class, parameters, is_moe = _size_measurement(subject["normalized_name"])
    record = {
        "provider": subject["provider"] or "__missing__",
        "harness": subject["harness"] or "__missing__",
        "size_class": size_class,
        "release_year_class": str(release_year) if release_year is not None else "__missing__",
        "release_year_scaled": (release_year - 2000) / 20 if release_year is not None else 0.0,
        "release_date_known": float(release_year is not None),
        "access_date_known": float(_parse_year(subject["access_date"]) is not None),
        "nominal_parameters_known": float(parameters > 0),
        "nominal_parameters_log": math.log1p(parameters) / 30 if parameters > 0 else 0.0,
        "nominal_parameters_is_moe": is_moe,
        "extra_setting_count": float(len(_SETTING.findall(subject["subject_features_extra"]))),
        "extra_setting_characters_log": math.log1p(len(subject["subject_features_extra"])),
        "harness_version_characters_log": math.log1p(len(subject["harness_version"])),
        "reasoning_effort_characters_log": math.log1p(len(subject["reasoning_effort"])),
        "system_identity": subject["normalized_name"] or "__missing__",
    }
    for key in SUBJECT_FIELDS:
        record["missing_" + key] = float(not subject[key])
    return record


def task_feature_record(item):
    """Measurements of one supplied item dictionary; text is never clipped."""
    if not isinstance(item, dict) or not set(item) <= set(ITEM_FIELDS) or any(
            type(value) is not str for value in item.values()):
        raise ValueError("Expected supplied item fields")
    item = {key: item.get(key, "") for key in ITEM_FIELDS}
    text = item["item_content"]
    length = len(text)
    words = _WORD.findall(text)
    nonascii = sum(ord(character) > 127 for character in text)
    record = {
        "task_characters_log": math.log1p(length),
        "task_words_log": math.log1p(len(text.split())),
        "task_lines_log": math.log1p(text.count("\n") + 1),
        "task_paragraphs_log": math.log1p(1 + sum(
            1 for block in re.split(r"\n\s*\n", text) if block.strip())),
        "task_sentences_log": math.log1p(len(_SENTENCE.findall(text)) + 1),
        "task_mean_word_characters": _fraction(sum(len(word) for word in words), len(words)),
        "task_digit_fraction": _fraction(sum(character.isdigit() for character in text), length),
        "task_letter_fraction": _fraction(sum(character.isalpha() for character in text), length),
        "task_uppercase_fraction": _fraction(sum(character.isupper() for character in text), length),
        "task_nonascii_fraction": _fraction(nonascii, length),
        "task_punctuation_fraction": _fraction(
            sum(character in ".,;:!?()[]{}<>\"'`-/\\|@#$%^&*_+=~" for character in text), length),
        "task_has_code_fence": float("```" in text),
        "task_backtick_count_log": math.log1p(text.count("`")),
        "task_has_math_marker": float("$" in text or "\\(" in text or "\\begin{" in text),
        "task_has_table_pipe": float("|" in text),
        "task_option_marker_count": float(len(_OPTION.findall(text))),
        "task_question_marks": float(text.count("?")),
        "task_bullet_count": float(len(_BULLET.findall(text))),
        "task_numbered_item_count": float(len(_NUMBERED.findall(text))),
        "task_numeric_span_count": float(len(re.findall(r"\d+(?:\.\d+)?", text))),
        "task_math_operator_count": float(sum(text.count(op) for op in "+-*/=<>")),
        "task_ends_with_question": float(text.strip().endswith("?")),
        "task_has_cjk": float(bool(re.search(r"[\u3400-\u9fff]", text))),
        "task_has_cyrillic": float(bool(re.search(r"[\u0400-\u04ff]", text))),
        "task_has_arabic": float(bool(re.search(r"[\u0600-\u06ff]", text))),
        "item_features_characters_log": math.log1p(len(item["item_features"])),
        "item_features_present": float(bool(item["item_features"])),
        "interactors_characters_log": math.log1p(len(item["interactors"])),
        "interactors_present": float(bool(item["interactors"])),
    }
    return record


def feature_record(subject, item, extra=None):
    """Merged record, including the diagnostic identity column.

    ``extra`` carries the declared profile columns when that group is selected;
    it is supplied by the caller so this module needs no catalog access.
    """
    checked_fields(subject, item)
    record = {**system_feature_record(subject), **task_feature_record(item)}
    if extra:
        if set(extra) != set(PROFILE_COLUMNS):
            raise ValueError("Extra columns must be exactly the declared profile columns")
        record.update(extra)
    return record


def select(record, groups=DEFAULT_GROUPS):
    """Project a record onto the declared columns of the requested groups."""
    return {name: record[name] for name in feature_names(groups)}


__all__ = ["CATALOG", "DEFAULT_GROUPS", "IDENTITY_GROUP", "ITEM_FIELDS", "PROFILE_COLUMNS",
           "PROFILE_GROUP", "SUBJECT_FIELDS", "TASK_GROUPS", "SYSTEM_GROUPS", "VERSION",
           "categorical_names", "feature_names", "feature_record", "select",
           "system_feature_record", "task_feature_record"]
