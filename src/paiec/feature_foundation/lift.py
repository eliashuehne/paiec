"""Sample-scoped, supplied-only features for the existing lift table interface.

Only bookkeeping keys are taken from the sample. No folds, labels, foundation
split assignments, research matches or across-row estimates enter the features.
"""
from collections import Counter
import json
from numbers import Integral
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .contracts import VERSION, file_hash, write_json
from .settings import FIELDS
from .sources import safe_source
from .tables import SCHEMAS

KEYS = ('benchmark', 'subject_id', 'item_id', 'trial')
TABLES = ('subject_links', 'run_settings', 'response_input_links', 'item_input_links', 'task_features')
PROJECTION_VERSION = 'foundation_lift_v1'
PROVENANCE_FILES = ('feature_dictionary.json', 'settings_policy.json')


def _keys(sample):
    if not set(KEYS) <= set(sample):
        raise ValueError(f'Lift sample needs all keys: {KEYS}')
    keys = sample[list(KEYS)].copy()
    if keys.isna().any().any():
        raise ValueError('Lift sample keys must be non-null')
    for name in KEYS[:-1]:
        if any(not isinstance(value, str) or not value for value in keys[name]):
            raise ValueError('Lift sample IDs must be nonempty strings')
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in keys.trial):
        raise ValueError('Lift sample trials must be integers')
    return keys.drop_duplicates().reset_index(drop=True)


def _check_bundle(bundle):
    if (bundle / 'INCOMPLETE').exists():
        raise ValueError('Incomplete feature bundle')
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if manifest['version'] != VERSION:
        raise ValueError('Unsupported feature bundle schema')
    for name in TABLES:
        relative = name + '.parquet'
        path = safe_source(bundle, relative)
        if file_hash(path) != manifest['artifacts'][relative]['sha256']:
            raise ValueError('Artifact hash mismatch: ' + relative)
        reader = pq.ParquetFile(path)
        if reader.schema_arrow != SCHEMAS[name] or reader.metadata.num_rows != manifest['logical_tables'][name]['rows']:
            raise ValueError('Artifact schema or count mismatch: ' + name)
    return manifest


def _selected_responses(bundle, keys):
    columns = [*KEYS, 'item_key', 'interactor_key', 'status']
    parts = []
    if len(keys):
        for batch in pq.ParquetFile(bundle / 'response_input_links.parquet').iter_batches(
                batch_size=65536, columns=columns):
            matches = batch.to_pandas().merge(keys, on=list(KEYS), how='inner', validate='many_to_one')
            if len(matches):
                parts.append(matches)
    if not parts:
        return pd.DataFrame(columns=columns)
    # Repeated source responses with identical inputs do not create ambiguity.
    return pd.concat(parts, ignore_index=True).drop_duplicates()


def _read_selected(bundle, name, column, values):
    values = list(values)
    if not values:
        return pd.DataFrame(columns=SCHEMAS[name].names)
    return pq.read_table(bundle / (name + '.parquet'), filters=[(column, 'in', values)]).to_pandas()


def _settings(bundle, subjects):
    rows = _read_selected(bundle, 'run_settings', 'system_key', subjects.system_key.dropna().unique())
    wide = {}
    for row in rows.itertuples(index=False):
        values = wide.setdefault(row.system_key, {})
        values['recorded_' + row.field] = json.loads(row.value_json) if row.status == 'known' else None
        if FIELDS[row.field] == 'string_list' and row.status == 'known':
            values['recorded_' + row.field] = row.value_json
        values['recorded_' + row.field + '_status'] = row.status
    records = []
    for row in subjects.itertuples(index=False):
        records.append({'benchmark': row.benchmark, 'subject_id': row.subject_id, **wide.get(row.system_key, {})})
    return pd.DataFrame(records, columns=['benchmark', 'subject_id', *[
        column for field in FIELDS for column in ('recorded_' + field, 'recorded_' + field + '_status')]])


def project_lift_features(bundle, sample):
    """Return one wide row per unique sample key plus aggregate input coverage.

    Ambiguous duplicate trial keys with different input variants are withheld.
    Exact duplicate source trials are harmless. Missing/protected task values
    remain null; their reasons are audit metadata, never model columns. The
    caller's fold/sealed policy remains authoritative and is never inspected.
    Consumed tables are byte/schema checked; the full validator is separate.
    """
    bundle = Path(bundle).resolve()
    keys = _keys(sample)
    _check_bundle(bundle)
    subjects = _read_selected(bundle, 'subject_links', 'benchmark', keys.benchmark.unique())
    subjects = subjects.merge(keys[['benchmark', 'subject_id']].drop_duplicates(),
                              on=['benchmark', 'subject_id'], validate='one_to_one')
    result = keys.merge(_settings(bundle, subjects), on=['benchmark', 'subject_id'],
                        how='left', validate='many_to_one', sort=False)
    responses = _selected_responses(bundle, keys)
    counts = responses.groupby(list(KEYS)).size().rename('_variants').reset_index()
    resolved = keys.merge(counts, on=list(KEYS), how='left', validate='one_to_one')
    unique = responses.merge(counts, on=list(KEYS), validate='many_to_one')
    unique = unique.loc[unique._variants == 1].drop(columns='_variants')
    links = _read_selected(bundle, 'item_input_links', 'item_key', unique.item_key.dropna().unique())
    links = links[['item_key', 'interactor_key', 'task_key']]
    unique = unique.merge(links, on=['item_key', 'interactor_key'], how='left', validate='many_to_one')
    resolved = resolved.merge(unique, on=list(KEYS), how='left', validate='one_to_one')
    reasons = pd.Series('matched', index=resolved.index, dtype='string')
    reasons.loc[resolved.task_key.isna()] = 'missing_task'
    reasons.loc[resolved.status.eq('withheld_protected')] = 'withheld_protected'
    reasons.loc[resolved._variants.gt(1)] = 'ambiguous_trial'
    reasons.loc[resolved._variants.isna()] = 'missing_response'
    matched = reasons.eq('matched') & resolved.status.eq('recorded')
    reasons.loc[reasons.eq('matched') & ~matched] = 'unresolved_response'
    resolved.loc[~matched, 'task_key'] = None
    tasks = _read_selected(bundle, 'task_features', 'task_key', resolved.task_key.dropna().unique())
    tasks = tasks.rename(columns={name: 'task_' + name for name in tasks if name != 'task_key'})
    projected = resolved[[*KEYS, 'task_key']].merge(tasks, on='task_key', how='left', validate='many_to_one')
    result = result.merge(projected.drop(columns='task_key'), on=list(KEYS), how='left', validate='one_to_one')
    for field, kind in FIELDS.items():
        dtype = {'integer': 'Int64', 'number': 'Float64', 'boolean': 'boolean'}.get(kind, 'string')
        result['recorded_' + field] = result['recorded_' + field].astype(dtype)
        result['recorded_' + field + '_status'] = result['recorded_' + field + '_status'].astype('string')
    for field in SCHEMAS['task_features']:
        if field.name != 'task_key':
            dtype = 'boolean' if pa.types.is_boolean(field.type) else 'Int64' if pa.types.is_integer(field.type) else 'string'
            result['task_' + field.name] = result['task_' + field.name].astype(dtype)
    return result, {'sample_rows': len(sample), 'unique_keys': len(keys),
                    'task_resolution': dict(sorted(Counter(reasons).items())),
                    'recorded_settings': {field: result['recorded_' + field + '_status'].fillna('missing_subject').value_counts().sort_index().to_dict() for field in FIELDS}}


def export_lift_features(bundle, sample_path, out):
    """Write a fresh supplied-only projection with sample/source/code receipts."""
    bundle, sample_path, out = Path(bundle).resolve(), Path(sample_path).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError('Use a fresh lift feature output directory')
    if bundle.is_relative_to(out) or out.is_relative_to(bundle) or sample_path.is_relative_to(out):
        raise ValueError('Lift outputs overlap input artifacts')
    before = {'bundle_manifest_sha256': file_hash(bundle / 'manifest.json'), 'sample_sha256': file_hash(sample_path)}
    for name in PROVENANCE_FILES:
        _check_entry(bundle, name)
    policy = json.loads((bundle / 'settings_policy.json').read_text())
    if policy['fields'] != FIELDS:
        raise ValueError('Unsupported recorded setting fields')
    sample = pq.read_table(sample_path, columns=list(KEYS)).to_pandas()
    features, coverage = project_lift_features(bundle, sample)
    out.mkdir(parents=True)
    (out / 'INCOMPLETE').write_text('Projection has not been published.\n')
    features.to_parquet(out / 'features.parquet', index=False)
    write_json(out / 'coverage.json', coverage)
    dictionary = json.loads((bundle / 'feature_dictionary.json').read_text())
    entries = {row['name']: row for row in dictionary}
    projected_dictionary = []
    for name in features.columns:
        if name in KEYS:
            continue
        status = name.startswith('recorded_') and name.endswith('_status')
        source_name = ('run.' + name[len('recorded_'):].removesuffix('_status') if name.startswith('recorded_')
                       else 'task.' + name[len('task_'):])
        projected_dictionary.append({'column': name, 'dtype': str(features[name].dtype),
                                     'source_feature': source_name, 'is_status': status,
                                     'missing_behavior': 'null; never zero-filled',
                                     'source_definition': entries[source_name]})
    write_json(out / 'dictionary.json', projected_dictionary)
    after = {'bundle_manifest_sha256': file_hash(bundle / 'manifest.json'), 'sample_sha256': file_hash(sample_path)}
    if before != after:
        raise ValueError('Projection inputs changed during export')
    manifest = {'version': PROJECTION_VERSION, 'policy': 'supplied_only', **before,
                'sample_columns_read': list(KEYS), 'keys': list(KEYS), 'rows': len(features),
                'bundle_artifacts': {name: _check_entry(bundle, name) for name in
                                     (*[name + '.parquet' for name in TABLES], *PROVENANCE_FILES)},
                'code_hashes': {path.name: file_hash(path) for path in sorted(Path(__file__).parent.glob('*.py'))},
                'artifacts': {name: {'sha256': file_hash(out / name)} for name in ('features.parquet', 'coverage.json', 'dictionary.json')}}
    write_json(out / 'manifest.json', manifest)
    (out / 'INCOMPLETE').unlink()
    return manifest


def _check_entry(bundle, relative):
    # Recheck consumed bytes immediately before publication as well.
    manifest = json.loads((bundle / 'manifest.json').read_text())
    path = safe_source(bundle, relative)
    actual = file_hash(path)
    if actual != manifest['artifacts'][relative]['sha256']:
        raise ValueError('Artifact hash mismatch: ' + relative)
    return actual
