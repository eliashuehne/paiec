# System enrichment

Collected model, harness and configuration facts included with the first public
v1 feature foundation. The snapshot contains 46 model/base/family profiles,
eight harness profiles and 189 catalog configuration mappings, together with
source evidence, review decisions and unresolved claims.

`snapshot/` preserves all 109 original files byte for byte. The feature recipe
pins their hashes and reads them directly from this checkout, independently of
where the original public dataset is stored. Original evidence identifiers and
filenames retain their source names.

Research facts remain separate from supplied data. A family match does not
verify a private checkpoint, documented capabilities do not establish enabled
run settings, and unresolved values remain unknown. The lift export excludes
these researched facts.

See the [preparation and usage guide](../../docs/feature-foundation.md) for the
current commands and the [coverage report](../feature_foundation/coverage-and-gaps-v1.md)
for coverage against the full dataset. `snapshot/README.md` describes the source
collection and its file layouts.
