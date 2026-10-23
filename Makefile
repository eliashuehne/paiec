PYTHON ?= .venv-laya-mlx/bin/python
CATALOGS := v1 collection_v1 collection_v2 collection_v3

.PHONY: check test validate-profiles

check: test validate-profiles
	git diff --check

test:
	PYTHONPATH=src:. $(PYTHON) -m unittest discover -s tests -p 'test_*.py'

validate-profiles:
	@set -e; for catalog in $(CATALOGS); do \
		PYTHONPATH=src:. $(PYTHON) scripts/validate_system_profiles.py \
			--catalog research/system_profiles/$$catalog; \
	done
