.PHONY: all pytest ruff ruff_changes coverage mypy

CMD:=poetry run
PYMODULE:=src
TESTS:=tests

# Run checks which do not change files
all: mypy ruff pytest

# Run the unit tests using `pytest`
pytest:
	$(CMD) pytest $(PYMODULE) $(TESTS)

# Run the unit tests using `pytest` with memray
pytest_memray:
	@rm -rf $(TESTS)/memray
	$(CMD) pytest --memray --memray-bin-path $(TESTS)/memray --memray-bin-prefix memory-usage $(PYMODULE) $(TESTS)
	@find $(TESTS)/memray -name "*.bin" -exec $(CMD) memray flamegraph {} \;

# Run ruff linter/formatter - does not change files
ruff:
	$(CMD) ruff check $(PYMODULE) $(TESTS)
	$(CMD) ruff format --diff $(PYMODULE) $(TESTS)	

# Run ruff linter/formatter - changes files
ruff_changes:
	$(CMD) ruff check --fix $(PYMODULE) $(TESTS)
	$(CMD) ruff format $(PYMODULE) $(TESTS)	

# Generate a unit test coverage report using `pytest-cov`
coverage:
	$(CMD) pytest --cov=$(PYMODULE) $(TESTS) --cov-report html

# Perform static type checking using `mypy`
mypy:
	$(CMD) mypy $(PYMODULE) $(TESTS)

# Build mkdocs
build:
	$(CMD) mkdocs build

# Serve mkdocs
serve:
	$(CMD) mkdocs serve