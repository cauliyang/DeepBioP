.PHONY: all clean build test stubs lint

all: build

clean:
	rm -rf build dist

lint:
	uvx ruff format
	uvx ruff check --fix --unsafe-fixes

test:
	uv run pytest -vsl tests

build:
	uv sync
	uv tool run maturin develop -r

# Regenerate type stubs for the Rust extension: deepbiop/<module>/__init__.pyi.
# The top-level deepbiop/__init__.pyi is dropped because deepbiop/__init__.py is
# pure Python and is what type checkers should read.
stubs:
	cargo run --bin stub_gen
	rm -f deepbiop/__init__.pyi
	uvx ruff check --fix --unsafe-fixes deepbiop/*/__init__.pyi
	uvx ruff format deepbiop/*/__init__.pyi
