.PHONY: clean build

clean:
	rm -rf build dist

build: clean
	uv sync
	cargo run --bin stub_gen
	ruff check --fix --unsafe-fixes
