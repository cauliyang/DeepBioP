.PHONY: clean build

# add build target to the default target
all: build

clean:
	rm -rf build dist

build: clean
	uv sync
	cargo run --bin stub_gen
	uvx ruff check --fix --unsafe-fixes
