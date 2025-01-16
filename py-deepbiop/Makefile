.PHONY: clean build

# add build target to the default target
all: build

clean:
	rm -rf build dist

build: clean
	uv sync
	uv tool run maturin develop -r
	cargo run --bin stub_gen
	ruff check --fix --unsafe-fixes
