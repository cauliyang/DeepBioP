.PHONY: clean build stubs

# add build target to the default target
all: build

clean:
	rm -rf build dist
	uvx ruff format
	uvx ruff check --fix --unsafe-fixes

test:
	uv run pytest -vsl tests

build:
	uv sync
	uv tool run maturin develop -r

# Generate Python type stubs (.pyi files)
stubs:
	@echo "Generating Python type stubs..."
	uv run --with maturin --with pyo3-stub-gen maturin develop -r
	uv run python -c "import pyo3_stub_gen; import deepbiop; pyo3_stub_gen.generate_stub_file('deepbiop', deepbiop, 'deepbiop')"
	@echo "✓ Stubs generated successfully"
	@echo ""
	@echo "Generated stubs:"
	@echo "  - deepbiop/__init__.pyi"
	@echo "  - deepbiop/fq.pyi"
	@echo "  - deepbiop/bam.pyi"
	@echo "  - deepbiop/fa.pyi"
	@echo "  - deepbiop/core.pyi"
	@echo "  - deepbiop/utils.pyi"
	@echo "  - deepbiop/vcf.pyi"
	@echo "  - deepbiop/gtf.pyi"
