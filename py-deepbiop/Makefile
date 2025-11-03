.PHONY: clean build stubs

# add build target to the default target
all: build

clean:
	rm -rf build dist

build: clean
	uv sync
	uvx ruff check --fix --unsafe-fixes

# Generate Python type stubs (.pyi files)
# Note: Automatic stub generation via stub_gen binary has linking issues on macOS
# due to PyO3 extension-module feature. Stubs are maintained in deepbiop/*.pyi
stubs:
	@echo "⚠️  Automatic stub generation not available on this platform"
	@echo ""
	@echo "Type stubs are manually maintained in:"
	@echo "  - deepbiop/__init__.pyi"
	@echo "  - deepbiop/fq.pyi"
	@echo "    • Filter classes: LengthFilter, QualityFilter, Deduplicator, Subsampler"
	@echo "    • Augmentation classes: ReverseComplement, Mutator, Sampler"
	@echo "  - deepbiop/bam.pyi"
	@echo "  - deepbiop/fa.pyi"
	@echo "  - deepbiop/core.pyi"
	@echo "  - deepbiop/utils.pyi"
	@echo ""
	@echo "Stubs are kept up-to-date and verified with ruff."
