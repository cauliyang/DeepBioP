# Python Stub Generation

## Current Status

Python type stubs (`.pyi` files) are **manually maintained** in the `deepbiop/` directory.

## Why Automatic Generation Doesn't Work

The `stub_gen` binary cannot be built or run due to PyO3's `extension-module` feature:

- **Extension modules** (`.so`/`.dylib` files for Python) use `extension-module` feature
- This feature prevents linking against `libpython` to create standalone extensions
- The `stub_gen` **binary** needs to link against `libpython` to run
- These requirements are mutually exclusive

## Technical Details

PyO3's `extension-module` feature is required for Python extensions and explicitly:
- Disables linking to libpython
- Uses `Py_LIMITED_API` for stable ABI
- Creates standalone `.so` files that Python loads

A stub generation binary would need:
- Link against libpython
- Import the extension module
- Execute Python code

This creates an architectural impossibility with the current setup.

## Maintaining Stubs

Stubs are kept in:
- `deepbiop/__init__.pyi`
- `deepbiop/fq.pyi`
- `deepbiop/bam.pyi`
- `deepbiop/fa.pyi`
- `deepbiop/core.pyi`
- `deepbiop/utils.pyi`
- `deepbiop/vcf.pyi`
- `deepbiop/gtf.pyi`

When adding new Python bindings:
1. Update the corresponding `.pyi` file
2. Run `uvx ruff check` to verify syntax
3. Ensure types match the Rust implementation

## Alternative Approach (Future)

To enable automatic stub generation:

1. Create a separate workspace crate without `extension-module`
2. Import all modules with `python` feature enabled
3. Use `pyo3-stub-gen` library directly
4. Generate stubs programmatically

This requires restructuring the workspace dependencies.
