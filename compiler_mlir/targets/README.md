# Smollu Target Definitions

This directory contains YAML definitions for different compilation targets. Each target specifies the ordered list of native function calls available for that platform.

## Schema

```yaml
name: <target-name>      # string, required - must match filename (without .yaml)
device_id: 0xNN          # integer or hex string, optional - device identifier
natives:                 # array of strings, required - ordered list of native functions
  - function_name_1
  - function_name_2
  ...
```

### Fields

- **name**: The target identifier. Must match the filename (e.g., `demo.yaml` → `name: demo`).
- **device_id**: Optional device identifier (0-255). Can be specified as decimal (`1`) or hex (`0x01`).
- **natives**: Ordered array of native function names. **The array index is the native call index used in bytecode**, so this order must remain stable across releases for binary compatibility.

## Adding a New Target

1. Create `<target-name>.yaml` in this directory
2. Define the `name` field to match the filename
3. Optionally specify `device_id`
4. List all native functions in the order they should be indexed
5. Document any platform-specific behavior in comments or external docs

## Important Rules

### Index Stability
The position of each native function in the `natives` array determines its index in the compiled bytecode:
- `natives[0]` → index 0
- `natives[1]` → index 1
- etc.

**Never reorder or remove entries** from an existing target's `natives` list, as this will break binary compatibility. To deprecate a function, keep it in the list but mark it as deprecated in documentation.

### Adding New Functions
Always **append** new native functions to the end of the `natives` array to maintain compatibility with existing bytecode.

### Renaming
If you need to rename a function:
1. Add the new name as a new entry at the end
2. Keep the old name in its original position (for backward compatibility)
3. Update the runtime to map both names to the same implementation
4. Document the deprecation

## Example Targets

### demo
Simple target for testing and demonstrations:
```yaml
name: demo
device_id: 0x00
natives:
  - print
  - rand
```

### rs-emulator
Rust-based emulator with LED matrix support:
```yaml
name: rs-emulator
device_id: 0x01
natives:
  - print
  - led_matrix_init
  - led_set
  - led_set_color
  - led_clear
  - led_set_row
  - led_set_col
  - led_get
  - delay_ms
```

## Usage

Specify the target when compiling:
```bash
smollu-mlir-compiler input.smol --target=rs-emulator -o output.smolbc
```

The compiler will:
1. Load the corresponding `rs-emulator.yaml`
2. Validate that all `native <name>` calls in your code exist in the target's `natives` list
3. Resolve each name to its integer index
4. Emit bytecode with the appropriate native call indices

## Directory Location

The compiler searches for target YAML files in this order:
1. `$SMOLLU_TARGETS_DIR` environment variable
2. `<binary_directory>/../targets/`
3. Built-in compiled path (for installed versions)

For development, set `SMOLLU_TARGETS_DIR` to point to your source tree's `compiler_mlir/targets/` directory.

