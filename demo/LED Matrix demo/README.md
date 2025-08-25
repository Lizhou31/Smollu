# LED Matrix Examples for Smollu VM

This directory contains example Smollu programs that demonstrate LED matrix hardware simulation capabilities.

## Available Examples

### 1. Basic LED Demo (`basic_led_demo.smol`)
- **Matrix Size**: 8x8
- **Features Demonstrated**:
  - LED matrix initialization with `led_matrix_init(rows, cols)`
  - Individual LED control with `led_set(row, col, state)`
  - Color LED control with `led_set_color(row, col, r, g, b)`
- **What it does**: Creates a basic pattern with corner LEDs and center square, then demonstrates different colors

### 2. Pattern Demo (`pattern_demo.smol`)
- **Matrix Size**: 4x8
- **Features Demonstrated**:
  - Matrix clearing with `led_clear()`
  - Row pattern setting with `led_set_row(row, bit_pattern)`
  - Column pattern setting with `led_set_col(col, bit_pattern)`
- **What it does**: Shows how to use bit patterns to control entire rows and columns efficiently

### 3. Animation Demo (`animation_demo.smol`)
- **Matrix Size**: 8x8  
- **Features Demonstrated**:
  - Animation loops using while loops
  - Moving LED patterns
  - Color cycling
  - Complex patterns (checkerboard)
- **What it does**: Animates a colored dot moving in a square pattern, then creates a checkerboard pattern

### 4. Interactive Demo (`interactive_demo.smol`)
- **Matrix Size**: 6x6
- **Features Demonstrated**:
  - LED state reading with `led_get(row, col)`
  - Complex pattern creation using loops
  - State verification and counting
  - Multiple color patterns
- **What it does**: Creates a border pattern, fills center with colors, adds diagonal lines, and counts active LEDs

## LED Matrix Native Functions

The following native functions are available for LED matrix control:

### Matrix Management
- `led_matrix_init(rows, cols)` - Initialize LED matrix with specified dimensions (1-64 each)

### LED Control  
- `led_set(row, col, state)` - Set LED on (state=1) or off (state=0) with white color
- `led_set_color(row, col, r, g, b)` - Set LED to specific RGB color (0-255 each)
- `led_clear()` - Turn off all LEDs in the matrix

### Pattern Control
- `led_set_row(row, pattern)` - Set entire row using bit pattern (LSB = rightmost LED)
- `led_set_col(col, pattern)` - Set entire column using bit pattern (LSB = topmost LED)

### State Reading
- `led_get(row, col)` - Returns 1 if LED is on, 0 if off

## How to Run Examples

1. Build the Smollu project:
   ```bash
   cmake --preset=default && cmake --build build
   ```

2. Navigate to the demo directory:
   ```bash
   cd build/demo/Simple\ demo
   ```

3. Compile an LED matrix example:
   ```bash
   ./smollu_compiler "../../../demo/LED Matrix Examples/basic_led_demo.smol" -o led_demo.smolbc
   ```

4. Run with the emulator GUI to see LED matrix visualization:
   ```bash
   ../../emulator/target/debug/smollu-emulator led_demo.smolbc
   ```

## Tips for LED Matrix Programming

1. **Coordinate System**: 
   - (0,0) is top-left corner
   - Row increases downward, column increases rightward

2. **Bit Patterns**:
   - For `led_set_row()`: bit 0 = leftmost LED, bit 7 = rightmost LED  
   - For `led_set_col()`: bit 0 = topmost LED, bit 7 = bottommost LED

3. **Colors**:
   - RGB values range from 0-255
   - Use (255,255,255) for bright white
   - Use (0,0,0) for off (same as `led_set(row, col, 0)`)

4. **Performance**:
   - Use `led_set_row()` and `led_set_col()` for efficient pattern setting
   - Use `led_clear()` to quickly turn off all LEDs

5. **Matrix Limits**:
   - Maximum matrix size is 64x64
   - Minimum matrix size is 1x1
   - Initialize matrix before using any LED functions