# Language Specification

## 1. Lexical basics
| Element             | Form                                    | Notes                      |
| ------------------- | --------------------------------------- | -------------------------- |
| **Identifiers**     | `[A-Za-z_][A-Za-z0-9_]*`                | Case-sensitive.            |
| **Integer literal** | `-? [0-9]+`                             | 32-bit signed.             |
| **Float literal**   | `-? [0-9]+ '.' [0-9]*`                  | 32-bit IEEE (no exponent). |
| **Boolean**         | `true` \| `false`                       |                            |
| **Nil**             | `nil`                                   | Single singleton value.    |
| **Comments**        | `--` to end-of-line                     | No block comments.         |
| **Whitespace**      | Spaces, tabs, newlines separate tokens. |                            |

## 2. Porgram structure
```BNF
<block>    ::= "{" <top_stat> "}"
<top_stat>  ::= <func_def> | <assign_stat> | <native_decl>
```
* function-IDs 0-254 (255 reserved)

* global slots 0-255

* native IDs 0x80-0x8F (hard-coded table)

### Example
```lua
-- top level code

native gpio_write(pin, value)  -- <native-decl>
LED_PIN = 1                    -- <global-assign>

function blink()
    gpio_write(LED_PIN, 1)
end
```

## 3. Values & types

| **Type**            | **Form**                                | **Notes**                  |
| ------------------- | --------------------------------------- | -------------------------- |
| **Integer literal** | `-? [0-9]+`                             | 32-bit signed.             |
| **Float literal**   | `-? [0-9]+ '.' [0-9]*`                  | 32-bit IEEE.               |
| **Boolean**         | `true` \| `false`                       |                            |
| **Nil**             | `nil`                                   | Single singleton value.    |

* Mixed arithmetic prmotes int -> float

* Any invalid type combination rases a runtime "type-mismatch" error and halts execution

## 4. Assignments & Operators
```BNF
<assign_stat> ::= "local "? <identifier> "=" <expr> ";"
<expr> ::= <m_exp> | <b_exp>

<m_exp> ::= (<integer> | <float>) 
 | "(" <m_exp> ")" 
 | <m_exp> <mop> <m_exp> 
 | <identifier>
<mop> ::= "+" | "-" | "*" | "/"

<b_exp> ::= <bool>
 | "(" <b_exp> ")"
 | "!" <b_exp>
 | <b_exp> <bop> <b_exp>
 | <identifier>
 | <m_exp> <eq_op> <m_exp>
 | <b_exp> <eq_op> <b_exp>
 | <m_exp> <cmp_op> <m_exp>
<bop> ::= "&&" | "||"
<eq_op> ::= "==" | "!="
<cmp_op> ::= ">" | ">=" | "<" | "<="
```

* local declares / binds an identifier to a function‑local slot (0‑255).

* When local is omitted, the assignment targets a global slot; the variable is auto‑created on first write.

### Example
```lua
local x = 5   -- local
y = 3.14      -- global
```

