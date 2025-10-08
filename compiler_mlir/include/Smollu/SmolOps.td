//===- SmolOps.td - Smol high-level operations ------------*- tablegen -*-===//
//
// Defines high-level operations for the Smol dialect with verifiers
//
//===----------------------------------------------------------------------===//

#ifndef SMOL_OPS
#define SMOL_OPS

include "Smollu/SmolDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Constant operations
//===----------------------------------------------------------------------===//

def Smol_ConstantIntOp : Smol_Op<"const_int", [Pure]> {
  let summary = "Integer constant";
  let description = [{
    Represents a 32-bit signed integer constant in the Smollu language.

    Example:
    ```mlir
    %0 = smol.const_int 42 : i32
    ```
  }];
  let arguments = (ins I32Attr:$value);
  let results = (outs I32:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";
}

def Smol_ConstantFloatOp : Smol_Op<"const_float", [Pure]> {
  let summary = "Float constant";
  let description = [{
    Represents a 32-bit IEEE float constant in the Smollu language.

    Example:
    ```mlir
    %0 = smol.const_float 3.14 : f32
    ```
  }];
  let arguments = (ins F32Attr:$value);
  let results = (outs F32:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";
}

def Smol_ConstantBoolOp : Smol_Op<"const_bool", [Pure]> {
  let summary = "Boolean constant";
  let description = [{
    Represents a boolean constant (true/false) in the Smollu language.

    Example:
    ```mlir
    %0 = smol.const_bool true : i1
    ```
  }];
  let arguments = (ins BoolAttr:$value);
  let results = (outs I1:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";
}

def Smol_ConstantNilOp : Smol_Op<"const_nil", [Pure]> {
  let summary = "Nil constant";
  let description = [{
    Represents a nil/null value in the Smollu language.

    Example:
    ```mlir
    %0 = smol.const_nil : !smol.value
    ```
  }];
  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Variable operations
//===----------------------------------------------------------------------===//

def Smol_VarDeclOp : Smol_Op<"var_decl"> {
  let summary = "Variable declaration";
  let description = [{
    Declares a variable with a name and scope (local or global).
    This operation is used during the high-level representation and will be
    lowered to slot-based storage later.

    Example:
    ```mlir
    %0 = smol.var_decl "x" {scope = "local"} : !smol.ref<i32>
    ```
  }];
  let arguments = (ins StrAttr:$name, StrAttr:$scope);
  let results = (outs AnyType:$ref);
}

def Smol_VarLoadOp : Smol_Op<"var_load"> {
  let summary = "Load variable value";
  let description = [{
    Loads the current value of a variable by name.

    Example:
    ```mlir
    %1 = smol.var_load "x" : i32
    ```
  }];
  let arguments = (ins StrAttr:$name);
  let results = (outs AnyType:$result);
}

def Smol_VarStoreOp : Smol_Op<"var_store"> {
  let summary = "Store variable value";
  let description = [{
    Stores a value to a variable by name.

    Example:
    ```mlir
    smol.var_store "x", %value : i32
    ```
  }];
  let arguments = (ins StrAttr:$name, AnyType:$value);
}

def Smol_CastOp : Smol_Op<"cast", [Pure]> {
  let summary = "Type coersion";
  let description = [{
    Type coerses a value to a different type.

    Example:
    ```mlir
    %1 = smol.cast %0 : i32 -> f32
    ```
  }];
  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
  let assemblyFormat = "$value attr-dict `:` type($value) `to` type($result)";
}

//===----------------------------------------------------------------------===//
// Arithmetic operations
//===----------------------------------------------------------------------===//

def Smol_AddOp : Smol_Op<"add", [Pure]> {
  let summary = "Addition operation";
  let description = [{
    Performs addition of two values. Supports integers and floats with
    automatic type promotion (int + float = float).

    Example:
    ```mlir
    %2 = smol.add %0, %1 : i32, i32 -> i32
    %3 = smol.add %0, %1 : i32, f32 -> f32
    ```
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);

  let hasVerifier = 1;
}

def Smol_SubOp : Smol_Op<"sub", [Pure]> {
  let summary = "Subtraction operation";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def Smol_MulOp : Smol_Op<"mul", [Pure]> {
  let summary = "Multiplication operation";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def Smol_DivOp : Smol_Op<"div", [Pure]> {
  let summary = "Division operation";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def Smol_ModOp : Smol_Op<"mod", [Pure]> {
  let summary = "Modulo operation";
  let description = [{
    Performs modulo operation on integers only.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def Smol_NegOp : Smol_Op<"neg", [Pure]> {
  let summary = "Unary negation";
  let description = [{
    Performs unary negation on a value (integer or float).

    Example:
    ```mlir
    %1 = smol.neg %0 : i32
    ```
  }];
  let arguments = (ins AnyType:$operand);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Comparison operations
//===----------------------------------------------------------------------===//

def Smol_EqOp : Smol_Op<"eq", [Pure]> {
  let summary = "Equality comparison";
  let description = [{
    Compares two values for equality. Returns a boolean result.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs I1:$result);
  let hasVerifier = 1;
}

def Smol_NeOp : Smol_Op<"ne", [Pure]> {
  let summary = "Not equal comparison";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs I1:$result);
  let hasVerifier = 1;
}

def Smol_LtOp : Smol_Op<"lt", [Pure]> {
  let summary = "Less than comparison";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs I1:$result);
  let hasVerifier = 1;
}

def Smol_LeOp : Smol_Op<"le", [Pure]> {
  let summary = "Less than or equal comparison";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs I1:$result);
  let hasVerifier = 1;
}

def Smol_GtOp : Smol_Op<"gt", [Pure]> {
  let summary = "Greater than comparison";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs I1:$result);
  let hasVerifier = 1;
}

def Smol_GeOp : Smol_Op<"ge", [Pure]> {
  let summary = "Greater than or equal comparison";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs I1:$result);
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Logical operations
//===----------------------------------------------------------------------===//

def Smol_AndOp : Smol_Op<"and", [Pure]> {
  let summary = "Logical AND";
  let description = [{
    Performs logical AND operation on boolean values.
  }];
  let arguments = (ins I1:$lhs, I1:$rhs);
  let results = (outs I1:$result);
}

def Smol_OrOp : Smol_Op<"or", [Pure]> {
  let summary = "Logical OR";
  let description = [{
    Performs logical OR operation on boolean values.
  }];
  let arguments = (ins I1:$lhs, I1:$rhs);
  let results = (outs I1:$result);
}

def Smol_NotOp : Smol_Op<"not", [Pure]> {
  let summary = "Logical NOT";
  let description = [{
    Performs logical NOT operation on a boolean value.
  }];
  let arguments = (ins I1:$operand);
  let results = (outs I1:$result);
}

//===----------------------------------------------------------------------===//
// Control flow operations
//===----------------------------------------------------------------------===//

def Smol_IfOp : Smol_Op<"if", [RecursiveMemoryEffects]> {
  let summary = "Conditional execution";
  let description = [{
    Executes the then region if condition is true, else region otherwise.

    Example:
    ```mlir
    smol.if %cond : i1 {
      // then block
    } else {
      // else block
    }
    ```
  }];
  let arguments = (ins I1:$condition);
  let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);

  let hasVerifier = 1;
}

def Smol_WhileOp : Smol_Op<"while", [RecursiveMemoryEffects]> {
  let summary = "While loop";
  let description = [{
    Executes body region while condition evaluates to true.
    The condition region should yield a boolean value.

    Example:
    ```mlir
    smol.while {
      // condition region
      %cond = smol.lt %i, %n : i32, i32 -> i1
      smol.yield %cond : i1
    } do {
      // body region
    }
    ```
  }];
  let regions = (region SizedRegion<1>:$condition, SizedRegion<1>:$body);

  let hasVerifier = 1;
}

def Smol_YieldOp : Smol_Op<"yield", [Terminator, ReturnLike]> {
  let summary = "Yield operation for regions";
  let description = [{
    Yields a value from a region (e.g., condition in while loop).

    Example:
    ```mlir
    smol.yield %cond : i1
    ```
  }];
  let arguments = (ins Variadic<AnyType>:$results);
}

//===----------------------------------------------------------------------===//
// Function operations
//===----------------------------------------------------------------------===//

def Smol_CallOp : Smol_Op<"call"> {
  let summary = "User function call";
  let description = [{
    Calls a user-defined function with arguments.

    Example:
    ```mlir
    %result = smol.call @my_func(%arg0, %arg1) : (i32, i32) -> i32
    ```
  }];
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$args);
  let results = (outs Optional<AnyType>:$result);

  let hasVerifier = 1;
}

def Smol_NativeCallOp : Smol_Op<"native_call"> {
  let summary = "Native function call";
  let description = [{
    Calls a native (C) function by name with arguments.

    Example:
    ```mlir
    smol.native_call "print"(%arg0) : (i32) -> ()
    %val = smol.native_call "gpio_read"() : () -> i32
    ```
  }];
  let arguments = (ins StrAttr:$name, Variadic<AnyType>:$args);
  let results = (outs Optional<AnyType>:$result);
}

def Smol_ReturnOp : Smol_Op<"return", [Terminator, ReturnLike]> {
  let summary = "Return from function";
  let description = [{
    Returns a value from a function.

    Example:
    ```mlir
    smol.return %value : i32
    smol.return  // void return
    ```
  }];
  let arguments = (ins Optional<AnyType>:$value);
}

#endif // SMOL_OPS
