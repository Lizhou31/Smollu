//===- SmolluOps.td - Smollu operations definition --------*- tablegen -*-===//
//
// Defines operations for the Smollu dialect
//
//===----------------------------------------------------------------------===//

#ifndef SMOLLU_OPS
#define SMOLLU_OPS

include "Smollu/SmolluDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"


//===----------------------------------------------------------------------===//
// Arithmetic operations
//===----------------------------------------------------------------------===//

def Smollu_AddOp : Smollu_Op<"add", [Pure]> {
  let summary = "Integer/float addition";
  let description = [{
    Performs addition of two values. Supports automatic type promotion
    from integer to float when operands have different types.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}

def Smollu_SubOp : Smollu_Op<"sub", [Pure]> {
  let summary = "Integer/float subtraction";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}

def Smollu_MulOp : Smollu_Op<"mul", [Pure]> {
  let summary = "Integer/float multiplication";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}

def Smollu_DivOp : Smollu_Op<"div", [Pure]> {
  let summary = "Integer/float division";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}

//===----------------------------------------------------------------------===//
// Comparison operations
//===----------------------------------------------------------------------===//

def Smollu_EqOp : Smollu_Op<"eq", [Pure]> {
  let summary = "Equality comparison";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs I1:$result);
}

def Smollu_NeOp : Smollu_Op<"ne", [Pure]> {
  let summary = "Not equal comparison";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs I1:$result);
}

def Smollu_LtOp : Smollu_Op<"lt", [Pure]> {
  let summary = "Less than comparison";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs I1:$result);
}

def Smollu_LeOp : Smollu_Op<"le", [Pure]> {
  let summary = "Less than or equal comparison";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs I1:$result);
}

def Smollu_GtOp : Smollu_Op<"gt", [Pure]> {
  let summary = "Greater than comparison";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs I1:$result);
}

def Smollu_GeOp : Smollu_Op<"ge", [Pure]> {
  let summary = "Greater than or equal comparison";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs I1:$result);
}

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

def Smollu_ConstantIntOp : Smollu_Op<"constant_int", [Pure]> {
  let summary = "Integer constant";
  let description = [{
    Represents a 32-bit signed integer constant.
  }];
  let arguments = (ins I32Attr:$value);
  let results = (outs I32:$result);
}

def Smollu_ConstantFloatOp : Smollu_Op<"constant_float", [Pure]> {
  let summary = "Float constant";
  let description = [{
    Represents a 32-bit IEEE float constant.
  }];
  let arguments = (ins F32Attr:$value);
  let results = (outs F32:$result);
}

def Smollu_ConstantBoolOp : Smollu_Op<"constant_bool", [Pure]> {
  let summary = "Boolean constant";
  let arguments = (ins BoolAttr:$value);
  let results = (outs I1:$result);
}

//===----------------------------------------------------------------------===//
// Variable operations
//===----------------------------------------------------------------------===//

def Smollu_GetGlobalOp : Smollu_Op<"get_global"> {
  let summary = "Load global variable";
  let description = [{
    Loads a value from a global variable slot (0-255).
  }];
  let arguments = (ins I8Attr:$slot);
  let results = (outs AnyType:$result);
}

def Smollu_SetGlobalOp : Smollu_Op<"set_global"> {
  let summary = "Store global variable";
  let description = [{
    Stores a value to a global variable slot (0-255).
  }];
  let arguments = (ins I8Attr:$slot, AnyType:$value);
}

def Smollu_GetLocalOp : Smollu_Op<"get_local"> {
  let summary = "Load local variable";
  let description = [{
    Loads a value from a local variable slot (0-255).
  }];
  let arguments = (ins I8Attr:$slot);
  let results = (outs AnyType:$result);
}

def Smollu_SetLocalOp : Smollu_Op<"set_local"> {
  let summary = "Store local variable";
  let description = [{
    Stores a value to a local variable slot (0-255).
  }];
  let arguments = (ins I8Attr:$slot, AnyType:$value);
}

//===----------------------------------------------------------------------===//
// Control flow operations
//===----------------------------------------------------------------------===//

def Smollu_IfOp : Smollu_Op<"if", [NoTerminator]> {
  let summary = "Conditional execution";
  let description = [{
    Executes then region if condition is true, else region otherwise.
  }];
  let arguments = (ins I1:$condition);
  let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);
}

def Smollu_WhileOp : Smollu_Op<"while", [NoTerminator]> {
  let summary = "While loop";
  let description = [{
    Executes body region while condition evaluates to true.
  }];
  let regions = (region SizedRegion<1>:$condition, SizedRegion<1>:$body);
}

//===----------------------------------------------------------------------===//
// Native function calls
//===----------------------------------------------------------------------===//

def Smollu_PrintOp : Smollu_Op<"print"> {
  let summary = "Print values";
  let description = [{
    Calls the native print function with variable arguments.
  }];
  let arguments = (ins Variadic<AnyType>:$args);
}

//===----------------------------------------------------------------------===//
// Program structure
//===----------------------------------------------------------------------===//

def Smollu_MainOp : Smollu_Op<"main", [NoTerminator]> {
  let summary = "Main program entry point";
  let description = [{
    Represents the main program block that gets executed.
  }];
  let regions = (region SizedRegion<1>:$body);
}

#endif // SMOLLU_OPS