//===- SmolDialect.cpp - Smol high-level dialect ---------------*- C++ -*-===//
//
// Implementation of the Smol high-level dialect with verifiers
//
//===----------------------------------------------------------------------===//

#include "Smollu/SmolDialect.h"
#include "Smollu/SmolOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::smol;

//===----------------------------------------------------------------------===//
// Smol Dialect
//===----------------------------------------------------------------------===//

#include "SmolDialect.cpp.inc"

void SmolDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "SmolOps.cpp.inc"
  >();
}

// Type parsing and printing (using default MLIR types)
Type SmolDialect::parseType(DialectAsmParser &parser) const {
  return Type();
}

void SmolDialect::printType(Type type, DialectAsmPrinter &printer) const {
  printer << type;
}

//===----------------------------------------------------------------------===//
// Smol Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "SmolOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Operation Verifiers
//===----------------------------------------------------------------------===//

// Helper function to check if a type is numeric (int or float)
static bool isNumericType(Type type) {
  return llvm::isa<IntegerType>(type) || llvm::isa<FloatType>(type);
}

// Helper function to check if two types are compatible for arithmetic
static bool areTypesCompatible(Type lhs, Type rhs) {
  // Same types are always compatible
  if (lhs == rhs)
    return true;

  // Int and float are compatible (will promote to float)
  if (isNumericType(lhs) && isNumericType(rhs))
    return true;

  return false;
}

// Helper function to determine result type for binary arithmetic
static Type getResultType(Type lhs, Type rhs) {
  // If either is float, result is float
  if (llvm::isa<FloatType>(lhs) || llvm::isa<FloatType>(rhs))
    return mlir::Float32Type::get(lhs.getContext());

  // Otherwise, result is integer
  return IntegerType::get(lhs.getContext(), 32);
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

LogicalResult AddOp::verify() {
  Type lhsType = getLhs().getType();
  Type rhsType = getRhs().getType();
  Type resultType = getResult().getType();

  // Check that operand types are numeric
  if (!isNumericType(lhsType) || !isNumericType(rhsType)) {
    return emitOpError("operands must be numeric types (int or float)");
  }

  // Check that operand types are compatible
  if (!areTypesCompatible(lhsType, rhsType)) {
    return emitOpError("incompatible operand types");
  }

  // Check that result type matches expected type
  Type expectedType = getResultType(lhsType, rhsType);
  if (resultType != expectedType) {
    return emitOpError("result type must match operand types (with promotion)");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

LogicalResult SubOp::verify() {
  Type lhsType = getLhs().getType();
  Type rhsType = getRhs().getType();
  Type resultType = getResult().getType();

  if (!isNumericType(lhsType) || !isNumericType(rhsType)) {
    return emitOpError("operands must be numeric types (int or float)");
  }

  if (!areTypesCompatible(lhsType, rhsType)) {
    return emitOpError("incompatible operand types");
  }

  Type expectedType = getResultType(lhsType, rhsType);
  if (resultType != expectedType) {
    return emitOpError("result type must match operand types (with promotion)");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

LogicalResult MulOp::verify() {
  Type lhsType = getLhs().getType();
  Type rhsType = getRhs().getType();
  Type resultType = getResult().getType();

  if (!isNumericType(lhsType) || !isNumericType(rhsType)) {
    return emitOpError("operands must be numeric types (int or float)");
  }

  if (!areTypesCompatible(lhsType, rhsType)) {
    return emitOpError("incompatible operand types");
  }

  Type expectedType = getResultType(lhsType, rhsType);
  if (resultType != expectedType) {
    return emitOpError("result type must match operand types (with promotion)");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

LogicalResult DivOp::verify() {
  Type lhsType = getLhs().getType();
  Type rhsType = getRhs().getType();
  Type resultType = getResult().getType();

  if (!isNumericType(lhsType) || !isNumericType(rhsType)) {
    return emitOpError("operands must be numeric types (int or float)");
  }

  if (!areTypesCompatible(lhsType, rhsType)) {
    return emitOpError("incompatible operand types");
  }

  Type expectedType = getResultType(lhsType, rhsType);
  if (resultType != expectedType) {
    return emitOpError("result type must match operand types (with promotion)");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ModOp
//===----------------------------------------------------------------------===//

LogicalResult ModOp::verify() {
  Type lhsType = getLhs().getType();
  Type rhsType = getRhs().getType();
  Type resultType = getResult().getType();

  // Mod only works on integers
  if (!llvm::isa<IntegerType>(lhsType) || !llvm::isa<IntegerType>(rhsType)) {
    return emitOpError("modulo operation requires integer operands");
  }

  if (!llvm::isa<IntegerType>(resultType)) {
    return emitOpError("modulo result must be integer type");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// NegOp
//===----------------------------------------------------------------------===//

LogicalResult NegOp::verify() {
  Type operandType = getOperand().getType();
  Type resultType = getResult().getType();

  if (!isNumericType(operandType)) {
    return emitOpError("operand must be numeric type (int or float)");
  }

  if (operandType != resultType) {
    return emitOpError("result type must match operand type");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Comparison operations
//===----------------------------------------------------------------------===//

LogicalResult EqOp::verify() {
  Type lhsType = getLhs().getType();
  Type rhsType = getRhs().getType();

  if (!areTypesCompatible(lhsType, rhsType)) {
    return emitOpError("comparison operands must have compatible types");
  }

  return success();
}

LogicalResult NeOp::verify() {
  Type lhsType = getLhs().getType();
  Type rhsType = getRhs().getType();

  if (!areTypesCompatible(lhsType, rhsType)) {
    return emitOpError("comparison operands must have compatible types");
  }

  return success();
}

LogicalResult LtOp::verify() {
  Type lhsType = getLhs().getType();
  Type rhsType = getRhs().getType();

  if (!isNumericType(lhsType) || !isNumericType(rhsType)) {
    return emitOpError("comparison requires numeric types");
  }

  if (!areTypesCompatible(lhsType, rhsType)) {
    return emitOpError("comparison operands must have compatible types");
  }

  return success();
}

LogicalResult LeOp::verify() {
  Type lhsType = getLhs().getType();
  Type rhsType = getRhs().getType();

  if (!isNumericType(lhsType) || !isNumericType(rhsType)) {
    return emitOpError("comparison requires numeric types");
  }

  if (!areTypesCompatible(lhsType, rhsType)) {
    return emitOpError("comparison operands must have compatible types");
  }

  return success();
}

LogicalResult GtOp::verify() {
  Type lhsType = getLhs().getType();
  Type rhsType = getRhs().getType();

  if (!isNumericType(lhsType) || !isNumericType(rhsType)) {
    return emitOpError("comparison requires numeric types");
  }

  if (!areTypesCompatible(lhsType, rhsType)) {
    return emitOpError("comparison operands must have compatible types");
  }

  return success();
}

LogicalResult GeOp::verify() {
  Type lhsType = getLhs().getType();
  Type rhsType = getRhs().getType();

  if (!isNumericType(lhsType) || !isNumericType(rhsType)) {
    return emitOpError("comparison requires numeric types");
  }

  if (!areTypesCompatible(lhsType, rhsType)) {
    return emitOpError("comparison operands must have compatible types");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Control flow operations
//===----------------------------------------------------------------------===//

LogicalResult IfOp::verify() {
  // Check that condition is boolean
  if (auto intTy = llvm::dyn_cast<IntegerType>(getCondition().getType())) {
    if (intTy.getWidth() != 1) {
      return emitOpError("condition must be i1 (boolean) type");
    }
  } else {
    return emitOpError("condition must be i1 (boolean) type");
  }

  // Check that then region is not empty
  if (getThenRegion().empty()) {
    return emitOpError("then region cannot be empty");
  }

  return success();
}

LogicalResult WhileOp::verify() {
  // Check that condition region exists and is not empty
  if (getCondition().empty()) {
    return emitOpError("condition region cannot be empty");
  }

  // Check that body region exists and is not empty
  if (getBody().empty()) {
    return emitOpError("body region cannot be empty");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Function operations
//===----------------------------------------------------------------------===//

LogicalResult CallOp::verify() {
  // Basic verification - callee symbol ref should be valid
  // Additional verification would check if the function exists in the symbol table
  return success();
}
