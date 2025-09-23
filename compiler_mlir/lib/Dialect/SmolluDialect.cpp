//===- SmolluDialect.cpp - Smollu dialect implementation --------*- C++ -*-===//
//
// Implementation of the Smollu MLIR dialect
//
//===----------------------------------------------------------------------===//

#include "Smollu/SmolluDialect.h"
#include "Smollu/SmolluOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"

using namespace mlir;
using namespace mlir::smollu;

#include "SmolluDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Smollu dialect
//===----------------------------------------------------------------------===//

void SmolluDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "SmolluOps.cpp.inc"
      >();
}

Type SmolluDialect::parseType(DialectAsmParser &parser) const {
  // Stub implementation - not needed for basic functionality
  return {};
}

void SmolluDialect::printType(Type type, DialectAsmPrinter &printer) const {
  // Stub implementation - not needed for basic functionality
}

#define GET_OP_CLASSES
#include "SmolluOps.cpp.inc"