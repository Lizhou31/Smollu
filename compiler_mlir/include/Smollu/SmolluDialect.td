//===- SmolluDialect.td - Smollu dialect definition -------*- tablegen -*-===//
//
// Defines the Smollu MLIR dialect
//
//===----------------------------------------------------------------------===//

#ifndef SMOLLU_DIALECT
#define SMOLLU_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Smollu dialect definition
//===----------------------------------------------------------------------===//

def Smollu_Dialect : Dialect {
  let name = "smollu";
  let summary = "A dialect for the Smollu language";
  let description = [{
    This dialect provides operations for the Smollu embedded scripting language,
    including arithmetic, control flow, and variable operations.
  }];
  let cppNamespace = "::mlir::smollu";
  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base Smollu operation definition
//===----------------------------------------------------------------------===//

class Smollu_Op<string mnemonic, list<Trait> traits = []> :
    Op<Smollu_Dialect, mnemonic, traits>;

#endif // SMOLLU_DIALECT