//===- SmolDialect.td - Smol high-level dialect -----------*- tablegen -*-===//
//
// Defines the high-level Smol MLIR dialect for the Smollu language
// This dialect represents language-level semantics before lowering to VM
//
//===----------------------------------------------------------------------===//

#ifndef SMOL_DIALECT
#define SMOL_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Smol dialect definition
//===----------------------------------------------------------------------===//

def Smol_Dialect : Dialect {
  let name = "smol";
  let summary = "High-level dialect for the Smollu language";
  let description = [{
    This dialect provides high-level operations for the Smollu embedded
    scripting language. It represents language semantics directly from the AST,
    without VM-specific lowering. Operations include:

    - Type-aware constants and variables
    - High-level control flow (if/while with proper regions)
    - Function calls (user-defined and native)
    - Arithmetic and logical operations with type checking

    This dialect is meant to be lowered to standard MLIR dialects (func, scf,
    arith) and eventually to the smolluasm dialect for VM code generation.
  }];
  let cppNamespace = "::mlir::smol";
  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base Smol operation definition
//===----------------------------------------------------------------------===//

class Smol_Op<string mnemonic, list<Trait> traits = []> :
    Op<Smol_Dialect, mnemonic, traits>;

#endif // SMOL_DIALECT
