//===- Passes.h - Smollu dialect passes ------------------------*- C++ -*-===//
//
// Declares transformation passes for the Smol dialect
//
//===----------------------------------------------------------------------===//

#ifndef SMOLLU_PASSES_H
#define SMOLLU_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace smol {

/// Create a pass that promotes integer operands to float in mixed arithmetic.
/// This implements the Smollu language rule: "Mixed arithmetic promotes int -> float"
std::unique_ptr<Pass> createPromoteNumericsPass();

/// Register all Smollu passes
void registerSmolPasses();

} // namespace smol
} // namespace mlir

#endif // SMOLLU_PASSES_H
