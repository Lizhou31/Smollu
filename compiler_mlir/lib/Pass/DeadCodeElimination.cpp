//===- DeadCodeElimination.cpp - Dead code elimination pass -----*- C++ -*-===//
//
// This pass removes unused code and variables in the Smol dialect:
// - Removes stores to variables that are never loaded
// - Removes operations whose results are never used (if they are Pure)
// - Uses MLIR's side effect system to determine which ops can be removed
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "Smollu/SmolOps.h"
#include "Smollu/SmolDialect.h"
#include <set>

using namespace mlir;
using namespace mlir::smol;

namespace {

/// Pass to eliminate dead code and unused variables
struct DeadCodeEliminationPass : public PassWrapper<DeadCodeEliminationPass,
                                                     OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DeadCodeEliminationPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<SmolDialect>();
  }

  StringRef getArgument() const final { return "smol-dce"; }

  StringRef getDescription() const final {
    return "Eliminate dead code and unused variables";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Iterate until no more changes are made
    bool changed = true;
    while (changed) {
      changed = false;

      // Step 1: Remove unused variable stores
      changed |= removeUnusedStores(module);

      // Step 2: Remove pure operations with unused results
      changed |= removeUnusedPureOps(module);
    }
  }

private:
  /// Collect all variables that are loaded (used)
  std::set<std::string> collectLoadedVariables(ModuleOp module) {
    std::set<std::string> loadedVars;

    module.walk([&](VarLoadOp loadOp) {
      loadedVars.insert(loadOp.getName().str());
    });

    return loadedVars;
  }

  /// Remove stores to variables that are never loaded
  bool removeUnusedStores(ModuleOp module) {
    std::set<std::string> loadedVars = collectLoadedVariables(module);

    llvm::SmallVector<VarStoreOp> toErase;
    module.walk([&](VarStoreOp storeOp) {
      std::string varName = storeOp.getName().str();

      // If this variable is never loaded, mark the store for removal
      if (loadedVars.find(varName) == loadedVars.end()) {
        toErase.push_back(storeOp);
      }
    });

    // Erase unused stores
    for (auto op : toErase) {
      op.erase();
    }

    return !toErase.empty();
  }

  /// Check if an operation is pure (no side effects)
  /// Uses MLIR's built-in side effect system
  bool isPureOp(Operation *op) {
    // Check if the operation implements memory effect interface
    if (auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
      // Check if operation has no side effects
      return effectInterface.hasNoEffect();
    }

    // If the operation doesn't implement the interface, assume it has side effects
    // (conservative approach for safety)
    return false;
  }

  /// Remove pure operations whose results are never used
  bool removeUnusedPureOps(ModuleOp module) {
    llvm::SmallVector<Operation*> toErase;

    module.walk([&](Operation *op) {
      // Skip operations with no results
      if (op->getNumResults() == 0) {
        return;
      }

      // Only remove pure operations (those with [Pure] trait in .td file)
      if (!isPureOp(op)) {
        return;
      }

      // Check if all results are unused
      bool allUnused = true;
      for (Value result : op->getResults()) {
        if (!result.use_empty()) {
          allUnused = false;
          break;
        }
      }

      if (allUnused) {
        toErase.push_back(op);
      }
    });

    // Erase unused operations
    for (auto op : toErase) {
      op->erase();
    }

    return !toErase.empty();
  }
};

} // namespace

namespace mlir {
namespace smol {

std::unique_ptr<Pass> createDeadCodeEliminationPass() {
  return std::make_unique<DeadCodeEliminationPass>();
}

} // namespace smol
} // namespace mlir
