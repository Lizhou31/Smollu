//===- ScopeVerification.cpp - Variable scope verification pass -*- C++ -*-===//
//
// This pass verifies that all variable uses follow proper scoping rules:
// - Variables must be declared (stored) before use (load) within their scope
// - Local variables are only accessible within their declaring function
// - Global variables are accessible from all functions
// - Handles block-level scoping for if/while constructs
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "Smollu/SmolOps.h"
#include "Smollu/SmolDialect.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include <set>

using namespace mlir;
using namespace mlir::smol;

namespace {

struct ScopeVerificationPass : public PassWrapper<ScopeVerificationPass,
                                                    OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScopeVerificationPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<SmolDialect>();
    registry.insert<mlir::func::FuncDialect>();
  }

  StringRef getArgument() const final { return "smol-verify-scope"; }

  StringRef getDescription() const final {
    return "Verify variable scope correctness (declare before use, local/global rules)";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool hasError = false;

    // Track variable declarations and their defining operations
    llvm::StringMap<std::set<Operation*>> localVarDefs;  // "funcName:varName" -> set of defining ops
    llvm::StringMap<std::set<Operation*>> globalVarDefs; // "varName" -> set of defining ops

    // First pass: collect all variable declarations
    module.walk([&](Operation *op) {
      if (auto storeOp = dyn_cast<VarStoreOp>(op)) {
        std::string varName = storeOp.getName().str();
        bool isLocal = storeOp.getIsLocal();
        Operation *parentFunc = op->getParentOfType<mlir::func::FuncOp>();

        if (isLocal) {
          // Local variable - scope is current function
          std::string funcName = parentFunc ?
            cast<mlir::func::FuncOp>(parentFunc).getName().str() : "__init";
          localVarDefs[funcName + ":" + varName].insert(op);
        } else {
          // Global variable
          globalVarDefs[varName].insert(op);
        }
      }
    });

    // Second pass: verify all variable loads
    module.walk([&](mlir::func::FuncOp funcOp) {
      DominanceInfo domInfo(funcOp);

      funcOp.walk([&](Operation *op) {
        if (auto loadOp = dyn_cast<VarLoadOp>(op)) {
          std::string varName = loadOp.getName().str();
          Operation *parentFunc = op->getParentOfType<mlir::func::FuncOp>();

          std::string currentFunc = parentFunc ?
            cast<mlir::func::FuncOp>(parentFunc).getName().str() : "__init";

          std::string localKey = currentFunc + ":" + varName;

          // Check if variable exists in current local scope or global scope
          auto localIt = localVarDefs.find(localKey);
          auto globalIt = globalVarDefs.find(varName);

          bool existsInLocal = localIt != localVarDefs.end();
          bool existsInGlobal = globalIt != globalVarDefs.end();

          if (!existsInLocal && !existsInGlobal) {
            op->emitError("variable '") << varName << "' used before declaration";
            hasError = true;
            return;
          }

          // Verify at least one definition dominates the use
          // Local variables take precedence over global variables
          if (existsInLocal) {
            bool dominated = false;
            for (Operation *defOp : localIt->second) {
              if (domInfo.dominates(defOp, op)) {
                dominated = true;
                break;
              }
            }
            if (!dominated) {
              op->emitError("variable '") << varName << "' may be used before initialization in some code paths";
              hasError = true;
            }
          } else if (existsInGlobal) {
            // For global variables, check dominance within the current function
            bool dominated = false;
            for (Operation *defOp : globalIt->second) {
              // Only check definitions in the same function
              Operation *defFunc = defOp->getParentOfType<mlir::func::FuncOp>();
              if (defFunc == parentFunc && domInfo.dominates(defOp, op)) {
                dominated = true;
                break;
              }
            }
            if (!dominated) {
              op->emitError("variable '") << varName << "' used before declaration";
              hasError = true;
            }
          }
        }
      });
    });

    if (hasError) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
namespace smol {

std::unique_ptr<Pass> createScopeVerificationPass() {
  return std::make_unique<ScopeVerificationPass>();
}

} // namespace smol
} // namespace mlir
