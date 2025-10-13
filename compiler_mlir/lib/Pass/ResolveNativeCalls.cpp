//===- ResolveNativeCalls.cpp - Resolve native calls to indices -*- C++ -*-===//
//
// This pass resolves native function calls to target-specific indices.
// It reads the target specification from module attributes or pass options,
// loads the native function table, and annotates each smol.native_call
// operation with its resolved index.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "Smollu/SmolOps.h"
#include "Smollu/SmolDialect.h"
#include "Smollu/NativeRegistry.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::smol;

namespace {

// Simple Levenshtein distance for suggestions
static int levenshteinDistance(llvm::StringRef s1, llvm::StringRef s2) {
    const size_t len1 = s1.size(), len2 = s2.size();
    std::vector<std::vector<int>> d(len1 + 1, std::vector<int>(len2 + 1));

    d[0][0] = 0;
    for (size_t i = 1; i <= len1; ++i) d[i][0] = i;
    for (size_t j = 1; j <= len2; ++j) d[0][j] = j;

    for (size_t i = 1; i <= len1; ++i) {
        for (size_t j = 1; j <= len2; ++j) {
            d[i][j] = std::min({
                d[i - 1][j] + 1,      // deletion
                d[i][j - 1] + 1,      // insertion
                d[i - 1][j - 1] + (s1[i - 1] == s2[j - 1] ? 0 : 1) // substitution
            });
        }
    }

    return d[len1][len2];
}

struct ResolveNativeCallsPass : public PassWrapper<ResolveNativeCallsPass,
                                                     OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ResolveNativeCallsPass)

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<SmolDialect>();
        registry.insert<mlir::func::FuncDialect>();
    }

    StringRef getArgument() const final { return "smol-resolve-native-calls"; }

    StringRef getDescription() const final {
        return "Resolve native function calls to target-specific indices";
    }

    void runOnOperation() override {
        ModuleOp module = getOperation();
        bool hasError = false;

        // Get target name from module attribute
        std::string target;
        if (auto attr = module->getAttrOfType<StringAttr>("smol.target")) {
            target = attr.getValue().str();
        }

        if (target.empty()) {
            module.emitError("No target specified. Set smol.target module attribute (use --target=<name> when compiling)");
            signalPassFailure();
            return;
        }

        // Load the native table
        NativeTable table = loadTargetTable(target, llvm::StringRef());
        if (table.name.empty()) {
            module.emitError("Failed to load target '") << target << "'";
            signalPassFailure();
            return;
        }

        // Walk all native_call operations
        module.walk([&](NativeCallOp callOp) {
            std::string name = callOp.getName().str();

            // Look up the index
            int index = findNativeIndex(table, name);
            if (index < 0) {
                // Native not found - emit error with suggestions
                auto diag = callOp.emitError("native function '")
                    << name << "' not found in target '" << target << "'";

                // Find closest matches using Levenshtein distance
                std::vector<std::pair<int, std::string>> distances;
                for (const auto &nativeName : table.natives) {
                    int dist = levenshteinDistance(name, nativeName);
                    distances.push_back({dist, nativeName});
                }
                std::sort(distances.begin(), distances.end());

                // Suggest up to 5 closest matches
                if (!distances.empty()) {
                    diag.attachNote() << "available natives in target '" << target << "':";
                    for (size_t i = 0; i < std::min(size_t(5), distances.size()); ++i) {
                        diag.attachNote() << "  - " << distances[i].second;
                    }
                }

                hasError = true;
                return;
            }

            // Annotate with the resolved index
            callOp->setAttr("smol.native_index",
                IntegerAttr::get(IntegerType::get(callOp.getContext(), 32), index));
        });

        if (hasError) {
            signalPassFailure();
        }
    }
};

} // anonymous namespace

namespace mlir {
namespace smol {

std::unique_ptr<Pass> createResolveNativeCallsPass() {
    return std::make_unique<ResolveNativeCallsPass>();
}

} // namespace smol
} // namespace mlir

