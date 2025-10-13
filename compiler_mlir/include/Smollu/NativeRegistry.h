//===- NativeRegistry.h - Target native function registry ------*- C++ -*-===//
//
// Loads and manages target-specific native function tables from YAML files.
//
//===----------------------------------------------------------------------===//

#ifndef SMOLLU_NATIVE_REGISTRY_H
#define SMOLLU_NATIVE_REGISTRY_H

#include "llvm/ADT/StringRef.h"
#include <optional>
#include <string>
#include <vector>
#include <cstdint>

namespace mlir {
namespace smol {

/// Represents a target's native function table loaded from YAML
struct NativeTable {
    std::string name;                      // Target name (e.g., "demo", "rs-emulator")
    std::optional<uint8_t> deviceId;       // Optional device ID (0-255)
    std::vector<std::string> natives;      // Ordered list of native function names
};

/// Load a target's native table from YAML
/// Searches in order: SMOLLU_TARGETS_DIR env, <binary>/../targets, built-in path
/// Returns empty NativeTable with empty name on failure (check .name.empty())
NativeTable loadTargetTable(llvm::StringRef targetName,
                             llvm::StringRef overrideDir = llvm::StringRef());

/// Find the index of a native function in the table
/// Returns -1 if not found
int findNativeIndex(const NativeTable &table, llvm::StringRef name);

/// Get a list of available targets in a directory
/// Returns empty vector on error
std::vector<std::string> listAvailableTargets(llvm::StringRef targetsDir);

} // namespace smol
} // namespace mlir

#endif // SMOLLU_NATIVE_REGISTRY_H

