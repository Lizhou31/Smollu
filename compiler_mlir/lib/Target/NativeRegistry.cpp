//===- NativeRegistry.cpp - Target native function registry ----*- C++ -*-===//
//
// Loads and manages target-specific native function tables from YAML files.
//
//===----------------------------------------------------------------------===//

#include "Smollu/NativeRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include <system_error>

using namespace llvm;
using namespace mlir::smol;

// YAML mapping structures
namespace llvm {
namespace yaml {

template <>
struct ScalarTraits<std::optional<uint8_t>> {
    static void output(const std::optional<uint8_t> &val, void *, raw_ostream &out) {
        if (val.has_value())
            out << (int)val.value();
        else
            out << "~";
    }

    static StringRef input(StringRef scalar, void *, std::optional<uint8_t> &val) {
        if (scalar.empty() || scalar == "~") {
            val = std::nullopt;
            return StringRef();
        }

        // Support hex (0x01) and decimal (1)
        unsigned long long num;
        if (scalar.starts_with("0x") || scalar.starts_with("0X")) {
            if (scalar.substr(2).getAsInteger(16, num))
                return "Invalid hex number";
        } else {
            if (scalar.getAsInteger(10, num))
                return "Invalid decimal number";
        }

        if (num > 255)
            return "Device ID must be 0-255";

        val = (uint8_t)num;
        return StringRef();
    }

    static QuotingType mustQuote(StringRef) { return QuotingType::None; }
};

template <>
struct MappingTraits<mlir::smol::NativeTable> {
    static void mapping(IO &io, mlir::smol::NativeTable &table) {
        io.mapRequired("name", table.name);
        io.mapOptional("device_id", table.deviceId);
        io.mapRequired("natives", table.natives);
    }
};

} // namespace yaml
} // namespace llvm

namespace mlir {
namespace smol {

/// Find the targets directory by searching in order:
/// 1. SMOLLU_TARGETS_DIR env variable
/// 2. overrideDir parameter (if non-empty)
/// 3. <binary_dir>/../targets
/// Returns empty string if not found
static std::string findTargetsDirectory(StringRef overrideDir) {
    // 1. Check environment variable
    if (const char *envDir = std::getenv("SMOLLU_TARGETS_DIR")) {
        if (sys::fs::exists(envDir))
            return envDir;
    }

    // 2. Check override directory
    if (!overrideDir.empty() && sys::fs::exists(overrideDir))
        return overrideDir.str();

    // 3. Check relative to binary (for installed/build tree)
    // Note: This is a simplified heuristic; a real implementation might use
    // CMake-generated paths or binary introspection
    SmallString<256> binaryPath;
    std::string mainExe = sys::fs::getMainExecutable(nullptr, nullptr);
    if (!mainExe.empty()) {
        binaryPath = mainExe;
        sys::path::remove_filename(binaryPath); // Remove binary name
        sys::path::append(binaryPath, "..", "targets");
        if (sys::fs::exists(binaryPath))
            return binaryPath.str().str();
    }

    return "";
}

NativeTable loadTargetTable(StringRef targetName, StringRef overrideDir) {
    NativeTable table;

    // Find targets directory
    std::string targetsDir = findTargetsDirectory(overrideDir);
    if (targetsDir.empty()) {
        errs() << "Error: Could not find targets directory\n";
        errs() << "Set SMOLLU_TARGETS_DIR environment variable or ensure targets/ is adjacent to the binary\n";
        return table; // Empty table
    }

    // Construct full path to target YAML file
    SmallString<256> yamlPath(targetsDir);
    sys::path::append(yamlPath, targetName + ".yaml");

    // Load the file
    auto bufferOrErr = MemoryBuffer::getFile(yamlPath);
    if (!bufferOrErr) {
        errs() << "Error: Could not open target file: " << yamlPath << "\n";
        errs() << "Error: " << bufferOrErr.getError().message() << "\n";

        // Try to list available targets
        auto available = listAvailableTargets(targetsDir);
        if (!available.empty()) {
            errs() << "Available targets: ";
            for (size_t i = 0; i < available.size(); ++i) {
                if (i > 0) errs() << ", ";
                errs() << available[i];
            }
            errs() << "\n";
        }
        return table; // Empty table
    }

    // Parse YAML
    yaml::Input yin(bufferOrErr.get()->getBuffer());
    yin >> table;

    if (yin.error()) {
        errs() << "Error: Failed to parse YAML file: " << yamlPath << "\n";
        table.name.clear(); // Mark as invalid
        return table;
    }

    // Validate that the name matches
    if (table.name != targetName) {
        errs() << "Warning: Target name in YAML (" << table.name
               << ") does not match filename (" << targetName << ")\n";
    }

    return table;
}

int findNativeIndex(const NativeTable &table, StringRef name) {
    for (size_t i = 0; i < table.natives.size(); ++i) {
        if (table.natives[i] == name)
            return (int)i;
    }
    return -1;
}

std::vector<std::string> listAvailableTargets(StringRef targetsDir) {
    std::vector<std::string> targets;

    std::error_code ec;
    for (sys::fs::directory_iterator it(targetsDir, ec), end; it != end && !ec; it.increment(ec)) {
        StringRef path = it->path();
        if (sys::path::extension(path) == ".yaml") {
            StringRef filename = sys::path::filename(path);
            StringRef stem = filename.drop_back(5); // Remove ".yaml"
            targets.push_back(stem.str());
        }
    }

    return targets;
}

} // namespace smol
} // namespace mlir

