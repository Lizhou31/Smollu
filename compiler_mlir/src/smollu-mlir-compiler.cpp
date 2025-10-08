//===- smollu-mlir-compiler.cpp - Smollu MLIR compiler driver --*- C++ -*-===//
//
// Main entry point for the Smollu MLIR compiler
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "Smollu/SmolDialect.h"
#include "Smollu/SmolOps.h"
#include "Smollu/SmolluParser.h"
#include "Smollu/SmolluASMEmitter.h"
#include "Smollu/Passes.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include <fstream>
#include <string>

using namespace mlir;

std::string readFile(const std::string &filename) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return "";
    }

    std::string content;
    std::string line;
    while (std::getline(file, line)) {
        content += line + "\n";
    }

    return content;
}

void printUsage(const char *progName) {
    std::cout << "Usage: " << progName << " <input.smol> [options]\n";
    std::cout << "\n";
    std::cout << "Options:\n";
    std::cout << "  -o <file>      Output bytecode file\n";
    std::cout << "  --emit-ast     Emit AST only (no bytecode generation)\n";
    std::cout << "  --emit-smol    Emit high-level Smol dialect MLIR to .smol.mlir file\n";
    std::cout << "  --emit-mlir    Emit MLIR representation to .mlir file\n";
    std::cout << "  --emit-asm     Emit assembly representation to .smolasm file\n";
    std::cout << "  -h, --help     Show this help message\n";
}

// Run the promotion pass to insert type casts for mixed arithmetic
bool runPromotionPass(mlir::ModuleOp module) {
    mlir::PassManager pm(module.getContext());
    pm.addPass(mlir::smol::createPromoteNumericsPass());

    if (mlir::failed(pm.run(module))) {
        std::cerr << "Error: Failed to run type promotion pass\n";
        return false;
    }

    return true;
}

// Run the scope verification pass to check variable scope correctness
bool runScopeVerificationPass(mlir::ModuleOp module) {
    mlir::PassManager pm(module.getContext());
    pm.addPass(mlir::smol::createScopeVerificationPass());

    if (mlir::failed(pm.run(module))) {
        std::cerr << "Error: Scope verification failed\n";
        return false;
    }

    return true;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string inputFile;
    std::string outputFile;
    bool emitASTOnly = false;
    bool emitSmol = false;
    bool emitMLIR = false;
    bool emitASM = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "-o") {
            if (i + 1 < argc) {
                outputFile = argv[++i];
            } else {
                std::cerr << "Error: -o requires an argument\n";
                return 1;
            }
        } else if (arg == "--emit-ast") {
            emitASTOnly = true;
        } else if (arg == "--emit-smol") {
            emitSmol = true;
        } else if (arg == "--emit-mlir") {
            emitMLIR = true;
        } else if (arg == "--emit-asm") {
            emitASM = true;
        } else if (inputFile.empty()) {
            inputFile = arg;
        } else {
            std::cerr << "Error: Unknown argument " << arg << "\n";
            return 1;
        }
    }

    if (inputFile.empty()) {
        std::cerr << "Error: No input file specified\n";
        return 1;
    }

    if (!emitASTOnly && !emitSmol && !emitMLIR && !emitASM && outputFile.empty()) {
        std::cerr << "Error: No output file specified (-o required for bytecode generation)\n";
        return 1;
    }

    // Read source file
    std::string sourceCode = readFile(inputFile);
    if (sourceCode.empty()) {
        return 1;
    }

    std::cout << "Parsing " << inputFile << "...\n";

    if (emitASTOnly) {
        // AST-only mode
        if (!parseSmolluToAST(sourceCode.c_str(), inputFile.c_str())) {
            std::cerr << "Error: Failed to parse source file\n";
            return 1;
        }
        std::cout << "Successfully parsed " << inputFile << " and emitted AST\n";
        return 0;
    }

    if (emitSmol) {
        // High-level Smol dialect mode
        MLIRContext context;
        context.loadDialect<mlir::smol::SmolDialect>();
        context.loadDialect<mlir::func::FuncDialect>();

        // Parse to Smol dialect
        mlir::ModuleOp module = parseSmolluToSmolDialect(&context, sourceCode.c_str(), true, inputFile.c_str());
        if (!module) {
            std::cerr << "Error: Failed to parse source file\n";
            return 1;
        }

        // Run type promotion pass
        if (!runPromotionPass(module)) {
            return 1;
        }

        // Run scope verification pass
        if (!runScopeVerificationPass(module)) {
            return 1;
        }

        // Generate output file name
        std::string smolFile = inputFile;
        size_t lastDot = smolFile.find_last_of('.');
        if (lastDot != std::string::npos) {
            smolFile = smolFile.substr(0, lastDot);
        }
        smolFile += ".smol.mlir";

        // Write Smol dialect MLIR to file
        std::ofstream outFile(smolFile);
        if (!outFile) {
            std::cerr << "Error: Could not create output file " << smolFile << "\n";
            return 1;
        }

        std::string mlirStr;
        llvm::raw_string_ostream strStream(mlirStr);
        // Print with debug info (source locations) - use pretty form for better readability
        mlir::OpPrintingFlags flags;
        flags.enableDebugInfo();
        flags.useLocalScope();
        module.print(strStream, flags);
        strStream.flush();
        outFile << mlirStr;
        outFile.close();

        std::cout << "Successfully generated Smol dialect MLIR to " << smolFile << "\n";
        return 0;
    }

    // MLIR mode (currently uses Smol dialect)
    // Initialize MLIR
    MLIRContext context;
    context.loadDialect<mlir::smol::SmolDialect>();
    context.loadDialect<mlir::func::FuncDialect>();

    // Parse Smollu source to MLIR (Smol dialect)
    mlir::ModuleOp module = parseSmolluToSmolDialect(&context, sourceCode.c_str(), !emitMLIR && !emitASM, inputFile.c_str());
    if (!module) {
        std::cerr << "Error: Failed to parse source file\n";
        return 1;
    }
    std::cout << "Module parsed successfully\n";

    // Run type promotion pass
    if (!runPromotionPass(module)) {
        return 1;
    }

    if (emitMLIR) {
        // Generate MLIR output file name
        std::string mlirFile = inputFile;
        size_t lastDot = mlirFile.find_last_of('.');
        if (lastDot != std::string::npos) {
            mlirFile = mlirFile.substr(0, lastDot);
        }
        mlirFile += ".mlir";

        // Write MLIR to file
        std::ofstream outFile(mlirFile);
        if (!outFile) {
            std::cerr << "Error: Could not create output file " << mlirFile << "\n";
            return 1;
        }

        // Print MLIR module to file
        std::string mlirStr;
        llvm::raw_string_ostream strStream(mlirStr);
        module.print(strStream);
        strStream.flush();
        outFile << mlirStr;
        outFile.close();

        std::cout << "Successfully generated MLIR to " << mlirFile << "\n";
        return 0;
    }

    if (emitASM) {
        std::cerr << "Error: Assembly emission is temporarily disabled\n";
        std::cerr << "       The ASM emitter is being refactored for the new Smol dialect\n";
        std::cerr << "       It will be re-enabled in a future phase\n";
        return 1;
    }

    // Bytecode generation is temporarily disabled during refactoring
    std::cerr << "Error: Bytecode emission is temporarily disabled\n";
    std::cerr << "       The bytecode emitter is being refactored for the new Smol dialect\n";
    std::cerr << "       It will be re-enabled in a future phase\n";
    std::cerr << "\n";
    std::cerr << "Available options:\n";
    std::cerr << "  --emit-smol    Generate high-level Smol dialect MLIR (.smol.mlir)\n";
    std::cerr << "  --emit-ast     Generate AST output\n";
    return 1;
}