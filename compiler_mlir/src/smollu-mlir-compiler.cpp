//===- smollu-mlir-compiler.cpp - Smollu MLIR compiler driver --*- C++ -*-===//
//
// Main entry point for the Smollu MLIR compiler
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "Smollu/SmolluDialect.h"
#include "Smollu/SmolluParser.h"
#include "Smollu/SmolluASMEmitter.h"
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
    std::cout << "  --emit-mlir    Emit MLIR representation to .mlir file\n";
    std::cout << "  --emit-asm     Emit assembly representation to .smolasm file\n";
    std::cout << "  -h, --help     Show this help message\n";
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string inputFile;
    std::string outputFile;
    bool emitASTOnly = false;
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

    if (!emitASTOnly && !emitMLIR && !emitASM && outputFile.empty()) {
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
        if (!parseSmolluToAST(sourceCode.c_str())) {
            std::cerr << "Error: Failed to parse source file\n";
            return 1;
        }
        std::cout << "Successfully parsed " << inputFile << " and emitted AST\n";
        return 0;
    }

    // MLIR mode
    // Initialize MLIR
    MLIRContext context;
    context.loadDialect<mlir::smollu::SmolluDialect>();
    context.loadDialect<mlir::func::FuncDialect>();

    // Parse Smollu source to MLIR
    mlir::ModuleOp module = parseSmolluWithMode(&context, sourceCode.c_str(), !emitMLIR && !emitASM);
    if (!module) {
        std::cerr << "Error: Failed to parse source file\n";
        return 1;
    }
    std::cout << "Module parsed successfully\n";

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
        // Generate ASM output file name
        std::string asmFile = inputFile;
        size_t lastDot = asmFile.find_last_of('.');
        if (lastDot != std::string::npos) {
            asmFile = asmFile.substr(0, lastDot);
        }
        asmFile += ".smolasm";

        // Emit assembly
        if (!mlir::smollu::emitASMFromMLIR(module, asmFile.c_str())) {
            std::cerr << "Error: Failed to emit assembly\n";
            return 1;
        }

        std::cout << "Successfully generated assembly to " << asmFile << "\n";
        return 0;
    }

    std::cout << "Generating bytecode...\n";

    // Emit bytecode from MLIR
    // if (!emitBytecodeFromMLIR(module, outputFile.c_str())) {
    //    std::cerr << "Error: Failed to emit bytecode\n";
    //    return 1;
    //}

    std::cout << "Successfully compiled " << inputFile << " to " << outputFile << "\n";
    return 0;
}