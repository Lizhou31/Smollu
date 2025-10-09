//===- SmolluParser.cpp - Unified Smollu parser implementation -*- C++ -*-===//
//
// Unified API implementation for Smollu parsing with AST-only and MLIR modes
//
//===----------------------------------------------------------------------===//

#include "Smollu/SmolluParser.h"
#include "Smollu/SmolMLIRGen.h"
#include <iostream>

using namespace mlir;
using namespace mlir::smollu;
using SmolluASTNode = mlir::smollu::SmolluASTNode;

SmolluParser::SmolluParser(MLIRContext *ctx, CompilationMode m)
    : context(ctx), mode(m) {
}

SmolluASTNode SmolluParser::parseToAST(const std::string &source, const std::string &filename) {
    SmolluASTParser astParser;
    return astParser.parseSourceFile(source, filename);
}

ModuleOp SmolluParser::parseToMLIR(const std::string &source) {
    if (mode != CompilationMode::MLIR_MODULE) {
        std::cerr << "Error: MLIR generation not available in AST_ONLY mode\n";
        return nullptr;
    }

    // First parse to AST
    SmolluASTNode ast = parseToAST(source);
    if (ast.type.empty()) {
        return nullptr;
    }

    // Generate high-level Smol dialect MLIR from AST (new approach)
    mlir::smol::SmolMLIRGenerator mlirGen(context);
    return mlirGen.generateMLIR(ast);
}

bool SmolluParser::parseAndEmitAST(const std::string &source, std::ostream &out) {
    SmolluASTNode ast = parseToAST(source);
    if (ast.type.empty()) {
        return false;
    }

    out << "\n=== AST Structure ===\n";
    SmolluASTParser astParser;
    astParser.printAST(ast, out);
    out << "=== End AST ===\n\n";
    return true;
}

ModuleOp SmolluParser::parseAndEmitMLIR(const std::string &source, bool printAST, std::ostream &out) {
    // First parse to AST
    SmolluASTNode ast = parseToAST(source);
    if (ast.type.empty()) {
        return nullptr;
    }

    // Optionally print AST
    if (printAST) {
        out << "\n=== AST Structure ===\n";
        SmolluASTParser astParser;
        astParser.printAST(ast, out);
        out << "=== End AST ===\n\n";
    }

    if (mode != CompilationMode::MLIR_MODULE) {
        std::cerr << "Error: MLIR generation not available in AST_ONLY mode\n";
        return nullptr;
    }

    // Generate high-level Smol dialect MLIR from AST (new approach)
    mlir::smol::SmolMLIRGenerator mlirGen(context);
    return mlirGen.generateMLIR(ast);
}

bool parseSmolluToAST(const char *source, const char *filename) {
    // Create a dummy context for AST-only parsing
    mlir::MLIRContext context;
    SmolluParser parser(&context, CompilationMode::AST_ONLY);
    SmolluASTNode ast = parser.parseToAST(std::string(source), std::string(filename));
    if (ast.type.empty()) {
        return false;
    }

    std::cout << "\n=== AST Structure ===\n";
    SmolluASTParser astParser;
    astParser.printAST(ast, std::cout);
    std::cout << "=== End AST ===\n\n";
    return true;
}

mlir::ModuleOp parseSmolluToSmolDialect(mlir::MLIRContext *context, const char *source, bool emitAST, const char *filename) {
    // Parse to AST
    SmolluParser parser(context, CompilationMode::MLIR_MODULE);
    SmolluASTNode ast = parser.parseToAST(std::string(source), std::string(filename));

    if (ast.type.empty()) {
        return nullptr;
    }

    // Optionally print AST
    if (emitAST) {
        std::cout << "\n=== AST Structure ===\n";
        SmolluASTParser astParser;
        astParser.printAST(ast, std::cout);
        std::cout << "=== End AST ===\n\n";
    }

    // Generate high-level Smol dialect MLIR from AST
    mlir::smol::SmolMLIRGenerator mlirGen(context);
    return mlirGen.generateMLIR(ast);
}