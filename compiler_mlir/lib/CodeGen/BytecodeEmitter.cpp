//===- BytecodeEmitter.cpp - MLIR to Smollu bytecode emitter ---*- C++ -*-===//
//
// Emits Smollu bytecode from MLIR representation
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "Smollu/SmolluOps.h"

#include <vector>
#include <fstream>
#include <cstring>

using namespace mlir;
using namespace mlir::smollu;

namespace {

// Smollu bytecode instruction opcodes
enum SmolluOpcode : uint8_t {
    NOP = 0x00,
    PUSH_NIL = 0x01,
    PUSH_TRUE = 0x02,
    PUSH_FALSE = 0x03,
    PUSH_I8 = 0x04,
    PUSH_I32 = 0x05,
    PUSH_F32 = 0x06,
    DUP = 0x07,
    POP = 0x08,
    SWAP = 0x09,

    LOAD_LOCAL = 0x10,
    STORE_LOCAL = 0x11,
    LOAD_GLOBAL = 0x12,
    STORE_GLOBAL = 0x13,

    ADD = 0x20,
    SUB = 0x21,
    MUL = 0x22,
    DIV = 0x23,
    MOD = 0x24,
    NEG = 0x25,
    NOT = 0x26,

    EQ = 0x30,
    NEQ = 0x31,
    LT = 0x32,
    LE = 0x33,
    GT = 0x34,
    GE = 0x35,

    JMP = 0x40,
    JMP_IF_TRUE = 0x41,
    JMP_IF_FALSE = 0x42,
    HALT = 0x43,

    CALL = 0x50,
    RET = 0x51,

    NCALL = 0x60,
    SLEEP_MS = 0x61
};

// Native function IDs (matching VM implementation)
enum NativeFunctionId : uint8_t {
    NATIVE_PRINT = 0x80
};

class SmolluBytecodeEmitter {
private:
    std::vector<uint8_t> bytecode;
    std::vector<uint16_t> nativeFunctions;

public:
    bool emitModule(ModuleOp module, const std::string &outputFile) {
        // Emit header placeholder
        emitHeader();

        // Find main function and emit it
        auto result = module.walk([&](mlir::func::FuncOp func) {
            if (func.getName() == "main") {
                emitFunction(func);
                return WalkResult::interrupt();
            }
            return WalkResult::advance();
        });

        // Update header with actual sizes
        updateHeader();

        // Write to file
        return writeToFile(outputFile);
    }

private:
    void emitHeader() {
        // Magic number "SMOL" (4 bytes)
        bytecode.push_back('S');
        bytecode.push_back('M');
        bytecode.push_back('O');
        bytecode.push_back('L');

        // Version (1 byte)
        bytecode.push_back(1);

        // Device ID (1 byte)
        bytecode.push_back(0);

        // Function count (1 byte) - placeholder
        bytecode.push_back(1); // main function

        // Native function count (1 byte) - placeholder
        bytecode.push_back(0);

        // Code size (4 bytes) - placeholder
        bytecode.push_back(0);
        bytecode.push_back(0);
        bytecode.push_back(0);
        bytecode.push_back(0);

        // Reserved (4 bytes)
        bytecode.push_back(0);
        bytecode.push_back(0);
        bytecode.push_back(0);
        bytecode.push_back(0);
    }

    void updateHeader() {
        // Update native function count
        bytecode[7] = static_cast<uint8_t>(nativeFunctions.size());

        // Calculate code size (total - header - native table)
        uint32_t codeSize = bytecode.size() - 16 - (nativeFunctions.size() * 2);
        bytecode[8] = codeSize & 0xFF;
        bytecode[9] = (codeSize >> 8) & 0xFF;
        bytecode[10] = (codeSize >> 16) & 0xFF;
        bytecode[11] = (codeSize >> 24) & 0xFF;

        // Insert native function table after header
        std::vector<uint8_t> nativeTable;
        for (uint16_t nativeId : nativeFunctions) {
            nativeTable.push_back(nativeId & 0xFF);
            nativeTable.push_back((nativeId >> 8) & 0xFF);
        }

        bytecode.insert(bytecode.begin() + 16, nativeTable.begin(), nativeTable.end());
    }

    void emitFunction(mlir::func::FuncOp func) {
        // Emit function body
        for (Block &block : func.getBlocks()) {
            emitBlock(block);
        }

        // End with HALT
        emit1(HALT);
    }

    void emitBlock(Block &block) {
        for (Operation &op : block.getOperations()) {
            emitOperation(&op);
        }
    }

    void emitOperation(Operation *op) {
        if (auto constOp = dyn_cast<ConstantIntOp>(op)) {
            emitConstantInt(constOp);
        } else if (auto constOp = dyn_cast<ConstantFloatOp>(op)) {
            emitConstantFloat(constOp);
        } else if (auto constOp = dyn_cast<ConstantBoolOp>(op)) {
            emitConstantBool(constOp);
        } else if (auto addOp = dyn_cast<AddOp>(op)) {
            emit1(ADD);
        } else if (auto subOp = dyn_cast<SubOp>(op)) {
            emit1(SUB);
        } else if (auto mulOp = dyn_cast<MulOp>(op)) {
            emit1(MUL);
        } else if (auto divOp = dyn_cast<DivOp>(op)) {
            emit1(DIV);
        } else if (auto eqOp = dyn_cast<EqOp>(op)) {
            emit1(EQ);
        } else if (auto neOp = dyn_cast<NeOp>(op)) {
            emit1(NEQ);
        } else if (auto ltOp = dyn_cast<LtOp>(op)) {
            emit1(LT);
        } else if (auto leOp = dyn_cast<LeOp>(op)) {
            emit1(LE);
        } else if (auto gtOp = dyn_cast<GtOp>(op)) {
            emit1(GT);
        } else if (auto geOp = dyn_cast<GeOp>(op)) {
            emit1(GE);
        } else if (auto getGlobalOp = dyn_cast<GetGlobalOp>(op)) {
            emit1(LOAD_GLOBAL);
            emit1(static_cast<uint8_t>(getGlobalOp.getSlot()));
        } else if (auto setGlobalOp = dyn_cast<SetGlobalOp>(op)) {
            emit1(STORE_GLOBAL);
            emit1(static_cast<uint8_t>(setGlobalOp.getSlot()));
        } else if (auto getLocalOp = dyn_cast<GetLocalOp>(op)) {
            emit1(LOAD_LOCAL);
            emit1(static_cast<uint8_t>(getLocalOp.getSlot()));
        } else if (auto setLocalOp = dyn_cast<SetLocalOp>(op)) {
            emit1(STORE_LOCAL);
            emit1(static_cast<uint8_t>(setLocalOp.getSlot()));
        } else if (auto printOp = dyn_cast<PrintOp>(op)) {
            emitPrint(printOp);
        } else if (auto ifOp = dyn_cast<IfOp>(op)) {
            emitIf(ifOp);
        } else if (auto whileOp = dyn_cast<WhileOp>(op)) {
            emitWhile(whileOp);
        } else if (isa<mlir::func::ReturnOp>(op)) {
            // Return from main - already handled by HALT
        } else {
            // Unknown operation - emit NOP
            emit1(NOP);
        }
    }

    void emitConstantInt(ConstantIntOp op) {
        int32_t value = op.getValue();
        if (value >= -128 && value <= 127) {
            emit1(PUSH_I8);
            emit1(static_cast<uint8_t>(value));
        } else {
            emit1(PUSH_I32);
            emit4(static_cast<uint32_t>(value));
        }
    }

    void emitConstantFloat(ConstantFloatOp op) {
        float value = op.getValue().convertToFloat();
        emit1(PUSH_F32);

        // Emit float as 4 bytes
        union { float f; uint32_t i; } converter;
        converter.f = value;
        emit4(converter.i);
    }

    void emitConstantBool(ConstantBoolOp op) {
        if (op.getValue()) {
            emit1(PUSH_TRUE);
        } else {
            emit1(PUSH_FALSE);
        }
    }

    void emitPrint(PrintOp op) {
        // Add print to native function table if not already present
        if (std::find(nativeFunctions.begin(), nativeFunctions.end(), NATIVE_PRINT) == nativeFunctions.end()) {
            nativeFunctions.push_back(NATIVE_PRINT);
        }

        // Emit native call
        emit1(NCALL);
        emit1(NATIVE_PRINT);
        emit1(static_cast<uint8_t>(op.getArgs().size()));
    }

    void emitIf(IfOp op) {
        // Emit condition (should already be on stack from previous operations)

        // JMP_IF_FALSE to else block (or end if no else)
        emit1(JMP_IF_FALSE);
        size_t elseJumpAddr = bytecode.size();
        emit2(0); // Placeholder for jump offset

        // Emit then block
        emitBlock(op.getThenRegion().front());

        // If there's an else region, emit JMP to skip it
        size_t endJumpAddr = 0;
        if (!op.getElseRegion().empty()) {
            emit1(JMP);
            endJumpAddr = bytecode.size();
            emit2(0); // Placeholder for jump offset
        }

        // Patch else jump
        size_t elseStart = bytecode.size();
        int16_t elseOffset = static_cast<int16_t>(elseStart - elseJumpAddr - 2);
        bytecode[elseJumpAddr] = elseOffset & 0xFF;
        bytecode[elseJumpAddr + 1] = (elseOffset >> 8) & 0xFF;

        // Emit else block if present
        if (!op.getElseRegion().empty()) {
            emitBlock(op.getElseRegion().front());

            // Patch end jump
            size_t end = bytecode.size();
            int16_t endOffset = static_cast<int16_t>(end - endJumpAddr - 2);
            bytecode[endJumpAddr] = endOffset & 0xFF;
            bytecode[endJumpAddr + 1] = (endOffset >> 8) & 0xFF;
        }
    }

    void emitWhile(WhileOp op) {
        size_t loopStart = bytecode.size();

        // Emit condition evaluation (simplified - should emit condition region)
        emitBlock(op.getCondition().front());

        // JMP_IF_FALSE to end of loop
        emit1(JMP_IF_FALSE);
        size_t exitJumpAddr = bytecode.size();
        emit2(0); // Placeholder for jump offset

        // Emit body
        emitBlock(op.getBody().front());

        // JMP back to condition
        emit1(JMP);
        int16_t backOffset = static_cast<int16_t>(loopStart - bytecode.size() - 2);
        emit2(static_cast<uint16_t>(backOffset));

        // Patch exit jump
        size_t loopEnd = bytecode.size();
        int16_t exitOffset = static_cast<int16_t>(loopEnd - exitJumpAddr - 2);
        bytecode[exitJumpAddr] = exitOffset & 0xFF;
        bytecode[exitJumpAddr + 1] = (exitOffset >> 8) & 0xFF;
    }

    void emit1(uint8_t byte) {
        bytecode.push_back(byte);
    }

    void emit2(uint16_t word) {
        bytecode.push_back(word & 0xFF);
        bytecode.push_back((word >> 8) & 0xFF);
    }

    void emit4(uint32_t dword) {
        bytecode.push_back(dword & 0xFF);
        bytecode.push_back((dword >> 8) & 0xFF);
        bytecode.push_back((dword >> 16) & 0xFF);
        bytecode.push_back((dword >> 24) & 0xFF);
    }

    bool writeToFile(const std::string &filename) {
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            return false;
        }

        file.write(reinterpret_cast<const char*>(bytecode.data()), bytecode.size());
        return file.good();
    }
};

} // anonymous namespace

bool emitBytecodeFromMLIR(mlir::ModuleOp module, const char *outputFile) {
    SmolluBytecodeEmitter emitter;
    return emitter.emitModule(module, std::string(outputFile));
}