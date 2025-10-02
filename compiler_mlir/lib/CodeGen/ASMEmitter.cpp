//===- ASMEmitter.cpp - MLIR to Smollu ASM emitter -------------*- C++ -*-===//
//
// Emits human-readable Smollu assembly from MLIR representation
//
//===----------------------------------------------------------------------===//

#include "Smollu/SmolluASMEmitter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "Smollu/SmolluOps.h"

#include <vector>
#include <fstream>
#include <sstream>
#include <map>

using namespace mlir;
using namespace mlir::smollu;

namespace {

class SmolluASMEmitter {
private:
    std::ostringstream asm_output;
    int labelCounter = 0;
    int currentAddress = 0;
    std::map<std::string, std::string> nativeFunctionMap;

public:
    SmolluASMEmitter() {
        // Map native function names to IDs
        nativeFunctionMap["print"] = "0x80";
        nativeFunctionMap["gpio_write"] = "0x81";
        nativeFunctionMap["led_matrix_init"] = "0x82";
        nativeFunctionMap["led_set_color"] = "0x83";
        nativeFunctionMap["led_set"] = "0x84";
        nativeFunctionMap["led_clear"] = "0x85";
        nativeFunctionMap["led_set_row"] = "0x86";
        nativeFunctionMap["led_set_col"] = "0x87";
        nativeFunctionMap["led_get"] = "0x88";
        nativeFunctionMap["delay_ms"] = "0x89";
    }

    bool emitModule(ModuleOp module, const std::string &outputFile) {
        // Emit header comment
        asm_output << "; Smollu Assembly Output\n";
        asm_output << "; Generated from MLIR\n\n";

        // Emit functions
        module.walk([&](mlir::func::FuncOp func) {
            emitFunction(func);
            return WalkResult::advance();
        });

        // Write to file
        return writeToFile(outputFile);
    }

private:
    std::string getNewLabel(const std::string &prefix = "L") {
        return prefix + std::to_string(labelCounter++);
    }

    void emit(const std::string &text) {
        asm_output << text;
    }

    void emitLine(const std::string &text = "") {
        asm_output << text << "\n";
    }

    void emitComment(const std::string &comment) {
        asm_output << "  ; " << comment << "\n";
    }

    void emitInstruction(const std::string &mnemonic, const std::string &operands = "", const std::string &comment = "") {
        asm_output << "    " << mnemonic;
        if (!operands.empty()) {
            // Pad mnemonic to 16 characters for alignment
            int padding = 16 - mnemonic.length();
            for (int i = 0; i < padding; i++) {
                asm_output << " ";
            }
            asm_output << operands;
        }
        if (!comment.empty()) {
            asm_output << "  ; " << comment;
        }
        asm_output << "\n";
        currentAddress++;
    }

    void emitLabel(const std::string &label) {
        asm_output << label << ":\n";
    }

    void emitFunction(mlir::func::FuncOp func) {
        std::string funcName = func.getName().str();

        emitLine();
        emitLine("; ==========================================");
        emitComment("Function: " + funcName);
        emitLine("; ==========================================");
        emitLabel(funcName);

        // Emit function body
        for (Block &block : func.getBlocks()) {
            emitBlock(block);
        }

        // End with HALT or RET depending on function
        if (funcName == "main" || funcName == "__init") {
            emitInstruction("HALT", "", "End of " + funcName);
        } else {
            emitInstruction("RET", "1", "Return 1 value");
        }

        emitLine();
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
        } else if (isa<AddOp>(op)) {
            emitInstruction("ADD", "", "a + b");
        } else if (isa<SubOp>(op)) {
            emitInstruction("SUB", "", "a - b");
        } else if (isa<MulOp>(op)) {
            emitInstruction("MUL", "", "a * b");
        } else if (isa<DivOp>(op)) {
            emitInstruction("DIV", "", "a / b");
        } else if (isa<ModOp>(op)) {
            emitInstruction("MOD", "", "a % b");
        } else if (isa<EqOp>(op)) {
            emitInstruction("EQ", "", "a == b");
        } else if (isa<NeOp>(op)) {
            emitInstruction("NEQ", "", "a != b");
        } else if (isa<LtOp>(op)) {
            emitInstruction("LT", "", "a < b");
        } else if (isa<LeOp>(op)) {
            emitInstruction("LE", "", "a <= b");
        } else if (isa<GtOp>(op)) {
            emitInstruction("GT", "", "a > b");
        } else if (isa<GeOp>(op)) {
            emitInstruction("GE", "", "a >= b");
        } else if (isa<AndOp>(op)) {
            emitInstruction("AND", "", "a && b");
        } else if (isa<OrOp>(op)) {
            emitInstruction("OR", "", "a || b");
        } else if (isa<NotOp>(op)) {
            emitInstruction("NOT", "", "!a");
        } else if (auto getGlobalOp = dyn_cast<GetGlobalOp>(op)) {
            std::ostringstream operands;
            operands << static_cast<int>(getGlobalOp.getSlot());
            emitInstruction("LOAD_GLOBAL", operands.str(), "Load global[" + operands.str() + "]");
        } else if (auto setGlobalOp = dyn_cast<SetGlobalOp>(op)) {
            std::ostringstream operands;
            operands << static_cast<int>(setGlobalOp.getSlot());
            emitInstruction("STORE_GLOBAL", operands.str(), "Store to global[" + operands.str() + "]");
        } else if (auto getLocalOp = dyn_cast<GetLocalOp>(op)) {
            std::ostringstream operands;
            operands << static_cast<int>(getLocalOp.getSlot());
            emitInstruction("LOAD_LOCAL", operands.str(), "Load local[" + operands.str() + "]");
        } else if (auto setLocalOp = dyn_cast<SetLocalOp>(op)) {
            std::ostringstream operands;
            operands << static_cast<int>(setLocalOp.getSlot());
            emitInstruction("STORE_LOCAL", operands.str(), "Store to local[" + operands.str() + "]");
        } else if (auto printOp = dyn_cast<PrintOp>(op)) {
            emitPrint(printOp);
        } else if (auto nativeCallOp = dyn_cast<NativeCallOp>(op)) {
            emitNativeCall(nativeCallOp);
        } else if (auto callOp = dyn_cast<CallOp>(op)) {
            emitCall(callOp);
        } else if (auto returnOp = dyn_cast<ReturnOp>(op)) {
            emitReturn(returnOp);
        } else if (auto ifOp = dyn_cast<IfOp>(op)) {
            emitIf(ifOp);
        } else if (auto whileOp = dyn_cast<WhileOp>(op)) {
            emitWhile(whileOp);
        } else if (isa<mlir::func::ReturnOp>(op)) {
            // Function return - skip, handled by RET instruction
        } else {
            emitComment("Unknown operation: " + op->getName().getStringRef().str());
            emitInstruction("NOP", "", "Unknown operation");
        }
    }

    void emitConstantInt(ConstantIntOp op) {
        int32_t value = op.getValue();
        std::ostringstream operands;

        if (value >= -128 && value <= 127) {
            operands << static_cast<int>(value);
            emitInstruction("PUSH_I8", operands.str(), "Push byte " + operands.str());
        } else {
            operands << value;
            emitInstruction("PUSH_I32", operands.str(), "Push int32 " + operands.str());
        }
    }

    void emitConstantFloat(ConstantFloatOp op) {
        float value = op.getValue().convertToFloat();
        std::ostringstream operands;
        operands << value;
        emitInstruction("PUSH_F32", operands.str(), "Push float " + operands.str());
    }

    void emitConstantBool(ConstantBoolOp op) {
        if (op.getValue()) {
            emitInstruction("PUSH_TRUE", "", "Push true");
        } else {
            emitInstruction("PUSH_FALSE", "", "Push false");
        }
    }

    void emitPrint(PrintOp op) {
        std::ostringstream operands;
        size_t argc = op.getArgs().size();
        operands << "0x80, " << argc;
        emitInstruction("NCALL", operands.str(), "print(" + std::to_string(argc) + " args)");
    }

    void emitNativeCall(NativeCallOp op) {
        std::string funcName = op.getName().str();
        size_t argc = op.getArgs().size();

        std::string nativeId = "0x80"; // default to print
        if (nativeFunctionMap.find(funcName) != nativeFunctionMap.end()) {
            nativeId = nativeFunctionMap[funcName];
        }

        std::ostringstream operands;
        operands << nativeId << ", " << argc;
        emitInstruction("NCALL", operands.str(), funcName + "(" + std::to_string(argc) + " args)");
    }

    void emitCall(CallOp op) {
        std::string funcName = op.getCalleeAttr().getValue().str();
        size_t argc = op.getArgs().size();

        std::ostringstream operands;
        operands << funcName << ", " << argc;
        emitInstruction("CALL", operands.str(), "Call " + funcName + "(" + std::to_string(argc) + " args)");
    }

    void emitReturn(ReturnOp op) {
        if (op.getValue()) {
            emitInstruction("RET", "1", "Return 1 value");
        } else {
            emitInstruction("RET", "0", "Return void");
        }
    }

    void emitIf(IfOp op) {
        std::string elseLabel = getNewLabel("else");
        std::string endLabel = getNewLabel("endif");

        emitComment("if statement");

        // Condition is already on stack
        emitInstruction("JMP_IF_FALSE", elseLabel, "Jump to else if false");

        // Then block
        emitComment("then block");
        emitBlock(op.getThenRegion().front());

        // Jump to end if we have an else
        if (!op.getElseRegion().empty()) {
            emitInstruction("JMP", endLabel, "Skip else block");
        }

        // Else block
        if (!op.getElseRegion().empty()) {
            emitLabel(elseLabel);
            emitComment("else block");
            emitBlock(op.getElseRegion().front());
            emitLabel(endLabel);
        } else {
            emitLabel(elseLabel);
        }

        emitComment("end if");
    }

    void emitWhile(WhileOp op) {
        std::string loopStart = getNewLabel("while_start");
        std::string loopEnd = getNewLabel("while_end");

        emitComment("while loop");
        emitLabel(loopStart);

        // Emit condition
        emitComment("condition");
        emitBlock(op.getCondition().front());

        emitInstruction("JMP_IF_FALSE", loopEnd, "Exit loop if false");

        // Emit body
        emitComment("loop body");
        emitBlock(op.getBody().front());

        emitInstruction("JMP", loopStart, "Loop back to condition");

        emitLabel(loopEnd);
        emitComment("end while");
    }

    bool writeToFile(const std::string &filename) {
        std::ofstream file(filename);
        if (!file) {
            return false;
        }

        file << asm_output.str();
        return file.good();
    }
};

} // anonymous namespace

namespace mlir {
namespace smollu {

bool emitASMFromMLIR(mlir::ModuleOp module, const char *outputFile) {
    SmolluASMEmitter emitter;
    return emitter.emitModule(module, std::string(outputFile));
}

} // namespace smollu
} // namespace mlir
