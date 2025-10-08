//===- PromoteNumerics.cpp - Numeric type promotion pass -------*- C++ -*-===//
//
// This pass automatically inserts cast operations to promote integer operands
// to float when mixing int and float in arithmetic operations, following the
// Smollu language specification: "Mixed arithmetic promotes int -> float"
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Smollu/SmolOps.h"
#include "Smollu/SmolDialect.h"

using namespace mlir;
using namespace mlir::smol;

namespace {


/// Helper function to check if a type is i1
static bool isI1(Type type) {
  auto intType = llvm::dyn_cast<IntegerType>(type);
  return intType && intType.getWidth() == 1;
}

/// Helper function to check if a type is i32
static bool isInt32(Type type) {
  auto intType = llvm::dyn_cast<IntegerType>(type);
  return intType && intType.getWidth() == 32;
}

/// Helper function to check if a type is f32
static bool isFloat32(Type type) {
  return llvm::isa<Float32Type>(type);
}

/// Helper function to check if a type is numeric (i32 or f32)
static bool isNumeric(Type type) {
  return isI1(type) || isInt32(type) || isFloat32(type);
}

/// Helper function to create a cast operation from i1 to i32
static Value promoteToI32(OpBuilder &builder, Location loc, Value val) {
  if (isI1(val.getType())) {
    return builder.create<CastOp>(loc, builder.getI32Type(), val);
  }
  return val;
}

/// Helper function to create a cast operation from i32 to f32
static Value promoteToFloat(OpBuilder &builder, Location loc, Value val) {
  if (isFloat32(val.getType())) {
    return val; // Already float, no need to cast
  }

  if (isInt32(val.getType()) || isI1(val.getType())) {
    Type f32Type = builder.getF32Type();
    return builder.create<CastOp>(loc, f32Type, val);
  }

  return val; // Not a numeric type, return as-is
}

/// Pattern to promote operands of binary arithmetic operations
template<typename OpType>
struct PromoteBinaryArithOp : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op, PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type lhsType = lhs.getType();
    Type rhsType = rhs.getType();

    // Skip if types are already the same
    if (lhsType == rhsType) {
      return failure();
    }

    // Check if both operands are numeric
    if (!isNumeric(lhsType) || !isNumeric(rhsType)) {
      return failure();
    }

    // If we have a mix of int and float, promote to float
    bool needsPromotionFloat = (isInt32(lhsType) && isFloat32(rhsType)) ||
                          (isFloat32(lhsType) && isInt32(rhsType)) ||
                          (isI1(lhsType) && isFloat32(rhsType)) ||
                          (isFloat32(lhsType) && isI1(rhsType));

    bool needsPromotionI32 = (isI1(lhsType) && isInt32(rhsType)) ||
                          (isInt32(lhsType) && isI1(rhsType));

    if (!needsPromotionFloat && !needsPromotionI32) {
      return failure();
    }

    // Promote operands to float
    Location loc = op.getLoc();
    Value promotedLhs = needsPromotionFloat ? promoteToFloat(rewriter, loc, lhs) : promoteToI32(rewriter, loc, lhs);
    Value promotedRhs = needsPromotionFloat ? promoteToFloat(rewriter, loc, rhs) : promoteToI32(rewriter, loc, rhs);

    // Create new operation with promoted operands
    Type resultType = needsPromotionFloat ? (Type)rewriter.getF32Type() : (Type)rewriter.getI32Type();
    auto newOp = rewriter.create<OpType>(loc, resultType, promotedLhs, promotedRhs);

    // Replace the original operation
    rewriter.replaceOp(op, newOp.getResult());

    return success();
  }
};

/// Pattern to promote operands of comparison operations
template<typename OpType>
struct PromoteComparisonOp : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op, PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type lhsType = lhs.getType();
    Type rhsType = rhs.getType();

    // Skip if types are already the same
    if (lhsType == rhsType) {
      return failure();
    }

    // Check if both operands are numeric
    if (!isNumeric(lhsType) || !isNumeric(rhsType)) {
      return failure();
    }

    // If we have a mix of int and float, promote to float
    bool needsPromotionFloat = (isInt32(lhsType) && isFloat32(rhsType)) ||
                          (isFloat32(lhsType) && isInt32(rhsType)) ||
                          (isI1(lhsType) && isFloat32(rhsType)) ||
                          (isFloat32(lhsType) && isI1(rhsType));

    bool needsPromotionI32 = (isI1(lhsType) && isInt32(rhsType)) ||
                          (isInt32(lhsType) && isI1(rhsType));

    if (!needsPromotionFloat && !needsPromotionI32) {
      return failure();
    }

    // Promote operands to float or i32
    Location loc = op.getLoc();
    Value promotedLhs = needsPromotionFloat ? promoteToFloat(rewriter, loc, lhs) : promoteToI32(rewriter, loc, lhs);
    Value promotedRhs = needsPromotionFloat ? promoteToFloat(rewriter, loc, rhs) : promoteToI32(rewriter, loc, rhs);

    // Create new comparison operation with promoted operands
    // Result type is always i1 for comparisons
    auto newOp = rewriter.create<OpType>(loc, rewriter.getI1Type(),
                                         promotedLhs, promotedRhs);

    // Replace the original operation
    rewriter.replaceOp(op, newOp.getResult());

    return success();
  }
};

/// Pass to promote numeric types in arithmetic and comparison operations
struct PromoteNumericsPass : public PassWrapper<PromoteNumericsPass,
                                                  OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PromoteNumericsPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<SmolDialect>();
  }

  StringRef getArgument() const final { return "smol-promote-numerics"; }

  StringRef getDescription() const final {
    return "Promote integer operands to float in mixed arithmetic";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // Add patterns for binary arithmetic operations
    patterns.add<PromoteBinaryArithOp<AddOp>>(context);
    patterns.add<PromoteBinaryArithOp<SubOp>>(context);
    patterns.add<PromoteBinaryArithOp<MulOp>>(context);
    patterns.add<PromoteBinaryArithOp<DivOp>>(context);
    patterns.add<PromoteBinaryArithOp<ModOp>>(context);

    // Add patterns for comparison operations
    patterns.add<PromoteComparisonOp<EqOp>>(context);
    patterns.add<PromoteComparisonOp<NeOp>>(context);
    patterns.add<PromoteComparisonOp<LtOp>>(context);
    patterns.add<PromoteComparisonOp<LeOp>>(context);
    patterns.add<PromoteComparisonOp<GtOp>>(context);
    patterns.add<PromoteComparisonOp<GeOp>>(context);

    // Apply patterns
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
namespace smol {

std::unique_ptr<Pass> createPromoteNumericsPass() {
  return std::make_unique<PromoteNumericsPass>();
}

} // namespace smol
} // namespace mlir
