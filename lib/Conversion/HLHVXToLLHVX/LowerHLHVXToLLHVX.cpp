#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "dsplang/Dialect/HLHVX/HLHVXDialect.h"
#include "dsplang/Dialect/LLHVX/LLHVXDialect.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#include "dsplang/Conversion/HLHVXToLLHVX/HLHVXToLLHVX.h"

namespace dsplang {

#define GEN_PASS_DEF_HLHVXTOLLHVXPASS
#include "dsplang/Conversion/HLHVXToLLHVX/Passes.h.inc"

#include "dsplang/Conversion/HLHVXToLLHVX/HLHVXToLLHVX.h.inc"

#include "dsplang/Dialect/HLHVX/HLHVXDialect.h"

struct LowerCastVectorToGenericOp
    : public mlir::OpRewritePattern<dsplang::CastVectorToGenericOp> {
  using mlir::OpRewritePattern<
      dsplang::CastVectorToGenericOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(dsplang::CastVectorToGenericOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto sourceType = mlir::cast<mlir::VectorType>(op.getOperand().getType());
    auto targetType = op.getResult().getType();
    if (sourceType.getShape() == targetType.getShape() &&
        sourceType.getElementType() == targetType.getElementType()) {
      rewriter.replaceOp(op, op.getOperand());
      return mlir::success();
    } else if (sourceType.getNumElements() *
                   sourceType.getElementTypeBitWidth() ==
               targetType.getNumElements() *
                   targetType.getElementTypeBitWidth()) {
      rewriter.replaceOpWithNewOp<mlir::vector::BitCastOp>(op, targetType,
                                                           op.getOperand());
      return mlir::success();
    }
    return mlir::failure();
  }
};

struct LowerCastIntegerToGenericOp
    : public mlir::OpRewritePattern<dsplang::CastIntegerToGenericOp> {
  using mlir::OpRewritePattern<
      dsplang::CastIntegerToGenericOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(dsplang::CastIntegerToGenericOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto sourceType = op.getOperand().getType();
    auto targetType = op.getResult().getType();
    if (sourceType == targetType) {
      rewriter.replaceOp(op, op.getOperand());
      return mlir::success();
    } else {
      rewriter.replaceOpWithNewOp<mlir::arith::BitcastOp>(op, targetType,
                                                          op.getOperand());
      return mlir::success();
    }
    return mlir::failure();
  }
};

struct LowerCastVectorToRegisterOp
    : public mlir::OpRewritePattern<dsplang::CastVectorToRegisterOp> {
  using mlir::OpRewritePattern<
      dsplang::CastVectorToRegisterOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(dsplang::CastVectorToRegisterOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto sourceType = mlir::cast<mlir::VectorType>(op.getOperand().getType());
    auto targetType = op.getResult().getType();
    if (sourceType.getNumElements() * sourceType.getElementTypeBitWidth() ==
        targetType.getWidth()) {
      auto intermediateVectorType =
          mlir::VectorType::get(1, rewriter.getI32Type());
      auto intermediateVector = rewriter.create<mlir::vector::BitCastOp>(
          op.getLoc(), intermediateVectorType, op.getOperand());
      auto zeroConstant = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
      auto intermediateIntegerExtract =
          rewriter.create<mlir::vector::ExtractElementOp>(
              op.getLoc(), rewriter.getI32Type(), intermediateVector,
              zeroConstant);
      rewriter.replaceOp(op, intermediateIntegerExtract);
      return mlir::success();
    }
    return mlir::failure();
  }
};

struct LowerCastVectorFromGenericOp
    : public mlir::OpRewritePattern<dsplang::CastVectorFromGenericOp> {
  using mlir::OpRewritePattern<
      dsplang::CastVectorFromGenericOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(dsplang::CastVectorFromGenericOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto sourceType = mlir::cast<mlir::VectorType>(op.getOperand().getType());
    auto targetType = op.getResult().getType();
    if (sourceType == targetType) {
      rewriter.replaceOp(op, op.getOperand());
      return mlir::success();
    } else if (sourceType.getNumElements() *
                   sourceType.getElementTypeBitWidth() ==
               targetType.getNumElements() *
                   targetType.getElementTypeBitWidth()) {
      rewriter.replaceOpWithNewOp<mlir::vector::BitCastOp>(op, targetType,
                                                           op.getOperand());
      return mlir::success();
    }
    return mlir::failure();
  }
};

struct LowerHLHVXOpsToLLHVXOps
    : impl::HLHVXToLLHVXPassBase<LowerHLHVXOpsToLLHVXOps> {
  using HLHVXToLLHVXPassBase::HLHVXToLLHVXPassBase;

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    populateGeneratedPDLLPatterns(patterns);
    patterns.add<LowerCastVectorToGenericOp>(&getContext());
    patterns.add<LowerCastIntegerToGenericOp>(&getContext());
    patterns.add<LowerCastVectorToRegisterOp>(&getContext());
    patterns.add<LowerCastVectorFromGenericOp>(&getContext());
    // walkAndApplyPatterns(getOperation(), std::move(patterns));
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace dsplang
