#include "dsplang/Conversion/DsplangToAffine/Passes.h"
#include "dsplang/Dialect/Dsplang/DsplangDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <unordered_map>

namespace {
struct AffineSelectiveVectorizePass
    : public mlir::PassWrapper<AffineSelectiveVectorizePass,
                               OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AffineSelectiveVectorizePass)
  StringRef getArgument() const override {
    return "dsplang-affine-selective-vectorize";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::affine::AffineDialect, mlir::func::FuncDialect,
                    mlir::vector::VectorDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    IRRewriter b(funcOp.getContext());

    funcOp.walk([&](mlir::affine::AffineForOp forOp) {
      if (forOp.getStepAsInt() > 1) {
        return WalkResult::advance();
      }
      std::vector<uint32_t> typeBitwidths;
      bool canVectorize = true;
      for (auto &bodyOp : forOp.getBody()->getOperations()) {
        if (!mlir::isa<mlir::affine::AffineDialect>(bodyOp.getDialect()) &&
            !mlir::isa<mlir::arith::ArithDialect>(bodyOp.getDialect())) {
          canVectorize = false;
          break;
        }
        if (bodyOp.getNumResults() == 0) {
          continue;
        }
        auto resultType = bodyOp.getResult(0).getType();
        typeBitwidths.push_back(resultType.getIntOrFloatBitWidth());
      }
      if (!canVectorize) {
        return WalkResult::advance();
      }
      if (typeBitwidths.size() < 1) {
        return WalkResult::advance();
      }
      if (!std::all_of(
              typeBitwidths.begin(), typeBitwidths.end(),
              [&](uint32_t type) { return type == typeBitwidths[0]; })) {
        return WalkResult::advance();
      }
      DenseSet<Operation *> loops{forOp};
      affine::vectorizeAffineLoops(forOp, loops, {1024 / typeBitwidths[0]}, {});
      return WalkResult::advance();
    });

    funcOp.walk([&](affine::AffineForOp loop) {
      for (auto trReadOp : loop.getRegion().getOps<vector::TransferReadOp>()) {
        VectorType readType = trReadOp.getVector().getType();
        SmallVector<bool> inBounds(readType.getRank(), true);
        trReadOp.setInBoundsAttr(b.getBoolArrayAttr(inBounds));
      }

      for (auto trWrtOp : loop.getRegion().getOps<vector::TransferWriteOp>()) {
        VectorType writeType = trWrtOp.getVector().getType();
        SmallVector<bool> inBounds(writeType.getRank(), true);
        trWrtOp.setInBoundsAttr(b.getBoolArrayAttr(inBounds));
      }
    });
  }
};
} // namespace

/// Create a Affine Vectorization pass.
std::unique_ptr<mlir::Pass> dsplang::createAffineSelectiveVectorizePass() {
  return std::make_unique<AffineSelectiveVectorizePass>();
}
