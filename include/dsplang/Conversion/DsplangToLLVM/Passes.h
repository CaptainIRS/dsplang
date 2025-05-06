// Adapted from LLVM MLIR Dsplang example

#ifndef DSPLANG_TO_LLVM_PASSES_H
#define DSPLANG_TO_LLVM_PASSES_H

#include <memory>

namespace mlir {
class Pass;
}

namespace dsplang {
/// Create a pass for lowering operations the remaining `Dsplang` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

/// Create a pass for vectorizing the Affine loops.
std::unique_ptr<mlir::Pass> createVectorizeAffineLoopsPass();

} // namespace dsplang

#endif // DSPLANG_TO_LLVM_PASSES_H
