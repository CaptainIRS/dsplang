// Adapted from LLVM MLIR Toy example

#ifndef DSPLANG_MLIRGEN_H
#define DSPLANG_MLIRGEN_H

#include <memory>

namespace mlir {
class MLIRContext;
template <typename OpTy> class OwningOpRef;
class ModuleOp;
} // namespace mlir

namespace dsplang {
class ModuleAST;

/// Emit IR for the given Dsplang moduleAST, returns a newly created MLIR module
/// or nullptr on failure.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                          ModuleAST &moduleAST);
} // namespace dsplang

#endif // DSPLANG_MLIRGEN_H
