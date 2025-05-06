#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"

#define GET_OP_CLASSES
#include "dsplang/Dialect/LLHVX/IR/LLHVXOps.h.inc"

#include "dsplang/Dialect/LLHVX/IR/LLHVXDialect.h.inc"

namespace llvm {

class CallInst;
class IRBuilderBase;
class StringRef;

} // namespace llvm

namespace mlir {

class Operation;

namespace LLVM {
class ModuleTranslation;
} // namespace LLVM

} // namespace mlir

namespace dsplang {

llvm::CallInst *createExternalLLVMIntrinsicCall(
    llvm::IRBuilderBase &builder,
    mlir::LLVM::ModuleTranslation &moduleTranslation, mlir::Operation *intrOp,
    llvm::StringRef intrinsicName);

} // namespace dsplang
