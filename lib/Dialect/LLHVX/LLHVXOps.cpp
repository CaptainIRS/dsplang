#include "dsplang/Dialect/LLHVX/LLHVXDialect.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/FoldUtils.h"
#include "llvm/IR/IRBuilder.h"

#include "dsplang/Dialect/LLHVX/IR/LLHVXDialect.cpp.inc"

void dsplang::LLHVXDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "dsplang/Dialect/LLHVX/IR/LLHVXOps.cpp.inc"
      >();
}

namespace dsplang {

static llvm::Function *
getNamedIntrinsicDeclaration(llvm::Module *M, llvm::StringRef fullName,
                             llvm::Type *resTy,
                             llvm::ArrayRef<llvm::Type *> argsTy) {
  auto *FT = llvm::FunctionType::get(resTy, argsTy, /*isVarArg=*/false);
  return llvm::cast<llvm::Function>(
      M->getOrInsertFunction(fullName, FT).getCallee());
}
llvm::CallInst *createExternalLLVMIntrinsicCall(
    llvm::IRBuilderBase &builder,
    mlir::LLVM::ModuleTranslation &moduleTranslation, mlir::Operation *intrOp,
    llvm::StringRef intrinsicName) {
  llvm::Type *resTy = nullptr;
  unsigned numResults = intrOp->getNumResults();
  if (numResults == 0)
    resTy = llvm::Type::getVoidTy(builder.getContext());
  else if (numResults == 1)
    resTy = moduleTranslation.convertType(*(intrOp->getResultTypes().begin()));
  else if (numResults > 1) {
    llvm::SmallVector<llvm::Type *> resTys;
    for (auto ty : intrOp->getResultTypes()) {
      llvm::errs() << "ty: " << ty << "\n";
      resTys.push_back(moduleTranslation.convertType(ty));
    }
    resTy = llvm::StructType::get(builder.getContext(), resTys);
  }
  auto operands = moduleTranslation.lookupValues(intrOp->getOperands());
  llvm::SmallVector<llvm::Type *> types;
  for (auto op : operands)
    types.push_back(op->getType());
  llvm::Module *module = builder.GetInsertBlock()->getModule();
  std::string fullName = intrinsicName.str();
  // replace "." with "_"
  std::replace(fullName.begin(), fullName.end(), '_', '.');

  llvm::Function *llvmIntr =
      getNamedIntrinsicDeclaration(module, fullName, resTy, types);
  return builder.CreateCall(llvmIntr, operands);
}
} // namespace dsplang

#define GET_OP_CLASSES
#include "dsplang/Dialect/LLHVX/IR/LLHVXOps.cpp.inc"
