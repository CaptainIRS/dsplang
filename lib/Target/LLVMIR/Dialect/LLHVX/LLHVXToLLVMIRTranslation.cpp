#include "dsplang/Target/LLVMIR/Dialect/LLHVX/LLHVXToLLVMIRTranslation.h"
#include "dsplang/Dialect/LLHVX/LLHVXDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/IRBuilder.h"

using namespace llvm;

namespace mlir {
class DialectRegistry;
class MLIRContext;
} // namespace mlir

namespace {
class LLHVXDialectLLVMIRTranslationInterface
    : public mlir::LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  llvm::LogicalResult convertOperation(
      mlir::Operation *op, llvm::IRBuilderBase &builder,
      mlir::LLVM::ModuleTranslation &moduleTranslation) const final {
    mlir::Operation &opInst = *op;
#include "dsplang/Dialect/LLHVX/IR/MLIRLLHVXConversions.inc"

    return failure();
  }
};
} // namespace

void dsplang::registerLLHVXDialectTranslation(mlir::DialectRegistry &registry) {
  registry.insert<dsplang::LLHVXDialect>();
  registry.addExtension(
      +[](mlir::MLIRContext *ctx, dsplang::LLHVXDialect *dialect) {
        dialect->addInterfaces<LLHVXDialectLLVMIRTranslationInterface>();
      });
}

void dsplang::registerLLHVXDialectTranslation(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  registerLLHVXDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}