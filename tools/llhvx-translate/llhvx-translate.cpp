#include "dsplang/Target/LLVMIR/Dialect/LLHVX/LLHVXToLLVMIRTranslation.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

void registerToLLVMIRTranslation() {
  mlir::TranslateFromMLIRRegistration registration(
      "mlir-to-llvmir", "Translate MLIR to LLVMIR",
      [](mlir::Operation *op, llvm::raw_ostream &output) {
        mlir::PassManager pm(op->getName());
        pm.addPass(mlir::createConvertVectorToLLVMPass());
        pm.addPass(mlir::createConvertFuncToLLVMPass());
        pm.addPass(mlir::createArithToLLVMConversionPass());
        if (mlir::failed(pm.run(op)))
          return mlir::failure();
        llvm::LLVMContext llvmContext;
        auto llvmModule = translateModuleToLLVMIR(op, llvmContext);
        if (!llvmModule)
          return llvm::failure();

        llvmModule->print(output, nullptr);
        return llvm::success();
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<mlir::arith::ArithDialect>();
        registry.insert<mlir::func::FuncDialect>();
        registry.insert<mlir::vector::VectorDialect>();
        registry.insert<mlir::LLVM::LLVMDialect>();
        dsplang::registerLLHVXDialectTranslation(registry);
        registerAllToLLVMIRTranslations(registry);
      });
}

int main(int argc, char **argv) {
  mlir::registerPassManagerCLOptions();
  // mlir::registerToLLVMIRTranslation();
  registerToLLVMIRTranslation();

  return llvm::failed(
      mlir::mlirTranslateMain(argc, argv, "MLIR to LLVMIR translation"));
}
