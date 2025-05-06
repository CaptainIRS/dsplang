#include "dsplang/Conversion/HLHVXToLLHVX/Passes.h"
#include "dsplang/Dialect/HLHVX/HLHVXDialect.h"
#include "dsplang/Dialect/LLHVX/LLHVXDialect.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<dsplang::HLHVXDialect>();
  registry.insert<dsplang::LLHVXDialect>();
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();

  dsplang::registerHLHVXToLLHVXPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Dsplang Pass Driver", registry));
}
