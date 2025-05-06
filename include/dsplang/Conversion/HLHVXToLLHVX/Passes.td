#ifndef CONVERSION_HLHVXTOHLHVX_PASSES_TD
#define CONVERSION_HLHVXTOHLHVX_PASSES_TD

include "mlir/Dialect/PDL/IR/PDLDialect.td"
include "mlir/Dialect/PDLInterp/IR/PDLInterpOps.td"
include "mlir/Dialect/Vector/IR/VectorOps.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/Dialect/LLVMIR/LLVMDialect.td"
include "mlir/Pass/PassBase.td"

def HLHVXToLLHVXPass : Pass<"hlhvx-to-llhvx"> {
  let summary = "Convert HLHVX to LLHVX";
  let description = [{
    Convert HLHVX operations to LLHVX operations.
  }];
  let dependentDialects = [
    "mlir::pdl::PDLDialect",
    "mlir::pdl_interp::PDLInterpDialect",
    "dsplang::LLHVXDialect",
    "dsplang::HLHVXDialect",
    "mlir::vector::VectorDialect",
    "mlir::arith::ArithDialect",
    "mlir::LLVM::LLVMDialect",
  ];
}

#endif // CONVERSION_HLHVXTOHLHVX_PASSES_TD
