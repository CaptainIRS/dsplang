add_mlir_pdll_library(ArithToHLHVXIncGen ArithToHLHVX.pdll ArithToHLHVX.h.inc)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name ArithToHLHVX)
add_public_tablegen_target(MLIRArithToHLHVXPasses)
