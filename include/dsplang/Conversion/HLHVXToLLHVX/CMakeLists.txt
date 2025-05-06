add_mlir_pdll_library(HLHVXToLLHVXIncGen HLHVXToLLHVX.pdll HLHVXToLLHVX.h.inc)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name HLHVXToLLHVX)
add_public_tablegen_target(MLIRHLHVXToLLHVXPasses)
