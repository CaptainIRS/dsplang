add_mlir_dialect(HLHVX hlhvx)
set(LLVM_TARGET_DEFINITIONS HLHVXOps.td)
mlir_tablegen(HLHVXOps.h.inc -gen-op-decls)
mlir_tablegen(HLHVXOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(HLHVXOpsIncGen)

set(LLVM_TARGET_DEFINITIONS HLHVXTypes.td)
mlir_tablegen(HLHVXTypeDefs.h.inc -gen-typedef-decls)
mlir_tablegen(HLHVXTypeDefs.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(HLHVXTypeDefsIncGen)

add_mlir_doc(HLHVX HLHVXOpsGen ./Dialect -gen-dialect-doc)
