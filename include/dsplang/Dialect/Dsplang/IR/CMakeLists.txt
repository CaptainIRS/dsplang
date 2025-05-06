add_mlir_dialect(Dsplang dsplang)
set(LLVM_TARGET_DEFINITIONS DsplangOps.td)
mlir_tablegen(DsplangOps.h.inc -gen-op-decls)
mlir_tablegen(DsplangOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(DsplangOpsIncGen)

set(LLVM_TARGET_DEFINITIONS DsplangCombine.td)
mlir_tablegen(DsplangCombine.inc -gen-rewriters)
add_public_tablegen_target(DsplangCombineIncGen)

set(LLVM_TARGET_DEFINITIONS ShapeInferenceInterface.td)
mlir_tablegen(ShapeInferenceOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ShapeInferenceOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(ShapeInferenceInterfaceIncGen)

add_mlir_doc(Dsplang DsplangOps ./Dialect -gen-dialect-doc)
