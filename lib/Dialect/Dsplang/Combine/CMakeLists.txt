add_llvm_library(DsplangCombine DsplangCombine.cpp)
add_dependencies(DsplangCombine DsplangOpsIncGen)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(DsplangCombine PRIVATE ${dialect_libs})
