add_subdirectory(Combine)

add_mlir_dialect_library(
  MLIRDsplangDialect
  DsplangOps.cpp
  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
  DsplangOpsIncGen
  DsplangCombine
  LINK_LIBS
  PUBLIC
  DsplangCombine
  MLIRIR
  MLIRPass)
