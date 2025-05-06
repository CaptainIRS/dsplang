include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_llvm_executable(llhvx-translate llhvx-translate.cpp)
llvm_update_compile_flags(llhvx-translate)

add_dependencies(llhvx-translate LLHVXOpsIncGen MLIRLLHVXToLLVMIRTranslation)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

target_link_libraries(
  llhvx-translate
  PRIVATE ${dialect_libs}
          ${translation_libs}
          MLIRLLHVXDialect
          MLIRLLHVXToLLVMIRTranslation
          MLIRIR
          MLIRParser
          MLIRPass
          MLIRTargetLLVMIRExport
          MLIRTargetLLVMIRImport)

target_link_libraries(llhvx-translate PUBLIC)
mlir_check_link_libraries(llhvx-translate)
