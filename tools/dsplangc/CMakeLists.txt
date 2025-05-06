include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_llvm_executable(dsplangc dsplangc.cpp)
llvm_update_compile_flags(dsplangc)

add_dependencies(dsplangc DsplangOpsIncGen ShapeInferenceInterfaceIncGen
                 LLHVXOpsIncGen DsplangLanguage MLIRLLHVXToLLVMIRTranslation)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

llvm_map_components_to_libnames(llvm_libs support)

target_link_libraries(
  dsplangc
  PRIVATE ${llvm_libs}
          ${dialect_libs}
          ${conversion_libs}
          ${extension_libs}
          ${translation_libs}
          DsplangLanguage
          LLVMHexagonAsmParser
          LLVMHexagonCodeGen
          LLVMHexagonDesc
          LLVMHexagonDisassembler
          LLVMHexagonInfo
          MLIRAnalysis
          MLIRIR
          MLIRParser
          MLIRPass
          MLIRFunctionInterfaces
          MLIRSideEffectInterfaces
          MLIRCastInterfaces
          MLIRTransforms)

target_link_libraries(dsplangc PUBLIC)
mlir_check_link_libraries(dsplangc)
