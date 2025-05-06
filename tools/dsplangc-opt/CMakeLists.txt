include_directories(${CMAKE_CURRENT_BINARY_DIR})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS ${dialect_libs} ${conversion_libs} MLIRHLHVXToLLHVX MLIRArithToHLHVX
         MLIROptLib MLIRPass)

add_llvm_executable(dsplangc-opt dsplangc-opt.cpp)

llvm_update_compile_flags(dsplangc-opt)
target_link_libraries(dsplangc-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(dsplangc-opt)
