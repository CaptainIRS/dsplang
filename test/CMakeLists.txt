find_program(
  LLVM_EXTERNAL_LIT
  NAMES llvm-lit lit.py lit
  PATHS "${LLVM_MAIN_SRC_DIR}/utils/lit"
  DOC "Path to lit.py")

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

set(LLHVX_TEST_DEPENDS FileCheck count not llhvx-translate)

message(STATUS "LLVM_EXTERNAL_LIT: ${LLVM_EXTERNAL_LIT}")

add_lit_testsuite(check-llhvx "Run the LLHVX translation tests"
                  ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${LLHVX_TEST_DEPENDS})
set_target_properties(check-llhvx PROPERTIES FOLDER "Tests")

add_lit_testsuites(LLHVX ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS
                   ${LLHVX_TEST_DEPENDS})
