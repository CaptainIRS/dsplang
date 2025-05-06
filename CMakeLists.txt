cmake_minimum_required(VERSION 3.20.0)
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()
project(dsplang LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD
    17
    CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_CLANG_TIDY clang-tidy
                         -checks=-*,clang-analyzer-*,-clang-analyzer-cplusplus*)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(DSPLANG_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(DSPLANG_SOURCE_DIR ${PROJECT_BINARY_DIR})
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
include_directories(${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)

find_program(
  CLANG_FORMAT_EXE
  NAMES "clang-format"
  DOC "Path to clang-format executable")

if(NOT CLANG_FORMAT_EXE)
  message(SEND_ERROR "clang-format executable not found")
endif()

file(GLOB_RECURSE ALL_SOURCE_FILES include/*.h lib/*.cpp tools/*.cpp)

add_custom_target(clangformat ALL COMMAND ${CLANG_FORMAT_EXE} -i -style=LLVM
                                          ${ALL_SOURCE_FILES})

find_program(
  CMAKE_FORMAT_EXE
  NAMES "cmake-format"
  DOC "Path to cmake-format executable")

if(NOT CMAKE_FORMAT_EXE)
  message(SEND_ERROR "cmake-format executable not found")
endif()
file(GLOB_RECURSE ALL_CMAKE_FILES *.cmake *CMakeLists.txt)

list(FILTER ALL_CMAKE_FILES EXCLUDE REGEX ${CMAKE_SOURCE_DIR}/build*)
list(FILTER ALL_CMAKE_FILES EXCLUDE REGEX ${CMAKE_INSTALL_PREFIX}/*)

add_custom_target(cmakeformat ALL COMMAND ${CMAKE_FORMAT_EXE} -i
                                          ${ALL_CMAKE_FILES})
