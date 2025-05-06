#ifndef DIALECT_HLHVX_H
#define DIALECT_HLHVX_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/FoldUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Compiler.h"

#include "dsplang/Dialect/HLHVX/IR/HLHVXDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "dsplang/Dialect/HLHVX/IR/HLHVXTypeDefs.h.inc"

#define GET_OP_CLASSES
#include "dsplang/Dialect/HLHVX/IR/HLHVXOps.h.inc"

#endif