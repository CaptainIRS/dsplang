#ifndef DIALECT_DSPLANGDIALECT_H
#define DIALECT_DSPLANGDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;

namespace dsplang {
#include "dsplang/Dialect/Dsplang/IR/ShapeInferenceOpInterfaces.h.inc"
}

#include "dsplang/Dialect/Dsplang/IR/DsplangDialect.h.inc"

#define GET_OP_CLASSES
#include "dsplang/Dialect/Dsplang/IR/DsplangOps.h.inc"

#endif