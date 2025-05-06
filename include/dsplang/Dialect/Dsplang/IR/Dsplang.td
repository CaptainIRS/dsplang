#ifndef DIALECT_DSPLANG_H
#define DIALECT_DSPLANG_H

include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "dsplang/Dialect/Dsplang/IR/ShapeInferenceInterface.td"

def Dsplang_Dialect : Dialect {
  let name = "dsplang";
  let cppNamespace = "dsplang";
}

#endif // DIALECT_DSPLANG_H
