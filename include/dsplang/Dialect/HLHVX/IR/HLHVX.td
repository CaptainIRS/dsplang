#ifndef DIALECT_HLHVX_H
#define DIALECT_HLHVX_H

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

def HLHVX_Dialect : Dialect {
    let name = "hlhvx";
    let cppNamespace = "dsplang";
}

#endif
