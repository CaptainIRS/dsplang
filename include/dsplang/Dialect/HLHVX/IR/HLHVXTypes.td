#ifndef DIALECT_HLHVX_TYPES_H
#define DIALECT_HLHVX_TYPES_H

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "dsplang/Dialect/HLHVX/IR/HLHVX.td"

class HLHVX_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<HLHVX_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def VectorOfQf32 : HLHVX_Type<"VectorOfQf32", "VectorOfQf32">;
def VectorOfQf16 : HLHVX_Type<"VectorOfQf16", "VectorOfQf16">;
def VectorPairOfQf32 : HLHVX_Type<"VectorPairOfQf32", "VectorPairOfQf32">;
def VectorPairOfQf16 : HLHVX_Type<"VectorPairOfQf16", "VectorPairOfQf16">;



#endif // DIALECT_HLHVX_TYPES_H