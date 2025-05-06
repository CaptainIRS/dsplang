#ifndef DIALECT_HLHVX_OPS_H
#define DIALECT_HLHVX_OPS_H

include "dsplang/Dialect/HLHVX/IR/HLHVX.td"
include "dsplang/Dialect/HLHVX/IR/HLHVXTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

class HLHVX_Op<string mnemonic, list<Trait> traits = []>:
    Op<HLHVX_Dialect, mnemonic, traits>;

def HVX_CastVectorToGenericOp : HLHVX_Op<
    "CastVectorToGeneric",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>]
> {
    let summary = "Cast to generic type";
    let description = [{
        Casts a vector of a specific type to a generic vector type.
    }];
    let arguments = (ins VectorOfLengthAndType<[32, 64, 128, 256, 1024, 2048], [AnyI1, AnyI8, AnyI16, AnyI32, F16, F32, BF16]>:$input);
    let results = (outs VectorOfLengthAndType<[32], [I32]>:$output);
}

def HVX_CastVectorToRegisterOp : HLHVX_Op<
    "CastVectorToRegister",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>]
> {
    let summary = "Cast to register type";
    let description = [{
        Casts a vector of a specific type to a 32/64-bit register type.
    }];
    let arguments = (ins VectorOfLengthAndType<[1, 2, 4, 8], [AnyI8, AnyI16, AnyI32, AnyI64]>:$input);
    let results = (outs I32:$output);
}

def HVX_CastVectorFromGenericOp : HLHVX_Op<
    "CastVectorFromGeneric",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>]
> {
    let summary = "Cast from generic type";
    let description = [{
        Casts a generic vector type to a specific vector type.
    }];
    let arguments = (ins VectorOfLengthAndType<[32, 64], [I32]>:$input);
    let results = (outs VectorOfLengthAndType<[32, 64, 128, 256, 1024, 2048], [AnyI1, AnyI8, AnyI16, AnyI32, F16, F32, BF16]>:$output);
}

def HVX_CastIntegerToGenericOp : HLHVX_Op<
    "CastIntegerToGeneric",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>]
> {
    let summary = "Cast to generic integer type";
    let description = [{
        Casts an integer of a specific type to a generic integer type.
    }];
    let arguments = (ins AnyInteger:$input);
    let results = (outs I32:$output);
}

def HVX_CastIntegerFromGenericOp : HLHVX_Op<
    "CastIntegerFromGeneric",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>]
> {
    let summary = "Cast from generic integer type";
    let description = [{
        Casts a generic integer type to a specific integer type.
    }];
    let arguments = (ins I32:$input);
    let results = (outs AnyInteger:$output);
}

include "dsplang/Dialect/HLHVX/IR/HLHVXOpsGen.td"


#endif // DIALECT_HLHVX_OPS_H
