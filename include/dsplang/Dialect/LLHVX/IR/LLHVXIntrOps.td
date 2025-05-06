#ifndef DIALECT_LLHVX_INTR_OPS_H
#define DIALECT_LLHVX_INTR_OPS_H

include "dsplang/Dialect/LLHVX/IR/LLHVX.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


class LLHVX_OneResultIntrOp<string mnemonic,
                     list<Trait> traits = [],
                     int numResults = 1> :
    ExtIntrOpBase</*opName =*/"intr." # mnemonic,
                  /*enumName =*/"hexagon.V6." # mnemonic,
                  traits,
                  numResults>;

class LLHVX_ZeroResultIntrOp<string mnemonic,
                     list<Trait> traits = [],
                     int numResults = 0> :
    ExtIntrOpBase</*opName =*/"intr." # mnemonic,
                  /*enumName =*/"hexagon.V6." # mnemonic,
                  traits,
                  numResults>;

class LLHVX_TwoResultIntrOp<string mnemonic,
                     list<Trait> traits = [],
                     int numResults = 2> :
    ExtIntrOpBase</*opName =*/"intr." # mnemonic,
                  /*enumName =*/"hexagon.V6." # mnemonic,
                  traits,
                  numResults>;

include "dsplang/Dialect/LLHVX/IR/LLHVXIntrOpsGen.td"

#endif // DIALECT_LLHVX_INTR_OPS_H