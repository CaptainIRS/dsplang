#ifndef DIALECT_LLHVX_H
#define DIALECT_LLHVX_H

include "mlir/IR/OpBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

def LLHVX_Dialect : Dialect {
    let name = "llhvx";
    let cppNamespace = "dsplang";
}

class ExtIntrOpBase<string opName,
                    string enumName,
                    list<Trait> traits,
                    int numResults
> : LLVM_OpBase<LLHVX_Dialect, opName, traits>,
    Results<!if(!gt(numResults, 1), (outs LLVM_Type:$res), !if(!gt(numResults, 0), (outs LLVM_Type:$res), (outs)))> {
    string baseLlvmBuilder = [{
      auto *inst = dsplang::createExternalLLVMIntrinsicCall(builder,
                        moduleTranslation, &opInst, "llvm.}] # enumName # [{");
      (void) inst;
    }];
    string baseLlvmBuilderCoda = !if(!gt(numResults, 1), "$res = inst;", !if(!gt(numResults, 0), "$res = inst;", ""));
    let llvmBuilder =  baseLlvmBuilder # baseLlvmBuilderCoda;
    string mlirBuilder = "";
    list<int> llvmArgIndices = [];
}

#endif // DIALECT_LLHVX_H
