// Adapted from LLVM MLIR Dsplang sample

#ifndef DSPLANG_OPS
#define DSPLANG_OPS

include "dsplang/Dialect/Dsplang/IR/Dsplang.td"

// Base class for dsplang dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Dsplang_Op<string mnemonic, list<Trait> traits = []> :
    Op<Dsplang_Dialect, mnemonic, traits>;


def DsplangTensor : TensorOf<[UI8, UI16, UI32, SI8, SI16, SI32, F16, F32]>;
def DsplangAnyTensor : TensorOf<[AnyI8, AnyI16, AnyI32, F16, F32]>;
def DsplangStaticShapeTensor : StaticShapeTensorOf<[UI8, UI16, UI32, SI8, SI16, SI32, F16, F32]>;
def DsplangMemref : MemRefOf<[UI8, UI16, UI32, SI8, SI16, SI32, F16, F32]>;
def DsplangAnyMemref : MemRefOf<[AnyI8, AnyI16, AnyI32, F16, F32]>;
//===----------------------------------------------------------------------===//
// Dsplang Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

// We define a dsplang operation by inheriting from our base 'Dsplang_Op' class above.
// Here we provide the mnemonic and a list of traits for the operation. The
// constant operation is marked as 'Pure' as it is a pure operation
// and may be removed if dead.
def ConstantSI8Op : Dsplang_Op<"constant.si8", [Pure]> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "constant.si8";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = dsplang.constant dense<[[1, 2, 3], [4, 5, 6]]>
                        : tensor<2x3xsi8>
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins IntElementsAttrOf<8>:$value);

  // The constant operation returns a single value of TensorType.
  let results = (outs TensorOf<[SI8]>);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Add custom build methods for the constant operation. These method populates
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<ConstantOp>(...)`.
  let builders = [
    // Build a constant with a given constant tensor value.
    OpBuilder<(ins "DenseIntElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,

    // Build a constant with a given constant int value.
    OpBuilder<(ins "int":$value)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}

def ConstantUI8Op : Dsplang_Op<"constant.ui8", [Pure]> {
  let summary = "constant.ui8";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = dsplang.constant dense<[[1, 2, 3], [4, 5, 6]]>
                        : tensor<2x3xui8>
    ```
  }];

  let arguments = (ins IntElementsAttrOf<8>:$value);
  let results = (outs TensorOf<[UI8]>);

  let hasCustomAssemblyFormat = 1;
  let builders = [
    OpBuilder<(ins "DenseIntElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,

    OpBuilder<(ins "int":$value)>
  ];
  let hasVerifier = 1;
}

def ConstantSI16Op : Dsplang_Op<"constant.si16", [Pure]> {
  let summary = "constant.si16";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = dsplang.constant dense<[[1, 2, 3], [4, 5, 6]]>
                        : tensor<2x3xsi16>
    ```
  }];

  let arguments = (ins IntElementsAttrOf<16>:$value);
  let results = (outs TensorOf<[SI16]>);

  let hasCustomAssemblyFormat = 1;
  let builders = [
    OpBuilder<(ins "DenseIntElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,

    OpBuilder<(ins "int":$value)>
  ];
  let hasVerifier = 1;
}

def ConstantUI16Op : Dsplang_Op<"constant.ui16", [Pure]> {
  let summary = "constant.ui16";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = dsplang.constant dense<[[1, 2, 3], [4, 5, 6]]>
                        : tensor<2x3xui16>
    ```
  }];

  let arguments = (ins IntElementsAttrOf<16>:$value);
  let results = (outs TensorOf<[UI16]>);

  let hasCustomAssemblyFormat = 1;
  let builders = [
    OpBuilder<(ins "DenseIntElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,

    OpBuilder<(ins "int":$value)>
  ];
  let hasVerifier = 1;
}

def ConstantSI32Op : Dsplang_Op<"constant.si32", [Pure]> {
  let summary = "constant.si32";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = dsplang.constant dense<[[1, 2, 3], [4, 5, 6]]>
                        : tensor<2x3xsi32>
    ```
  }];

  let arguments = (ins IntElementsAttrOf<32>:$value);
  let results = (outs TensorOf<[SI32]>);

  let hasCustomAssemblyFormat = 1;
  let builders = [
    OpBuilder<(ins "DenseIntElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,

    OpBuilder<(ins "int":$value)>
  ];
  let hasVerifier = 1;
}

def ConstantUI32Op : Dsplang_Op<"constant.ui32", [Pure]> {
  let summary = "constant.ui32";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = dsplang.constant dense<[[1, 2, 3], [4, 5, 6]]>
                        : tensor<2x3xui32>
    ```
  }];

  let arguments = (ins IntElementsAttrOf<32>:$value);
  let results = (outs TensorOf<[UI32]>);

  let hasCustomAssemblyFormat = 1;
  let builders = [
    OpBuilder<(ins "DenseIntElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,

    OpBuilder<(ins "int":$value)>
  ];
  let hasVerifier = 1;
}

def ConstantF16Op : Dsplang_Op<"constant.f16", [Pure]> {
  let summary = "constant.f16";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = dsplang.constant dense<[[1, 2, 3], [4, 5, 6]]>
                        : tensor<2x3xf16>
    ```
  }];

  let arguments = (ins FloatElementsAttr<16>:$value);
  let results = (outs TensorOf<[F16]>);

  let hasCustomAssemblyFormat = 1;
  let builders = [
    OpBuilder<(ins "DenseFPElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,

    OpBuilder<(ins "float":$value)>
  ];
  let hasVerifier = 1;
}

def ConstantF32Op : Dsplang_Op<"constant.f32", [Pure]> {
  let summary = "constant.f32";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = dsplang.constant dense<[[1, 2, 3], [4, 5, 6]]>
                        : tensor<2x3xf32>
    ```
  }];

  let arguments = (ins FloatElementsAttr<32>:$value);
  let results = (outs TensorOf<[F32]>);

  let hasCustomAssemblyFormat = 1;
  let builders = [
    OpBuilder<(ins "DenseFPElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,

    OpBuilder<(ins "float":$value)>
  ];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

def AddOp : Dsplang_Op<"add",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins DsplangTensor:$lhs, DsplangTensor:$rhs);
  let results = (outs DsplangTensor);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Allow building an AddOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

def CastOp : Dsplang_Op<"cast", [
     DeclareOpInterfaceMethods<CastOpInterface>,
     DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
     Pure,
     SameOperandsAndResultShape
  ]> {
  let summary = "shape cast operation";
  let description = [{
    The "cast" operation converts a tensor from one type to an equivalent type
    without changing any data elements. The source and destination types must
    both be tensor types with the same element type. If both are ranked, then
    shape is required to match. The operation is invalid if converting to a
    mismatching constant dimension.
  }];

  let arguments = (ins DsplangTensor:$input);
  let results = (outs DsplangTensor:$output);

  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}

//===----------------------------------------------------------------------===//
// AssignOp
//===----------------------------------------------------------------------===//
def AssignOp : Dsplang_Op<"assign"> {
  let summary = "tensor assignment operation";
  let description = [{
    The "assign" operation assigns a tensor to another tensor.
  }];

  let arguments = (ins DsplangAnyMemref:$lhs, DsplangTensor:$rhs);
  // let results = (outs DsplangTensor);

  // // Indicate that the operation has a custom parser and printer method.
  // let hasCustomAssemblyFormat = 1;
  let assemblyFormat = [{
    $lhs `=` $rhs attr-dict `:` type($rhs) `to` type($lhs)
  }];

  // Allow building an AssignOp with from the two input operands.
  // let builders = [
  //   OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  // ];
}


//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

def FuncOp : Dsplang_Op<"func", [
    FunctionOpInterface, IsolatedFromAbove
  ]> {
  let summary = "user defined function operation";
  let description = [{
    The "dsplang.func" operation represents a user defined function. These are
    callable SSA-region operations that contain dsplang computations.

    Example:

    ```mlir
    dsplang.func @main() {
      %0 = dsplang.constant dense<5.500000e+00> : tensor<f64>
      %1 = dsplang.reshape(%0 : tensor<f64>) to tensor<2x2xf64>
      dsplang.print %1 : tensor<2x2xf64>
      dsplang.return
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
  >];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Returns the region on the function operation that is callable.
    Region *getCallableRegion() { return &getBody(); }
  }];
  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// GenericCallOp
//===----------------------------------------------------------------------===//

def GenericCallOp : Dsplang_Op<"generic_call",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "generic call operation";
  let description = [{
    Generic calls represent calls to a user defined function that needs to
    be specialized for the shape of its arguments. The callee name is attached
    as a symbol reference via an attribute. The arguments list must match the
    arguments expected by the callee. For example:

    ```mlir
     %4 = dsplang.generic_call @my_func(%1, %3)
           : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    ```

    This is only valid if a function named "my_func" exists and takes two
    arguments.
  }];

  // The generic call operation takes a symbol reference attribute as the
  // callee, and inputs for the call.
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<DsplangTensor>:$inputs,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );

  // The generic call operation returns a single value of TensorType.
  let results = (outs DsplangTensor);

  // Specialize assembly printing and parsing using a declarative format.
  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];

  // Add custom build methods for the generic call operation.
  let builders = [
    OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments)>
  ];
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

def MulOp : Dsplang_Op<"mul",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "element-wise multiplication operation";
  let description = [{
    The "mul" operation performs element-wise multiplication between two
    tensors. The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins DsplangTensor:$lhs, DsplangTensor:$rhs);
  let results = (outs DsplangTensor);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Allow building a MulOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

//===----------------------------------------------------------------------===//
// PrintOp
//===----------------------------------------------------------------------===//

def PrintOp : Dsplang_Op<"print"> {
  let summary = "print operation";
  let description = [{
    The "print" builtin operation prints a given input tensor, and produces
    no results.
  }];

  // The print operation takes an input tensor to print.
  let arguments = (ins AnyTypeOf<[DsplangAnyTensor, DsplangAnyMemref]>:$input);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

//===----------------------------------------------------------------------===//
// AndOp
//===----------------------------------------------------------------------===//

def AndOp : Dsplang_Op<"and",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "element-wise logical and operation";
  let description = [{
    The "and" operation performs element-wise logical and between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins DsplangTensor:$lhs, DsplangTensor:$rhs);
  let results = (outs DsplangTensor);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Allow building an AndOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

//===----------------------------------------------------------------------===//
// OrOp
//===----------------------------------------------------------------------===//

def OrOp : Dsplang_Op<"or",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "element-wise logical or operation";
  let description = [{
    The "or" operation performs element-wise logical or between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins DsplangTensor:$lhs, DsplangTensor:$rhs);
  let results = (outs DsplangTensor);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Allow building an OrOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

//===----------------------------------------------------------------------===//
// XorOp
//===----------------------------------------------------------------------===//

def XorOp : Dsplang_Op<"xor",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "element-wise logical xor operation";
  let description = [{
    The "xor" operation performs element-wise logical xor between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins DsplangTensor:$lhs, DsplangTensor:$rhs);
  let results = (outs DsplangTensor);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Allow building an XorOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

def SubOp : Dsplang_Op<"sub",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "element-wise subtraction operation";
  let description = [{
    The "sub" operation performs element-wise subtraction between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins DsplangTensor:$lhs, DsplangTensor:$rhs);
  let results = (outs DsplangTensor);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Allow building a SubOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

def ReshapeOp : Dsplang_Op<"reshape", [Pure]> {
  let summary = "tensor reshape operation";
  let description = [{
    Reshape operation is transforming its input tensor into a new tensor with
    the same number of elements but different shapes. For example:

    ```mlir
       %0 = dsplang.reshape (%arg1 : tensor<10xf64>) to tensor<5x2xf64>
    ```
  }];

  let arguments = (ins DsplangTensor:$input);

  // We expect that the reshape operation returns a statically shaped tensor.
  let results = (outs DsplangStaticShapeTensor);

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)
  }];

  // Enable registering canonicalization patterns with this operation.
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// BufferizeOp
//===----------------------------------------------------------------------===//

def BufferizeOp : Dsplang_Op<"bufferize", [Pure]> {
  let summary = "memref to tensor bufferize operation";
  let description = [{
    Bufferize operation is transforming its input memref into a new tensor.

    ```mlir
       %0 = dsplang.bufferize (%arg1 : memref<10xf64>) to tensor<10xf64>
    ```
  }];

  let arguments = (ins DsplangAnyMemref:$input);

  // We expect that the reshape operation returns a statically shaped tensor.
  let results = (outs DsplangStaticShapeTensor);

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)
  }];
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def ReturnOp : Dsplang_Op<"return", [Pure, HasParent<"FuncOp">,
                                 Terminator]> {
  let summary = "return operation";
  let description = [{
    The "return" operation represents a return operation within a function.
    The operation takes an optional tensor operand and produces no results.
    The operand type must match the signature of the function that contains
    the operation. For example:

    ```mlir
      dsplang.func @foo() -> tensor<2xf64> {
        ...
        dsplang.return %0 : tensor<2xf64>
      }
    ```
  }];

  // The return operation takes an optional input operand to return. This
  // value must match the return type of the enclosing function.
  let arguments = (ins Variadic<DsplangTensor>:$input);

  // The return operation only emits the input in the format if it is present.
  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

  // Allow building a ReturnOp with no return operand.
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
  ];

  // Provide extra utility definitions on the c++ operation class definition.
  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}

#endif // Dsplang_OpS
