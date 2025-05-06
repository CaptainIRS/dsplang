// Adapted from LLVM MLIR Toy example

#ifndef DSPLANG_COMBINE
#define DSPLANG_COMBINE

include "mlir/IR/PatternBase.td"
include "dsplang/Dialect/Dsplang/IR/DsplangOps.td"

/// Note: The DRR definition used for defining patterns is shown below:
///
/// class Pattern<
///    dag sourcePattern, list<dag> resultPatterns,
///    list<dag> additionalConstraints = [],
///    dag benefitsAdded = (addBenefit 0)
/// >;

//===----------------------------------------------------------------------===//
// Basic Pattern-Match and Rewrite
//===----------------------------------------------------------------------===//

// Reshape(Reshape(x)) = Reshape(x)
def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)),
                                   (ReshapeOp $arg)>;

//===----------------------------------------------------------------------===//
// Pattern-Match and Rewrite using Native Code Call
//===----------------------------------------------------------------------===//

// Native Code Calls may be used for more complex transformations using inline
// C++ and C++ helper functions.

// Reshape(Constant(x)) = x'
def ReshapeConstant :
  NativeCodeCall<"$0.reshape(::llvm::cast<ShapedType>($1.getType()))">;
def FoldConstantSI8ReshapeOptPattern : Pat<
  (ReshapeOp:$res (ConstantSI8Op $arg)),
  (ConstantSI8Op (ReshapeConstant $arg, $res))>;
def FoldConstantUI8ReshapeOptPattern : Pat<
  (ReshapeOp:$res (ConstantUI8Op $arg)),
  (ConstantUI8Op (ReshapeConstant $arg, $res))>;
def FoldConstantSI16ReshapeOptPattern : Pat<
  (ReshapeOp:$res (ConstantSI16Op $arg)),
  (ConstantSI16Op (ReshapeConstant $arg, $res))>;
def FoldConstantUI16ReshapeOptPattern : Pat<
  (ReshapeOp:$res (ConstantUI16Op $arg)),
  (ConstantUI16Op (ReshapeConstant $arg, $res))>;
def FoldConstantSI32ReshapeOptPattern : Pat<
  (ReshapeOp:$res (ConstantSI32Op $arg)),
  (ConstantSI32Op (ReshapeConstant $arg, $res))>;
def FoldConstantUI32ReshapeOptPattern : Pat<
  (ReshapeOp:$res (ConstantUI32Op $arg)),
  (ConstantUI32Op (ReshapeConstant $arg, $res))>;
def FoldConstantF16ReshapeOptPattern : Pat<
  (ReshapeOp:$res (ConstantF16Op $arg)),
  (ConstantF16Op (ReshapeConstant $arg, $res))>;
def FoldConstantF32ReshapeOptPattern : Pat<
  (ReshapeOp:$res (ConstantF32Op $arg)),
  (ConstantF32Op (ReshapeConstant $arg, $res))>;

//===----------------------------------------------------------------------===//
// Pattern-Match and Rewrite with Constraints
//===----------------------------------------------------------------------===//

// DRR allows for constraint checking when the transformation is conditional
// on operand properties.

// Reshape(x) = x, where input and output shapes are identical
def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;
def RedundantReshapeOptPattern : Pat<
  (ReshapeOp:$res $arg), (replaceWithValue $arg),
  [(TypesAreIdentical $res, $arg)]>;

#endif // DSPLANG_COMBINE
