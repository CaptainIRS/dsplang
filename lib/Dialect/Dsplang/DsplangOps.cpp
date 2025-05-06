// Adapted from LLVM MLIR Toy example

#include "dsplang/Dialect/Dsplang/DsplangDialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <string>

#include "dsplang/Dialect/Dsplang/IR/DsplangDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// DsplangInlinerInterface
//===----------------------------------------------------------------------===//

/// This class defines the interface for handling inlining with Dsplang
/// operations.
struct DsplangInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  /// All call operations within dsplang can be inlined.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  /// All operations within dsplang can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }

  // All functions within dsplang can be inlined.
  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  /// Handle the given inlined terminator(dsplang.return) by replacing it with a
  /// new operation as necessary.
  void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {
    // Only "dsplang.return" needs to be handled here.
    auto returnOp = cast<dsplang::ReturnOp>(op);

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }

  /// Attempts to materialize a conversion for a type mismatch between a call
  /// from this dialect, and a callable region. This method should generate an
  /// operation that takes 'input' as the only operand, and produces a single
  /// result of 'resultType'. If a conversion can not be generated, nullptr
  /// should be returned.
  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type resultType,
                                       Location conversionLoc) const final {
    return builder.create<dsplang::CastOp>(conversionLoc, resultType, input);
  }
};

//===----------------------------------------------------------------------===//
// DsplangDialect
//===----------------------------------------------------------------------===//

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void dsplang::DsplangDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "dsplang/Dialect/Dsplang/IR/DsplangOps.cpp.inc"
      >();
  addInterfaces<DsplangInlinerInterface>();
}

//===----------------------------------------------------------------------===//
// Dsplang Operations
//===----------------------------------------------------------------------===//

/// A generalized parser for binary operations. This parses the different forms
/// of 'printBinaryOp' below.
static mlir::ParseResult parseBinaryOp(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2> operands;
  SMLoc operandsLoc = parser.getCurrentLocation();
  Type type;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/2) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type))
    return mlir::failure();

  // If the type is a function type, it contains the input and result types of
  // this operation.
  if (FunctionType funcType = llvm::dyn_cast<FunctionType>(type)) {
    if (parser.resolveOperands(operands, funcType.getInputs(), operandsLoc,
                               result.operands))
      return mlir::failure();
    result.addTypes(funcType.getResults());
    return mlir::success();
  }

  // Otherwise, the parsed type is the type of both operands and results.
  if (parser.resolveOperands(operands, type, result.operands))
    return mlir::failure();
  result.addTypes(type);
  return mlir::success();
}

/// A generalized printer for binary operations. It prints in two different
/// forms depending on if all of the types match.
static void printBinaryOp(mlir::OpAsmPrinter &printer, mlir::Operation *op) {
  printer << " " << op->getOperands();
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : ";

  // If all of the types are the same, print the type directly.
  Type resultType = *op->result_type_begin();
  if (llvm::all_of(op->getOperandTypes(),
                   [=](Type type) { return type == resultType; })) {
    printer << resultType;
    return;
  }

  // Otherwise, print a functional type.
  printer.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}

//===----------------------------------------------------------------------===//
// ConstantSI8Op
//===----------------------------------------------------------------------===//

/// Build a constant.si8 operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
void dsplang::ConstantSI8Op::build(mlir::OpBuilder &builder,
                                   mlir::OperationState &state, int value) {
  auto dataType = RankedTensorType::get({}, builder.getIntegerType(8, true));
  auto dataAttribute = DenseIntElementsAttr::get(dataType, value);
  ConstantSI8Op::build(builder, state, dataType, dataAttribute);
}

//===----------------------------------------------------------------------===//
// ConstantUI8Op
//===----------------------------------------------------------------------===//

/// Build a constant.ui8 operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
void dsplang::ConstantUI8Op::build(mlir::OpBuilder &builder,
                                   mlir::OperationState &state, int value) {
  auto dataType = RankedTensorType::get({}, builder.getIntegerType(8, false));
  auto dataAttribute = DenseIntElementsAttr::get(dataType, value);
  ConstantUI8Op::build(builder, state, dataType, dataAttribute);
}

//===----------------------------------------------------------------------===//
// ConstantSI16Op
//===----------------------------------------------------------------------===//

/// Build a constant.si16 operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
void dsplang::ConstantSI16Op::build(mlir::OpBuilder &builder,
                                    mlir::OperationState &state, int value) {
  auto dataType = RankedTensorType::get({}, builder.getIntegerType(16, true));
  auto dataAttribute = DenseIntElementsAttr::get(dataType, value);
  ConstantSI16Op::build(builder, state, dataType, dataAttribute);
}

//===----------------------------------------------------------------------===//
// ConstantUI16Op
//===----------------------------------------------------------------------===//

/// Build a constant.ui16 operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
void dsplang::ConstantUI16Op::build(mlir::OpBuilder &builder,
                                    mlir::OperationState &state, int value) {
  auto dataType = RankedTensorType::get({}, builder.getIntegerType(16, false));
  auto dataAttribute = DenseIntElementsAttr::get(dataType, value);
  ConstantUI16Op::build(builder, state, dataType, dataAttribute);
}

//===----------------------------------------------------------------------===//
// ConstantSI32Op
//===----------------------------------------------------------------------===//

/// Build a constant.si32 operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
void dsplang::ConstantSI32Op::build(mlir::OpBuilder &builder,
                                    mlir::OperationState &state, int value) {
  auto dataType = RankedTensorType::get({}, builder.getIntegerType(32, true));
  auto dataAttribute = DenseIntElementsAttr::get(dataType, value);
  ConstantSI32Op::build(builder, state, dataType, dataAttribute);
}

//===----------------------------------------------------------------------===//
// ConstantUI32Op
//===----------------------------------------------------------------------===//

/// Build a constant.ui32 operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
void dsplang::ConstantUI32Op::build(mlir::OpBuilder &builder,
                                    mlir::OperationState &state, int value) {
  auto dataType = RankedTensorType::get({}, builder.getIntegerType(32, false));
  auto dataAttribute = DenseIntElementsAttr::get(dataType, value);
  ConstantUI32Op::build(builder, state, dataType, dataAttribute);
}

//===----------------------------------------------------------------------===//
// ConstantF16Op
//===----------------------------------------------------------------------===//

/// Build a constant.f16 operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
void dsplang::ConstantF16Op::build(mlir::OpBuilder &builder,
                                   mlir::OperationState &state, float value) {
  auto dataType = RankedTensorType::get({}, builder.getF16Type());
  auto dataAttribute = DenseFPElementsAttr::get(dataType, value);
  ConstantF16Op::build(builder, state, dataType, dataAttribute);
}

//===----------------------------------------------------------------------===//
// ConstantF32Op
//===----------------------------------------------------------------------===//

/// Build a constant.f32 operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
void dsplang::ConstantF32Op::build(mlir::OpBuilder &builder,
                                   mlir::OperationState &state, float value) {
  auto dataType = RankedTensorType::get({}, builder.getF32Type());
  auto dataAttribute = DenseFPElementsAttr::get(dataType, value);
  ConstantF32Op::build(builder, state, dataType, dataAttribute);
}

/// The 'OpAsmParser' class provides a collection of methods for parsing
/// various punctuation, as well as attributes, operands, types, etc. Each of
/// these methods returns a `ParseResult`. This class is a wrapper around
/// `LogicalResult` that can be converted to a boolean `true` value on failure,
/// or `false` on success. This allows for easily chaining together a set of
/// parser rules. These rules are used to populate an `mlir::OperationState`
/// similarly to the `build` methods described above.
mlir::ParseResult dsplang::ConstantSI8Op::parse(mlir::OpAsmParser &parser,
                                                mlir::OperationState &result) {
  mlir::DenseIntElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(value, "value", result.attributes))
    return failure();

  result.addTypes(value.getType());
  return success();
}
mlir::ParseResult dsplang::ConstantUI8Op::parse(mlir::OpAsmParser &parser,
                                                mlir::OperationState &result) {
  mlir::DenseIntElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(value, "value", result.attributes))
    return failure();

  result.addTypes(value.getType());
  return success();
}
mlir::ParseResult dsplang::ConstantSI16Op::parse(mlir::OpAsmParser &parser,
                                                 mlir::OperationState &result) {
  mlir::DenseIntElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(value, "value", result.attributes))
    return failure();

  result.addTypes(value.getType());
  return success();
}
mlir::ParseResult dsplang::ConstantUI16Op::parse(mlir::OpAsmParser &parser,
                                                 mlir::OperationState &result) {
  mlir::DenseIntElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(value, "value", result.attributes))
    return failure();

  result.addTypes(value.getType());
  return success();
}
mlir::ParseResult dsplang::ConstantSI32Op::parse(mlir::OpAsmParser &parser,
                                                 mlir::OperationState &result) {
  mlir::DenseIntElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(value, "value", result.attributes))
    return failure();

  result.addTypes(value.getType());
  return success();
}
mlir::ParseResult dsplang::ConstantUI32Op::parse(mlir::OpAsmParser &parser,
                                                 mlir::OperationState &result) {
  mlir::DenseIntElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(value, "value", result.attributes))
    return failure();

  result.addTypes(value.getType());
  return success();
}
mlir::ParseResult dsplang::ConstantF16Op::parse(mlir::OpAsmParser &parser,
                                                mlir::OperationState &result) {
  mlir::DenseElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(value, "value", result.attributes))
    return failure();

  result.addTypes(value.getType());
  return success();
}
mlir::ParseResult dsplang::ConstantF32Op::parse(mlir::OpAsmParser &parser,
                                                mlir::OperationState &result) {
  mlir::DenseElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(value, "value", result.attributes))
    return failure();

  result.addTypes(value.getType());
  return success();
}

/// The 'OpAsmPrinter' class is a stream that allows for formatting
/// strings, attributes, operands, types, etc.
void dsplang::ConstantSI8Op::print(mlir::OpAsmPrinter &printer) {
  printer << " ";
  printer.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  printer << getValue();
}
void dsplang::ConstantUI8Op::print(mlir::OpAsmPrinter &printer) {
  printer << " ";
  printer.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  printer << getValue();
}
void dsplang::ConstantSI16Op::print(mlir::OpAsmPrinter &printer) {
  printer << " ";
  printer.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  printer << getValue();
}
void dsplang::ConstantUI16Op::print(mlir::OpAsmPrinter &printer) {
  printer << " ";
  printer.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  printer << getValue();
}
void dsplang::ConstantSI32Op::print(mlir::OpAsmPrinter &printer) {
  printer << " ";
  printer.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  printer << getValue();
}
void dsplang::ConstantUI32Op::print(mlir::OpAsmPrinter &printer) {
  printer << " ";
  printer.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  printer << getValue();
}
void dsplang::ConstantF16Op::print(mlir::OpAsmPrinter &printer) {
  printer << " ";
  printer.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  printer << getValue();
}
void dsplang::ConstantF32Op::print(mlir::OpAsmPrinter &printer) {
  printer << " ";
  printer.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  printer << getValue();
}

/// Verifier for the constant operation. This corresponds to the
/// `let hasVerifier = 1` in the op definition.
llvm::LogicalResult dsplang::ConstantSI8Op::verify() {
  // If the return type of the constant is not an unranked tensor, the shape
  // must match the shape of the attribute holding the data.
  auto resultType =
      llvm::dyn_cast<mlir::RankedTensorType>(getResult().getType());
  if (!resultType)
    return success();

  // Check that the rank of the attribute type matches the rank of the constant
  // result type.
  auto attrType = llvm::cast<mlir::RankedTensorType>(getValue().getType());
  if (attrType.getRank() != resultType.getRank()) {
    return emitOpError("return type must match the one of the attached value "
                       "attribute: ")
           << attrType.getRank() << " != " << resultType.getRank();
  }

  // Check that each of the dimensions match between the two types.
  for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
    if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
      return emitOpError(
                 "return type shape mismatches its attribute at dimension ")
             << dim << ": " << attrType.getShape()[dim]
             << " != " << resultType.getShape()[dim];
    }
  }
  return mlir::success();
}
llvm::LogicalResult dsplang::ConstantUI8Op::verify() {
  // If the return type of the constant is not an unranked tensor, the shape
  // must match the shape of the attribute holding the data.
  auto resultType =
      llvm::dyn_cast<mlir::RankedTensorType>(getResult().getType());
  if (!resultType)
    return success();

  // Check that the rank of the attribute type matches the rank of the constant
  // result type.
  auto attrType = llvm::cast<mlir::RankedTensorType>(getValue().getType());
  if (attrType.getRank() != resultType.getRank()) {
    return emitOpError("return type must match the one of the attached value "
                       "attribute: ")
           << attrType.getRank() << " != " << resultType.getRank();
  }

  // Check that each of the dimensions match between the two types.
  for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
    if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
      return emitOpError(
                 "return type shape mismatches its attribute at dimension ")
             << dim << ": " << attrType.getShape()[dim]
             << " != " << resultType.getShape()[dim];
    }
  }
  return mlir::success();
}
llvm::LogicalResult dsplang::ConstantSI16Op::verify() {
  // If the return type of the constant is not an unranked tensor, the shape
  // must match the shape of the attribute holding the data.
  auto resultType =
      llvm::dyn_cast<mlir::RankedTensorType>(getResult().getType());
  if (!resultType)
    return success();

  // Check that the rank of the attribute type matches the rank of the constant
  // result type.
  auto attrType = llvm::cast<mlir::RankedTensorType>(getValue().getType());
  if (attrType.getRank() != resultType.getRank()) {
    return emitOpError("return type must match the one of the attached value "
                       "attribute: ")
           << attrType.getRank() << " != " << resultType.getRank();
  }

  // Check that each of the dimensions match between the two types.
  for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
    if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
      return emitOpError(
                 "return type shape mismatches its attribute at dimension ")
             << dim << ": " << attrType.getShape()[dim]
             << " != " << resultType.getShape()[dim];
    }
  }
  return mlir::success();
}
llvm::LogicalResult dsplang::ConstantUI16Op::verify() {
  // If the return type of the constant is not an unranked tensor, the shape
  // must match the shape of the attribute holding the data.
  auto resultType =
      llvm::dyn_cast<mlir::RankedTensorType>(getResult().getType());
  if (!resultType)
    return success();

  // Check that the rank of the attribute type matches the rank of the constant
  // result type.
  auto attrType = llvm::cast<mlir::RankedTensorType>(getValue().getType());
  if (attrType.getRank() != resultType.getRank()) {
    return emitOpError("return type must match the one of the attached value "
                       "attribute: ")
           << attrType.getRank() << " != " << resultType.getRank();
  }

  // Check that each of the dimensions match between the two types.
  for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
    if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
      return emitOpError(
                 "return type shape mismatches its attribute at dimension ")
             << dim << ": " << attrType.getShape()[dim]
             << " != " << resultType.getShape()[dim];
    }
  }
  return mlir::success();
}
llvm::LogicalResult dsplang::ConstantSI32Op::verify() {
  // If the return type of the constant is not an unranked tensor, the shape
  // must match the shape of the attribute holding the data.
  auto resultType =
      llvm::dyn_cast<mlir::RankedTensorType>(getResult().getType());
  if (!resultType)
    return success();

  // Check that the rank of the attribute type matches the rank of the constant
  // result type.
  auto attrType = llvm::cast<mlir::RankedTensorType>(getValue().getType());
  if (attrType.getRank() != resultType.getRank()) {
    return emitOpError("return type must match the one of the attached value "
                       "attribute: ")
           << attrType.getRank() << " != " << resultType.getRank();
  }

  // Check that each of the dimensions match between the two types.
  for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
    if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
      return emitOpError(
                 "return type shape mismatches its attribute at dimension ")
             << dim << ": " << attrType.getShape()[dim]
             << " != " << resultType.getShape()[dim];
    }
  }
  return mlir::success();
}
llvm::LogicalResult dsplang::ConstantUI32Op::verify() {
  // If the return type of the constant is not an unranked tensor, the shape
  // must match the shape of the attribute holding the data.
  auto resultType =
      llvm::dyn_cast<mlir::RankedTensorType>(getResult().getType());
  if (!resultType)
    return success();

  // Check that the rank of the attribute type matches the rank of the constant
  // result type.
  auto attrType = llvm::cast<mlir::RankedTensorType>(getValue().getType());
  if (attrType.getRank() != resultType.getRank()) {
    return emitOpError("return type must match the one of the attached value "
                       "attribute: ")
           << attrType.getRank() << " != " << resultType.getRank();
  }

  // Check that each of the dimensions match between the two types.
  for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
    if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
      return emitOpError(
                 "return type shape mismatches its attribute at dimension ")
             << dim << ": " << attrType.getShape()[dim]
             << " != " << resultType.getShape()[dim];
    }
  }
  return mlir::success();
}
llvm::LogicalResult dsplang::ConstantF16Op::verify() {
  // If the return type of the constant is not an unranked tensor, the shape
  // must match the shape of the attribute holding the data.
  auto resultType =
      llvm::dyn_cast<mlir::RankedTensorType>(getResult().getType());
  if (!resultType)
    return success();

  // Check that the rank of the attribute type matches the rank of the constant
  // result type.
  auto attrType = llvm::cast<mlir::RankedTensorType>(getValue().getType());
  if (attrType.getRank() != resultType.getRank()) {
    return emitOpError("return type must match the one of the attached value "
                       "attribute: ")
           << attrType.getRank() << " != " << resultType.getRank();
  }

  // Check that each of the dimensions match between the two types.
  for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
    if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
      return emitOpError(
                 "return type shape mismatches its attribute at dimension ")
             << dim << ": " << attrType.getShape()[dim]
             << " != " << resultType.getShape()[dim];
    }
  }
  return mlir::success();
}
llvm::LogicalResult dsplang::ConstantF32Op::verify() {
  // If the return type of the constant is not an unranked tensor, the shape
  // must match the shape of the attribute holding the data.
  auto resultType =
      llvm::dyn_cast<mlir::RankedTensorType>(getResult().getType());
  if (!resultType)
    return success();

  // Check that the rank of the attribute type matches the rank of the constant
  // result type.
  auto attrType = llvm::cast<mlir::RankedTensorType>(getValue().getType());
  if (attrType.getRank() != resultType.getRank()) {
    return emitOpError("return type must match the one of the attached value "
                       "attribute: ")
           << attrType.getRank() << " != " << resultType.getRank();
  }

  // Check that each of the dimensions match between the two types.
  for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
    if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
      return emitOpError(
                 "return type shape mismatches its attribute at dimension ")
             << dim << ": " << attrType.getShape()[dim]
             << " != " << resultType.getShape()[dim];
    }
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

void dsplang::AddOp::build(mlir::OpBuilder &builder,
                           mlir::OperationState &state, mlir::Value lhs,
                           mlir::Value rhs) {
  auto rhsType = mlir::cast<TensorType>(rhs.getType());
  state.addTypes(UnrankedTensorType::get(rhsType.getElementType()));
  state.addOperands({lhs, rhs});
}

mlir::ParseResult dsplang::AddOp::parse(mlir::OpAsmParser &parser,
                                        mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void dsplang::AddOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

/// Infer the output shape of the AddOp, this is required by the shape inference
/// interface.
void dsplang::AddOp::inferShapes() { getResult().setType(getLhs().getType()); }

//===----------------------------------------------------------------------===//
// AssignOp
//===----------------------------------------------------------------------===//
/// Build an assign operation.
// void dsplang::AssignOp::build(mlir::OpBuilder &builder,
//                               mlir::OperationState &state, mlir::Value lhs,
//                               mlir::Value rhs) {
//   auto rhsType = mlir::cast<TensorType>(rhs.getType());
//   state.addTypes(UnrankedTensorType::get(rhsType.getElementType()));
//   state.addOperands({lhs, rhs});
// }

// mlir::ParseResult dsplang::AssignOp::parse(mlir::OpAsmParser &parser,
//                                            mlir::OperationState &result) {
//   return parseBinaryOp(parser, result);
// }
// void dsplang::AssignOp::print(mlir::OpAsmPrinter &p) {
//   printBinaryOp(p, *this);
// }

/// Infer the output shape of the AssignOp, this is required by the shape
/// inference interface.
// void dsplang::AssignOp::inferShapes() {
// The result type is the same as the lhs type.
// getResult().setType(getRhs().getType());
// }

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

/// Infer the output shape of the CastOp, this is required by the shape
/// inference interface.
void dsplang::CastOp::inferShapes() {
  getResult().setType(getInput().getType());
}

/// Returns true if the given set of input and result types are compatible with
/// this cast operation. This is required by the `CastOpInterface` to verify
/// this operation and provide other additional utilities.
bool dsplang::CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  // The inputs must be Tensors with the same element type.
  TensorType input = llvm::dyn_cast<TensorType>(inputs.front());
  TensorType output = llvm::dyn_cast<TensorType>(outputs.front());
  if (!input || !output || input.getElementType() != output.getElementType())
    return false;
  // The shape is required to match if both types are ranked.
  return !input.hasRank() || !output.hasRank() || input == output;
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

void dsplang::FuncOp::build(mlir::OpBuilder &builder,
                            mlir::OperationState &state, llvm::StringRef name,
                            mlir::FunctionType type,
                            llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  // FunctionOpInterface provides a convenient `build` method that will populate
  // the state of our FuncOp, and create an entry block.
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

mlir::ParseResult dsplang::FuncOp::parse(mlir::OpAsmParser &parser,
                                         mlir::OperationState &result) {
  // Dispatch to the FunctionOpInterface provided utility method that parses the
  // function operation.
  auto buildFuncType =
      [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
         llvm::ArrayRef<mlir::Type> results,
         mlir::function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return mlir::function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void dsplang::FuncOp::print(mlir::OpAsmPrinter &p) {
  // Dispatch to the FunctionOpInterface provided utility method that prints the
  // function operation.
  mlir::function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

//===----------------------------------------------------------------------===//
// GenericCallOp
//===----------------------------------------------------------------------===//

void dsplang::GenericCallOp::build(mlir::OpBuilder &builder,
                                   mlir::OperationState &state,
                                   StringRef callee,
                                   ArrayRef<mlir::Value> arguments) {
  // Generic call always returns an unranked Tensor initially.
  state.addTypes(UnrankedTensorType::get(builder.getF32Type()));
  state.addOperands(arguments);
  state.addAttribute("callee",
                     mlir::SymbolRefAttr::get(builder.getContext(), callee));
}

/// Return the callee of the generic call operation, this is required by the
/// call interface.
CallInterfaceCallable dsplang::GenericCallOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

/// Set the callee for the generic call operation, this is required by the call
/// interface.
void dsplang::GenericCallOp::setCalleeFromCallable(
    CallInterfaceCallable callee) {
  (*this)->setAttr("callee", cast<SymbolRefAttr>(callee));
}

/// Get the argument operands to the called function, this is required by the
/// call interface.
Operation::operand_range dsplang::GenericCallOp::getArgOperands() {
  return getInputs();
}

/// Get the argument operands to the called function as a mutable range, this is
/// required by the call interface.
MutableOperandRange dsplang::GenericCallOp::getArgOperandsMutable() {
  return getInputsMutable();
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

void dsplang::MulOp::build(mlir::OpBuilder &builder,
                           mlir::OperationState &state, mlir::Value lhs,
                           mlir::Value rhs) {
  auto rhsType = mlir::cast<TensorType>(rhs.getType());
  state.addTypes(UnrankedTensorType::get(rhsType.getElementType()));
  state.addOperands({lhs, rhs});
}

mlir::ParseResult dsplang::MulOp::parse(mlir::OpAsmParser &parser,
                                        mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void dsplang::MulOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

/// Infer the output shape of the MulOp, this is required by the shape inference
/// interface.
void dsplang::MulOp::inferShapes() { getResult().setType(getLhs().getType()); }

//===----------------------------------------------------------------------===//
// AndOp
//===----------------------------------------------------------------------===//

void dsplang::AndOp::build(mlir::OpBuilder &builder,
                           mlir::OperationState &state, mlir::Value lhs,
                           mlir::Value rhs) {
  auto rhsType = mlir::cast<TensorType>(rhs.getType());
  state.addTypes(UnrankedTensorType::get(rhsType.getElementType()));
  state.addOperands({lhs, rhs});
}

mlir::ParseResult dsplang::AndOp::parse(mlir::OpAsmParser &parser,
                                        mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void dsplang::AndOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

/// Infer the output shape of the AndOp, this is required by the shape inference
/// interface.
void dsplang::AndOp::inferShapes() { getResult().setType(getLhs().getType()); }

//===----------------------------------------------------------------------===//
// OrOp
//===----------------------------------------------------------------------===//

void dsplang::OrOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                          mlir::Value lhs, mlir::Value rhs) {
  auto rhsType = mlir::cast<TensorType>(rhs.getType());
  state.addTypes(UnrankedTensorType::get(rhsType.getElementType()));
  state.addOperands({lhs, rhs});
}

mlir::ParseResult dsplang::OrOp::parse(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void dsplang::OrOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

/// Infer the output shape of the OrOp, this is required by the shape inference
/// interface.
void dsplang::OrOp::inferShapes() { getResult().setType(getLhs().getType()); }

//===----------------------------------------------------------------------===//
// XorOp
//===----------------------------------------------------------------------===//

void dsplang::XorOp::build(mlir::OpBuilder &builder,
                           mlir::OperationState &state, mlir::Value lhs,
                           mlir::Value rhs) {
  auto rhsType = mlir::cast<TensorType>(rhs.getType());
  state.addTypes(UnrankedTensorType::get(rhsType.getElementType()));
  state.addOperands({lhs, rhs});
}

mlir::ParseResult dsplang::XorOp::parse(mlir::OpAsmParser &parser,
                                        mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void dsplang::XorOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

/// Infer the output shape of the XorOp, this is required by the shape inference
/// interface.
void dsplang::XorOp::inferShapes() { getResult().setType(getLhs().getType()); }

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

void dsplang::SubOp::build(mlir::OpBuilder &builder,
                           mlir::OperationState &state, mlir::Value lhs,
                           mlir::Value rhs) {
  auto rhsType = mlir::cast<TensorType>(rhs.getType());
  state.addTypes(UnrankedTensorType::get(rhsType.getElementType()));
  state.addOperands({lhs, rhs});
}

mlir::ParseResult dsplang::SubOp::parse(mlir::OpAsmParser &parser,
                                        mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void dsplang::SubOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

/// Infer the output shape of the SubOp, this is required by the shape inference
/// interface.
void dsplang::SubOp::inferShapes() { getResult().setType(getLhs().getType()); }

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult dsplang::ReturnOp::verify() {
  // We know that the parent operation is a function, because of the 'HasParent'
  // trait attached to the operation definition.
  auto function = cast<FuncOp>((*this)->getParentOp());

  /// ReturnOps can only have a single optional operand.
  if (getNumOperands() > 1)
    return emitOpError() << "expects at most 1 return operand";

  // The operand number and types must match the function signature.
  const auto &results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError() << "does not return the same number of values ("
                         << getNumOperands() << ") as the enclosing function ("
                         << results.size() << ")";

  // If the operation does not have an input, we are done.
  if (!hasOperand())
    return mlir::success();

  auto inputType = *operand_type_begin();
  auto resultType = results.front();

  // Check that the result type of the function matches the operand type.
  if (inputType == resultType ||
      llvm::isa<mlir::UnrankedTensorType>(inputType) ||
      llvm::isa<mlir::UnrankedTensorType>(resultType))
    return mlir::success();

  return emitError() << "type of return operand (" << inputType
                     << ") doesn't match function result type (" << resultType
                     << ")";
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "dsplang/Dialect/Dsplang/IR/DsplangOps.cpp.inc"
