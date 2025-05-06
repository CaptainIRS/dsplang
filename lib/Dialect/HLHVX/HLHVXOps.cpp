#include "dsplang/Dialect/HLHVX/HLHVXDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

#include "dsplang/Dialect/HLHVX/IR/HLHVXDialect.cpp.inc"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

void dsplang::HLHVXDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "dsplang/Dialect/HLHVX/IR/HLHVXOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "dsplang/Dialect/HLHVX/IR/HLHVXTypeDefs.cpp.inc"
      >();
}

namespace dsplang {

llvm::LogicalResult dsplang::CastVectorToGenericOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  uint64_t length = 32;
  if (attributes.contains("length")) {
    length = mlir::cast<mlir::IntegerAttr>(attributes.get("length")).getUInt();
  }
  inferredReturnTypes.push_back(
      mlir::VectorType::get(length, mlir::IntegerType::get(context, 32)));
  return mlir::success();
}

llvm::LogicalResult dsplang::CastVectorFromGenericOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  uint64_t length = 32;
  if (attributes.contains("length")) {
    length = mlir::cast<mlir::IntegerAttr>(attributes.get("length")).getUInt();
  } else {
    llvm::errs() << "Missing length attribute\n";
    return mlir::failure();
  }
  std::string type = "i32";
  if (attributes.contains("type")) {
    type =
        mlir::cast<mlir::StringAttr>(attributes.get("type")).getValue().str();
  } else {
    llvm::errs() << "Missing type attribute\n";
    return mlir::failure();
  }
  auto signedness = mlir::IntegerType::SignednessSemantics::Signless;
  if (type[0] == 'u') {
    signedness = mlir::IntegerType::SignednessSemantics::Unsigned;
    type = type.substr(1);
  } else if (type[0] == 's') {
    signedness = mlir::IntegerType::SignednessSemantics::Signed;
    type = type.substr(1);
  }
  if (type == "i8") {
    inferredReturnTypes.push_back(mlir::VectorType::get(
        length, mlir::IntegerType::get(context, 8, signedness)));
  } else if (type == "i16") {
    inferredReturnTypes.push_back(mlir::VectorType::get(
        length, mlir::IntegerType::get(context, 16, signedness)));
  } else if (type == "i32") {
    inferredReturnTypes.push_back(mlir::VectorType::get(
        length, mlir::IntegerType::get(context, 32, signedness)));
  } else if (type == "f16") {
    inferredReturnTypes.push_back(
        mlir::VectorType::get(length, mlir::Float16Type::get(context)));
  } else if (type == "f32") {
    inferredReturnTypes.push_back(
        mlir::VectorType::get(length, mlir::Float32Type::get(context)));
  } else if (type == "bf16") {
    inferredReturnTypes.push_back(
        mlir::VectorType::get(length, mlir::BFloat16Type::get(context)));
  } else {
    llvm::errs() << "Unknown type: " << type << "\n";
    return mlir::failure();
  }
  return mlir::success();
}

llvm::LogicalResult dsplang::CastIntegerFromGenericOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  std::string type = "i32";
  if (attributes.contains("type")) {
    type =
        mlir::cast<mlir::StringAttr>(attributes.get("type")).getValue().str();
  } else {
    llvm::errs() << "Missing type attribute\n";
    return mlir::failure();
  }
  auto signedness = mlir::IntegerType::SignednessSemantics::Signless;
  if (type[0] == 'u') {
    signedness = mlir::IntegerType::SignednessSemantics::Unsigned;
    type = type.substr(1);
  } else if (type[0] == 's') {
    signedness = mlir::IntegerType::SignednessSemantics::Signed;
    type = type.substr(1);
  }
  if (type == "i8") {
    inferredReturnTypes.push_back(
        mlir::IntegerType::get(context, 8, signedness));
  } else if (type == "i16") {
    inferredReturnTypes.push_back(
        mlir::IntegerType::get(context, 16, signedness));
  } else if (type == "i32") {
    inferredReturnTypes.push_back(
        mlir::IntegerType::get(context, 32, signedness));
  } else {
    llvm::errs() << "Unknown type: " << type << "\n";
    return mlir::failure();
  }
  return mlir::success();
}

} // namespace dsplang

LLVM_ATTRIBUTE_UNUSED static mlir::OptionalParseResult
generatedTypeParser(mlir::AsmParser &parser, mlir::StringRef *mnemonic,
                    mlir::Type &value);
LLVM_ATTRIBUTE_UNUSED static mlir::LogicalResult
generatedTypePrinter(mlir::Type def, mlir::AsmPrinter &printer);

#define GET_OP_CLASSES
#include "dsplang/Dialect/HLHVX/IR/HLHVXOps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "dsplang/Dialect/HLHVX/IR/HLHVXTypeDefs.cpp.inc"
