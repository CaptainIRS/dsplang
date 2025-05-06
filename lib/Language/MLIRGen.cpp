// Adapted from LLVM MLIR Toy example

#include "dsplang/Language/MLIRGen.h"
#include "dsplang/Dialect/Dsplang/DsplangDialect.h"
#include "dsplang/Language/AST.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "dsplang/Language/Lexer.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include <cassert>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <vector>

using namespace dsplang;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace {

/// Implementation of a simple MLIR emission from the Dsplang AST.
///
/// This will emit operations that are specific to the Dsplang language,
/// preserving the semantics of the language and (hopefully) allow to perform
/// accurate analysis and transformation based on these high level semantics.
class MLIRGenImpl {
  VarType::Type currentType;

public:
  MLIRGenImpl(mlir::MLIRContext &context)
      : currentType(VarType::Type::f32), builder(&context) {}

  /// Public API: convert the AST for a Dsplang module (source file) to an MLIR
  /// Module operation.
  mlir::ModuleOp mlirGen(ModuleAST &moduleAST) {
    // We create an empty MLIR module and codegen functions one at a time and
    // add them to the module.
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

    for (FunctionAST &f : moduleAST)
      mlirGen(f);

    // Verify the module after we have finished constructing it, this will check
    // the structural properties of the IR and invoke any specific verifiers we
    // have on the Dsplang operations.
    if (failed(mlir::verify(theModule))) {
      theModule.emitError("module verification error");
      return nullptr;
    }

    return theModule;
  }

private:
  /// A "module" matches a Dsplang source file: containing a list of functions.
  mlir::ModuleOp theModule;

  /// The builder is a helper class to create IR inside a function. The builder
  /// is stateful, in particular it keeps an "insertion point": this is where
  /// the next operations will be introduced.
  mlir::OpBuilder builder;

  /// The symbol table maps a variable name to a value in the current scope.
  /// Entering a function creates a new scope, and the function arguments are
  /// added to the mapping. When the processing of a function is terminated, the
  /// scope is destroyed and the mappings created in this scope are dropped.
  llvm::ScopedHashTable<StringRef, mlir::Value> symbolTable;

  /// Helper conversion for a Dsplang AST location to an MLIR location.
  mlir::Location loc(const dsplang::Location &loc) {
    return mlir::FileLineColLoc::get(builder.getStringAttr(*loc.file), loc.line,
                                     loc.col);
  }

  /// Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
  llvm::LogicalResult declare(llvm::StringRef var, mlir::Value value) {
    if (symbolTable.count(var))
      return mlir::failure();
    symbolTable.insert(var, value);
    return mlir::success();
  }

  /// Create the prototype for an MLIR function with as many arguments as the
  /// provided Dsplang AST prototype.
  dsplang::FuncOp mlirGen(PrototypeAST &proto) {
    auto location = loc(proto.loc());

    // This is a generic function, the return type will be inferred later.
    // Arguments type are uniformly unranked tensors.
    llvm::SmallVector<mlir::Type, 4> argTypes;
    for (auto &arg : proto.getArgs()) {
      auto type = mlir::cast<MemRefType>(getType(arg->getType(), true));
      if (type.getElementType().isInteger()) {
        if (!type.getElementType().isSignlessInteger()) {
          type = mlir::MemRefType::get(
              type.getShape(),
              mlir::IntegerType::get(
                  type.getContext(),
                  type.getElementType().getIntOrFloatBitWidth()));
        }
      }
      if (!type)
        return nullptr;
      argTypes.push_back(type);
    }
    auto funcType = builder.getFunctionType(argTypes, std::nullopt);
    return builder.create<dsplang::FuncOp>(location, proto.getName(), funcType);
  }

  /// Emit a new function and add it to the MLIR module.
  dsplang::FuncOp mlirGen(FunctionAST &funcAST) {
    // Create a scope in the symbol table to hold variable declarations.
    ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);

    // Create an MLIR function for the given prototype.
    builder.setInsertionPointToEnd(theModule.getBody());
    dsplang::FuncOp function = mlirGen(*funcAST.getProto());
    if (!function)
      return nullptr;

    // Let's start the body of the function now!
    mlir::Block &entryBlock = function.front();
    auto protoArgs = funcAST.getProto()->getArgs();

    // Declare all the function arguments in the symbol table.
    // for (const auto nameValue :
    //      llvm::zip(protoArgs, entryBlock.getArguments())) {
    //   if (failed(declare(std::get<0>(nameValue)->getName(),
    //                      std::get<1>(nameValue))))
    //     return nullptr;
    // }

    // Set the insertion point in the builder to the beginning of the function
    // body, it will be used throughout the codegen to create operations in this
    // function.
    builder.setInsertionPointToStart(&entryBlock);

    for (size_t i = 0; i < protoArgs.size(); ++i) {
      auto &arg = protoArgs[i];
      if (!arg->isPointer()) {
        auto type = getType(arg->getType());
        auto bufferizedArgument = builder.create<dsplang::BufferizeOp>(
            function.getLoc(), type, entryBlock.getArgument(i));
        auto reshaped = builder.create<dsplang::ReshapeOp>(
            function.getLoc(), type, bufferizedArgument);
        if (failed(declare(arg->getName(), reshaped)))
          return nullptr;
      } else {
        // auto reshaped = builder.create<dsplang::BufferizeOp>(
        //     function.getLoc(), getType(arg->getType()),
        //     entryBlock.getArgument(i));
        if (failed(declare(arg->getName(), entryBlock.getArgument(i))))
          return nullptr;
      }
    }

    // Emit the body of the function.
    if (mlir::failed(mlirGen(*funcAST.getBody()))) {
      function.erase();
      return nullptr;
    }

    // Implicitly return void if no return statement was emitted.
    // FIXME: we may fix the parser instead to always return the last expression
    // (this would possibly help the REPL case later)
    ReturnOp returnOp;
    if (!entryBlock.empty())
      returnOp = dyn_cast<ReturnOp>(entryBlock.back());
    if (!returnOp) {
      builder.create<ReturnOp>(loc(funcAST.getProto()->loc()));
    } else if (returnOp.hasOperand()) {
      // Otherwise, if this return operation has an operand then add a result to
      // the function.
      function.setType(builder.getFunctionType(
          function.getFunctionType().getInputs(), getType(VarType{})));
    }

    // If this function isn't main, then set the visibility to private.
    if (funcAST.getProto()->getName() != "main" &&
        funcAST.getProto()->getName() != "dsplanglib")
      function.setPrivate();

    return function;
  }

  /// Emit a binary operation
  mlir::Value mlirGen(BinaryExprAST &binop) {
    // First emit the operations for each side of the operation before emitting
    // the operation itself. For example if the expression is `a + foo(a)`
    // 1) First it will visiting the LHS, which will return a reference to the
    //    value holding `a`. This value should have been emitted at declaration
    //    time and registered in the symbol table, so nothing would be
    //    codegen'd. If the value is not in the symbol table, an error has been
    //    emitted and nullptr is returned.
    // 2) Then the RHS is visited (recursively) and a call to `foo` is emitted
    //    and the result value is returned. If an error occurs we get a nullptr
    //    and propagate.
    //
    mlir::Value lhs = mlirGen(*binop.getLHS());
    if (!lhs)
      return nullptr;
    mlir::Value rhs = mlirGen(*binop.getRHS());
    if (!rhs)
      return nullptr;
    auto location = loc(binop.loc());

    // Derive the operation name from the binary operator. At the moment we only
    // support '+' and '*'.
    switch (binop.getOp()) {
    case '+':
      return builder.create<AddOp>(location, lhs, rhs);
    case '-':
      return builder.create<SubOp>(location, lhs, rhs);
    case '*':
      return builder.create<MulOp>(location, lhs, rhs);
    case '&':
      return builder.create<AndOp>(location, lhs, rhs);
    case '|':
      return builder.create<OrOp>(location, lhs, rhs);
    case '^':
      return builder.create<XorOp>(location, lhs, rhs);
    case '=':
      builder.create<AssignOp>(location, lhs, rhs);
      return lhs;
    }

    emitError(location, "invalid binary operator '") << binop.getOp() << "'";
    return nullptr;
  }

  /// This is a reference to a variable in an expression. The variable is
  /// expected to have been declared and so should have a value in the symbol
  /// table, otherwise emit an error and return nullptr.
  mlir::Value mlirGen(VariableExprAST &expr) {
    if (auto variable = symbolTable.lookup(expr.getName()))
      return variable;

    emitError(loc(expr.loc()), "error: unknown variable '")
        << expr.getName() << "'";
    return nullptr;
  }

  /// Emit a return operation. This will return failure if any generation fails.
  llvm::LogicalResult mlirGen(ReturnExprAST &ret) {
    auto location = loc(ret.loc());

    // 'return' takes an optional expression, handle that case here.
    mlir::Value expr = nullptr;
    if (ret.getExpr().has_value()) {
      if (!(expr = mlirGen(**ret.getExpr())))
        return mlir::failure();
    }

    // Otherwise, this return operation has zero operands.
    builder.create<ReturnOp>(location,
                             expr ? ArrayRef(expr) : ArrayRef<mlir::Value>());
    return mlir::success();
  }

  /// Emit a literal/constant array. It will be emitted as a flattened array of
  /// data in an Attribute attached to a `dsplang.constant` operation.
  /// See documentation on [Attributes](LangRef.md#attributes) for more details.
  /// Here is an excerpt:
  ///
  ///   Attributes are the mechanism for specifying constant data in MLIR in
  ///   places where a variable is never allowed [...]. They consist of a name
  ///   and a concrete attribute value. The set of expected attributes, their
  ///   structure, and their interpretation are all contextually dependent on
  ///   what they are attached to.
  ///
  /// Example, the source level statement:
  ///   var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  /// will be converted to:
  ///   %0 = "dsplang.constant"() {value: dense<tensor<2x3xf64>,
  ///     [[1.000000e+00, 2.000000e+00, 3.000000e+00],
  ///      [4.000000e+00, 5.000000e+00, 6.000000e+00]]>} : () -> tensor<2x3xf64>
  ///
  mlir::Value mlirGen(LiteralExprAST &lit) {
    auto type = getType(lit.getDims(), currentType);

    // The type of this attribute is tensor of 32-bit floating-point with the
    // shape of the literal.
    // mlir::Type elementType = builder.getF32Type();
    // auto dataType = mlir::RankedTensorType::get(lit.getDims(), elementType);

    // Build the MLIR op `dsplang.constant`. This invokes the
    // `ConstantOp::build` method.
    if (currentType == VarType::Type::si8) {
      std::vector<char> data;
      data.reserve(std::accumulate(lit.getDims().begin(), lit.getDims().end(),
                                   1, std::multiplies<int>()));
      collectData<char>(lit, data);
      auto dataType = mlir::RankedTensorType::get(
          lit.getDims(), builder.getIntegerType(8, true));
      return builder.create<ConstantSI8Op>(
          loc(lit.loc()), type,
          DenseIntElementsAttr::get(dataType, llvm::ArrayRef(data)));
    }
    if (currentType == VarType::Type::ui8) {
      std::vector<unsigned char> data;
      data.reserve(std::accumulate(lit.getDims().begin(), lit.getDims().end(),
                                   1, std::multiplies<int>()));
      collectData<unsigned char>(lit, data);
      auto dataType = mlir::RankedTensorType::get(
          lit.getDims(), builder.getIntegerType(8, false));
      return builder.create<ConstantUI8Op>(
          loc(lit.loc()), type,
          DenseIntElementsAttr::get(dataType, llvm::ArrayRef(data)));
    }
    if (currentType == VarType::Type::si16) {
      std::vector<short> data;
      data.reserve(std::accumulate(lit.getDims().begin(), lit.getDims().end(),
                                   1, std::multiplies<int>()));
      collectData<short>(lit, data);
      auto dataType = mlir::RankedTensorType::get(
          lit.getDims(), builder.getIntegerType(16, true));
      return builder.create<ConstantSI16Op>(
          loc(lit.loc()), type,
          DenseIntElementsAttr::get(dataType, llvm::ArrayRef(data)));
    }
    if (currentType == VarType::Type::ui16) {
      std::vector<unsigned short> data;
      data.reserve(std::accumulate(lit.getDims().begin(), lit.getDims().end(),
                                   1, std::multiplies<int>()));
      collectData<unsigned short>(lit, data);
      auto dataType = mlir::RankedTensorType::get(
          lit.getDims(), builder.getIntegerType(16, false));
      return builder.create<ConstantUI16Op>(
          loc(lit.loc()), type,
          DenseIntElementsAttr::get(dataType, llvm::ArrayRef(data)));
    }
    if (currentType == VarType::Type::si32) {
      std::vector<int> data;
      data.reserve(std::accumulate(lit.getDims().begin(), lit.getDims().end(),
                                   1, std::multiplies<int>()));
      collectData<int>(lit, data);
      auto dataType = mlir::RankedTensorType::get(
          lit.getDims(), builder.getIntegerType(32, true));
      return builder.create<ConstantSI32Op>(
          loc(lit.loc()), type,
          DenseIntElementsAttr::get(dataType, llvm::ArrayRef(data)));
    }
    if (currentType == VarType::Type::ui32) {
      std::vector<unsigned int> data;
      data.reserve(std::accumulate(lit.getDims().begin(), lit.getDims().end(),
                                   1, std::multiplies<int>()));
      collectData<unsigned int>(lit, data);
      auto dataType = mlir::RankedTensorType::get(
          lit.getDims(), builder.getIntegerType(32, false));
      return builder.create<ConstantUI32Op>(
          loc(lit.loc()), type,
          DenseIntElementsAttr::get(dataType, llvm::ArrayRef(data)));
    }
    if (currentType == VarType::Type::f16) {
      std::vector<float> data;
      data.reserve(std::accumulate(lit.getDims().begin(), lit.getDims().end(),
                                   1, std::multiplies<int>()));
      collectData<float>(lit, data);
      auto dataType =
          mlir::RankedTensorType::get(lit.getDims(), builder.getF16Type());
      return builder.create<ConstantF16Op>(
          loc(lit.loc()), type,
          DenseFPElementsAttr::get(dataType, llvm::ArrayRef(data)));
    }
    if (currentType == VarType::Type::f32) {
      std::vector<float> data;
      data.reserve(std::accumulate(lit.getDims().begin(), lit.getDims().end(),
                                   1, std::multiplies<int>()));
      collectData<float>(lit, data);
      auto dataType =
          mlir::RankedTensorType::get(lit.getDims(), builder.getF32Type());
      return builder.create<ConstantF32Op>(
          loc(lit.loc()), type,
          DenseFPElementsAttr::get(dataType, llvm::ArrayRef(data)));
    }
    emitError(loc(lit.loc()), "invalid type for literal");
    return nullptr;
  }

  /// Recursive helper function to accumulate the data that compose an array
  /// literal. It flattens the nested structure in the supplied vector. For
  /// example with this array:
  ///  [[1, 2], [3, 4]]
  /// we will generate:
  ///  [ 1, 2, 3, 4 ]
  /// Individual numbers are represented as doubles.
  /// Attributes are the way MLIR attaches constant to operations.
  template <typename T> void collectData(ExprAST &expr, std::vector<T> &data) {
    if (auto *lit = dyn_cast<LiteralExprAST>(&expr)) {
      for (auto &value : lit->getValues())
        collectData(*value, data);
      return;
    }

    assert(isa<NumberExprAST>(expr) && "expected literal or number expr");
    data.push_back(cast<NumberExprAST>(expr).getValue());
  }

  /// Emit a call expression.
  mlir::Value mlirGen(CallExprAST &call) {
    llvm::StringRef callee = call.getCallee();
    auto location = loc(call.loc());

    // Codegen the operands first.
    SmallVector<mlir::Value, 4> operands;
    for (auto &expr : call.getArgs()) {
      auto arg = mlirGen(*expr);
      if (!arg)
        return nullptr;
      operands.push_back(arg);
    }

    if (callee == "zero") {

    } else if (callee == "min") {

    } else if (callee == "max") {

    } else if (callee == "abs") {
    }

    // Otherwise this is a call to a user-defined function. Calls to
    // user-defined functions are mapped to a custom call that takes the callee
    // name as an attribute.
    return builder.create<GenericCallOp>(location, callee, operands);
  }

  /// Emit a print expression. It emits specific operations print(x).
  llvm::LogicalResult mlirGen(PrintExprAST &call) {
    auto arg = mlirGen(*call.getArg());
    if (!arg)
      return mlir::failure();

    builder.create<PrintOp>(loc(call.loc()), arg);
    return mlir::success();
  }

  /// Emit a constant for a single number (FIXME: semantic? broadcast?)
  mlir::Value mlirGen(NumberExprAST &num) {
    return builder.create<ConstantF32Op>(loc(num.loc()), num.getValue());
  }

  /// Dispatch codegen for the right expression subclass using RTTI.
  mlir::Value mlirGen(ExprAST &expr) {
    switch (expr.getKind()) {
    case dsplang::ExprAST::Expr_BinOp:
      return mlirGen(cast<BinaryExprAST>(expr));
    case dsplang::ExprAST::Expr_Var:
      return mlirGen(cast<VariableExprAST>(expr));
    case dsplang::ExprAST::Expr_Literal:
      return mlirGen(cast<LiteralExprAST>(expr));
    case dsplang::ExprAST::Expr_Call:
      return mlirGen(cast<CallExprAST>(expr));
    case dsplang::ExprAST::Expr_Num:
      return mlirGen(cast<NumberExprAST>(expr));
    default:
      emitError(loc(expr.loc()))
          << "MLIR codegen encountered an unhandled expr kind '"
          << Twine(expr.getKind()) << "'";
      return nullptr;
    }
  }

  /// Handle a variable declaration, we'll codegen the expression that forms the
  /// initializer and record the value in the symbol table before returning it.
  /// Future expressions will be able to reference this variable through symbol
  /// table lookup.
  mlir::Value mlirGen(VarDeclExprAST &vardecl) {
    currentType = vardecl.getType().type;
    auto *init = vardecl.getInitVal();
    if (!init) {
      emitError(loc(vardecl.loc()),
                "missing initializer in variable declaration");
      return nullptr;
    }

    mlir::Value value = mlirGen(*init);
    if (!value)
      return nullptr;

    value = builder.create<ReshapeOp>(loc(vardecl.loc()),
                                      getType(vardecl.getType()), value);

    // Register the value in the symbol table.
    if (failed(declare(vardecl.getName(), value)))
      return nullptr;
    return value;
  }

  /// Codegen a list of expression, return failure if one of them hit an error.
  llvm::LogicalResult mlirGen(ExprASTList &blockAST) {
    ScopedHashTableScope<StringRef, mlir::Value> varScope(symbolTable);
    for (auto &expr : blockAST) {
      // Specific handling for variable declarations, return statement, and
      // print. These can only appear in block list and not in nested
      // expressions.
      if (auto *vardecl = dyn_cast<VarDeclExprAST>(expr.get())) {
        if (!mlirGen(*vardecl))
          return mlir::failure();
        continue;
      }
      if (auto *ret = dyn_cast<ReturnExprAST>(expr.get()))
        return mlirGen(*ret);
      if (auto *print = dyn_cast<PrintExprAST>(expr.get())) {
        if (mlir::failed(mlirGen(*print)))
          return mlir::success();
        continue;
      }

      // Generic expression dispatch codegen.
      if (!mlirGen(*expr))
        return mlir::failure();
    }
    return mlir::success();
  }

  /// Build a tensor type from a list of shape dimensions.
  mlir::Type getType(ArrayRef<int64_t> shape, VarType::Type type,
                     bool isMemref = false) {
    Type mlirType = builder.getF32Type();
    if (type == VarType::Type::si8)
      mlirType = builder.getIntegerType(8, true);
    else if (type == VarType::Type::ui8)
      mlirType = builder.getIntegerType(8, false);
    else if (type == VarType::Type::si16)
      mlirType = builder.getIntegerType(16, true);
    else if (type == VarType::Type::ui16)
      mlirType = builder.getIntegerType(16, false);
    else if (type == VarType::Type::si32)
      mlirType = builder.getIntegerType(32, true);
    else if (type == VarType::Type::ui32)
      mlirType = builder.getIntegerType(32, false);
    else if (type == VarType::Type::f16)
      mlirType = builder.getF16Type();
    else if (type == VarType::Type::f32)
      mlirType = builder.getF32Type();
    if (!isMemref) {
      if (shape.empty())
        return mlir::UnrankedTensorType::get(mlirType);

      // Otherwise, we use the given shape.
      return mlir::RankedTensorType::get(shape, mlirType);
    } else {
      if (shape.empty())
        return mlir::UnrankedMemRefType::get(mlirType, 0);
      return mlir::MemRefType::get(shape, mlirType);
    }
  }

  /// Build an MLIR type from a Dsplang AST variable type (forward to the
  /// generic getType above).
  mlir::Type getType(const VarType &type, bool isMemRef = false) {
    return getType(type.shape, type.type, isMemRef);
  }
};

} // namespace

namespace dsplang {

// The public API for codegen.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                          ModuleAST &moduleAST) {
  return MLIRGenImpl(context).mlirGen(moduleAST);
}

} // namespace dsplang
