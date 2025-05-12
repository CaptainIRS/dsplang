#include "dsplang/Conversion/ArithToHLHVX/Passes.h"
#include "dsplang/Conversion/DsplangToAffine/Passes.h"
#include "dsplang/Conversion/DsplangToLLVM/Passes.h"
#include "dsplang/Conversion/HLHVXToLLHVX/Passes.h"
#include "dsplang/Dialect/Dsplang/DsplangDialect.h"
#include "dsplang/Dialect/HLHVX/HLHVXDialect.h"
#include "dsplang/Dialect/LLHVX/LLHVXDialect.h"
#include "dsplang/Language/AST.h"
#include "dsplang/Language/Lexer.h"
#include "dsplang/Language/MLIRGen.h"
#include "dsplang/Language/Parser.h"
#include "dsplang/Target/LLVMIR/Dialect/LLHVX/LLHVXToLLVMIRTranslation.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToControlFlow//SCFToControlFlow.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"

#include <cassert>
#include <memory>
#include <string>
#include <system_error>
#include <utility>

namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input dsplang file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("a.out"));

namespace {
enum InputType { DSPLANG, MLIR };
} // namespace
static cl::opt<enum InputType> inputType(
    "x", cl::init(DSPLANG), cl::desc("Decided the kind of output desired"),
    cl::values(clEnumValN(DSPLANG, "dsplang",
                          "load the input file as a dsplang source.")),
    cl::values(clEnumValN(MLIR, "mlir",
                          "load the input file as an MLIR file")));

namespace {
enum Action {
  DumpAST,
  DumpMLIR,
  DumpMLIRAffine,
  DumpMLIRAffineFusion,
  DumpMLIRAffineVectorize,
  DumpMLIRHLHVX,
  DumpMLIRLLHVX,
  DumpMLIRLLVM,
  DumpLLVMIR,
  DumpLLVMIROpt,
  DumpObject
};
} // namespace
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(DumpMLIRAffine, "mlir-affine",
                          "output the MLIR dump after affine lowering")),
    cl::values(clEnumValN(DumpMLIRAffineFusion, "mlir-affine-fusion",
                          "output the MLIR dump after affine loop fusion")),
    cl::values(
        clEnumValN(DumpMLIRAffineVectorize, "mlir-affine-vectorize",
                   "output the MLIR dump after affine super-vectorization")),
    cl::values(clEnumValN(DumpMLIRHLHVX, "mlir-hlhvx",
                          "output the MLIR dump after HLHVX lowering")),
    cl::values(clEnumValN(DumpMLIRLLHVX, "mlir-llhvx",
                          "output the MLIR dump after LLHVX lowering")),
    cl::values(clEnumValN(DumpMLIRLLVM, "mlir-llvm",
                          "output the MLIR dump after llvm lowering")),
    cl::values(clEnumValN(DumpLLVMIR, "llvm", "output the LLVM IR dump")),
    cl::values(clEnumValN(DumpLLVMIROpt, "llvm-opt",
                          "output the LLVM IR dump after optimization")),
    cl::init(DumpObject));

static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));

std::unique_ptr<dsplang::ModuleAST> parseInputFile(llvm::StringRef filename) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }
  auto buffer = fileOrErr.get()->getBuffer();
  dsplang::LexerBuffer lexer(buffer.begin(), buffer.end(),
                             std::string(filename));
  dsplang::Parser parser(lexer);
  return parser.parseModule();
}

int loadMLIR(mlir::MLIRContext &context,
             mlir::OwningOpRef<mlir::ModuleOp> &module) {
  if (inputType != InputType::MLIR &&
      !llvm::StringRef(inputFilename).ends_with(".mlir")) {
    auto moduleAST = parseInputFile(inputFilename);
    if (!moduleAST)
      return 6;
    module = mlirGen(context, *moduleAST);
    return !module ? 1 : 0;
  }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return -1;
  }

  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }
  return 0;
}

int loadAndProcessMLIR(mlir::MLIRContext &context,
                       mlir::OwningOpRef<mlir::ModuleOp> &module) {
  if (int error = loadMLIR(context, module))
    return error;

  mlir::PassManager pm(module.get()->getName());
  // Apply any generic pass manager command line options and run the pipeline.
  if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
    return 4;

  // Check to see what granularity of MLIR we are compiling to.
  bool isLoweringToAffine = emitAction >= Action::DumpMLIRAffine;
  bool isLoweringToAffineFusion = emitAction >= Action::DumpMLIRAffineFusion;
  bool isLoweringToAffineVectorize =
      emitAction >= Action::DumpMLIRAffineVectorize;
  bool isLoweringToHLHVX = emitAction >= Action::DumpMLIRHLHVX;
  bool isLoweringToLLHVX = emitAction >= Action::DumpMLIRLLHVX;
  bool isLoweringToLLVM = emitAction >= Action::DumpMLIRLLVM;

  if (enableOpt || isLoweringToAffine) {
    // Inline all functions into main/dsplanglib and then delete them.
    pm.addPass(mlir::createInlinerPass());

    // Now that there is only one function, we can infer the shapes of each of
    // the operations.
    mlir::OpPassManager &optPM = pm.nest<dsplang::FuncOp>();
    optPM.addPass(dsplang::createShapeInferencePass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
  }

  if (isLoweringToAffine) {
    pm.addPass(dsplang::createLowerToAffinePass());
  }
  if (isLoweringToAffineFusion) {
    mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
    optPM.addPass(mlir::affine::createLoopFusionPass());
    optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
  }
  if (isLoweringToAffineVectorize) {
    mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
    optPM.addPass(dsplang::createAffineSelectiveVectorizePass());
  }
  if (isLoweringToHLHVX) {
    pm.addPass(dsplang::createArithToHLHVXPass());
  }
  if (isLoweringToLLHVX) {
    pm.addPass(dsplang::createHLHVXToLLHVXPass());
  }

  if (isLoweringToLLVM) {
    mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());

    // Add optimizations if enabled.
    if (enableOpt) {
      optPM.addPass(mlir::affine::createLoopFusionPass());
      optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
    }

    pm.addPass(dsplang::createLowerToLLVMPass());
    pm.addPass(createReconcileUnrealizedCastsPass());
    pm.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());
  }

  if (mlir::failed(pm.run(*module)))
    return 4;
  return 0;
}

int dumpAST() {
  if (inputType == InputType::MLIR) {
    llvm::errs() << "Can't dump a Dsplang AST when the input is MLIR\n";
    return 5;
  }

  auto moduleAST = parseInputFile(inputFilename);
  if (!moduleAST)
    return 1;

  dump(*moduleAST);
  return 0;
}

int processMLIRLLVM(mlir::ModuleOp module) {
  // Register the translation to LLVM IR with the MLIR context.
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());
  dsplang::registerLLHVXDialectTranslation(*module->getContext());
  // Convert the module to LLVM IR in a new LLVM IR context.
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  if (emitAction == Action::DumpLLVMIR) {
    llvmModule->print(llvm::errs(), nullptr);
    return 0;
  }

  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  auto targetTriple = llvm::Triple("hexagon-unknown-none-elf");
  auto cpu = "hexagonv73";
  auto features = "+hvxv73,+hvx-length128b";

  std::string error;
  auto target = llvm::TargetRegistry::lookupTarget(targetTriple, error);

  if (!target) {
    llvm::errs() << error;
    return 1;
  }

  llvm::TargetOptions opt;
  auto targetMachine = target->createTargetMachine(targetTriple, cpu, features,
                                                   opt, llvm::Reloc::PIC_);

  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;
  llvm::PassBuilder passBuilder(targetMachine);
  passBuilder.registerModuleAnalyses(mam);
  passBuilder.registerCGSCCAnalyses(cgam);
  passBuilder.registerFunctionAnalyses(fam);
  passBuilder.registerLoopAnalyses(lam);
  passBuilder.crossRegisterProxies(lam, fam, cgam, mam);
  auto passManager =
      passBuilder.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
  passManager.run(*llvmModule, mam);

  if (emitAction == Action::DumpLLVMIROpt) {
    llvmModule->print(llvm::errs(), nullptr);
    return 0;
  }

  llvmModule->setDataLayout(targetMachine->createDataLayout());
  llvmModule->setTargetTriple(targetTriple);

  std::error_code EC;
  llvm::raw_fd_ostream dest(outputFilename, EC, llvm::sys::fs::OF_None);

  if (EC) {
    llvm::errs() << "Could not open file: " << EC.message();
    return 1;
  }

  llvm::legacy::PassManager pass;
  auto FileType = llvm::CodeGenFileType::ObjectFile;

  if (targetMachine->addPassesToEmitFile(pass, dest, nullptr, FileType)) {
    llvm::errs() << "TargetMachine can't emit a file of this type";
    return 1;
  }

  pass.run(*llvmModule);
  dest.flush();
  return 0;
}

int main(int argc, char **argv) {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "dsplang compiler\n");

  if (emitAction == Action::DumpAST)
    return dumpAST();

  // If we aren't dumping the AST, then we are compiling with/to MLIR.
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::func::registerAllExtensions(registry);
  mlir::LLVM::registerInlinerInterface(registry);

  mlir::registerAllPasses();
  mlir::registerConversionPasses();

  mlir::MLIRContext context(registry);
  context.getOrLoadDialect<dsplang::DsplangDialect>();
  context.getOrLoadDialect<dsplang::LLHVXDialect>();
  context.getOrLoadDialect<dsplang::HLHVXDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (int error = loadAndProcessMLIR(context, module))
    return error;

  // If we aren't exporting to non-mlir, then we are done.
  bool isOutputingMLIR = emitAction <= Action::DumpMLIRLLVM;
  if (isOutputingMLIR) {
    module->dump();
    return 0;
  }

  return processMLIRLLVM(*module);
}
