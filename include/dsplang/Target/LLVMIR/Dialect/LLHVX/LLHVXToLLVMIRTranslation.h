#ifndef DIALECT_LLHVX_TRANSLATION_H
#define DIALECT_LLHVX_TRANSLATION_H

namespace mlir {
class DialectRegistry;
class MLIRContext;
} // namespace mlir

namespace dsplang {

void registerLLHVXDialectTranslation(mlir::DialectRegistry &registry);
void registerLLHVXDialectTranslation(mlir::MLIRContext &context);

} // namespace dsplang

#endif // DIALECT_LLHVX_TRANSLATION_H