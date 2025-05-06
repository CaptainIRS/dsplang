// RUN: llhvx-translate %s -mlir-to-llvmir -split-input-file | FileCheck %s

// CHECK-LABEL: define <32 x i32> @vd0_128B
func.func @vd0_128B() -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vd0.128B()
    %0 = "llhvx.intr.vd0_128B"() : () -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define void @vgathermhw_128B
func.func @vgathermhw_128B(%A : !llvm.ptr, %B : i32, %C : i32, %D : vector<64 x i32>) {
    // CHECK: call void @llvm.hexagon.V6.vgathermhw.128B(
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}})
    "llhvx.intr.vgathermhw_128B"(%A, %B, %C, %D) : (!llvm.ptr, i32, i32, vector<64 x i32>) -> ()
    return
}

// CHECK-LABEL: define <32 x i32> @valignbi_128B
func.func @valignbi_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.valignbi.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 {{[0-9]+}})
    %C = arith.constant 1 : i32
    %0 = "llhvx.intr.valignbi_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @pred_and_128B
func.func @pred_and_128B(%A : vector<128 x i1>, %B : vector<128 x i1>, %C : vector<32xi32>, %D : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.pred.and.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <128 x i1> %{{[0-9]+}})
    %0 = "llhvx.intr.pred_and_128B"(%A, %B) : (vector<128 x i1>, vector<128 x i1>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %C, %D) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vaddcarry_128B
func.func @vaddcarry_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : vector<128 x i1>) -> vector<32 x i32> {
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <128 x i1> %{{[0-9]+}})
    %0 = "llhvx.intr.vaddcarry_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, vector<128 x i1>) -> !llvm.struct<(vector<32 x i32>, vector<128 x i1>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(vector<32 x i32>, vector<128 x i1>)>
    return %1 : vector<32 x i32>
}
