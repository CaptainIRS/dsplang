// RUN: llhvx-translate %s -mlir-to-llvmir -split-input-file | FileCheck %s

// CHECK-LABEL: define <64 x i32> @vrmpybub_rtt_128B
func.func @vrmpybub_rtt_128B(%A : vector<32 x i32>, %B : i64) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vrmpybub.rtt.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i64 %{{[0-9]+}})
    %0 = "llhvx.intr.vrmpybub_rtt_128B"(%A, %B) : (vector<32 x i32>, i64) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vrmpybub_rtt_acc_128B
func.func @vrmpybub_rtt_acc_128B(%A : vector<64 x i32>, %B : vector<32 x i32>, %C : i64) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vrmpybub.rtt.acc.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i64 %{{[0-9]+}})
    %0 = "llhvx.intr.vrmpybub_rtt_acc_128B"(%A, %B, %C) : (vector<64 x i32>, vector<32 x i32>, i64) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vrmpyub_rtt_128B
func.func @vrmpyub_rtt_128B(%A : vector<32 x i32>, %B : i64) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vrmpyub.rtt.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i64 %{{[0-9]+}})
    %0 = "llhvx.intr.vrmpyub_rtt_128B"(%A, %B) : (vector<32 x i32>, i64) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vrmpyub_rtt_acc_128B
func.func @vrmpyub_rtt_acc_128B(%A : vector<64 x i32>, %B : vector<32 x i32>, %C : i64) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vrmpyub.rtt.acc.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i64 %{{[0-9]+}})
    %0 = "llhvx.intr.vrmpyub_rtt_acc_128B"(%A, %B, %C) : (vector<64 x i32>, vector<32 x i32>, i64) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vL32b_pred_ai_128B
func.func @vL32b_pred_ai_128B(%B : !llvm.ptr) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vL32b.pred.ai.128B(
    // CHECK-SAME: i1 true,
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: i32 {{[0-9]+}})
    %A = arith.constant 1 : i1
    %C = arith.constant 1 : i32
    %0 = "llhvx.intr.vL32b_pred_ai_128B"(%A, %B, %C) : (i1, !llvm.ptr, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vL32b_npred_ai_128B
func.func @vL32b_npred_ai_128B(%B : !llvm.ptr) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vL32b.npred.ai.128B(
    // CHECK-SAME: i1 true,
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: i32 {{[0-9]+}})
    %A = arith.constant 1 : i1
    %C = arith.constant 1 : i32
    %0 = "llhvx.intr.vL32b_npred_ai_128B"(%A, %B, %C) : (i1, !llvm.ptr, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vL32b_nt_pred_ai_128B
func.func @vL32b_nt_pred_ai_128B(%B : !llvm.ptr) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vL32b.nt.pred.ai.128B(
    // CHECK-SAME: i1 true,
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: i32 {{[0-9]+}})
    %A = arith.constant 1 : i1
    %C = arith.constant 1 : i32
    %0 = "llhvx.intr.vL32b_nt_pred_ai_128B"(%A, %B, %C) : (i1, !llvm.ptr, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vL32b_nt_npred_ai_128B
func.func @vL32b_nt_npred_ai_128B(%B : !llvm.ptr) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vL32b.nt.npred.ai.128B(
    // CHECK-SAME: i1 true,
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: i32 {{[0-9]+}})
    %A = arith.constant 1 : i1
    %C = arith.constant 1 : i32
    %0 = "llhvx.intr.vL32b_nt_npred_ai_128B"(%A, %B, %C) : (i1, !llvm.ptr, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vL32b_pred_pi_128B
func.func @vL32b_pred_pi_128B(%B : !llvm.ptr) -> vector<32 x i32> {
    // CHECK: call { <32 x i32>, ptr } @llvm.hexagon.V6.vL32b.pred.pi.128B(
    // CHECK-SAME: i1 true,
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: i32 {{[0-9]+}})
    %A = arith.constant 1 : i1
    %C = arith.constant 1 : i32
    %0 = "llhvx.intr.vL32b_pred_pi_128B"(%A, %B, %C) : (i1, !llvm.ptr, i32) -> !llvm.struct<(vector<32 x i32>, !llvm.ptr)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(vector<32 x i32>, !llvm.ptr)>
    func.return %1 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vL32b_npred_pi_128B
func.func @vL32b_npred_pi_128B(%B : !llvm.ptr) -> vector<32 x i32> {
    // CHECK: call { <32 x i32>, ptr } @llvm.hexagon.V6.vL32b.npred.pi.128B(
    // CHECK-SAME: i1 true,
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: i32 {{[0-9]+}})
    %A = arith.constant 1 : i1
    %C = arith.constant 1 : i32
    %0 = "llhvx.intr.vL32b_npred_pi_128B"(%A, %B, %C) : (i1, !llvm.ptr, i32) -> !llvm.struct<(vector<32 x i32>, !llvm.ptr)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(vector<32 x i32>, !llvm.ptr)>
    func.return %1 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vL32b_nt_pred_pi_128B
func.func @vL32b_nt_pred_pi_128B(%B : !llvm.ptr) -> vector<32 x i32> {
    // CHECK: call { <32 x i32>, ptr } @llvm.hexagon.V6.vL32b.nt.pred.pi.128B(
    // CHECK-SAME: i1 true,
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: i32 {{[0-9]+}})
    %A = arith.constant 1 : i1
    %C = arith.constant 1 : i32
    %0 = "llhvx.intr.vL32b_nt_pred_pi_128B"(%A, %B, %C) : (i1, !llvm.ptr, i32) -> !llvm.struct<(vector<32 x i32>, !llvm.ptr)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(vector<32 x i32>, !llvm.ptr)>
    func.return %1 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vL32b_nt_npred_pi_128B
func.func @vL32b_nt_npred_pi_128B(%B : !llvm.ptr) -> vector<32 x i32> {
    // CHECK: call { <32 x i32>, ptr } @llvm.hexagon.V6.vL32b.nt.npred.pi.128B(
    // CHECK-SAME: i1 true,
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: i32 {{[0-9]+}})
    %A = arith.constant 1 : i1
    %C = arith.constant 1 : i32
    %0 = "llhvx.intr.vL32b_nt_npred_pi_128B"(%A, %B, %C) : (i1, !llvm.ptr, i32) -> !llvm.struct<(vector<32 x i32>, !llvm.ptr)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(vector<32 x i32>, !llvm.ptr)>
    func.return %1 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vL32b_pred_ppu_128B
func.func @vL32b_pred_ppu_128B(%B : !llvm.ptr, %C : i32) -> vector<32 x i32> {
    // CHECK: call { <32 x i32>, ptr } @llvm.hexagon.V6.vL32b.pred.ppu.128B(
    // CHECK-SAME: i1 true,
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %A = arith.constant 1 : i1
    %0 = "llhvx.intr.vL32b_pred_ppu_128B"(%A, %B, %C) : (i1, !llvm.ptr, i32) -> !llvm.struct<(vector<32 x i32>, !llvm.ptr)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(vector<32 x i32>, !llvm.ptr)>
    func.return %1 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vL32b_npred_ppu_128B
func.func @vL32b_npred_ppu_128B(%B : !llvm.ptr, %C : i32) -> vector<32 x i32> {
    // CHECK: call { <32 x i32>, ptr } @llvm.hexagon.V6.vL32b.npred.ppu.128B(
    // CHECK-SAME: i1 true,
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %A = arith.constant 1 : i1
    %0 = "llhvx.intr.vL32b_npred_ppu_128B"(%A, %B, %C) : (i1, !llvm.ptr, i32) -> !llvm.struct<(vector<32 x i32>, !llvm.ptr)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(vector<32 x i32>, !llvm.ptr)>
    func.return %1 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vL32b_nt_pred_ppu_128B
func.func @vL32b_nt_pred_ppu_128B(%B : !llvm.ptr, %C : i32) -> vector<32 x i32> {
    // CHECK: call { <32 x i32>, ptr } @llvm.hexagon.V6.vL32b.nt.pred.ppu.128B(
    // CHECK-SAME: i1 true,
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %A = arith.constant 1 : i1
    %0 = "llhvx.intr.vL32b_nt_pred_ppu_128B"(%A, %B, %C) : (i1, !llvm.ptr, i32) -> !llvm.struct<(vector<32 x i32>, !llvm.ptr)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(vector<32 x i32>, !llvm.ptr)>
    func.return %1 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vL32b_nt_npred_ppu_128B
func.func @vL32b_nt_npred_ppu_128B(%B : !llvm.ptr, %C : i32) -> vector<32 x i32> {
    // CHECK: call { <32 x i32>, ptr } @llvm.hexagon.V6.vL32b.nt.npred.ppu.128B(
    // CHECK-SAME: i1 true,
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %A = arith.constant 1 : i1
    %0 = "llhvx.intr.vL32b_nt_npred_ppu_128B"(%A, %B, %C) : (i1, !llvm.ptr, i32) -> !llvm.struct<(vector<32 x i32>, !llvm.ptr)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(vector<32 x i32>, !llvm.ptr)>
    func.return %1 : vector<32 x i32>
}

// CHECK-LABEL: define void @vS32b_pred_ai_128B
func.func @vS32b_pred_ai_128B(%B : !llvm.ptr, %D : vector<32 x i32>) {
    // CHECK: call void @llvm.hexagon.V6.vS32b.pred.ai.128B(
    // CHECK-SAME: i1 true,
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: i32 {{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %A = arith.constant 1 : i1
    %C = arith.constant 1 : i32
    "llhvx.intr.vS32b_pred_ai_128B"(%A, %B, %C, %D) : (i1, !llvm.ptr, i32, vector<32 x i32>) -> ()
    return
}

// CHECK-LABEL: define void @vS32b_npred_ai_128B
func.func @vS32b_npred_ai_128B(%B : !llvm.ptr, %D : vector<32 x i32>) {
    // CHECK: call void @llvm.hexagon.V6.vS32b.npred.ai.128B(
    // CHECK-SAME: i1 true,
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: i32 {{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %A = arith.constant 1 : i1
    %C = arith.constant 1 : i32
    "llhvx.intr.vS32b_npred_ai_128B"(%A, %B, %C, %D) : (i1, !llvm.ptr, i32, vector<32 x i32>) -> ()
    return
}

// CHECK-LABEL: define void @vS32Ub_pred_ai_128B
func.func @vS32Ub_pred_ai_128B(%B : !llvm.ptr, %D : vector<32 x i32>) {
    // CHECK: call void @llvm.hexagon.V6.vS32Ub.pred.ai.128B(
    // CHECK-SAME: i1 true,
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: i32 {{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %A = arith.constant 1 : i1
    %C = arith.constant 1 : i32
    "llhvx.intr.vS32Ub_pred_ai_128B"(%A, %B, %C, %D) : (i1, !llvm.ptr, i32, vector<32 x i32>) -> ()
    return
}

// CHECK-LABEL: define void @vS32Ub_npred_ai_128B
func.func @vS32Ub_npred_ai_128B(%B : !llvm.ptr, %D : vector<32 x i32>) {
    // CHECK: call void @llvm.hexagon.V6.vS32Ub.npred.ai.128B(
    // CHECK-SAME: i1 true,
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: i32 {{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %A = arith.constant 1 : i1
    %C = arith.constant 1 : i32
    "llhvx.intr.vS32Ub_npred_ai_128B"(%A, %B, %C, %D) : (i1, !llvm.ptr, i32, vector<32 x i32>) -> ()
    return
}

// CHECK-LABEL: define void @vS32b_nt_pred_ai_128B
func.func @vS32b_nt_pred_ai_128B(%B : !llvm.ptr, %D : vector<32 x i32>) {
    // CHECK: call void @llvm.hexagon.V6.vS32b.nt.pred.ai.128B(
    // CHECK-SAME: i1 true,
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: i32 {{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %A = arith.constant 1 : i1
    %C = arith.constant 1 : i32
    "llhvx.intr.vS32b_nt_pred_ai_128B"(%A, %B, %C, %D) : (i1, !llvm.ptr, i32, vector<32 x i32>) -> ()
    return
}

// CHECK-LABEL: define void @vS32b_nt_npred_ai_128B
func.func @vS32b_nt_npred_ai_128B(%B : !llvm.ptr, %D : vector<32 x i32>) {
    // CHECK: call void @llvm.hexagon.V6.vS32b.nt.npred.ai.128B(
    // CHECK-SAME: i1 true,
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: i32 {{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %A = arith.constant 1 : i1
    %C = arith.constant 1 : i32
    "llhvx.intr.vS32b_nt_npred_ai_128B"(%A, %B, %C, %D) : (i1, !llvm.ptr, i32, vector<32 x i32>) -> ()
    return
}

// CHECK-LABEL: define ptr @vS32b_pred_pi_128B
func.func @vS32b_pred_pi_128B(%B : !llvm.ptr, %D : vector<32 x i32>) -> !llvm.ptr {
    // CHECK: call ptr @llvm.hexagon.V6.vS32b.pred.pi.128B(
    // CHECK-SAME: i1 true,
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: i32 {{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %A = arith.constant 1 : i1
    %C = arith.constant 1 : i32
    %0 = "llhvx.intr.vS32b_pred_pi_128B"(%A, %B, %C, %D) : (i1, !llvm.ptr, i32, vector<32 x i32>) -> !llvm.ptr
    return %0 : !llvm.ptr
}

// CHECK-LABEL: define ptr @vS32b_npred_pi_128B
func.func @vS32b_npred_pi_128B(%B : !llvm.ptr, %D : vector<32 x i32>) -> !llvm.ptr {
    // CHECK: call ptr @llvm.hexagon.V6.vS32b.npred.pi.128B(
    // CHECK-SAME: i1 true,
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: i32 {{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %A = arith.constant 1 : i1
    %C = arith.constant 1 : i32
    %0 = "llhvx.intr.vS32b_npred_pi_128B"(%A, %B, %C, %D) : (i1, !llvm.ptr, i32, vector<32 x i32>) -> !llvm.ptr
    return %0 : !llvm.ptr
}

// CHECK-LABEL: define ptr @vS32Ub_pred_pi_128B
func.func @vS32Ub_pred_pi_128B(%B : !llvm.ptr, %D : vector<32 x i32>) -> !llvm.ptr {
    // CHECK: call ptr @llvm.hexagon.V6.vS32Ub.pred.pi.128B(
    // CHECK-SAME: i1 true,
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: i32 {{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %A = arith.constant 1 : i1
    %C = arith.constant 1 : i32
    %0 = "llhvx.intr.vS32Ub_pred_pi_128B"(%A, %B, %C, %D) : (i1, !llvm.ptr, i32, vector<32 x i32>) -> !llvm.ptr
    return %0 : !llvm.ptr
}

// CHECK-LABEL: define ptr @vS32Ub_npred_pi_128B
func.func @vS32Ub_npred_pi_128B(%B : !llvm.ptr, %D : vector<32 x i32>) -> !llvm.ptr {
    // CHECK: call ptr @llvm.hexagon.V6.vS32Ub.npred.pi.128B(
    // CHECK-SAME: i1 true,
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: i32 {{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %A = arith.constant 1 : i1
    %C = arith.constant 1 : i32
    %0 = "llhvx.intr.vS32Ub_npred_pi_128B"(%A, %B, %C, %D) : (i1, !llvm.ptr, i32, vector<32 x i32>) -> !llvm.ptr
    return %0 : !llvm.ptr
}

// CHECK-LABEL: define ptr @vS32b_nt_pred_pi_128B
func.func @vS32b_nt_pred_pi_128B(%B : !llvm.ptr, %D : vector<32 x i32>) -> !llvm.ptr {
    // CHECK: call ptr @llvm.hexagon.V6.vS32b.nt.pred.pi.128B(
    // CHECK-SAME: i1 true,
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: i32 {{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %A = arith.constant 1 : i1
    %C = arith.constant 1 : i32
    %0 = "llhvx.intr.vS32b_nt_pred_pi_128B"(%A, %B, %C, %D) : (i1, !llvm.ptr, i32, vector<32 x i32>) -> !llvm.ptr
    return %0 : !llvm.ptr
}

// CHECK-LABEL: define ptr @vS32b_nt_npred_pi_128B
func.func @vS32b_nt_npred_pi_128B(%B : !llvm.ptr, %D : vector<32 x i32>) -> !llvm.ptr {
    // CHECK: call ptr @llvm.hexagon.V6.vS32b.nt.npred.pi.128B(
    // CHECK-SAME: i1 true,
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: i32 {{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %A = arith.constant 1 : i1
    %C = arith.constant 1 : i32
    %0 = "llhvx.intr.vS32b_nt_npred_pi_128B"(%A, %B, %C, %D) : (i1, !llvm.ptr, i32, vector<32 x i32>) -> !llvm.ptr
    return %0 : !llvm.ptr
}

// CHECK-LABEL: define ptr @vS32b_pred_ppu_128B
func.func @vS32b_pred_ppu_128B(%B : !llvm.ptr, %C : i32, %D : vector<32 x i32>) -> !llvm.ptr {
    // CHECK: call ptr @llvm.hexagon.V6.vS32b.pred.ppu.128B(
    // CHECK-SAME: i1 true,
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %A = arith.constant 1 : i1
    %0 = "llhvx.intr.vS32b_pred_ppu_128B"(%A, %B, %C, %D) : (i1, !llvm.ptr, i32, vector<32 x i32>) -> !llvm.ptr
    return %0 : !llvm.ptr
}

// CHECK-LABEL: define ptr @vS32b_npred_ppu_128B
func.func @vS32b_npred_ppu_128B(%B : !llvm.ptr, %C : i32, %D : vector<32 x i32>) -> !llvm.ptr {
    // CHECK: call ptr @llvm.hexagon.V6.vS32b.npred.ppu.128B(
    // CHECK-SAME: i1 true,
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %A = arith.constant 1 : i1
    %0 = "llhvx.intr.vS32b_npred_ppu_128B"(%A, %B, %C, %D) : (i1, !llvm.ptr, i32, vector<32 x i32>) -> !llvm.ptr
    return %0 : !llvm.ptr
}

// CHECK-LABEL: define ptr @vS32Ub_pred_ppu_128B
func.func @vS32Ub_pred_ppu_128B(%B : !llvm.ptr, %C : i32, %D : vector<32 x i32>) -> !llvm.ptr {
    // CHECK: call ptr @llvm.hexagon.V6.vS32Ub.pred.ppu.128B(
    // CHECK-SAME: i1 true,
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %A = arith.constant 1 : i1
    %0 = "llhvx.intr.vS32Ub_pred_ppu_128B"(%A, %B, %C, %D) : (i1, !llvm.ptr, i32, vector<32 x i32>) -> !llvm.ptr
    return %0 : !llvm.ptr
}

// CHECK-LABEL: define ptr @vS32Ub_npred_ppu_128B
func.func @vS32Ub_npred_ppu_128B(%B : !llvm.ptr, %C : i32, %D : vector<32 x i32>) -> !llvm.ptr {
    // CHECK: call ptr @llvm.hexagon.V6.vS32Ub.npred.ppu.128B(
    // CHECK-SAME: i1 true,
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %A = arith.constant 1 : i1
    %0 = "llhvx.intr.vS32Ub_npred_ppu_128B"(%A, %B, %C, %D) : (i1, !llvm.ptr, i32, vector<32 x i32>) -> !llvm.ptr
    return %0 : !llvm.ptr
}

// CHECK-LABEL: define ptr @vS32b_nt_pred_ppu_128B
func.func @vS32b_nt_pred_ppu_128B(%B : !llvm.ptr, %C : i32, %D : vector<32 x i32>) -> !llvm.ptr {
    // CHECK: call ptr @llvm.hexagon.V6.vS32b.nt.pred.ppu.128B(
    // CHECK-SAME: i1 true,
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %A = arith.constant 1 : i1
    %0 = "llhvx.intr.vS32b_nt_pred_ppu_128B"(%A, %B, %C, %D) : (i1, !llvm.ptr, i32, vector<32 x i32>) -> !llvm.ptr
    return %0 : !llvm.ptr
}

// CHECK-LABEL: define ptr @vS32b_nt_npred_ppu_128B
func.func @vS32b_nt_npred_ppu_128B(%B : !llvm.ptr, %C : i32, %D : vector<32 x i32>) -> !llvm.ptr {
    // CHECK: call ptr @llvm.hexagon.V6.vS32b.nt.npred.ppu.128B(
    // CHECK-SAME: i1 true,
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %A = arith.constant 1 : i1
    %0 = "llhvx.intr.vS32b_nt_npred_ppu_128B"(%A, %B, %C, %D) : (i1, !llvm.ptr, i32, vector<32 x i32>) -> !llvm.ptr
    return %0 : !llvm.ptr
}

// CHECK-LABEL: define i32 @extractw_128B
func.func @extractw_128B(%A : vector<32 x i32>, %B : i32) -> i32 {
    // CHECK: call i32 @llvm.hexagon.V6.extractw.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.extractw_128B"(%A, %B) : (vector<32 x i32>, i32) -> i32
    return %0 : i32
}

// CHECK-LABEL: define <32 x i32> @hi_128B
func.func @hi_128B(%A : vector<64 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.hi.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.hi_128B"(%A) : (vector<64 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @lo_128B
func.func @lo_128B(%A : vector<64 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.lo.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.lo_128B"(%A) : (vector<64 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @lvsplatw_128B
func.func @lvsplatw_128B(%A : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.lvsplatw_128B"(%A) : (i32) -> vector<32 x i32>
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

// CHECK-LABEL: define <32 x i32> @pred_and_n_128B
func.func @pred_and_n_128B(%A : vector<128 x i1>, %B : vector<128 x i1>, %C : vector<32xi32>, %D : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.pred.and.n.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <128 x i1> %{{[0-9]+}})
    %0 = "llhvx.intr.pred_and_n_128B"(%A, %B) : (vector<128 x i1>, vector<128 x i1>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %C, %D) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @pred_not_128B
func.func @pred_not_128B(%A : vector<128 x i1>, %B : vector<32xi32>, %C : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.pred.not.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}})
    %0 = "llhvx.intr.pred_not_128B"(%A) : (vector<128 x i1>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %B, %C) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @pred_or_128B
func.func @pred_or_128B(%A : vector<128 x i1>, %B : vector<128 x i1>, %C : vector<32xi32>, %D : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.pred.or.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <128 x i1> %{{[0-9]+}})
    %0 = "llhvx.intr.pred_or_128B"(%A, %B) : (vector<128 x i1>, vector<128 x i1>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %C, %D) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @pred_or_n_128B
func.func @pred_or_n_128B(%A : vector<128 x i1>, %B : vector<128 x i1>, %C : vector<32xi32>, %D : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.pred.or.n.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <128 x i1> %{{[0-9]+}})
    %0 = "llhvx.intr.pred_or_n_128B"(%A, %B) : (vector<128 x i1>, vector<128 x i1>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %C, %D) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @pred_scalar2_128B
func.func @pred_scalar2_128B(%A : i32, %B : vector<32xi32>, %C : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.pred.scalar2.128B(
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.pred_scalar2_128B"(%A) : (i32) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %B, %C) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @pred_xor_128B
func.func @pred_xor_128B(%A : vector<128 x i1>, %B : vector<128 x i1>, %C : vector<32xi32>, %D : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.pred.xor.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <128 x i1> %{{[0-9]+}})
    %0 = "llhvx.intr.pred_xor_128B"(%A, %B) : (vector<128 x i1>, vector<128 x i1>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %C, %D) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define void @vS32b_nqpred_ai_128B
func.func @vS32b_nqpred_ai_128B(%A : vector<128 x i1>, %B : !llvm.ptr, %C : vector<32 x i32>) {
    // CHECK: call void @llvm.hexagon.V6.vS32b.nqpred.ai.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    "llhvx.intr.vS32b_nqpred_ai_128B"(%A, %B, %C) : (vector<128 x i1>, !llvm.ptr, vector<32 x i32>) -> ()
    return
}

// CHECK-LABEL: define void @vS32b_nt_nqpred_ai_128B
func.func @vS32b_nt_nqpred_ai_128B(%A : vector<128 x i1>, %B : !llvm.ptr, %C : vector<32 x i32>) {
    // CHECK: call void @llvm.hexagon.V6.vS32b.nt.nqpred.ai.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    "llhvx.intr.vS32b_nt_nqpred_ai_128B"(%A, %B, %C) : (vector<128 x i1>, !llvm.ptr, vector<32 x i32>) -> ()
    return
}

// CHECK-LABEL: define void @vS32b_nt_qpred_ai_128B
func.func @vS32b_nt_qpred_ai_128B(%A : vector<128 x i1>, %B : !llvm.ptr, %C : vector<32 x i32>) {
    // CHECK: call void @llvm.hexagon.V6.vS32b.nt.qpred.ai.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    "llhvx.intr.vS32b_nt_qpred_ai_128B"(%A, %B, %C) : (vector<128 x i1>, !llvm.ptr, vector<32 x i32>) -> ()
    return
}

// CHECK-LABEL: define void @vS32b_qpred_ai_128B
func.func @vS32b_qpred_ai_128B(%A : vector<128 x i1>, %B : !llvm.ptr, %C : vector<32 x i32>) {
    // CHECK: call void @llvm.hexagon.V6.vS32b.qpred.ai.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    "llhvx.intr.vS32b_qpred_ai_128B"(%A, %B, %C) : (vector<128 x i1>, !llvm.ptr, vector<32 x i32>) -> ()
    return
}

// CHECK-LABEL: define <32 x i32> @vabsdiffh_128B
func.func @vabsdiffh_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vabsdiffh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vabsdiffh_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vabsdiffub_128B
func.func @vabsdiffub_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vabsdiffub.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vabsdiffub_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vabsdiffuh_128B
func.func @vabsdiffuh_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vabsdiffuh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vabsdiffuh_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vabsdiffw_128B
func.func @vabsdiffw_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vabsdiffw.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vabsdiffw_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vabsh_128B
func.func @vabsh_128B(%A : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vabsh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vabsh_128B"(%A) : (vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vabsh_sat_128B
func.func @vabsh_sat_128B(%A : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vabsh.sat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vabsh_sat_128B"(%A) : (vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vabsw_128B
func.func @vabsw_128B(%A : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vabsw.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vabsw_128B"(%A) : (vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vabsw_sat_128B
func.func @vabsw_sat_128B(%A : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vabsw.sat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vabsw_sat_128B"(%A) : (vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vaddb_128B
func.func @vaddb_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vaddb.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vaddb_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vaddb_dv_128B
func.func @vaddb_dv_128B(%A : vector<64 x i32>, %B : vector<64 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vaddb.dv.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vaddb_dv_128B"(%A, %B) : (vector<64 x i32>, vector<64 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vaddbnq_128B
func.func @vaddbnq_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vaddbnq_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vaddbq_128B
func.func @vaddbq_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vaddbq.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vaddbq_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vaddh_128B
func.func @vaddh_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vaddh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vaddh_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vaddh_dv_128B
func.func @vaddh_dv_128B(%A : vector<64 x i32>, %B : vector<64 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vaddh.dv.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vaddh_dv_128B"(%A, %B) : (vector<64 x i32>, vector<64 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vaddhnq_128B
func.func @vaddhnq_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vaddhnq.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vaddhnq_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vaddhq_128B
func.func @vaddhq_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vaddhq.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vaddhq_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vaddhsat_128B
func.func @vaddhsat_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vaddhsat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vaddhsat_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vaddhsat_dv_128B
func.func @vaddhsat_dv_128B(%A : vector<64 x i32>, %B : vector<64 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vaddhsat.dv.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vaddhsat_dv_128B"(%A, %B) : (vector<64 x i32>, vector<64 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vaddhw_128B
func.func @vaddhw_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vaddhw.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vaddhw_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vaddubh_128B
func.func @vaddubh_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vaddubh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vaddubh_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vaddubsat_128B
func.func @vaddubsat_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vaddubsat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vaddubsat_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vaddubsat_dv_128B
func.func @vaddubsat_dv_128B(%A : vector<64 x i32>, %B : vector<64 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vaddubsat.dv.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vaddubsat_dv_128B"(%A, %B) : (vector<64 x i32>, vector<64 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vadduhsat_128B
func.func @vadduhsat_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vadduhsat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vadduhsat_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vadduhsat_dv_128B
func.func @vadduhsat_dv_128B(%A : vector<64 x i32>, %B : vector<64 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vadduhsat.dv.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vadduhsat_dv_128B"(%A, %B) : (vector<64 x i32>, vector<64 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vadduhw_128B
func.func @vadduhw_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vadduhw.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vadduhw_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vaddw_128B
func.func @vaddw_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vaddw.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vaddw_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vaddw_dv_128B
func.func @vaddw_dv_128B(%A : vector<64 x i32>, %B : vector<64 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vaddw.dv.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vaddw_dv_128B"(%A, %B) : (vector<64 x i32>, vector<64 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vaddwnq_128B
func.func @vaddwnq_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vaddwnq.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vaddwnq_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vaddwq_128B
func.func @vaddwq_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vaddwq.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vaddwq_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vaddwsat_128B
func.func @vaddwsat_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vaddwsat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vaddwsat_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vaddwsat_dv_128B
func.func @vaddwsat_dv_128B(%A : vector<64 x i32>, %B : vector<64 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vaddwsat.dv.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vaddwsat_dv_128B"(%A, %B) : (vector<64 x i32>, vector<64 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @valignb_128B
func.func @valignb_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.valignb.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.valignb_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
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

// CHECK-LABEL: define <32 x i32> @vand_128B
func.func @vand_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vand.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vand_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vandqrt_128B
func.func @vandqrt_128B(%A : vector<128 x i1>, %B : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vandqrt.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vandqrt_128B"(%A, %B) : (vector<128 x i1>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vandqrt_acc_128B
func.func @vandqrt_acc_128B(%A : vector<32 x i32>, %B : vector<128 x i1>, %C : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vandqrt.acc.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vandqrt_acc_128B"(%A, %B, %C) : (vector<32 x i32>, vector<128 x i1>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vandvrt_128B
func.func @vandvrt_128B(%A : vector<32 x i32>, %B : i32, %C : vector<32xi32>, %D : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vandvrt.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vandvrt_128B"(%A, %B) : (vector<32 x i32>, i32) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %C, %D) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vandvrt_acc_128B
func.func @vandvrt_acc_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : i32, %D : vector<32xi32>, %E : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vandvrt.acc.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vandvrt_acc_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, i32) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %D, %E) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vaslh_128B
func.func @vaslh_128B(%A : vector<32 x i32>, %B : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vaslh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vaslh_128B"(%A, %B) : (vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vaslhv_128B
func.func @vaslhv_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vaslhv.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vaslhv_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vaslw_128B
func.func @vaslw_128B(%A : vector<32 x i32>, %B : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vaslw.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vaslw_128B"(%A, %B) : (vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vaslw_acc_128B
func.func @vaslw_acc_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vaslw.acc.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vaslw_acc_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vaslwv_128B
func.func @vaslwv_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vaslwv.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vaslwv_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vasrh_128B
func.func @vasrh_128B(%A : vector<32 x i32>, %B : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vasrh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vasrh_128B"(%A, %B) : (vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vasrhbrndsat_128B
func.func @vasrhbrndsat_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vasrhbrndsat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vasrhbrndsat_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vasrhubrndsat_128B
func.func @vasrhubrndsat_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vasrhubrndsat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vasrhubrndsat_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vasrhubsat_128B
func.func @vasrhubsat_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vasrhubsat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vasrhubsat_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vasrhv_128B
func.func @vasrhv_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vasrhv.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vasrhv_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vasrw_128B
func.func @vasrw_128B(%A : vector<32 x i32>, %B : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vasrw.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vasrw_128B"(%A, %B) : (vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vasrw_acc_128B
func.func @vasrw_acc_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vasrw.acc.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vasrw_acc_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vasrwh_128B
func.func @vasrwh_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vasrwh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vasrwh_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vasrwhrndsat_128B
func.func @vasrwhrndsat_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vasrwhrndsat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vasrwhrndsat_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vasrwhsat_128B
func.func @vasrwhsat_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vasrwhsat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vasrwhsat_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vasrwuhsat_128B
func.func @vasrwuhsat_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vasrwuhsat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vasrwuhsat_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vasrwv_128B
func.func @vasrwv_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vasrwv.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vasrwv_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vassign_128B
func.func @vassign_128B(%A : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vassign.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vassign_128B"(%A) : (vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vassignp_128B
func.func @vassignp_128B(%A : vector<64 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vassignp.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vassignp_128B"(%A) : (vector<64 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vavgh_128B
func.func @vavgh_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vavgh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vavgh_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vavghrnd_128B
func.func @vavghrnd_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vavghrnd.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vavghrnd_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vavgub_128B
func.func @vavgub_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vavgub.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vavgub_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vavgubrnd_128B
func.func @vavgubrnd_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vavgubrnd.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vavgubrnd_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vavguh_128B
func.func @vavguh_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vavguh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vavguh_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vavguhrnd_128B
func.func @vavguhrnd_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vavguhrnd.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vavguhrnd_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vavgw_128B
func.func @vavgw_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vavgw.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vavgw_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vavgwrnd_128B
func.func @vavgwrnd_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vavgwrnd.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vavgwrnd_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vcl0h_128B
func.func @vcl0h_128B(%A : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vcl0h.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vcl0h_128B"(%A) : (vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vcl0w_128B
func.func @vcl0w_128B(%A : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vcl0w.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vcl0w_128B"(%A) : (vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vcombine_128B
func.func @vcombine_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vcombine.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vcombine_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vd0_128B
func.func @vd0_128B() -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vd0.128B()
    %0 = "llhvx.intr.vd0_128B"() : () -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vdealb_128B
func.func @vdealb_128B(%A : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vdealb.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vdealb_128B"(%A) : (vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vdealb4w_128B
func.func @vdealb4w_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vdealb4w.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vdealb4w_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vdealh_128B
func.func @vdealh_128B(%A : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vdealh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vdealh_128B"(%A) : (vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vdealvdd_128B
func.func @vdealvdd_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vdealvdd.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vdealvdd_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vdelta_128B
func.func @vdelta_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vdelta.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vdelta_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vdmpybus_128B
func.func @vdmpybus_128B(%A : vector<32 x i32>, %B : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vdmpybus.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vdmpybus_128B"(%A, %B) : (vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vdmpybus_acc_128B
func.func @vdmpybus_acc_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vdmpybus.acc.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vdmpybus_acc_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vdmpybus_dv_128B
func.func @vdmpybus_dv_128B(%A : vector<64 x i32>, %B : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vdmpybus.dv.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vdmpybus_dv_128B"(%A, %B) : (vector<64 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vdmpybus_dv_acc_128B
func.func @vdmpybus_dv_acc_128B(%A : vector<64 x i32>, %B : vector<64 x i32>, %C : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vdmpybus.dv.acc.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vdmpybus_dv_acc_128B"(%A, %B, %C) : (vector<64 x i32>, vector<64 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vdmpyhb_128B
func.func @vdmpyhb_128B(%A : vector<32 x i32>, %B : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vdmpyhb.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vdmpyhb_128B"(%A, %B) : (vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vdmpyhb_acc_128B
func.func @vdmpyhb_acc_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vdmpyhb.acc.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vdmpyhb_acc_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vdmpyhb_dv_128B
func.func @vdmpyhb_dv_128B(%A : vector<64 x i32>, %B : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vdmpyhb.dv.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vdmpyhb_dv_128B"(%A, %B) : (vector<64 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vdmpyhb_dv_acc_128B
func.func @vdmpyhb_dv_acc_128B(%A : vector<64 x i32>, %B : vector<64 x i32>, %C : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vdmpyhb.dv.acc.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vdmpyhb_dv_acc_128B"(%A, %B, %C) : (vector<64 x i32>, vector<64 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vdmpyhisat_128B
func.func @vdmpyhisat_128B(%A : vector<64 x i32>, %B : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vdmpyhisat.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vdmpyhisat_128B"(%A, %B) : (vector<64 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vdmpyhisat_acc_128B
func.func @vdmpyhisat_acc_128B(%A : vector<32 x i32>, %B : vector<64 x i32>, %C : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vdmpyhisat.acc.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vdmpyhisat_acc_128B"(%A, %B, %C) : (vector<32 x i32>, vector<64 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vdmpyhsat_128B
func.func @vdmpyhsat_128B(%A : vector<32 x i32>, %B : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vdmpyhsat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vdmpyhsat_128B"(%A, %B) : (vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vdmpyhsat_acc_128B
func.func @vdmpyhsat_acc_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vdmpyhsat.acc.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vdmpyhsat_acc_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vdmpyhsuisat_128B
func.func @vdmpyhsuisat_128B(%A : vector<64 x i32>, %B : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vdmpyhsuisat.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vdmpyhsuisat_128B"(%A, %B) : (vector<64 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vdmpyhsuisat_acc_128B
func.func @vdmpyhsuisat_acc_128B(%A : vector<32 x i32>, %B : vector<64 x i32>, %C : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vdmpyhsuisat.acc.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vdmpyhsuisat_acc_128B"(%A, %B, %C) : (vector<32 x i32>, vector<64 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vdmpyhsusat_128B
func.func @vdmpyhsusat_128B(%A : vector<32 x i32>, %B : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vdmpyhsusat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vdmpyhsusat_128B"(%A, %B) : (vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vdmpyhsusat_acc_128B
func.func @vdmpyhsusat_acc_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vdmpyhsusat.acc.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vdmpyhsusat_acc_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vdmpyhvsat_128B
func.func @vdmpyhvsat_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vdmpyhvsat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vdmpyhvsat_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vdmpyhvsat_acc_128B
func.func @vdmpyhvsat_acc_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vdmpyhvsat.acc.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vdmpyhvsat_acc_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vdsaduh_128B
func.func @vdsaduh_128B(%A : vector<64 x i32>, %B : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vdsaduh.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vdsaduh_128B"(%A, %B) : (vector<64 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vdsaduh_acc_128B
func.func @vdsaduh_acc_128B(%A : vector<64 x i32>, %B : vector<64 x i32>, %C : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vdsaduh.acc.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vdsaduh_acc_128B"(%A, %B, %C) : (vector<64 x i32>, vector<64 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @veqb_128B
func.func @veqb_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : vector<32xi32>, %D : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.veqb.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.veqb_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %C, %D) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @veqb_and_128B
func.func @veqb_and_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : vector<32xi32>, %E : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.veqb.and.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.veqb_and_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %D, %E) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @veqb_or_128B
func.func @veqb_or_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : vector<32xi32>, %E : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.veqb.or.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.veqb_or_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %D, %E) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @veqb_xor_128B
func.func @veqb_xor_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : vector<32xi32>, %E : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.veqb.xor.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.veqb_xor_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %D, %E) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @veqh_128B
func.func @veqh_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : vector<32xi32>, %D : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.veqh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.veqh_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %C, %D) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @veqh_and_128B
func.func @veqh_and_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : vector<32xi32>, %E : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.veqh.and.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.veqh_and_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %D, %E) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @veqh_or_128B
func.func @veqh_or_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : vector<32xi32>, %E : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.veqh.or.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.veqh_or_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %D, %E) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @veqh_xor_128B
func.func @veqh_xor_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : vector<32xi32>, %E : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.veqh.xor.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.veqh_xor_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %D, %E) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @veqw_128B
func.func @veqw_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : vector<32xi32>, %D : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.veqw.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.veqw_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %C, %D) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @veqw_and_128B
func.func @veqw_and_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : vector<32xi32>, %E : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.veqw.and.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.veqw_and_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %D, %E) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @veqw_or_128B
func.func @veqw_or_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : vector<32xi32>, %E : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.veqw.or.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.veqw_or_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %D, %E) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @veqw_xor_128B
func.func @veqw_xor_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : vector<32xi32>, %E : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.veqw.xor.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.veqw_xor_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %D, %E) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vgtb_128B
func.func @vgtb_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : vector<32xi32>, %D : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vgtb.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vgtb_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %C, %D) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vgtb_and_128B
func.func @vgtb_and_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : vector<32xi32>, %E : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vgtb.and.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vgtb_and_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %D, %E) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vgtb_or_128B
func.func @vgtb_or_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : vector<32xi32>, %E : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vgtb.or.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vgtb_or_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %D, %E) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vgtb_xor_128B
func.func @vgtb_xor_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : vector<32xi32>, %E : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vgtb.xor.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vgtb_xor_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %D, %E) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vgth_128B
func.func @vgth_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : vector<32xi32>, %D : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vgth.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vgth_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %C, %D) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vgth_and_128B
func.func @vgth_and_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : vector<32xi32>, %E : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vgth.and.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vgth_and_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %D, %E) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vgth_or_128B
func.func @vgth_or_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : vector<32xi32>, %E : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vgth.or.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vgth_or_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %D, %E) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vgth_xor_128B
func.func @vgth_xor_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : vector<32xi32>, %E : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vgth.xor.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vgth_xor_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %D, %E) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vgtub_128B
func.func @vgtub_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : vector<32xi32>, %D : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vgtub.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vgtub_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %C, %D) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vgtub_and_128B
func.func @vgtub_and_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : vector<32xi32>, %E : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vgtub.and.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vgtub_and_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %D, %E) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vgtub_or_128B
func.func @vgtub_or_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : vector<32xi32>, %E : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vgtub.or.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vgtub_or_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %D, %E) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vgtub_xor_128B
func.func @vgtub_xor_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : vector<32xi32>, %E : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vgtub.xor.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vgtub_xor_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %D, %E) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vgtuh_128B
func.func @vgtuh_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : vector<32xi32>, %D : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vgtuh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vgtuh_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %C, %D) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vgtuh_and_128B
func.func @vgtuh_and_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : vector<32xi32>, %E : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vgtuh.and.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vgtuh_and_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %D, %E) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vgtuh_or_128B
func.func @vgtuh_or_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : vector<32xi32>, %E : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vgtuh.or.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vgtuh_or_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %D, %E) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vgtuh_xor_128B
func.func @vgtuh_xor_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : vector<32xi32>, %E : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vgtuh.xor.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vgtuh_xor_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %D, %E) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vgtuw_128B
func.func @vgtuw_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : vector<32xi32>, %D : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vgtuw.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vgtuw_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %C, %D) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vgtuw_and_128B
func.func @vgtuw_and_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : vector<32xi32>, %E : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vgtuw.and.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vgtuw_and_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %D, %E) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vgtuw_or_128B
func.func @vgtuw_or_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : vector<32xi32>, %E : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vgtuw.or.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vgtuw_or_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %D, %E) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vgtuw_xor_128B
func.func @vgtuw_xor_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : vector<32xi32>, %E : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vgtuw.xor.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vgtuw_xor_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %D, %E) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vgtw_128B
func.func @vgtw_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : vector<32xi32>, %D : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vgtw.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vgtw_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %C, %D) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vgtw_and_128B
func.func @vgtw_and_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : vector<32xi32>, %E : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vgtw.and.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vgtw_and_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %D, %E) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vgtw_or_128B
func.func @vgtw_or_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : vector<32xi32>, %E : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vgtw.or.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vgtw_or_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %D, %E) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vgtw_xor_128B
func.func @vgtw_xor_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : vector<32xi32>, %E : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vgtw.xor.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vgtw_xor_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %D, %E) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vinsertwr_128B
func.func @vinsertwr_128B(%A : vector<32 x i32>, %B : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vinsertwr.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vinsertwr_128B"(%A, %B) : (vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vlalignb_128B
func.func @vlalignb_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vlalignb.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vlalignb_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vlalignbi_128B
func.func @vlalignbi_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vlalignbi.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 {{[0-9]+}})
    %C = arith.constant 1 : i32
    %0 = "llhvx.intr.vlalignbi_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vlsrh_128B
func.func @vlsrh_128B(%A : vector<32 x i32>, %B : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vlsrh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vlsrh_128B"(%A, %B) : (vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vlsrhv_128B
func.func @vlsrhv_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vlsrhv.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vlsrhv_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vlsrw_128B
func.func @vlsrw_128B(%A : vector<32 x i32>, %B : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vlsrw.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vlsrw_128B"(%A, %B) : (vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vlsrwv_128B
func.func @vlsrwv_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vlsrwv.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vlsrwv_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vlutvvb_128B
func.func @vlutvvb_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vlutvvb.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vlutvvb_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vlutvvb_oracc_128B
func.func @vlutvvb_oracc_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vlutvvb.oracc.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vlutvvb_oracc_128B"(%A, %B, %C, %D) : (vector<32 x i32>, vector<32 x i32>, vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vlutvwh_128B
func.func @vlutvwh_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vlutvwh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vlutvwh_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vlutvwh_oracc_128B
func.func @vlutvwh_oracc_128B(%A : vector<64 x i32>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vlutvwh.oracc.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vlutvwh_oracc_128B"(%A, %B, %C, %D) : (vector<64 x i32>, vector<32 x i32>, vector<32 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmaxh_128B
func.func @vmaxh_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmaxh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmaxh_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmaxub_128B
func.func @vmaxub_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmaxub.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmaxub_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmaxuh_128B
func.func @vmaxuh_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmaxuh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmaxuh_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmaxw_128B
func.func @vmaxw_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmaxw.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmaxw_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vminh_128B
func.func @vminh_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vminh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vminh_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vminub_128B
func.func @vminub_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vminub.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vminub_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vminuh_128B
func.func @vminuh_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vminuh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vminuh_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vminw_128B
func.func @vminw_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vminw.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vminw_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpabus_128B
func.func @vmpabus_128B(%A : vector<64 x i32>, %B : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpabus.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vmpabus_128B"(%A, %B) : (vector<64 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpabus_acc_128B
func.func @vmpabus_acc_128B(%A : vector<64 x i32>, %B : vector<64 x i32>, %C : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpabus.acc.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vmpabus_acc_128B"(%A, %B, %C) : (vector<64 x i32>, vector<64 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpabusv_128B
func.func @vmpabusv_128B(%A : vector<64 x i32>, %B : vector<64 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpabusv.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpabusv_128B"(%A, %B) : (vector<64 x i32>, vector<64 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpabuuv_128B
func.func @vmpabuuv_128B(%A : vector<64 x i32>, %B : vector<64 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpabuuv.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpabuuv_128B"(%A, %B) : (vector<64 x i32>, vector<64 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpahb_128B
func.func @vmpahb_128B(%A : vector<64 x i32>, %B : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpahb.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vmpahb_128B"(%A, %B) : (vector<64 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpahb_acc_128B
func.func @vmpahb_acc_128B(%A : vector<64 x i32>, %B : vector<64 x i32>, %C : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpahb.acc.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vmpahb_acc_128B"(%A, %B, %C) : (vector<64 x i32>, vector<64 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpybus_128B
func.func @vmpybus_128B(%A : vector<32 x i32>, %B : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpybus.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vmpybus_128B"(%A, %B) : (vector<32 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpybus_acc_128B
func.func @vmpybus_acc_128B(%A : vector<64 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpybus.acc.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vmpybus_acc_128B"(%A, %B, %C) : (vector<64 x i32>, vector<32 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpybusv_128B
func.func @vmpybusv_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpybusv.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpybusv_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpybusv_acc_128B
func.func @vmpybusv_acc_128B(%A : vector<64 x i32>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpybusv.acc.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpybusv_acc_128B"(%A, %B, %C) : (vector<64 x i32>, vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpybv_128B
func.func @vmpybv_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpybv.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpybv_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpybv_acc_128B
func.func @vmpybv_acc_128B(%A : vector<64 x i32>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpybv.acc.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpybv_acc_128B"(%A, %B, %C) : (vector<64 x i32>, vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmpyewuh_128B
func.func @vmpyewuh_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmpyewuh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyewuh_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpyh_128B
func.func @vmpyh_128B(%A : vector<32 x i32>, %B : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpyh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyh_128B"(%A, %B) : (vector<32 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpyhsat_acc_128B
func.func @vmpyhsat_acc_128B(%A : vector<64 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpyhsat.acc.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyhsat_acc_128B"(%A, %B, %C) : (vector<64 x i32>, vector<32 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmpyhsrs_128B
func.func @vmpyhsrs_128B(%A : vector<32 x i32>, %B : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmpyhsrs.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyhsrs_128B"(%A, %B) : (vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmpyhss_128B
func.func @vmpyhss_128B(%A : vector<32 x i32>, %B : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmpyhss.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyhss_128B"(%A, %B) : (vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpyhus_128B
func.func @vmpyhus_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpyhus.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyhus_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpyhus_acc_128B
func.func @vmpyhus_acc_128B(%A : vector<64 x i32>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpyhus.acc.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyhus_acc_128B"(%A, %B, %C) : (vector<64 x i32>, vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpyhv_128B
func.func @vmpyhv_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpyhv.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyhv_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpyhv_acc_128B
func.func @vmpyhv_acc_128B(%A : vector<64 x i32>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpyhv.acc.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyhv_acc_128B"(%A, %B, %C) : (vector<64 x i32>, vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmpyhvsrs_128B
func.func @vmpyhvsrs_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmpyhvsrs.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyhvsrs_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmpyieoh_128B
func.func @vmpyieoh_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmpyieoh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyieoh_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmpyiewh_acc_128B
func.func @vmpyiewh_acc_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmpyiewh.acc.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyiewh_acc_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmpyiewuh_128B
func.func @vmpyiewuh_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmpyiewuh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyiewuh_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmpyiewuh_acc_128B
func.func @vmpyiewuh_acc_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmpyiewuh.acc.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyiewuh_acc_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmpyih_128B
func.func @vmpyih_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmpyih.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyih_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmpyih_acc_128B
func.func @vmpyih_acc_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmpyih.acc.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyih_acc_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmpyihb_128B
func.func @vmpyihb_128B(%A : vector<32 x i32>, %B : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmpyihb.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyihb_128B"(%A, %B) : (vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmpyihb_acc_128B
func.func @vmpyihb_acc_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmpyihb.acc.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyihb_acc_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmpyiowh_128B
func.func @vmpyiowh_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmpyiowh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyiowh_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmpyiwb_128B
func.func @vmpyiwb_128B(%A : vector<32 x i32>, %B : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmpyiwb.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyiwb_128B"(%A, %B) : (vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmpyiwb_acc_128B
func.func @vmpyiwb_acc_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmpyiwb.acc.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyiwb_acc_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmpyiwh_128B
func.func @vmpyiwh_128B(%A : vector<32 x i32>, %B : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmpyiwh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyiwh_128B"(%A, %B) : (vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmpyiwh_acc_128B
func.func @vmpyiwh_acc_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmpyiwh.acc.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyiwh_acc_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmpyowh_128B
func.func @vmpyowh_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmpyowh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyowh_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmpyowh_rnd_128B
func.func @vmpyowh_rnd_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmpyowh.rnd.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyowh_rnd_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmpyowh_rnd_sacc_128B
func.func @vmpyowh_rnd_sacc_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmpyowh.rnd.sacc.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyowh_rnd_sacc_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmpyowh_sacc_128B
func.func @vmpyowh_sacc_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmpyowh.sacc.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyowh_sacc_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpyub_128B
func.func @vmpyub_128B(%A : vector<32 x i32>, %B : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpyub.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyub_128B"(%A, %B) : (vector<32 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpyub_acc_128B
func.func @vmpyub_acc_128B(%A : vector<64 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpyub.acc.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyub_acc_128B"(%A, %B, %C) : (vector<64 x i32>, vector<32 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpyubv_128B
func.func @vmpyubv_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpyubv.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyubv_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpyubv_acc_128B
func.func @vmpyubv_acc_128B(%A : vector<64 x i32>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpyubv.acc.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyubv_acc_128B"(%A, %B, %C) : (vector<64 x i32>, vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpyuh_128B
func.func @vmpyuh_128B(%A : vector<32 x i32>, %B : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpyuh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyuh_128B"(%A, %B) : (vector<32 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpyuh_acc_128B
func.func @vmpyuh_acc_128B(%A : vector<64 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpyuh.acc.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyuh_acc_128B"(%A, %B, %C) : (vector<64 x i32>, vector<32 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpyuhv_128B
func.func @vmpyuhv_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpyuhv.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyuhv_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpyuhv_acc_128B
func.func @vmpyuhv_acc_128B(%A : vector<64 x i32>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpyuhv.acc.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyuhv_acc_128B"(%A, %B, %C) : (vector<64 x i32>, vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmux_128B
func.func @vmux_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmux.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmux_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vnavgh_128B
func.func @vnavgh_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vnavgh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vnavgh_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vnavgub_128B
func.func @vnavgub_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vnavgub.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vnavgub_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vnavgw_128B
func.func @vnavgw_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vnavgw.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vnavgw_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vnormamth_128B
func.func @vnormamth_128B(%A : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vnormamth.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vnormamth_128B"(%A) : (vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vnormamtw_128B
func.func @vnormamtw_128B(%A : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vnormamtw.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vnormamtw_128B"(%A) : (vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vnot_128B
func.func @vnot_128B(%A : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vnot.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vnot_128B"(%A) : (vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vor_128B
func.func @vor_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vor.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vor_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vpackeb_128B
func.func @vpackeb_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vpackeb.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vpackeb_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vpackeh_128B
func.func @vpackeh_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vpackeh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vpackeh_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vpackhb_sat_128B
func.func @vpackhb_sat_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vpackhb.sat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vpackhb_sat_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vpackhub_sat_128B
func.func @vpackhub_sat_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vpackhub.sat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vpackhub_sat_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vpackob_128B
func.func @vpackob_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vpackob.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vpackob_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vpackoh_128B
func.func @vpackoh_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vpackoh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vpackoh_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vpackwh_sat_128B
func.func @vpackwh_sat_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vpackwh.sat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vpackwh_sat_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vpackwuh_sat_128B
func.func @vpackwuh_sat_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vpackwuh.sat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vpackwuh_sat_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vpopcounth_128B
func.func @vpopcounth_128B(%A : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vpopcounth.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vpopcounth_128B"(%A) : (vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vrdelta_128B
func.func @vrdelta_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vrdelta.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vrdelta_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vrmpybus_128B
func.func @vrmpybus_128B(%A : vector<32 x i32>, %B : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vrmpybus.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vrmpybus_128B"(%A, %B) : (vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vrmpybus_acc_128B
func.func @vrmpybus_acc_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vrmpybus.acc.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vrmpybus_acc_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vrmpybusi_128B
func.func @vrmpybusi_128B(%A : vector<64 x i32>, %B : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vrmpybusi.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: i32 {{[0-9]+}})
    %C = arith.constant 1 : i32
    %0 = "llhvx.intr.vrmpybusi_128B"(%A, %B, %C) : (vector<64 x i32>, i32, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vrmpybusi_acc_128B
func.func @vrmpybusi_acc_128B(%A : vector<64 x i32>, %B : vector<64 x i32>, %C : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vrmpybusi.acc.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: i32 {{[0-9]+}})
    %D = arith.constant 1 : i32
    %0 = "llhvx.intr.vrmpybusi_acc_128B"(%A, %B, %C, %D) : (vector<64 x i32>, vector<64 x i32>, i32, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vrmpybusv_128B
func.func @vrmpybusv_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vrmpybusv.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vrmpybusv_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vrmpybusv_acc_128B
func.func @vrmpybusv_acc_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vrmpybusv.acc.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vrmpybusv_acc_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vrmpybv_128B
func.func @vrmpybv_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vrmpybv.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vrmpybv_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vrmpybv_acc_128B
func.func @vrmpybv_acc_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vrmpybv.acc.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vrmpybv_acc_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vrmpyub_128B
func.func @vrmpyub_128B(%A : vector<32 x i32>, %B : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vrmpyub.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vrmpyub_128B"(%A, %B) : (vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vrmpyub_acc_128B
func.func @vrmpyub_acc_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vrmpyub.acc.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vrmpyub_acc_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vrmpyubi_128B
func.func @vrmpyubi_128B(%A : vector<64 x i32>, %B : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vrmpyubi.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: i32 {{[0-9]+}})
    %C = arith.constant 1 : i32
    %0 = "llhvx.intr.vrmpyubi_128B"(%A, %B, %C) : (vector<64 x i32>, i32, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vrmpyubi_acc_128B
func.func @vrmpyubi_acc_128B(%A : vector<64 x i32>, %B : vector<64 x i32>, %C : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vrmpyubi.acc.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: i32 {{[0-9]+}})
    %D = arith.constant 1 : i32
    %0 = "llhvx.intr.vrmpyubi_acc_128B"(%A, %B, %C, %D) : (vector<64 x i32>, vector<64 x i32>, i32, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vrmpyubv_128B
func.func @vrmpyubv_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vrmpyubv.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vrmpyubv_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vrmpyubv_acc_128B
func.func @vrmpyubv_acc_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vrmpyubv.acc.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vrmpyubv_acc_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vror_128B
func.func @vror_128B(%A : vector<32 x i32>, %B : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vror.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vror_128B"(%A, %B) : (vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vroundhb_128B
func.func @vroundhb_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vroundhb.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vroundhb_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vroundhub_128B
func.func @vroundhub_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vroundhub.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vroundhub_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vroundwh_128B
func.func @vroundwh_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vroundwh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vroundwh_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vroundwuh_128B
func.func @vroundwuh_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vroundwuh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vroundwuh_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vrsadubi_128B
func.func @vrsadubi_128B(%A : vector<64 x i32>, %B : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vrsadubi.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: i32 {{[0-9]+}})
    %C = arith.constant 1 : i32
    %0 = "llhvx.intr.vrsadubi_128B"(%A, %B, %C) : (vector<64 x i32>, i32, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vrsadubi_acc_128B
func.func @vrsadubi_acc_128B(%A : vector<64 x i32>, %B : vector<64 x i32>, %C : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vrsadubi.acc.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: i32 {{[0-9]+}})
    %D = arith.constant 1 : i32
    %0 = "llhvx.intr.vrsadubi_acc_128B"(%A, %B, %C, %D) : (vector<64 x i32>, vector<64 x i32>, i32, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vsathub_128B
func.func @vsathub_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vsathub.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsathub_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vsatwh_128B
func.func @vsatwh_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vsatwh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsatwh_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vsb_128B
func.func @vsb_128B(%A : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vsb.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsb_128B"(%A) : (vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vsh_128B
func.func @vsh_128B(%A : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vsh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsh_128B"(%A) : (vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vshufeh_128B
func.func @vshufeh_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vshufeh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vshufeh_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vshuffb_128B
func.func @vshuffb_128B(%A : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vshuffb.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vshuffb_128B"(%A) : (vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vshuffeb_128B
func.func @vshuffeb_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vshuffeb.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vshuffeb_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vshuffh_128B
func.func @vshuffh_128B(%A : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vshuffh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vshuffh_128B"(%A) : (vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vshuffob_128B
func.func @vshuffob_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vshuffob.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vshuffob_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vshuffvdd_128B
func.func @vshuffvdd_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vshuffvdd_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vshufoeb_128B
func.func @vshufoeb_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vshufoeb.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vshufoeb_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vshufoeh_128B
func.func @vshufoeh_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vshufoeh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vshufoeh_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vshufoh_128B
func.func @vshufoh_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vshufoh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vshufoh_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vsubb_128B
func.func @vsubb_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vsubb.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsubb_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vsubb_dv_128B
func.func @vsubb_dv_128B(%A : vector<64 x i32>, %B : vector<64 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vsubb.dv.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsubb_dv_128B"(%A, %B) : (vector<64 x i32>, vector<64 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vsubbnq_128B
func.func @vsubbnq_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vsubbnq.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsubbnq_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vsubbq_128B
func.func @vsubbq_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vsubbq.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsubbq_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vsubh_128B
func.func @vsubh_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vsubh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsubh_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vsubh_dv_128B
func.func @vsubh_dv_128B(%A : vector<64 x i32>, %B : vector<64 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vsubh.dv.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsubh_dv_128B"(%A, %B) : (vector<64 x i32>, vector<64 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vsubhnq_128B
func.func @vsubhnq_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vsubhnq.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsubhnq_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vsubhq_128B
func.func @vsubhq_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vsubhq.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsubhq_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vsubhsat_128B
func.func @vsubhsat_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vsubhsat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsubhsat_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vsubhsat_dv_128B
func.func @vsubhsat_dv_128B(%A : vector<64 x i32>, %B : vector<64 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vsubhsat.dv.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsubhsat_dv_128B"(%A, %B) : (vector<64 x i32>, vector<64 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vsubhw_128B
func.func @vsubhw_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vsubhw.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsubhw_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vsububh_128B
func.func @vsububh_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vsububh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsububh_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vsububsat_128B
func.func @vsububsat_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vsububsat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsububsat_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vsububsat_dv_128B
func.func @vsububsat_dv_128B(%A : vector<64 x i32>, %B : vector<64 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vsububsat.dv.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsububsat_dv_128B"(%A, %B) : (vector<64 x i32>, vector<64 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vsubuhsat_128B
func.func @vsubuhsat_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vsubuhsat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsubuhsat_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vsubuhsat_dv_128B
func.func @vsubuhsat_dv_128B(%A : vector<64 x i32>, %B : vector<64 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vsubuhsat.dv.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsubuhsat_dv_128B"(%A, %B) : (vector<64 x i32>, vector<64 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vsubuhw_128B
func.func @vsubuhw_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vsubuhw.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsubuhw_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vsubw_128B
func.func @vsubw_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vsubw.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsubw_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vsubw_dv_128B
func.func @vsubw_dv_128B(%A : vector<64 x i32>, %B : vector<64 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vsubw.dv.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsubw_dv_128B"(%A, %B) : (vector<64 x i32>, vector<64 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vsubwnq_128B
func.func @vsubwnq_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vsubwnq.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsubwnq_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vsubwq_128B
func.func @vsubwq_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vsubwq.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsubwq_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vsubwsat_128B
func.func @vsubwsat_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vsubwsat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsubwsat_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vsubwsat_dv_128B
func.func @vsubwsat_dv_128B(%A : vector<64 x i32>, %B : vector<64 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vsubwsat.dv.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsubwsat_dv_128B"(%A, %B) : (vector<64 x i32>, vector<64 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vswap_128B
func.func @vswap_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vswap.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vswap_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vtmpyb_128B
func.func @vtmpyb_128B(%A : vector<64 x i32>, %B : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vtmpyb.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vtmpyb_128B"(%A, %B) : (vector<64 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vtmpyb_acc_128B
func.func @vtmpyb_acc_128B(%A : vector<64 x i32>, %B : vector<64 x i32>, %C : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vtmpyb.acc.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vtmpyb_acc_128B"(%A, %B, %C) : (vector<64 x i32>, vector<64 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vtmpybus_128B
func.func @vtmpybus_128B(%A : vector<64 x i32>, %B : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vtmpybus.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vtmpybus_128B"(%A, %B) : (vector<64 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vtmpybus_acc_128B
func.func @vtmpybus_acc_128B(%A : vector<64 x i32>, %B : vector<64 x i32>, %C : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vtmpybus.acc.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vtmpybus_acc_128B"(%A, %B, %C) : (vector<64 x i32>, vector<64 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vtmpyhb_128B
func.func @vtmpyhb_128B(%A : vector<64 x i32>, %B : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vtmpyhb.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vtmpyhb_128B"(%A, %B) : (vector<64 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vtmpyhb_acc_128B
func.func @vtmpyhb_acc_128B(%A : vector<64 x i32>, %B : vector<64 x i32>, %C : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vtmpyhb.acc.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vtmpyhb_acc_128B"(%A, %B, %C) : (vector<64 x i32>, vector<64 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vunpackb_128B
func.func @vunpackb_128B(%A : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vunpackb.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vunpackb_128B"(%A) : (vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vunpackh_128B
func.func @vunpackh_128B(%A : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vunpackh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vunpackh_128B"(%A) : (vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vunpackob_128B
func.func @vunpackob_128B(%A : vector<64 x i32>, %B : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vunpackob.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vunpackob_128B"(%A, %B) : (vector<64 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vunpackoh_128B
func.func @vunpackoh_128B(%A : vector<64 x i32>, %B : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vunpackoh.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vunpackoh_128B"(%A, %B) : (vector<64 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vunpackub_128B
func.func @vunpackub_128B(%A : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vunpackub.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vunpackub_128B"(%A) : (vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vunpackuh_128B
func.func @vunpackuh_128B(%A : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vunpackuh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vunpackuh_128B"(%A) : (vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vxor_128B
func.func @vxor_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vxor.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vxor_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vzb_128B
func.func @vzb_128B(%A : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vzb.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vzb_128B"(%A) : (vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vzh_128B
func.func @vzh_128B(%A : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vzh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vzh_128B"(%A) : (vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @lvsplatb_128B
func.func @lvsplatb_128B(%A : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.lvsplatb.128B(
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.lvsplatb_128B"(%A) : (i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @lvsplath_128B
func.func @lvsplath_128B(%A : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.lvsplath_128B"(%A) : (i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @pred_scalar2v2_128B
func.func @pred_scalar2v2_128B(%A : i32, %B : vector<32xi32>, %C : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.pred.scalar2v2.128B(
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.pred_scalar2v2_128B"(%A) : (i32) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %B, %C) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @shuffeqh_128B
func.func @shuffeqh_128B(%A : vector<128 x i1>, %B : vector<128 x i1>, %C : vector<32xi32>, %D : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.shuffeqh.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <128 x i1> %{{[0-9]+}})
    %0 = "llhvx.intr.shuffeqh_128B"(%A, %B) : (vector<128 x i1>, vector<128 x i1>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %C, %D) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @shuffeqw_128B
func.func @shuffeqw_128B(%A : vector<128 x i1>, %B : vector<128 x i1>, %C : vector<32xi32>, %D : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.shuffeqw.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <128 x i1> %{{[0-9]+}})
    %0 = "llhvx.intr.shuffeqw_128B"(%A, %B) : (vector<128 x i1>, vector<128 x i1>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %C, %D) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vaddbsat_128B
func.func @vaddbsat_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vaddbsat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vaddbsat_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vaddbsat_dv_128B
func.func @vaddbsat_dv_128B(%A : vector<64 x i32>, %B : vector<64 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vaddbsat.dv.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vaddbsat_dv_128B"(%A, %B) : (vector<64 x i32>, vector<64 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vaddcarry_128B
func.func @vaddcarry_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : vector<128 x i1>) -> vector<32 x i32> {
    // CHECK: call { <32 x i32>, <128 x i1> } @llvm.hexagon.V6.vaddcarry.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <128 x i1> %{{[0-9]+}})
    %0 = "llhvx.intr.vaddcarry_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, vector<128 x i1>) -> !llvm.struct<(vector<32 x i32>, vector<128 x i1>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(vector<32 x i32>, vector<128 x i1>)>
    func.return %1 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vaddclbh_128B
func.func @vaddclbh_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vaddclbh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vaddclbh_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vaddclbw_128B
func.func @vaddclbw_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vaddclbw.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vaddclbw_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vaddhw_acc_128B
func.func @vaddhw_acc_128B(%A : vector<64 x i32>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vaddhw.acc.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vaddhw_acc_128B"(%A, %B, %C) : (vector<64 x i32>, vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vaddubh_acc_128B
func.func @vaddubh_acc_128B(%A : vector<64 x i32>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vaddubh.acc.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vaddubh_acc_128B"(%A, %B, %C) : (vector<64 x i32>, vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vaddububb_sat_128B
func.func @vaddububb_sat_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vaddububb.sat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vaddububb_sat_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vadduhw_acc_128B
func.func @vadduhw_acc_128B(%A : vector<64 x i32>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vadduhw.acc.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vadduhw_acc_128B"(%A, %B, %C) : (vector<64 x i32>, vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vadduwsat_128B
func.func @vadduwsat_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vadduwsat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vadduwsat_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vadduwsat_dv_128B
func.func @vadduwsat_dv_128B(%A : vector<64 x i32>, %B : vector<64 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vadduwsat.dv.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vadduwsat_dv_128B"(%A, %B) : (vector<64 x i32>, vector<64 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vandnqrt_128B
func.func @vandnqrt_128B(%A : vector<128 x i1>, %B : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vandnqrt.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vandnqrt_128B"(%A, %B) : (vector<128 x i1>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vandnqrt_acc_128B
func.func @vandnqrt_acc_128B(%A : vector<32 x i32>, %B : vector<128 x i1>, %C : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vandnqrt.acc.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vandnqrt_acc_128B"(%A, %B, %C) : (vector<32 x i32>, vector<128 x i1>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vandvnqv_128B
func.func @vandvnqv_128B(%A : vector<128 x i1>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vandvnqv.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vandvnqv_128B"(%A, %B) : (vector<128 x i1>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vandvqv_128B
func.func @vandvqv_128B(%A : vector<128 x i1>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vandvqv.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vandvqv_128B"(%A, %B) : (vector<128 x i1>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vasrhbsat_128B
func.func @vasrhbsat_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vasrhbsat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vasrhbsat_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vasruwuhrndsat_128B
func.func @vasruwuhrndsat_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vasruwuhrndsat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vasruwuhrndsat_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vasrwuhrndsat_128B
func.func @vasrwuhrndsat_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vasrwuhrndsat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vasrwuhrndsat_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vlsrb_128B
func.func @vlsrb_128B(%A : vector<32 x i32>, %B : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vlsrb.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vlsrb_128B"(%A, %B) : (vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vlutvvb_nm_128B
func.func @vlutvvb_nm_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vlutvvb.nm.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vlutvvb_nm_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vlutvvb_oracci_128B
func.func @vlutvvb_oracci_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vlutvvb.oracci.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 {{[0-9]+}})
    %D = arith.constant 1 : i32
    %0 = "llhvx.intr.vlutvvb_oracci_128B"(%A, %B, %C, %D) : (vector<32 x i32>, vector<32 x i32>, vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vlutvvbi_128B
func.func @vlutvvbi_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vlutvvbi.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 {{[0-9]+}})
    %C = arith.constant 1 : i32
    %0 = "llhvx.intr.vlutvvbi_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vlutvwh_nm_128B
func.func @vlutvwh_nm_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vlutvwh.nm.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vlutvwh_nm_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vlutvwh_oracci_128B
func.func @vlutvwh_oracci_128B(%A : vector<64 x i32>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vlutvwh.oracci.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 {{[0-9]+}})
    %D = arith.constant 1 : i32
    %0 = "llhvx.intr.vlutvwh_oracci_128B"(%A, %B, %C, %D) : (vector<64 x i32>, vector<32 x i32>, vector<32 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vlutvwhi_128B
func.func @vlutvwhi_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vlutvwhi.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 {{[0-9]+}})
    %C = arith.constant 1 : i32
    %0 = "llhvx.intr.vlutvwhi_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmaxb_128B
func.func @vmaxb_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmaxb.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmaxb_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vminb_128B
func.func @vminb_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vminb.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vminb_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpauhb_128B
func.func @vmpauhb_128B(%A : vector<64 x i32>, %B : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpauhb.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vmpauhb_128B"(%A, %B) : (vector<64 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpauhb_acc_128B
func.func @vmpauhb_acc_128B(%A : vector<64 x i32>, %B : vector<64 x i32>, %C : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpauhb.acc.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vmpauhb_acc_128B"(%A, %B, %C) : (vector<64 x i32>, vector<64 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpyewuh_64_128B
func.func @vmpyewuh_64_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpyewuh.64.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyewuh_64_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmpyiwub_128B
func.func @vmpyiwub_128B(%A : vector<32 x i32>, %B : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmpyiwub.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyiwub_128B"(%A, %B) : (vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmpyiwub_acc_128B
func.func @vmpyiwub_acc_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmpyiwub.acc.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyiwub_acc_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpyowh_64_acc_128B
func.func @vmpyowh_64_acc_128B(%A : vector<64 x i32>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpyowh.64.acc.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyowh_64_acc_128B"(%A, %B, %C) : (vector<64 x i32>, vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vrounduhub_128B
func.func @vrounduhub_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vrounduhub.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vrounduhub_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vrounduwuh_128B
func.func @vrounduwuh_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vrounduwuh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vrounduwuh_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vsatuwuh_128B
func.func @vsatuwuh_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vsatuwuh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsatuwuh_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vsubbsat_128B
func.func @vsubbsat_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vsubbsat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsubbsat_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vsubbsat_dv_128B
func.func @vsubbsat_dv_128B(%A : vector<64 x i32>, %B : vector<64 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vsubbsat.dv.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsubbsat_dv_128B"(%A, %B) : (vector<64 x i32>, vector<64 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vsubcarry_128B
func.func @vsubcarry_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : vector<128 x i1>) -> vector<32 x i32> {
    // CHECK: call { <32 x i32>, <128 x i1> } @llvm.hexagon.V6.vsubcarry.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <128 x i1> %{{[0-9]+}})
    %0 = "llhvx.intr.vsubcarry_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, vector<128 x i1>) -> !llvm.struct<(vector<32 x i32>, vector<128 x i1>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(vector<32 x i32>, vector<128 x i1>)>
    func.return %1 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vsubububb_sat_128B
func.func @vsubububb_sat_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vsubububb.sat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsubububb_sat_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vsubuwsat_128B
func.func @vsubuwsat_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vsubuwsat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsubuwsat_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vsubuwsat_dv_128B
func.func @vsubuwsat_dv_128B(%A : vector<64 x i32>, %B : vector<64 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vsubuwsat.dv.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsubuwsat_dv_128B"(%A, %B) : (vector<64 x i32>, vector<64 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vabsb_128B
func.func @vabsb_128B(%A : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vabsb.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vabsb_128B"(%A) : (vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vabsb_sat_128B
func.func @vabsb_sat_128B(%A : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vabsb.sat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vabsb_sat_128B"(%A) : (vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vaslh_acc_128B
func.func @vaslh_acc_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vaslh.acc.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vaslh_acc_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vasrh_acc_128B
func.func @vasrh_acc_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vasrh.acc.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vasrh_acc_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vasruhubrndsat_128B
func.func @vasruhubrndsat_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vasruhubrndsat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vasruhubrndsat_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vasruhubsat_128B
func.func @vasruhubsat_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vasruhubsat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vasruhubsat_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vasruwuhsat_128B
func.func @vasruwuhsat_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vasruwuhsat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vasruwuhsat_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vavgb_128B
func.func @vavgb_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vavgb.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vavgb_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vavgbrnd_128B
func.func @vavgbrnd_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vavgbrnd.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vavgbrnd_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vavguw_128B
func.func @vavguw_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vavguw.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vavguw_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vavguwrnd_128B
func.func @vavguwrnd_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vavguwrnd.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vavguwrnd_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vdd0_128B
func.func @vdd0_128B() -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vdd0.128B()
    %0 = "llhvx.intr.vdd0_128B"() : () -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define void @vgathermh_128B
func.func @vgathermh_128B(%A : !llvm.ptr, %B : i32, %C : i32, %D : vector<32 x i32>) {
    // CHECK: call void @llvm.hexagon.V6.vgathermh.128B(
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    "llhvx.intr.vgathermh_128B"(%A, %B, %C, %D) : (!llvm.ptr, i32, i32, vector<32 x i32>) -> ()
    return
}

// CHECK-LABEL: define void @vgathermhq_128B
func.func @vgathermhq_128B(%A : !llvm.ptr, %B : vector<128 x i1>, %C : i32, %D : i32, %E : vector<32 x i32>) {
    // CHECK: call void @llvm.hexagon.V6.vgathermhq.128B(
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    "llhvx.intr.vgathermhq_128B"(%A, %B, %C, %D, %E) : (!llvm.ptr, vector<128 x i1>, i32, i32, vector<32 x i32>) -> ()
    return
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

// CHECK-LABEL: define void @vgathermhwq_128B
func.func @vgathermhwq_128B(%A : !llvm.ptr, %B : vector<128 x i1>, %C : i32, %D : i32, %E : vector<64 x i32>) {
    // CHECK: call void @llvm.hexagon.V6.vgathermhwq.128B(
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}})
    "llhvx.intr.vgathermhwq_128B"(%A, %B, %C, %D, %E) : (!llvm.ptr, vector<128 x i1>, i32, i32, vector<64 x i32>) -> ()
    return
}

// CHECK-LABEL: define void @vgathermw_128B
func.func @vgathermw_128B(%A : !llvm.ptr, %B : i32, %C : i32, %D : vector<32 x i32>) {
    // CHECK: call void @llvm.hexagon.V6.vgathermw.128B(
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    "llhvx.intr.vgathermw_128B"(%A, %B, %C, %D) : (!llvm.ptr, i32, i32, vector<32 x i32>) -> ()
    return
}

// CHECK-LABEL: define void @vgathermwq_128B
func.func @vgathermwq_128B(%A : !llvm.ptr, %B : vector<128 x i1>, %C : i32, %D : i32, %E : vector<32 x i32>) {
    // CHECK: call void @llvm.hexagon.V6.vgathermwq.128B(
    // CHECK-SAME: ptr %{{[0-9]+}},
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    "llhvx.intr.vgathermwq_128B"(%A, %B, %C, %D, %E) : (!llvm.ptr, vector<128 x i1>, i32, i32, vector<32 x i32>) -> ()
    return
}

// CHECK-LABEL: define <32 x i32> @vlut4_128B
func.func @vlut4_128B(%A : vector<32 x i32>, %B : i64) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vlut4.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i64 %{{[0-9]+}})
    %0 = "llhvx.intr.vlut4_128B"(%A, %B) : (vector<32 x i32>, i64) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpabuu_128B
func.func @vmpabuu_128B(%A : vector<64 x i32>, %B : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpabuu.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vmpabuu_128B"(%A, %B) : (vector<64 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpabuu_acc_128B
func.func @vmpabuu_acc_128B(%A : vector<64 x i32>, %B : vector<64 x i32>, %C : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpabuu.acc.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vmpabuu_acc_128B"(%A, %B, %C) : (vector<64 x i32>, vector<64 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmpahhsat_128B
func.func @vmpahhsat_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i64) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmpahhsat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i64 %{{[0-9]+}})
    %0 = "llhvx.intr.vmpahhsat_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i64) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmpauhuhsat_128B
func.func @vmpauhuhsat_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i64) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmpauhuhsat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i64 %{{[0-9]+}})
    %0 = "llhvx.intr.vmpauhuhsat_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i64) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmpsuhuhsat_128B
func.func @vmpsuhuhsat_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i64) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmpsuhuhsat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i64 %{{[0-9]+}})
    %0 = "llhvx.intr.vmpsuhuhsat_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i64) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpyh_acc_128B
func.func @vmpyh_acc_128B(%A : vector<64 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpyh.acc.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyh_acc_128B"(%A, %B, %C) : (vector<64 x i32>, vector<32 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmpyuhe_128B
func.func @vmpyuhe_128B(%A : vector<32 x i32>, %B : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmpyuhe.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyuhe_128B"(%A, %B) : (vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmpyuhe_acc_128B
func.func @vmpyuhe_acc_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : i32) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmpyuhe.acc.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyuhe_acc_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, i32) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vnavgb_128B
func.func @vnavgb_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vnavgb.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vnavgb_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vprefixqb_128B
func.func @vprefixqb_128B(%A : vector<128 x i1>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vprefixqb.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}})
    %0 = "llhvx.intr.vprefixqb_128B"(%A) : (vector<128 x i1>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vprefixqh_128B
func.func @vprefixqh_128B(%A : vector<128 x i1>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vprefixqh.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}})
    %0 = "llhvx.intr.vprefixqh_128B"(%A) : (vector<128 x i1>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vprefixqw_128B
func.func @vprefixqw_128B(%A : vector<128 x i1>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vprefixqw.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}})
    %0 = "llhvx.intr.vprefixqw_128B"(%A) : (vector<128 x i1>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define void @vscattermh_128B
func.func @vscattermh_128B(%A : i32, %B : i32, %C : vector<32 x i32>, %D : vector<32 x i32>) {
    // CHECK: call void @llvm.hexagon.V6.vscattermh.128B(
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    "llhvx.intr.vscattermh_128B"(%A, %B, %C, %D) : (i32, i32, vector<32 x i32>, vector<32 x i32>) -> ()
    return
}

// CHECK-LABEL: define void @vscattermh_add_128B
func.func @vscattermh_add_128B(%A : i32, %B : i32, %C : vector<32 x i32>, %D : vector<32 x i32>) {
    // CHECK: call void @llvm.hexagon.V6.vscattermh.add.128B(
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    "llhvx.intr.vscattermh_add_128B"(%A, %B, %C, %D) : (i32, i32, vector<32 x i32>, vector<32 x i32>) -> ()
    return
}

// CHECK-LABEL: define void @vscattermhq_128B
func.func @vscattermhq_128B(%A : vector<128 x i1>, %B : i32, %C : i32, %D : vector<32 x i32>, %E : vector<32 x i32>) {
    // CHECK: call void @llvm.hexagon.V6.vscattermhq.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    "llhvx.intr.vscattermhq_128B"(%A, %B, %C, %D, %E) : (vector<128 x i1>, i32, i32, vector<32 x i32>, vector<32 x i32>) -> ()
    return
}

// CHECK-LABEL: define void @vscattermhw_128B
func.func @vscattermhw_128B(%A : i32, %B : i32, %C : vector<64 x i32>, %D : vector<32 x i32>) {
    // CHECK: call void @llvm.hexagon.V6.vscattermhw.128B(
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    "llhvx.intr.vscattermhw_128B"(%A, %B, %C, %D) : (i32, i32, vector<64 x i32>, vector<32 x i32>) -> ()
    return
}

// CHECK-LABEL: define void @vscattermhw_add_128B
func.func @vscattermhw_add_128B(%A : i32, %B : i32, %C : vector<64 x i32>, %D : vector<32 x i32>) {
    // CHECK: call void @llvm.hexagon.V6.vscattermhw.add.128B(
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    "llhvx.intr.vscattermhw_add_128B"(%A, %B, %C, %D) : (i32, i32, vector<64 x i32>, vector<32 x i32>) -> ()
    return
}

// CHECK-LABEL: define void @vscattermhwq_128B
func.func @vscattermhwq_128B(%A : vector<128 x i1>, %B : i32, %C : i32, %D : vector<64 x i32>, %E : vector<32 x i32>) {
    // CHECK: call void @llvm.hexagon.V6.vscattermhwq.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    "llhvx.intr.vscattermhwq_128B"(%A, %B, %C, %D, %E) : (vector<128 x i1>, i32, i32, vector<64 x i32>, vector<32 x i32>) -> ()
    return
}

// CHECK-LABEL: define void @vscattermw_128B
func.func @vscattermw_128B(%A : i32, %B : i32, %C : vector<32 x i32>, %D : vector<32 x i32>) {
    // CHECK: call void @llvm.hexagon.V6.vscattermw.128B(
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    "llhvx.intr.vscattermw_128B"(%A, %B, %C, %D) : (i32, i32, vector<32 x i32>, vector<32 x i32>) -> ()
    return
}

// CHECK-LABEL: define void @vscattermw_add_128B
func.func @vscattermw_add_128B(%A : i32, %B : i32, %C : vector<32 x i32>, %D : vector<32 x i32>) {
    // CHECK: call void @llvm.hexagon.V6.vscattermw.add.128B(
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    "llhvx.intr.vscattermw_add_128B"(%A, %B, %C, %D) : (i32, i32, vector<32 x i32>, vector<32 x i32>) -> ()
    return
}

// CHECK-LABEL: define void @vscattermwq_128B
func.func @vscattermwq_128B(%A : vector<128 x i1>, %B : i32, %C : i32, %D : vector<32 x i32>, %E : vector<32 x i32>) {
    // CHECK: call void @llvm.hexagon.V6.vscattermwq.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: i32 %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    "llhvx.intr.vscattermwq_128B"(%A, %B, %C, %D, %E) : (vector<128 x i1>, i32, i32, vector<32 x i32>, vector<32 x i32>) -> ()
    return
}

// CHECK-LABEL: define <32 x i32> @vaddcarryo_128B
func.func @vaddcarryo_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call { <32 x i32>, <128 x i1> } @llvm.hexagon.V6.vaddcarryo.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vaddcarryo_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> !llvm.struct<(vector<32 x i32>, vector<128 x i1>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(vector<32 x i32>, vector<128 x i1>)>
    func.return %1 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vaddcarrysat_128B
func.func @vaddcarrysat_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : vector<128 x i1>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vaddcarrysat.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <128 x i1> %{{[0-9]+}})
    %0 = "llhvx.intr.vaddcarrysat_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, vector<128 x i1>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vasr_into_128B
func.func @vasr_into_128B(%A : vector<64 x i32>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vasr.into.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vasr_into_128B"(%A, %B, %C) : (vector<64 x i32>, vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vrotr_128B
func.func @vrotr_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vrotr.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vrotr_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vsatdw_128B
func.func @vsatdw_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vsatdw.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsatdw_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vsubcarryo_128B
func.func @vsubcarryo_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call { <32 x i32>, <128 x i1> } @llvm.hexagon.V6.vsubcarryo.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsubcarryo_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> !llvm.struct<(vector<32 x i32>, vector<128 x i1>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(vector<32 x i32>, vector<128 x i1>)>
    func.return %1 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @v6mpyhubs10_128B
func.func @v6mpyhubs10_128B(%A : vector<64 x i32>, %B : vector<64 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.v6mpyhubs10.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 {{[0-9]+}})
    %C = arith.constant 1 : i32
    %0 = "llhvx.intr.v6mpyhubs10_128B"(%A, %B, %C) : (vector<64 x i32>, vector<64 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @v6mpyhubs10_vxx_128B
func.func @v6mpyhubs10_vxx_128B(%A : vector<64 x i32>, %B : vector<64 x i32>, %C : vector<64 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.v6mpyhubs10.vxx.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 {{[0-9]+}})
    %D = arith.constant 1 : i32
    %0 = "llhvx.intr.v6mpyhubs10_vxx_128B"(%A, %B, %C, %D) : (vector<64 x i32>, vector<64 x i32>, vector<64 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @v6mpyvubs10_128B
func.func @v6mpyvubs10_128B(%A : vector<64 x i32>, %B : vector<64 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.v6mpyvubs10.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 {{[0-9]+}})
    %C = arith.constant 1 : i32
    %0 = "llhvx.intr.v6mpyvubs10_128B"(%A, %B, %C) : (vector<64 x i32>, vector<64 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @v6mpyvubs10_vxx_128B
func.func @v6mpyvubs10_vxx_128B(%A : vector<64 x i32>, %B : vector<64 x i32>, %C : vector<64 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.v6mpyvubs10.vxx.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: i32 {{[0-9]+}})
    %D = arith.constant 1 : i32
    %0 = "llhvx.intr.v6mpyvubs10_vxx_128B"(%A, %B, %C, %D) : (vector<64 x i32>, vector<64 x i32>, vector<64 x i32>, i32) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vabs_hf_128B
func.func @vabs_hf_128B(%A : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vabs.hf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vabs_hf_128B"(%A) : (vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vabs_sf_128B
func.func @vabs_sf_128B(%A : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vabs.sf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vabs_sf_128B"(%A) : (vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vadd_hf_128B
func.func @vadd_hf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vadd.hf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vadd_hf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vadd_hf_hf_128B
func.func @vadd_hf_hf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vadd.hf.hf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vadd_hf_hf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vadd_qf16_128B
func.func @vadd_qf16_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vadd.qf16.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vadd_qf16_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vadd_qf16_mix_128B
func.func @vadd_qf16_mix_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vadd.qf16.mix.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vadd_qf16_mix_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vadd_qf32_128B
func.func @vadd_qf32_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vadd.qf32.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vadd_qf32_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vadd_qf32_mix_128B
func.func @vadd_qf32_mix_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vadd.qf32.mix.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vadd_qf32_mix_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vadd_sf_128B
func.func @vadd_sf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vadd.sf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vadd_sf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vadd_sf_hf_128B
func.func @vadd_sf_hf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vadd.sf.hf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vadd_sf_hf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vadd_sf_sf_128B
func.func @vadd_sf_sf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vadd.sf.sf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vadd_sf_sf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vassign_fp_128B
func.func @vassign_fp_128B(%A : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vassign.fp.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vassign_fp_128B"(%A) : (vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vconv_hf_qf16_128B
func.func @vconv_hf_qf16_128B(%A : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vconv.hf.qf16.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vconv_hf_qf16_128B"(%A) : (vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vconv_hf_qf32_128B
func.func @vconv_hf_qf32_128B(%A : vector<64 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vconv.hf.qf32.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vconv_hf_qf32_128B"(%A) : (vector<64 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vconv_sf_qf32_128B
func.func @vconv_sf_qf32_128B(%A : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vconv.sf.qf32.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vconv_sf_qf32_128B"(%A) : (vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vcvt_b_hf_128B
func.func @vcvt_b_hf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vcvt.b.hf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vcvt_b_hf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vcvt_h_hf_128B
func.func @vcvt_h_hf_128B(%A : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vcvt.h.hf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vcvt_h_hf_128B"(%A) : (vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vcvt_hf_b_128B
func.func @vcvt_hf_b_128B(%A : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vcvt.hf.b.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vcvt_hf_b_128B"(%A) : (vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vcvt_hf_h_128B
func.func @vcvt_hf_h_128B(%A : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vcvt.hf.h.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vcvt_hf_h_128B"(%A) : (vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vcvt_hf_sf_128B
func.func @vcvt_hf_sf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vcvt.hf.sf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vcvt_hf_sf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vcvt_hf_ub_128B
func.func @vcvt_hf_ub_128B(%A : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vcvt.hf.ub.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vcvt_hf_ub_128B"(%A) : (vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vcvt_hf_uh_128B
func.func @vcvt_hf_uh_128B(%A : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vcvt.hf.uh.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vcvt_hf_uh_128B"(%A) : (vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vcvt_sf_hf_128B
func.func @vcvt_sf_hf_128B(%A : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vcvt.sf.hf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vcvt_sf_hf_128B"(%A) : (vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vcvt_ub_hf_128B
func.func @vcvt_ub_hf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vcvt.ub.hf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vcvt_ub_hf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vcvt_uh_hf_128B
func.func @vcvt_uh_hf_128B(%A : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vcvt.uh.hf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vcvt_uh_hf_128B"(%A) : (vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vdmpy_sf_hf_128B
func.func @vdmpy_sf_hf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vdmpy.sf.hf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vdmpy_sf_hf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vdmpy_sf_hf_acc_128B
func.func @vdmpy_sf_hf_acc_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vdmpy.sf.hf.acc.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vdmpy_sf_hf_acc_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vfmax_hf_128B
func.func @vfmax_hf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vfmax.hf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vfmax_hf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vfmax_sf_128B
func.func @vfmax_sf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vfmax.sf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vfmax_sf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vfmin_hf_128B
func.func @vfmin_hf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vfmin.hf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vfmin_hf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vfmin_sf_128B
func.func @vfmin_sf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vfmin.sf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vfmin_sf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vfneg_hf_128B
func.func @vfneg_hf_128B(%A : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vfneg.hf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vfneg_hf_128B"(%A) : (vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vfneg_sf_128B
func.func @vfneg_sf_128B(%A : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vfneg.sf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vfneg_sf_128B"(%A) : (vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vgthf_128B
func.func @vgthf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : vector<32xi32>, %D : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vgthf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vgthf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %C, %D) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vgthf_and_128B
func.func @vgthf_and_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : vector<32xi32>, %E : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vgthf.and.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vgthf_and_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %D, %E) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vgthf_or_128B
func.func @vgthf_or_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : vector<32xi32>, %E : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vgthf.or.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vgthf_or_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %D, %E) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vgthf_xor_128B
func.func @vgthf_xor_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : vector<32xi32>, %E : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vgthf.xor.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vgthf_xor_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %D, %E) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vgtsf_128B
func.func @vgtsf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : vector<32xi32>, %D : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vgtsf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vgtsf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %C, %D) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vgtsf_and_128B
func.func @vgtsf_and_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : vector<32xi32>, %E : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vgtsf.and.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vgtsf_and_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %D, %E) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vgtsf_or_128B
func.func @vgtsf_or_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : vector<32xi32>, %E : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vgtsf.or.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vgtsf_or_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %D, %E) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vgtsf_xor_128B
func.func @vgtsf_xor_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : vector<32xi32>, %E : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vgtsf.xor.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vgtsf_xor_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %D, %E) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vmax_hf_128B
func.func @vmax_hf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmax.hf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmax_hf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmax_sf_128B
func.func @vmax_sf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmax.sf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmax_sf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmin_hf_128B
func.func @vmin_hf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmin.hf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmin_hf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmin_sf_128B
func.func @vmin_sf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmin.sf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmin_sf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmpy_hf_hf_128B
func.func @vmpy_hf_hf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmpy.hf.hf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpy_hf_hf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmpy_hf_hf_acc_128B
func.func @vmpy_hf_hf_acc_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmpy.hf.hf.acc.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpy_hf_hf_acc_128B"(%A, %B, %C) : (vector<32 x i32>, vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmpy_qf16_128B
func.func @vmpy_qf16_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmpy.qf16.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpy_qf16_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmpy_qf16_hf_128B
func.func @vmpy_qf16_hf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmpy.qf16.hf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpy_qf16_hf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmpy_qf16_mix_hf_128B
func.func @vmpy_qf16_mix_hf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmpy.qf16.mix.hf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpy_qf16_mix_hf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmpy_qf32_128B
func.func @vmpy_qf32_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpy_qf32_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpy_qf32_hf_128B
func.func @vmpy_qf32_hf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpy.qf32.hf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpy_qf32_hf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpy_qf32_mix_hf_128B
func.func @vmpy_qf32_mix_hf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpy.qf32.mix.hf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpy_qf32_mix_hf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpy_qf32_qf16_128B
func.func @vmpy_qf32_qf16_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpy.qf32.qf16.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpy_qf32_qf16_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmpy_qf32_sf_128B
func.func @vmpy_qf32_sf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.sf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpy_qf32_sf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpy_sf_hf_128B
func.func @vmpy_sf_hf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpy.sf.hf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpy_sf_hf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpy_sf_hf_acc_128B
func.func @vmpy_sf_hf_acc_128B(%A : vector<64 x i32>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpy.sf.hf.acc.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpy_sf_hf_acc_128B"(%A, %B, %C) : (vector<64 x i32>, vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmpy_sf_sf_128B
func.func @vmpy_sf_sf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmpy.sf.sf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpy_sf_sf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vsub_hf_128B
func.func @vsub_hf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vsub.hf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsub_hf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vsub_hf_hf_128B
func.func @vsub_hf_hf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vsub.hf.hf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsub_hf_hf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vsub_qf16_128B
func.func @vsub_qf16_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vsub.qf16.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsub_qf16_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vsub_qf16_mix_128B
func.func @vsub_qf16_mix_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vsub.qf16.mix.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsub_qf16_mix_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vsub_qf32_128B
func.func @vsub_qf32_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vsub.qf32.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsub_qf32_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vsub_qf32_mix_128B
func.func @vsub_qf32_mix_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vsub.qf32.mix.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsub_qf32_mix_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vsub_sf_128B
func.func @vsub_sf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vsub.sf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsub_sf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vsub_sf_hf_128B
func.func @vsub_sf_hf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vsub.sf.hf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsub_sf_hf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vsub_sf_sf_128B
func.func @vsub_sf_sf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vsub.sf.sf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsub_sf_sf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vasrvuhubrndsat_128B
func.func @vasrvuhubrndsat_128B(%A : vector<64 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vasrvuhubrndsat.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vasrvuhubrndsat_128B"(%A, %B) : (vector<64 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vasrvuhubsat_128B
func.func @vasrvuhubsat_128B(%A : vector<64 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vasrvuhubsat.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vasrvuhubsat_128B"(%A, %B) : (vector<64 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vasrvwuhrndsat_128B
func.func @vasrvwuhrndsat_128B(%A : vector<64 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vasrvwuhrndsat.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vasrvwuhrndsat_128B"(%A, %B) : (vector<64 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vasrvwuhsat_128B
func.func @vasrvwuhsat_128B(%A : vector<64 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vasrvwuhsat.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vasrvwuhsat_128B"(%A, %B) : (vector<64 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmpyuhvs_128B
func.func @vmpyuhvs_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmpyuhvs.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpyuhvs_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vadd_sf_bf_128B
func.func @vadd_sf_bf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vadd.sf.bf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vadd_sf_bf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <32 x i32> @vconv_h_hf_128B
func.func @vconv_h_hf_128B(%A : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vconv.h.hf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vconv_h_hf_128B"(%A) : (vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vconv_hf_h_128B
func.func @vconv_hf_h_128B(%A : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vconv.hf.h.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vconv_hf_h_128B"(%A) : (vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vconv_sf_w_128B
func.func @vconv_sf_w_128B(%A : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vconv.sf.w.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vconv_sf_w_128B"(%A) : (vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vconv_w_sf_128B
func.func @vconv_w_sf_128B(%A : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vconv.w.sf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vconv_w_sf_128B"(%A) : (vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vcvt_bf_sf_128B
func.func @vcvt_bf_sf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vcvt.bf.sf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vcvt_bf_sf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vgtbf_128B
func.func @vgtbf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>, %C : vector<32xi32>, %D : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vgtbf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vgtbf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %C, %D) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vgtbf_and_128B
func.func @vgtbf_and_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : vector<32xi32>, %E : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vgtbf.and.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vgtbf_and_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %D, %E) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vgtbf_or_128B
func.func @vgtbf_or_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : vector<32xi32>, %E : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vgtbf.or.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vgtbf_or_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %D, %E) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vgtbf_xor_128B
func.func @vgtbf_xor_128B(%A : vector<128 x i1>, %B : vector<32 x i32>, %C : vector<32 x i32>, %D : vector<32xi32>, %E : vector<32xi32>) -> vector<32xi32> {
    // CHECK: call <128 x i1> @llvm.hexagon.V6.vgtbf.xor.128B(
    // CHECK-SAME: <128 x i1> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vgtbf_xor_128B"(%A, %B, %C) : (vector<128 x i1>, vector<32 x i32>, vector<32 x i32>) -> vector<128 x i1>
    %1 = "llhvx.intr.vaddbq_128B"(%0, %D, %E) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    return %1 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i32> @vmax_bf_128B
func.func @vmax_bf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmax.bf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmax_bf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <32 x i32> @vmin_bf_128B
func.func @vmin_bf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<32 x i32> {
    // CHECK: call <32 x i32> @llvm.hexagon.V6.vmin.bf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmin_bf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
    return %0 : vector<32 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpy_sf_bf_128B
func.func @vmpy_sf_bf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpy.sf.bf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpy_sf_bf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vmpy_sf_bf_acc_128B
func.func @vmpy_sf_bf_acc_128B(%A : vector<64 x i32>, %B : vector<32 x i32>, %C : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vmpy.sf.bf.acc.128B(
    // CHECK-SAME: <64 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vmpy_sf_bf_acc_128B"(%A, %B, %C) : (vector<64 x i32>, vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

// CHECK-LABEL: define <64 x i32> @vsub_sf_bf_128B
func.func @vsub_sf_bf_128B(%A : vector<32 x i32>, %B : vector<32 x i32>) -> vector<64 x i32> {
    // CHECK: call <64 x i32> @llvm.hexagon.V6.vsub.sf.bf.128B(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}},
    // CHECK-SAME: <32 x i32> %{{[0-9]+}})
    %0 = "llhvx.intr.vsub_sf_bf_128B"(%A, %B) : (vector<32 x i32>, vector<32 x i32>) -> vector<64 x i32>
    return %0 : vector<64 x i32>
}

