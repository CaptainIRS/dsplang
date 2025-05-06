// RUN: dsplangc-opt %s --hlhvx-to-llhvx

func.func @CastVectorToRegister.noop(%A : vector<4 x i8>) -> i32 {
  %0 = "hlhvx.CastVectorToRegister"(%A) : (vector<4 x i8>) -> i32
  func.return %0 : i32
}

func.func @CastVectorToGeneric.noop(%A : vector<32 x i32>) -> vector<32 x i32> {
  %0 = "hlhvx.CastVectorToGeneric"(%A) { length = 32 : ui32 } : (vector<32 x i32>) -> vector<32 x i32>
  func.return %0 : vector<32 x i32>
}

func.func @CastVectorToGeneric.cast(%A : vector<64 x i16>) -> vector<32 x i32> {
  %0 = "hlhvx.CastVectorToGeneric"(%A) { length = 32 : ui32 } : (vector<64 x i16>) -> vector<32 x i32>
  func.return %0 : vector<32 x i32>
}

// func.func @test(%A : vector<128 x si8>, %B : vector<128 x si8>) -> vector<128 x i8> {
//   %0 = "vector.bitcast"(%A) : (vector<128 x si8>) -> vector<128 x i8>
//   func.return %0 : vector<128 x i8>
// }

func.func @vadd.Vb.VbVb.highlevel(%A : vector<128 x si8>, %B : vector<128 x si8>) -> vector<128 x si8> {
  %0 = "hlhvx.vadd.Vb.VbVb"(%A, %B) : (vector<128 x si8>, vector<128 x si8>) -> vector<128 x si8>
  func.return %0 : vector<128 x si8>
}

func.func @vadd.Vb.VbVb.intermediate(%A : vector<128 x si8>, %B : vector<128 x si8>) -> vector<128 x si8> {
  %0 = "hlhvx.CastVectorToGeneric"(%A) { length = 32 : ui32 } : (vector<128 x si8>) -> vector<32 x i32>
  %1 = "hlhvx.CastVectorToGeneric"(%B) { length = 32 : ui32 } : (vector<128 x si8>) -> vector<32 x i32>
  %2 = "llhvx.intr.vaddb_128B"(%0, %1) : (vector<32 x i32>, vector<32 x i32>) -> vector<32 x i32>
  %3 = "hlhvx.CastVectorFromGeneric"(%2) {length = 128 : ui32, type = "si8" } : (vector<32 x i32>) -> vector<128 x si8>
  func.return %3 : vector<128 x si8>
}

func.func @vmpy.Wuh.VubRub.highlevel() {
  %A = arith.constant dense<0> : vector<128 x ui8>
  %B = arith.constant dense<0> : vector<4 x ui8>
  %0 = "hlhvx.vmpy.Wuh.VubRub"(%A, %B) : (vector<128 x ui8>, vector<4 x ui8>) -> vector<128 x ui16>
  func.return
}

func.func @vadd.Vw.VwVwQ.carry.highlevel(%A : vector<32 x si32>, %B : vector<32 x si32>, %C : vector<128 x i1>) -> vector<32 x si32> {
  %0 = "hlhvx.vadd.Vw.VwVwQ.carry"(%A, %B, %C) : (vector<32 x si32>, vector<32 x si32>, vector<128 x i1>) -> vector<32 x si32>
  func.return %0 : vector<32 x si32>
}
