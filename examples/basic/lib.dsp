def dsplanglib(c<10000, f32>, d<10000, f32>, e*<10000, f32>) {
  var f<10000, f32> = c * d;
  var g<10000, f32> = c + d;
  e = f * g;
}
