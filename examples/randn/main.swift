// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation

/// A random source consistent with NumPy
///
///  This implementation matches:
///  [NumPy's older randomkit.c](https://github.com/numpy/numpy/blob/v1.0/numpy/random/mtrand/randomkit.c)
///
public struct TorchRandomSource: RandomNumberGenerator {

  struct State {
    var key = [UInt32](repeating: 0, count: 624)
    var pos: Int = 0
    var nextGauss: Double? = nil
  }

  var state: State

  /// Initialize with a random seed
  ///
  /// - Parameters
  ///     - seed: Seed for underlying Mersenne Twister 19937 generator
  /// - Returns random source
  public init(seed: UInt32) {
    state = .init()
    var s = seed & 0xffff_ffff
    for i in 0..<state.key.count {
      state.key[i] = s
      s = UInt32((UInt64(1_812_433_253) * UInt64(s ^ (s >> 30)) + UInt64(i) + 1) & 0xffff_ffff)
    }
    state.pos = state.key.count
    state.nextGauss = nil
  }

  /// Generate next UInt32 using fast 32bit Mersenne Twister
  mutating func nextUInt32() -> UInt32 {
    let n = 624
    let m = 397
    let matrixA: UInt64 = 0x9908_b0df
    let upperMask: UInt32 = 0x8000_0000
    let lowerMask: UInt32 = 0x7fff_ffff

    var y: UInt32
    if state.pos == state.key.count {
      for i in 0..<(n - m) {
        y = (state.key[i] & upperMask) | (state.key[i + 1] & lowerMask)
        state.key[i] = state.key[i + m] ^ (y >> 1) ^ UInt32((UInt64(~(y & 1)) + 1) & matrixA)
      }
      for i in (n - m)..<(n - 1) {
        y = (state.key[i] & upperMask) | (state.key[i + 1] & lowerMask)
        state.key[i] = state.key[i + (m - n)] ^ (y >> 1) ^ UInt32((UInt64(~(y & 1)) + 1) & matrixA)
      }
      y = (state.key[n - 1] & upperMask) | (state.key[0] & lowerMask)
      state.key[n - 1] = state.key[m - 1] ^ (y >> 1) ^ UInt32((UInt64(~(y & 1)) + 1) & matrixA)
      state.pos = 0
    }
    y = state.key[state.pos]
    state.pos += 1

    y ^= (y >> 11)
    y ^= (y << 7) & 0x9d2c_5680
    y ^= (y << 15) & 0xefc6_0000
    y ^= (y >> 18)

    return y
  }

  public mutating func next() -> UInt64 {
    let high = nextUInt32()
    let low = nextUInt32()
    return (UInt64(high) << 32) | UInt64(low)
  }

  /// Generate next random double value
  mutating func nextDouble() -> Double {
    let a = next()
    return Double(a & 9_007_199_254_740_991) * (1.0 / 9007199254740992.0)
  }

  /// Generate next random float value
  mutating func nextFloat() -> Float {
    let a = nextUInt32()
    return Float(a & 16_777_215) * (1.0 / 16777216.0)
  }

  /// Generate next random value from a standard normal
  mutating func nextGauss() -> Double {
    if let nextGauss = state.nextGauss {
      state.nextGauss = nil
      return nextGauss
    }
    // Box-Muller transform
    let u1: Double = nextDouble()
    let u2: Double = 1 - nextDouble()
    let radius = sqrt(-2.0 * log(u2))
    let theta = 2.0 * .pi * u1
    state.nextGauss = radius * sin(theta)
    return radius * cos(theta)
  }

  /// Generates a random value from a normal distribution with given mean and standard deviation.
  mutating func nextNormal(mean: Double = 0.0, stdev: Double = 1.0) -> Double {
    nextGauss() * stdev + mean
  }

  /// Generates an array of random values from a normal distribution with given mean and standard deviation.
  public mutating func normalArrayDouble(count: Int, mean: Double = 0.0, stdev: Double = 1.0)
    -> [Double]
  {
    (0..<count).map { _ in nextNormal(mean: mean, stdev: stdev) }
  }

  public mutating func normalArray(count: Int, mean: Float = 0.0, stdev: Float = 1.0) -> [Float] {
    guard count >= 16 else {
      return normalArrayDouble(count: count, mean: Double(mean), stdev: Double(stdev)).map {
        Float($0)
      }
    }
    var data = (0..<count).map { _ in nextFloat() }
    for i in stride(from: 0, to: count - 15, by: 16) {
      for j in 0..<8 {
        let u1 = 1 - data[i + j]
        let u2 = data[i + j + 8]
        let radius = sqrt(-2.0 * log(u1))
        let theta = 2.0 * .pi * u2
        data[i + j] = radius * cos(theta) * stdev + mean
        data[i + j + 8] = radius * sin(theta) * stdev + mean
      }
    }
    if count % 16 != 0 {
      for i in (count - 16)..<count {
        data[i] = nextFloat()
      }
      let i = count - 16
      for j in 0..<8 {
        let u1 = 1 - data[i + j]
        let u2 = data[i + j + 8]
        let radius = sqrt(-2.0 * log(u1))
        let theta = 2.0 * .pi * u2
        data[i + j] = radius * cos(theta) * stdev + mean
        data[i + j + 8] = radius * sin(theta) * stdev + mean
      }
    }
    return data
  }
}

var torchRandomSource = TorchRandomSource(seed: 4)
print(torchRandomSource.normalArray(count: 17))

public struct NVRandomSource {
  public let seed: UInt64
  private var offset: UInt32

  /// Initialize with a random seed
  ///
  /// - Parameters
  ///     - seed: Seed for underlying Mersenne Twister 19937 generator
  /// - Returns random source
  public init(seed: UInt64) {
    self.seed = seed
    offset = 0
  }

  static private let philoxM: (UInt32, UInt32) = (0xD251_1F53, 0xCD9E_8D57)
  static private let philoxW: (UInt32, UInt32) = (0x9E37_79B9, 0xBB67_AE85)

  private func philox4Round(counter: inout [[UInt32]], key: [[UInt32]]) {
    for i in 0..<counter[0].count {
      let v1: UInt64 = UInt64(counter[0][i]) * UInt64(Self.philoxM.0)
      let v2: UInt64 = UInt64(counter[2][i]) * UInt64(Self.philoxM.1)
      counter[0][i] = UInt32(v2 >> 32) ^ counter[1][i] ^ key[0][i]
      counter[1][i] = UInt32(v2 & 0xffff_ffff)
      counter[2][i] = UInt32(v1 >> 32) ^ counter[3][i] ^ key[1][i]
      counter[3][i] = UInt32(v1 & 0xffff_ffff)
    }
  }

  private func philox4_32(counter: inout [[UInt32]], key: inout [[UInt32]], rounds: Int = 10) {
    for _ in 0..<(rounds - 1) {
      philox4Round(counter: &counter, key: key)
      for (i, element) in key[0].enumerated() {
        key[0][i] = element &+ Self.philoxW.0
      }
      for (i, element) in key[1].enumerated() {
        key[1][i] = element &+ Self.philoxW.1
      }
    }
    philox4Round(counter: &counter, key: key)
  }

  private func boxMuller(_ counter1: [UInt32], _ counter2: [UInt32]) -> [Float] {
    // Box-Muller transform
    return zip(counter1, counter2).map {
      let u: Double = Double($0) * 2.3283064e-10 + (2.3283064e-10 / 2)
      let v: Double = Double($1) * (2.3283064e-10 * 2.0 * .pi) + (2.3283064e-10 * .pi)
      let radius = sqrt(-2.0 * log(u))
      return Float(radius * sin(v))
    }
  }

  public mutating func normalArray(count: Int, mean: Float = 0.0, stdev: Float = 1.0) -> [Float] {
    var counter: [[UInt32]] = [
      Array(repeating: offset, count: count),
      Array(repeating: 0, count: count),
      Array(0..<UInt32(count)),
      Array(repeating: 0, count: count),
    ]
    offset += 1
    var key: [[UInt32]] = [
      Array(repeating: UInt32(seed & 0xffff_ffff), count: count),
      Array(repeating: UInt32(seed >> 32), count: count),
    ]
    philox4_32(counter: &counter, key: &key)
    return boxMuller(counter[0], counter[1])
  }
}

var nvRandomSource = NVRandomSource(seed: 0)
print(nvRandomSource.normalArray(count: 12))
