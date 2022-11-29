import Foundation
import Glibc
import PythonKit

let pickletools = Python.import("pickletools")

enum OpError: Error {
  case endOfFile
  case value
}

protocol ArgumentDescriptor {
  var name: String { get }
  var n: Int { get }
  mutating func read(_: UnsafeMutablePointer<FILE>) -> Result<Any, Error>
}

enum Argument {
  struct UInt1: ArgumentDescriptor {
    var name: String { "uint1" }
    var n: Int { 1 }
    func read(_ handle: UnsafeMutablePointer<FILE>) -> Result<Any, Error> {
      var result: Swift.UInt8 = 0
      let len = fread(&result, 1, 1, handle)
      guard len >= 1 else { return .failure(OpError.endOfFile) }
      return .success(result as Any)
    }
  }

  struct UInt2: ArgumentDescriptor {
    var name: String { "uint2" }
    var n: Int { 2 }
    func read(_ handle: UnsafeMutablePointer<FILE>) -> Result<Any, Error> {
      var result: UInt16 = 0
      let len = fread(&result, 2, 1, handle)
      guard len >= 1 else { return .failure(OpError.endOfFile) }
      return .success(result as Any)
    }
  }

  struct UInt4: ArgumentDescriptor {
    var name: String { "uint4" }
    var n: Int { 4 }
    func read(_ handle: UnsafeMutablePointer<FILE>) -> Result<Any, Error> {
      var result: UInt32 = 0
      let len = fread(&result, 4, 1, handle)
      guard len >= 1 else { return .failure(OpError.endOfFile) }
      return .success(result as Any)
    }
  }

  struct UInt8: ArgumentDescriptor {
    var name: String { "uint8" }
    var n: Int { 8 }
    func read(_ handle: UnsafeMutablePointer<FILE>) -> Result<Any, Error> {
      var result: UInt64 = 0
      let len = fread(&result, 8, 1, handle)
      guard len >= 1 else { return .failure(OpError.endOfFile) }
      return .success(result as Any)
    }
  }

  struct Int4: ArgumentDescriptor {
    var name: String { "int4" }
    var n: Int { 4 }
    func read(_ handle: UnsafeMutablePointer<FILE>) -> Result<Any, Error> {
      var result: Int32 = 0
      let len = fread(&result, 4, 1, handle)
      guard len >= 1 else { return .failure(OpError.endOfFile) }
      return .success(result as Any)
    }
  }

  struct Long1: ArgumentDescriptor {
    var name: String { "long1" }
    var n: Int = 0
    mutating func read(_ handle: UnsafeMutablePointer<FILE>) -> Result<Any, Error> {
      var n1: Swift.UInt8 = 0
      let len1 = fread(&n1, 1, 1, handle)
      guard len1 >= 1 else { return .failure(OpError.endOfFile) }
      n = Int(n1)
      guard n > 0 else { return .success(Data() as Any) }
      let buffer = UnsafeMutablePointer<CChar>.allocate(capacity: n)
      let len2 = fread(buffer, 1, n, handle)
      defer {
        buffer.deallocate()
      }
      guard len2 >= n else { return .failure(OpError.endOfFile) }
      let result = Data(bytes: buffer, count: n)
      return .success(result as Any)
    }
  }

  struct Long4: ArgumentDescriptor {
    var name: String { "long4" }
    var n: Int = 0
    mutating func read(_ handle: UnsafeMutablePointer<FILE>) -> Result<Any, Error> {
      var n4: UInt32 = 0
      let len1 = fread(&n4, 4, 1, handle)
      guard len1 >= 1 else { return .failure(OpError.endOfFile) }
      n = Int(n4)
      guard n > 0 else { return .success(Data() as Any) }
      let buffer = UnsafeMutablePointer<CChar>.allocate(capacity: n)
      let len2 = fread(buffer, 1, n, handle)
      defer {
        buffer.deallocate()
      }
      guard len2 >= n else { return .failure(OpError.endOfFile) }
      let result = Data(bytes: buffer, count: n)
      return .success(result as Any)
    }
  }

  struct Float8: ArgumentDescriptor {
    var name: String { "float8" }
    var n: Int { 8 }
    func read(_ handle: UnsafeMutablePointer<FILE>) -> Result<Any, Error> {
      var result: Float64 = 0
      let len = fread(&result, 8, 1, handle)
      guard len >= 1 else { return .failure(OpError.endOfFile) }
      return .success(result as Any)
    }
  }

  struct UnicodeString1: ArgumentDescriptor {
    var name: String { "unicodestring1" }
    var n: Int = 0
    mutating func read(_ handle: UnsafeMutablePointer<FILE>) -> Result<Any, Error> {
      var n1: Swift.UInt8 = 0
      let len1 = fread(&n1, 1, 1, handle)
      guard len1 >= 1 else { return .failure(OpError.endOfFile) }
      n = Int(n1)
      let buffer = UnsafeMutablePointer<CChar>.allocate(capacity: n + 1)
      buffer[n] = 0
      let len2 = fread(buffer, 1, n, handle)
      defer {
        buffer.deallocate()
      }
      guard len2 >= n else { return .failure(OpError.endOfFile) }
      guard let result = String(utf8String: UnsafePointer(buffer)) else {
        return .failure(OpError.value)
      }
      return .success(result as Any)
    }
  }

  struct UnicodeString4: ArgumentDescriptor {
    var name: String { "unicodestring4" }
    var n: Int = 0
    mutating func read(_ handle: UnsafeMutablePointer<FILE>) -> Result<Any, Error> {
      var n1: UInt32 = 0
      let len1 = fread(&n1, 4, 1, handle)
      guard len1 >= 1 else { return .failure(OpError.endOfFile) }
      n = Int(n1)
      let buffer = UnsafeMutablePointer<CChar>.allocate(capacity: n + 1)
      buffer[n] = 0
      let len2 = fread(buffer, 1, n, handle)
      defer {
        buffer.deallocate()
      }
      guard len2 >= n else { return .failure(OpError.endOfFile) }
      guard let result = String(utf8String: UnsafePointer(buffer)) else {
        return .failure(OpError.value)
      }
      return .success(result as Any)
    }
  }

  struct UnicodeString8: ArgumentDescriptor {
    var name: String { "unicodestring8" }
    var n: Int = 0
    mutating func read(_ handle: UnsafeMutablePointer<FILE>) -> Result<Any, Error> {
      var n1: UInt64 = 0
      let len1 = fread(&n1, 8, 1, handle)
      guard len1 >= 1 else { return .failure(OpError.endOfFile) }
      n = Int(n1)
      let buffer = UnsafeMutablePointer<CChar>.allocate(capacity: n + 1)
      buffer[n] = 0
      let len2 = fread(buffer, 1, n, handle)
      defer {
        buffer.deallocate()
      }
      guard len2 >= n else { return .failure(OpError.endOfFile) }
      guard let result = String(utf8String: UnsafePointer(buffer)) else {
        return .failure(OpError.value)
      }
      return .success(result as Any)
    }
  }

  struct UnicodeStringNL: ArgumentDescriptor {
    var name: String { "unicodestringnl" }
    var n: Int { -1 }
    private let stripquotes: Bool
    init(stripquotes: Bool = true) {
      self.stripquotes = stripquotes
    }
    func read(_ handle: UnsafeMutablePointer<FILE>) -> Result<Any, Error> {
      var buffer: UnsafeMutablePointer<CChar>? = nil
      var bufferSize: Int = 0
      let len = withUnsafeMutablePointer(to: &buffer) {
        getline($0, &bufferSize, handle)
      }
      guard let buffer = buffer, len > 0, bufferSize > 0 else { return .failure(OpError.endOfFile) }
      defer {
        buffer.deallocate()
      }
      guard var result = String(utf8String: UnsafePointer(buffer)) else {
        return .failure(OpError.value)
      }
      if result.hasSuffix("\n") {
        result = String(result.prefix(upTo: result.index(before: result.endIndex)))
      }
      guard stripquotes else {
        return .success(result as Any)
      }
      if result.hasPrefix("\"") {
        if !result.hasSuffix("\"") && result.count >= 2 {
          return .failure(OpError.value)
        }
        result = String(
          result[result.index(after: result.startIndex)..<result.index(before: result.endIndex)])
      }
      if result.hasPrefix("'") {
        if !result.hasSuffix("'") && result.count >= 2 {
          return .failure(OpError.value)
        }
        result = String(
          result[result.index(after: result.startIndex)..<result.index(before: result.endIndex)])
      }
      return .success(result as Any)
    }
  }

  struct String1: ArgumentDescriptor {
    var name: String { "string1" }
    var n: Int = 0
    mutating func read(_ handle: UnsafeMutablePointer<FILE>) -> Result<Any, Error> {
      var n1: Swift.UInt8 = 0
      let len1 = fread(&n1, 1, 1, handle)
      guard len1 >= 1 else { return .failure(OpError.endOfFile) }
      n = Int(n1)
      let buffer = UnsafeMutablePointer<CChar>.allocate(capacity: n + 1)
      buffer[n] = 0
      let len2 = fread(buffer, 1, n, handle)
      defer {
        buffer.deallocate()
      }
      guard len2 >= n else { return .failure(OpError.endOfFile) }
      let result = String(cString: UnsafePointer(buffer))
      return .success(result as Any)
    }
  }

  struct String4: ArgumentDescriptor {
    var name: String { "string4" }
    var n: Int = 0
    mutating func read(_ handle: UnsafeMutablePointer<FILE>) -> Result<Any, Error> {
      var n1: UInt32 = 0
      let len1 = fread(&n1, 4, 1, handle)
      guard len1 >= 1 else { return .failure(OpError.endOfFile) }
      n = Int(n1)
      let buffer = UnsafeMutablePointer<CChar>.allocate(capacity: n + 1)
      buffer[n] = 0
      let len2 = fread(buffer, 1, n, handle)
      defer {
        buffer.deallocate()
      }
      guard len2 >= n else { return .failure(OpError.endOfFile) }
      let result = String(cString: UnsafePointer(buffer))
      return .success(result as Any)
    }
  }

  struct Bytes1: ArgumentDescriptor {
    var name: String { "bytes1" }
    var n: Int = 0
    mutating func read(_ handle: UnsafeMutablePointer<FILE>) -> Result<Any, Error> {
      var n1: Swift.UInt8 = 0
      let len1 = fread(&n1, 1, 1, handle)
      guard len1 >= 1 else { return .failure(OpError.endOfFile) }
      n = Int(n1)
      guard n > 0 else { return .success(Data() as Any) }
      let buffer = UnsafeMutablePointer<CChar>.allocate(capacity: n)
      let len2 = fread(buffer, 1, n, handle)
      defer {
        buffer.deallocate()
      }
      guard len2 >= n else { return .failure(OpError.endOfFile) }
      let result = Data(bytes: buffer, count: n)
      return .success(result as Any)
    }
  }

  struct Bytes4: ArgumentDescriptor {
    var name: String { "bytes4" }
    var n: Int = 0
    mutating func read(_ handle: UnsafeMutablePointer<FILE>) -> Result<Any, Error> {
      var n4: UInt32 = 0
      let len1 = fread(&n4, 4, 1, handle)
      guard len1 >= 1 else { return .failure(OpError.endOfFile) }
      n = Int(n4)
      guard n > 0 else { return .success(Data() as Any) }
      let buffer = UnsafeMutablePointer<CChar>.allocate(capacity: n)
      let len2 = fread(buffer, 1, n, handle)
      defer {
        buffer.deallocate()
      }
      guard len2 >= n else { return .failure(OpError.endOfFile) }
      let result = Data(bytes: buffer, count: n)
      return .success(result as Any)
    }
  }

  struct Bytes8: ArgumentDescriptor {
    var name: String { "bytes8" }
    var n: Int = 0
    mutating func read(_ handle: UnsafeMutablePointer<FILE>) -> Result<Any, Error> {
      var n8: UInt64 = 0
      let len1 = fread(&n8, 8, 1, handle)
      guard len1 >= 1 else { return .failure(OpError.endOfFile) }
      n = Int(n8)
      guard n > 0 else { return .success(Data() as Any) }
      let buffer = UnsafeMutablePointer<CChar>.allocate(capacity: n)
      let len2 = fread(buffer, 1, n, handle)
      defer {
        buffer.deallocate()
      }
      guard len2 >= n else { return .failure(OpError.endOfFile) }
      let result = Data(bytes: buffer, count: n)
      return .success(result as Any)
    }
  }

  struct ByteArray8: ArgumentDescriptor {
    var name: String { "bytearray8" }
    var n: Int = 0
    mutating func read(_ handle: UnsafeMutablePointer<FILE>) -> Result<Any, Error> {
      var n8: UInt64 = 0
      let len1 = fread(&n8, 8, 1, handle)
      guard len1 >= 1 else { return .failure(OpError.endOfFile) }
      n = Int(n8)
      guard n > 0 else { return .success(Data() as Any) }
      let buffer = UnsafeMutablePointer<CChar>.allocate(capacity: n)
      let len2 = fread(buffer, 1, n, handle)
      defer {
        buffer.deallocate()
      }
      guard len2 >= n else { return .failure(OpError.endOfFile) }
      let result = Data(bytes: buffer, count: n)
      return .success(result as Any)
    }
  }

  struct StringNL: ArgumentDescriptor {
    var name: String { "stringnl" }
    var n: Int { -1 }
    private let stripquotes: Bool
    init(stripquotes: Bool = true) {
      self.stripquotes = stripquotes
    }
    func read(_ handle: UnsafeMutablePointer<FILE>) -> Result<Any, Error> {
      var buffer: UnsafeMutablePointer<CChar>? = nil
      var bufferSize: Int = 0
      let len = withUnsafeMutablePointer(to: &buffer) {
        getline($0, &bufferSize, handle)
      }
      guard let buffer = buffer, len > 0, bufferSize > 0 else { return .failure(OpError.endOfFile) }
      defer {
        buffer.deallocate()
      }
      var result = String(cString: UnsafePointer(buffer))
      if result.hasSuffix("\n") {
        result = String(result.prefix(upTo: result.index(before: result.endIndex)))
      }
      guard stripquotes else {
        return .success(result as Any)
      }
      if result.hasPrefix("\"") {
        if !result.hasSuffix("\"") && result.count >= 2 {
          return .failure(OpError.value)
        }
        result = String(
          result[result.index(after: result.startIndex)..<result.index(before: result.endIndex)])
      }
      if result.hasPrefix("'") {
        if !result.hasSuffix("'") && result.count >= 2 {
          return .failure(OpError.value)
        }
        result = String(
          result[result.index(after: result.startIndex)..<result.index(before: result.endIndex)])
      }
      return .success(result as Any)
    }
  }

  struct StringNLPair: ArgumentDescriptor {
    var name: String { "stringnl_noescape_pair" }
    var n: Int { -1 }
    var first: StringNL
    var second: StringNL
    init(stripquotes: Bool = true) {
      first = StringNL(stripquotes: stripquotes)
      second = StringNL(stripquotes: stripquotes)
    }
    func read(_ handle: UnsafeMutablePointer<FILE>) -> Result<Any, Error> {
      return first.read(handle).flatMap { firstSuccess in
        second.read(handle).map {
          (firstSuccess, $0) as Any
        }
      }
    }
  }

  struct DecimalNLShort: ArgumentDescriptor {
    var name: String { "decimalnl_short" }
    var n: Int { -1 }
    func read(_ handle: UnsafeMutablePointer<FILE>) -> Result<Any, Error> {
      let stringnl = StringNL(stripquotes: false)
      return stringnl.read(handle).flatMap {
        let string = $0 as! String
        guard let short = Int(string) else { return .failure(OpError.value) }
        return .success(short as Any)
      }
    }
  }

  struct DecimalNLLong: ArgumentDescriptor {
    var name: String { "decimalnl_long" }
    var n: Int { -1 }
    func read(_ handle: UnsafeMutablePointer<FILE>) -> Result<Any, Error> {
      let stringnl = StringNL(stripquotes: false)
      return stringnl.read(handle).flatMap {
        let string = $0 as! String
        guard string.hasSuffix("L") else {
          guard let long = Int(string) else { return .failure(OpError.value) }
          return .success(long as Any)
        }
        guard let long = Int(String(string.dropLast())) else { return .failure(OpError.value) }
        return .success(long as Any)
      }
    }
  }

  struct FloatNL: ArgumentDescriptor {
    var name: String { "floatnl" }
    var n: Int { -1 }
    func read(_ handle: UnsafeMutablePointer<FILE>) -> Result<Any, Error> {
      let stringnl = StringNL(stripquotes: false)
      return stringnl.read(handle).flatMap {
        let string = $0 as! String
        guard let short = Float64(string) else { return .failure(OpError.value) }
        return .success(short as Any)
      }
    }
  }
}

struct Instruction {
  var name: String
  var code: UInt8
  var arg: ArgumentDescriptor?
}

let instructions: [Instruction] = [
  Instruction(name: "INT", code: 0x49, arg: Argument.DecimalNLShort()),
  Instruction(name: "BININT", code: 0x4A, arg: Argument.Int4()),
  Instruction(name: "BININT1", code: 0x4B, arg: Argument.UInt1()),
  Instruction(name: "BININT2", code: 0x4D, arg: Argument.UInt2()),
  Instruction(name: "LONG", code: 0x4C, arg: Argument.DecimalNLLong()),
  Instruction(name: "LONG1", code: 0x8A, arg: Argument.Long1()),
  Instruction(name: "LONG4", code: 0x8B, arg: Argument.Long4()),
  Instruction(name: "STRING", code: 0x53, arg: Argument.StringNL(stripquotes: true)),
  Instruction(name: "BINSTRING", code: 0x54, arg: Argument.String4()),
  Instruction(name: "SHORT_BINSTRING", code: 0x55, arg: Argument.String1()),
  Instruction(name: "BINBYTES", code: 0x42, arg: Argument.Bytes4()),
  Instruction(name: "SHORT_BINBYTES", code: 0x43, arg: Argument.Bytes1()),
  Instruction(name: "BINBYTES8", code: 0x8e, arg: Argument.Bytes8()),
  Instruction(name: "BYTEARRAY8", code: 0x96, arg: Argument.ByteArray8()),
  Instruction(name: "NEXT_BUFFER", code: 0x97, arg: nil),
  Instruction(name: "READONLY_BUFFER", code: 0x98, arg: nil),
  Instruction(name: "NONE", code: 0x4E, arg: nil),
  Instruction(name: "NEWTRUE", code: 0x88, arg: nil),
  Instruction(name: "NEWFALSE", code: 0x89, arg: nil),
  Instruction(name: "UNICODE", code: 0x56, arg: Argument.UnicodeStringNL()),
  Instruction(name: "SHORT_BINUNICODE", code: 0x8c, arg: Argument.UnicodeString1()),
  Instruction(name: "BINUNICODE", code: 0x58, arg: Argument.UnicodeString4()),
  Instruction(name: "BINUNICODE8", code: 0x8D, arg: Argument.UnicodeString8()),
  Instruction(name: "FLOAT", code: 0x46, arg: Argument.FloatNL()),
  Instruction(name: "BINFLOAT", code: 0x47, arg: Argument.Float8()),
  Instruction(name: "EMPTY_LIST", code: 0x5D, arg: nil),
  Instruction(name: "APPEND", code: 0x61, arg: nil),
  Instruction(name: "APPENDS", code: 0x65, arg: nil),
  Instruction(name: "LIST", code: 0x6C, arg: nil),
  Instruction(name: "EMPTY_TUPLE", code: 0x29, arg: nil),
  Instruction(name: "TUPLE", code: 0x74, arg: nil),
  Instruction(name: "TUPLE1", code: 0x85, arg: nil),
  Instruction(name: "TUPLE2", code: 0x86, arg: nil),
  Instruction(name: "TUPLE3", code: 0x87, arg: nil),
  Instruction(name: "EMPTY_DICT", code: 0x7D, arg: nil),
  Instruction(name: "DICT", code: 0x64, arg: nil),
  Instruction(name: "SETITEM", code: 0x73, arg: nil),
  Instruction(name: "SETITEMS", code: 0x75, arg: nil),
  Instruction(name: "EMPTY_SET", code: 0x8F, arg: nil),
  Instruction(name: "ADDITEMS", code: 0x90, arg: nil),
  Instruction(name: "FROZENSET", code: 0x91, arg: nil),
  Instruction(name: "POP", code: 0x30, arg: nil),
  Instruction(name: "DUP", code: 0x32, arg: nil),
  Instruction(name: "MARK", code: 0x28, arg: nil),
  Instruction(name: "POP_MARK", code: 0x31, arg: nil),
  Instruction(name: "GET", code: 0x67, arg: Argument.DecimalNLShort()),
  Instruction(name: "BINGET", code: 0x68, arg: Argument.UInt1()),
  Instruction(name: "LONG_BINGET", code: 0x6A, arg: Argument.UInt4()),
  Instruction(name: "PUT", code: 0x70, arg: Argument.DecimalNLShort()),
  Instruction(name: "BINPUT", code: 0x71, arg: Argument.UInt1()),
  Instruction(name: "LONG_BINPUT", code: 0x72, arg: Argument.UInt4()),
  Instruction(name: "MEMOIZE", code: 0x94, arg: nil),
  Instruction(name: "EXT1", code: 0x82, arg: Argument.UInt1()),
  Instruction(name: "EXT2", code: 0x83, arg: Argument.UInt2()),
  Instruction(name: "EXT4", code: 0x84, arg: Argument.Int4()),
  Instruction(name: "GLOBAL", code: 0x63, arg: Argument.StringNLPair(stripquotes: false)),
  Instruction(name: "STACK_GLOBAL", code: 0x93, arg: nil),
  Instruction(name: "REDUCE", code: 0x52, arg: nil),
  Instruction(name: "BUILD", code: 0x62, arg: nil),
  Instruction(name: "INST", code: 0x69, arg: Argument.StringNLPair(stripquotes: false)),
  Instruction(name: "OBJ", code: 0x6F, arg: nil),
  Instruction(name: "NEWOBJ", code: 0x81, arg: nil),
  Instruction(name: "NEWOBJ_EX", code: 0x92, arg: nil),
  Instruction(name: "PROTO", code: 0x80, arg: Argument.UInt1()),
  Instruction(name: "STOP", code: 0x2E, arg: nil),
  Instruction(name: "FRAME", code: 0x95, arg: Argument.UInt8()),
  Instruction(name: "PERSID", code: 0x50, arg: Argument.StringNL(stripquotes: false)),
  Instruction(name: "BINPERSID", code: 0x51, arg: nil),
]

var instructionMapping = [UInt8: Instruction]()
for instruction in instructions {
  instructionMapping[instruction.code] = instruction
}

let filename = "/home/liu/workspace/swift-diffusion/archive/data.pkl"
let handle = fopen(filename, "rb")!
var opcode: UInt8 = 0
var ops = [(UInt8, Any?, Int)]()
while true {
  let pos = ftell(handle)
  let len = fread(&opcode, 1, 1, handle)
  guard len > 0, var instruction = instructionMapping[opcode] else { break }
  do {
    let arg = try instruction.arg?.read(handle).get()
    ops.append((opcode, arg, pos))
  } catch {
    break
  }
}
fclose(handle)

let pyf = Python.open(filename, "rb")
let pyops = Python.list(pickletools.genops(pyf))
print(pyops[0..<42])
