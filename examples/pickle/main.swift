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
      var result: UInt8 = 0
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

  struct String1: ArgumentDescriptor {
    var name: String { "string1" }
    var n: Int = 0
    mutating func read(_ handle: UnsafeMutablePointer<FILE>) -> Result<Any, Error> {
      var n1: UInt8 = 0
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
}

struct Instruction {
  var name: String
  var code: UInt8
  var arg: ArgumentDescriptor?
}

let instructions: [Instruction] = [
  Instruction(name: "PROTO", code: 0x80, arg: Argument.UInt1()),
  Instruction(name: "EMPTY_DICT", code: 0x7D, arg: nil),
  Instruction(name: "BINPUT", code: 0x71, arg: Argument.UInt1()),
  Instruction(name: "MARK", code: 0x28, arg: nil),
  Instruction(name: "BINUNICODE", code: 0x58, arg: Argument.UnicodeString4()),
  Instruction(name: "BININT1", code: 0x4B, arg: Argument.UInt1()),
  Instruction(name: "BININT", code: 0x4A, arg: Argument.Int4()),
  Instruction(name: "BININT2", code: 0x4D, arg: Argument.UInt2()),
  Instruction(name: "GLOBAL", code: 0x63, arg: Argument.StringNLPair(stripquotes: false)),
  Instruction(name: "TUPLE", code: 0x74, arg: nil),
  Instruction(name: "BINPERSID", code: 0x51, arg: nil),
  Instruction(name: "TUPLE1", code: 0x85, arg: nil),
  Instruction(name: "NEWFALSE", code: 0x89, arg: nil),
  Instruction(name: "EMPTY_TUPLE", code: 0x29, arg: nil),
  Instruction(name: "REDUCE", code: 0x52, arg: nil),
  Instruction(name: "BINGET", code: 0x68, arg: Argument.UInt1()),
  Instruction(name: "TUPLE2", code: 0x86, arg: nil),
  Instruction(name: "LONG_BINPUT", code: 0x72, arg: Argument.UInt4()),
  Instruction(name: "SETITEMS", code: 0x75, arg: nil),
  Instruction(name: "NONE", code: 0x4E, arg: nil),
  Instruction(name: "SETITEM", code: 0x73, arg: nil),
  Instruction(name: "EMPTY_LIST", code: 0x5D, arg: nil),
  Instruction(name: "BINFLOAT", code: 0x47, arg: Argument.Float8()),
  Instruction(name: "APPEND", code: 0x61, arg: nil),
  Instruction(name: "NEWTRUE", code: 0x88, arg: nil),
  Instruction(name: "TUPLE3", code: 0x87, arg: nil),
  Instruction(name: "BUILD", code: 0x62, arg: nil),
  Instruction(name: "STOP", code: 0x2E, arg: nil),
  Instruction(name: "SHORT_BINSTRING", code: 0x55, arg: Argument.String1()),
  Instruction(name: "PERSID", code: 0x50, arg: Argument.StringNL(stripquotes: false)),
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
