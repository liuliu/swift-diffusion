import Foundation
import Glibc

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

enum InstructionOpcode: UInt8, CustomStringConvertible {
  case INT = 0x49
  case BININT = 0x4A
  case BININT1 = 0x4B
  case BININT2 = 0x4D
  case LONG = 0x4C
  case LONG1 = 0x8A
  case LONG4 = 0x8B
  case STRING = 0x53
  case BINSTRING = 0x54
  case SHORT_BINSTRING = 0x55
  case BINBYTES = 0x42
  case SHORT_BINBYTES = 0x43
  case BINBYTES8 = 0x8e
  case BYTEARRAY8 = 0x96
  case NEXT_BUFFER = 0x97
  case READONLY_BUFFER = 0x98
  case NONE = 0x4E
  case NEWTRUE = 0x88
  case NEWFALSE = 0x89
  case UNICODE = 0x56
  case SHORT_BINUNICODE = 0x8c
  case BINUNICODE = 0x58
  case BINUNICODE8 = 0x8D
  case FLOAT = 0x46
  case BINFLOAT = 0x47
  case EMPTY_LIST = 0x5D
  case APPEND = 0x61
  case APPENDS = 0x65
  case LIST = 0x6C
  case EMPTY_TUPLE = 0x29
  case TUPLE = 0x74
  case TUPLE1 = 0x85
  case TUPLE2 = 0x86
  case TUPLE3 = 0x87
  case EMPTY_DICT = 0x7D
  case DICT = 0x64
  case SETITEM = 0x73
  case SETITEMS = 0x75
  case EMPTY_SET = 0x8F
  case ADDITEMS = 0x90
  case FROZENSET = 0x91
  case POP = 0x30
  case DUP = 0x32
  case MARK = 0x28
  case POP_MARK = 0x31
  case GET = 0x67
  case BINGET = 0x68
  case LONG_BINGET = 0x6A
  case PUT = 0x70
  case BINPUT = 0x71
  case LONG_BINPUT = 0x72
  case MEMOIZE = 0x94
  case EXT1 = 0x82
  case EXT2 = 0x83
  case EXT4 = 0x84
  case GLOBAL = 0x63
  case STACK_GLOBAL = 0x93
  case REDUCE = 0x52
  case BUILD = 0x62
  case INST = 0x69
  case OBJ = 0x6F
  case NEWOBJ = 0x81
  case NEWOBJ_EX = 0x92
  case PROTO = 0x80
  case STOP = 0x2E
  case FRAME = 0x95
  case PERSID = 0x50
  case BINPERSID = 0x51

  var description: String {
    switch self {
    case .INT:
      return "INT"
    case .BININT:
      return "BININT"
    case .BININT1:
      return "BININT1"
    case .BININT2:
      return "BININT2"
    case .LONG:
      return "LONG"
    case .LONG1:
      return "LONG1"
    case .LONG4:
      return "LONG4"
    case .STRING:
      return "STRING"
    case .BINSTRING:
      return "BINSTRING"
    case .SHORT_BINSTRING:
      return "SHORT_BINSTRING"
    case .BINBYTES:
      return "BINBYTES"
    case .SHORT_BINBYTES:
      return "SHORT_BINBYTES"
    case .BINBYTES8:
      return "BINBYTES8"
    case .BYTEARRAY8:
      return "BYTEARRAY8"
    case .NEXT_BUFFER:
      return "NEXT_BUFFER"
    case .READONLY_BUFFER:
      return "READONLY_BUFFER"
    case .NONE:
      return "NONE"
    case .NEWTRUE:
      return "NEWTRUE"
    case .NEWFALSE:
      return "NEWFALSE"
    case .UNICODE:
      return "UNICODE"
    case .SHORT_BINUNICODE:
      return "SHORT_BINUNICODE"
    case .BINUNICODE:
      return "BINUNICODE"
    case .BINUNICODE8:
      return "BINUNICODE8"
    case .FLOAT:
      return "FLOAT"
    case .BINFLOAT:
      return "BINFLOAT"
    case .EMPTY_LIST:
      return "EMPTY_LIST"
    case .APPEND:
      return "APPEND"
    case .APPENDS:
      return "APPENDS"
    case .LIST:
      return "LIST"
    case .EMPTY_TUPLE:
      return "EMPTY_TUPLE"
    case .TUPLE:
      return "TUPLE"
    case .TUPLE1:
      return "TUPLE1"
    case .TUPLE2:
      return "TUPLE2"
    case .TUPLE3:
      return "TUPLE3"
    case .EMPTY_DICT:
      return "EMPTY_DICT"
    case .DICT:
      return "DICT"
    case .SETITEM:
      return "SETITEM"
    case .SETITEMS:
      return "SETITEMS"
    case .EMPTY_SET:
      return "EMPTY_SET"
    case .ADDITEMS:
      return "ADDITEMS"
    case .FROZENSET:
      return "FROZENSET"
    case .POP:
      return "POP"
    case .DUP:
      return "DUP"
    case .MARK:
      return "MARK"
    case .POP_MARK:
      return "POP_MARK"
    case .GET:
      return "GET"
    case .BINGET:
      return "BINGET"
    case .LONG_BINGET:
      return "LONG_BINGET"
    case .PUT:
      return "PUT"
    case .BINPUT:
      return "BINPUT"
    case .LONG_BINPUT:
      return "LONG_BINPUT"
    case .MEMOIZE:
      return "MEMOIZE"
    case .EXT1:
      return "EXT1"
    case .EXT2:
      return "EXT2"
    case .EXT4:
      return "EXT4"
    case .GLOBAL:
      return "GLOBAL"
    case .STACK_GLOBAL:
      return "STACK_GLOBAL"
    case .REDUCE:
      return "REDUCE"
    case .BUILD:
      return "BUILD"
    case .INST:
      return "INST"
    case .OBJ:
      return "OBJ"
    case .NEWOBJ:
      return "NEWOBJ"
    case .NEWOBJ_EX:
      return "NEWOBJ_EX"
    case .PROTO:
      return "PROTO"
    case .STOP:
      return "STOP"
    case .FRAME:
      return "FRAME"
    case .PERSID:
      return "PERSID"
    case .BINPERSID:
      return "BINPERSID"
    }
  }
}

struct Instruction {
  var opcode: InstructionOpcode
  var arg: ArgumentDescriptor?
  init(_ opcode: InstructionOpcode, arg: ArgumentDescriptor? = nil) {
    self.opcode = opcode
    self.arg = arg
  }
}

let instructions: [Instruction] = [
  Instruction(.INT, arg: Argument.DecimalNLShort()),
  Instruction(.BININT, arg: Argument.Int4()),
  Instruction(.BININT1, arg: Argument.UInt1()),
  Instruction(.BININT2, arg: Argument.UInt2()),
  Instruction(.LONG, arg: Argument.DecimalNLLong()),
  Instruction(.LONG1, arg: Argument.Long1()),
  Instruction(.LONG4, arg: Argument.Long4()),
  Instruction(.STRING, arg: Argument.StringNL(stripquotes: true)),
  Instruction(.BINSTRING, arg: Argument.String4()),
  Instruction(.SHORT_BINSTRING, arg: Argument.String1()),
  Instruction(.BINBYTES, arg: Argument.Bytes4()),
  Instruction(.SHORT_BINBYTES, arg: Argument.Bytes1()),
  Instruction(.BINBYTES8, arg: Argument.Bytes8()),
  Instruction(.BYTEARRAY8, arg: Argument.ByteArray8()),
  Instruction(.NEXT_BUFFER),
  Instruction(.READONLY_BUFFER),
  Instruction(.NONE),
  Instruction(.NEWTRUE),
  Instruction(.NEWFALSE),
  Instruction(.UNICODE, arg: Argument.UnicodeStringNL()),
  Instruction(.SHORT_BINUNICODE, arg: Argument.UnicodeString1()),
  Instruction(.BINUNICODE, arg: Argument.UnicodeString4()),
  Instruction(.BINUNICODE8, arg: Argument.UnicodeString8()),
  Instruction(.FLOAT, arg: Argument.FloatNL()),
  Instruction(.BINFLOAT, arg: Argument.Float8()),
  Instruction(.EMPTY_LIST),
  Instruction(.APPEND),
  Instruction(.APPENDS),
  Instruction(.LIST),
  Instruction(.EMPTY_TUPLE),
  Instruction(.TUPLE),
  Instruction(.TUPLE1),
  Instruction(.TUPLE2),
  Instruction(.TUPLE3),
  Instruction(.EMPTY_DICT),
  Instruction(.DICT),
  Instruction(.SETITEM),
  Instruction(.SETITEMS),
  Instruction(.EMPTY_SET),
  Instruction(.ADDITEMS),
  Instruction(.FROZENSET),
  Instruction(.POP),
  Instruction(.DUP),
  Instruction(.MARK),
  Instruction(.POP_MARK),
  Instruction(.GET, arg: Argument.DecimalNLShort()),
  Instruction(.BINGET, arg: Argument.UInt1()),
  Instruction(.LONG_BINGET, arg: Argument.UInt4()),
  Instruction(.PUT, arg: Argument.DecimalNLShort()),
  Instruction(.BINPUT, arg: Argument.UInt1()),
  Instruction(.LONG_BINPUT, arg: Argument.UInt4()),
  Instruction(.MEMOIZE),
  Instruction(.EXT1, arg: Argument.UInt1()),
  Instruction(.EXT2, arg: Argument.UInt2()),
  Instruction(.EXT4, arg: Argument.Int4()),
  Instruction(.GLOBAL, arg: Argument.StringNLPair(stripquotes: false)),
  Instruction(.STACK_GLOBAL),
  Instruction(.REDUCE),
  Instruction(.BUILD),
  Instruction(.INST, arg: Argument.StringNLPair(stripquotes: false)),
  Instruction(.OBJ),
  Instruction(.NEWOBJ),
  Instruction(.NEWOBJ_EX),
  Instruction(.PROTO, arg: Argument.UInt1()),
  Instruction(.STOP),
  Instruction(.FRAME, arg: Argument.UInt8()),
  Instruction(.PERSID, arg: Argument.StringNL(stripquotes: false)),
  Instruction(.BINPERSID),
]

var instructionMapping = [UInt8: Instruction]()
for instruction in instructions {
  instructionMapping[instruction.opcode.rawValue] = instruction
}

let filename = "/home/liu/workspace/swift-diffusion/archive/data.pkl"
let handle = fopen(filename, "rb")!
var opcode: UInt8 = 0
var ops = [(InstructionOpcode, Any?, Int)]()
while true {
  let pos = ftell(handle)
  let len = fread(&opcode, 1, 1, handle)
  guard len > 0, var instruction = instructionMapping[opcode] else { break }
  do {
    let arg = try instruction.arg?.read(handle).get()
    ops.append((instruction.opcode, arg, pos))
  } catch {
    break
  }
}
fclose(handle)
for op in ops {
  print(op)
}
