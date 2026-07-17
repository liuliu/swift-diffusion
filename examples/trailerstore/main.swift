import Foundation
import NNC

#if canImport(Darwin)
  import Darwin
#elseif canImport(Glibc)
  import Glibc
#endif

private let defaultAlignment = 16_384

private struct Options {
  let inputPath: String
  let outputPath: String
  let force: Bool
  let alignment: Int
}

private enum TrailerStoreToolError: Error, CustomStringConvertible {
  case invalidUsage(String)
  case missingInput(String)
  case outputExists(String)
  case cannotCreateTemporaryDirectory(String)
  case cannotReadTensor(String)
  case cannotWriteTensor(String)
  case cannotCreateTrailerStore(String)

  var description: String {
    switch self {
    case .invalidUsage(let message):
      return message
    case .missingInput(let path):
      return "input file does not exist: \(path)"
    case .outputExists(let path):
      return "output file already exists: \(path)"
    case .cannotCreateTemporaryDirectory(let path):
      return "cannot create temporary directory: \(path)"
    case .cannotReadTensor(let key):
      return "cannot read tensor: \(key)"
    case .cannotWriteTensor(let key):
      return "cannot write tensor: \(key)"
    case .cannotCreateTrailerStore(let path):
      return "cannot create trailer store: \(path)"
    }
  }
}

private func usage() -> String {
  return """
    Usage: trailerstore [--force] [--alignment N] <input.ckpt> [output.ckpt]

    If output is omitted, the tool writes <input>_trailer.<ext>.
    """
}

private func derivedOutputPath(from inputPath: String) -> String {
  let path = inputPath as NSString
  let ext = path.pathExtension
  if ext.isEmpty {
    return inputPath + "_trailer"
  }
  let base = path.deletingPathExtension
  return base + "_trailer." + ext
}

private func parseOptions(_ arguments: [String]) throws -> Options {
  var force = false
  var alignment = defaultAlignment
  var positional = [String]()
  var i = 1
  while i < arguments.count {
    let argument = arguments[i]
    switch argument {
    case "--help", "-h":
      throw TrailerStoreToolError.invalidUsage(usage())
    case "--force", "-f":
      force = true
    case "--alignment":
      i += 1
      guard i < arguments.count, let parsed = Int(arguments[i]), parsed > 0 else {
        throw TrailerStoreToolError.invalidUsage("invalid --alignment value\n\n\(usage())")
      }
      alignment = parsed
    default:
      if argument.hasPrefix("-") {
        throw TrailerStoreToolError.invalidUsage("unknown option: \(argument)\n\n\(usage())")
      }
      positional.append(argument)
    }
    i += 1
  }
  guard positional.count == 1 || positional.count == 2 else {
    throw TrailerStoreToolError.invalidUsage(usage())
  }
  let inputPath = positional[0]
  let outputPath = positional.count == 2 ? positional[1] : derivedOutputPath(from: inputPath)
  return Options(inputPath: inputPath, outputPath: outputPath, force: force, alignment: alignment)
}

private func externalStorePath(for filePath: String) -> String {
  return filePath.appending("-tensordata")
}

private func existingExternalStorePath(for filePath: String) -> String? {
  let path = externalStorePath(for: filePath)
  return FileManager.default.fileExists(atPath: path) ? path : nil
}

private func codecForRead(_ codec: DynamicGraph.Store.Codec) -> DynamicGraph.Store.Codec {
  var base = codec
  let isExternal = base.contains(.externalData) || base.contains(.externalOnDemand)
  base.subtract([.externalData, .externalOnDemand, .jit])
  return isExternal ? base.union([.externalData, .jit]) : base.union([.jit])
}

private func codecForWrite(
  _ codec: DynamicGraph.Store.Codec, tensor: AnyTensor
) -> DynamicGraph.Store.Codec {
  var base = codec
  base.subtract([.externalData, .externalOnDemand, .jit])
  guard !base.contains(.ezm7) else { return base }
  let squeezedDims = tensor.shape.reduce(0) { count, dim in count + (dim > 1 ? 1 : 0) }
  guard squeezedDims > 1 else { return base }
  return base.union([.externalData])
}

private func moveTensorDataToExternalStore(
  graph: DynamicGraph, inputPath: String, sqlitePath: String, externalPath: String, alignment: Int
) throws {
  let sourceExternalPath = existingExternalStorePath(for: inputPath)
  var movedCount = 0
  var inlineCount = 0
  var totalCount = 0
  let sourceResult = try graph.openStore(
    inputPath, flags: [.readOnly], externalStore: sourceExternalPath
  ) { sourceStore in
    let keys = sourceStore.keys
    totalCount = keys.count

    func shouldWriteFirst(_ key: String) -> Bool {
      key.contains("ada_ln")
    }

    let outputResult = try graph.openStore(
      sqlitePath, externalStore: externalPath, chunkSize: alignment
    ) { outputStore in
      try outputStore.withTransaction {
        for writeFirst in [true, false] {
          for key in keys where shouldWriteFirst(key) == writeFirst {
            guard let sourceCodec = sourceStore.codec(for: key) else { continue }
            guard
              let tensor = sourceStore.read(
                key, kind: .CPU, codec: codecForRead(sourceCodec))
            else {
              throw TrailerStoreToolError.cannotReadTensor(key)
            }
            let writeCodec = codecForWrite(sourceCodec, tensor: tensor)
            do {
              try outputStore.write(key, tensor: tensor, strict: true, codec: writeCodec)
            } catch {
              throw TrailerStoreToolError.cannotWriteTensor(key)
            }
            if writeCodec.contains(.externalData) {
              movedCount += 1
            } else {
              inlineCount += 1
            }
            let written = movedCount + inlineCount
            if written % 100 == 0 || written == totalCount {
              print("rewrote \(written)/\(totalCount) tensors")
            }
          }
        }
      }
      outputStore.vacuum()
    }
    switch outputResult {
    case .success:
      break
    case .failure:
      throw TrailerStoreToolError.invalidUsage("cannot open temporary store: \(sqlitePath)")
    }
  }
  switch sourceResult {
  case .success:
    print("externalized \(movedCount) tensors; kept \(inlineCount) inline")
  case .failure:
    throw TrailerStoreToolError.invalidUsage("cannot open input store: \(inputPath)")
  }
}

private func makeTrailerStore(options: Options) throws {
  let fileManager = FileManager.default
  guard fileManager.fileExists(atPath: options.inputPath) else {
    throw TrailerStoreToolError.missingInput(options.inputPath)
  }
  if fileManager.fileExists(atPath: options.outputPath) {
    guard options.force else { throw TrailerStoreToolError.outputExists(options.outputPath) }
    try fileManager.removeItem(atPath: options.outputPath)
  }

  let outputURL = URL(fileURLWithPath: options.outputPath)
  let outputDirectory = outputURL.deletingLastPathComponent()
  let outputFileName = outputURL.lastPathComponent.isEmpty ? "output" : outputURL.lastPathComponent
  let temporaryDirectory = outputDirectory.appendingPathComponent(
    ".\(outputFileName).trailerstore-\(UUID().uuidString)")
  do {
    try fileManager.createDirectory(at: temporaryDirectory, withIntermediateDirectories: false)
  } catch {
    throw TrailerStoreToolError.cannotCreateTemporaryDirectory(temporaryDirectory.path)
  }
  defer { try? fileManager.removeItem(at: temporaryDirectory) }

  let sqlitePath = temporaryDirectory.appendingPathComponent("store.ckpt").path
  let externalPath = temporaryDirectory.appendingPathComponent("store.ckpt-tensordata").path
  let graph = DynamicGraph()
  try moveTensorDataToExternalStore(
    graph: graph, inputPath: options.inputPath, sqlitePath: sqlitePath, externalPath: externalPath,
    alignment: options.alignment)
  if !fileManager.fileExists(atPath: externalPath) {
    guard fileManager.createFile(atPath: externalPath, contents: Data()) else {
      throw TrailerStoreToolError.cannotCreateTrailerStore(options.outputPath)
    }
  }

  do {
    try DynamicGraph.Store.makeTrailerStore(
      sqlitePath, externalStore: externalPath, to: options.outputPath, alignment: options.alignment)
  } catch {
    throw TrailerStoreToolError.cannotCreateTrailerStore(options.outputPath)
  }

  guard DynamicGraph.Store.isTrailerStore(options.outputPath) else {
    throw TrailerStoreToolError.cannotCreateTrailerStore(options.outputPath)
  }
  print("wrote trailer store: \(options.outputPath)")
}

if CommandLine.arguments.contains("--help") || CommandLine.arguments.contains("-h") {
  print(usage())
  exit(0)
}

do {
  let options = try parseOptions(CommandLine.arguments)
  try makeTrailerStore(options: options)
} catch let error as TrailerStoreToolError {
  fputs("\(error.description)\n", stderr)
  exit(1)
} catch {
  fputs("\(error)\n", stderr)
  exit(1)
}
