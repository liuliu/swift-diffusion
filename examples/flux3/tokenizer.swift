import Foundation

// Qwen's byte-level BPE, loaded directly from the original tokenizer.json.
// This intentionally preserves whitespace and Unicode normalization semantics.
struct Flux3Tokenizer {
  private struct Pair: Hashable {
    let first: String
    let second: String
  }
  private let vocabulary: [String: Int32]
  private let ranks: [Pair: Int]
  private let special: [String: Int32]
  private let specialPattern: NSRegularExpression
  private let pattern: NSRegularExpression
  let padding: Int32

  init(path: String) throws {
    let json =
      try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path)))
      as! [String: Any]
    let model = json["model"] as! [String: Any]
    precondition(model["type"] as? String == "BPE")
    vocabulary = (model["vocab"] as! [String: NSNumber]).mapValues { $0.int32Value }
    var pairs = [Pair: Int]()
    for (rank, merge) in (model["merges"] as! [String]).enumerated() {
      let parts = merge.split(separator: " ", omittingEmptySubsequences: false)
      precondition(parts.count == 2)
      pairs[Pair(first: String(parts[0]), second: String(parts[1]))] = rank
    }
    ranks = pairs
    special = Dictionary(
      uniqueKeysWithValues: (json["added_tokens"] as! [[String: Any]]).map {
        ($0["content"] as! String, ($0["id"] as! NSNumber).int32Value)
      })
    padding = special["<|endoftext|>"]!
    specialPattern = try NSRegularExpression(
      pattern: special.keys.sorted { $0.count > $1.count }
        .map(NSRegularExpression.escapedPattern(for:)).joined(separator: "|"))
    let pre = json["pre_tokenizer"] as! [String: Any]
    let split = (pre["pretokenizers"] as! [[String: Any]])[0]
    pattern = try NSRegularExpression(pattern: (split["pattern"] as! [String: String])["Regex"]!)
  }

  private static let byteEncoder: [UInt8: String] = {
    var bytes = Array(33...126) + Array(161...172) + Array(174...255)
    var scalars = bytes
    var nextScalar = 256
    for byte in 0..<256 where !bytes.contains(byte) {
      bytes.append(byte)
      scalars.append(nextScalar)
      nextScalar += 1
    }
    return Dictionary(
      uniqueKeysWithValues: zip(bytes, scalars).map {
        (UInt8($0.0), String(Unicode.Scalar($0.1)!))
      })
  }()

  private func ordinary(_ text: String) -> [Int32] {
    let normalized = text.precomposedStringWithCanonicalMapping
    let ns = normalized as NSString
    return pattern.matches(in: normalized, range: NSRange(location: 0, length: ns.length)).flatMap {
      match in
      var parts = ns.substring(with: match.range).utf8.map { Self.byteEncoder[$0]! }
      while parts.count > 1 {
        var best: (Int, Int)?
        for i in 0..<(parts.count - 1) {
          if let rank = ranks[Pair(first: parts[i], second: parts[i + 1])],
            rank < (best?.1 ?? Int.max)
          {
            best = (i, rank)
          }
        }
        guard let (i, _) = best else { break }
        parts[i] += parts[i + 1]
        parts.remove(at: i + 1)
      }
      return parts.map { vocabulary[$0]! }
    }
  }

  func encode(_ text: String) -> [Int32] {
    let ns = text as NSString
    var result = [Int32]()
    var offset = 0
    for match in specialPattern.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
      result += ordinary(
        ns.substring(with: NSRange(location: offset, length: match.range.location - offset)))
      result.append(special[ns.substring(with: match.range)]!)
      offset = match.range.location + match.range.length
    }
    result += ordinary(ns.substring(from: offset))
    return result
  }

  func prompt(_ text: String) -> (tokens: [Int32], valid: Int) {
    let chat = "<|im_start|>user\n\(text)<|im_end|>\n<|im_start|>assistant\n"
    var tokens = Array(encode(chat).prefix(4_096))
    let valid = tokens.count
    let length = min((valid + 79) / 80 * 80, 4_096)
    tokens += Array(repeating: padding, count: length - valid)
    return (tokens, valid)
  }
}
