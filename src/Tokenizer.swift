import Foundation

public struct CLIPTokenizer {
  public struct Pair: Hashable, Equatable {
    public var first: String
    public var second: String
    public init(first: String, second: String) {
      self.first = first
      self.second = second
    }
  }
  public let vocabulary: [String: Int32]
  public let bpeRanks: [Pair: Int]
  public let unknownToken: Int32
  public let startToken: Int32
  public let endToken: Int32
  public init(vocabulary: String, merges: String) {
    let vocabJSONData = try! Data(contentsOf: URL(fileURLWithPath: vocabulary))
    let decoder = JSONDecoder()
    self.vocabulary = try! decoder.decode([String: Int32].self, from: vocabJSONData)
    let bpeMerges = (try! String(contentsOf: URL(fileURLWithPath: merges), encoding: .utf8))
      .trimmingCharacters(in: .whitespacesAndNewlines).split(separator: "\n")[
        1..<(49152 - 256 - 2 + 1)]
    var bpeRanks = [Pair: Int]()
    for (i, merge) in bpeMerges.enumerated() {
      let splits = merge.split(separator: " ", maxSplits: 2)
      bpeRanks[Pair(first: String(splits[0]), second: String(splits[1]))] = i
    }
    self.bpeRanks = bpeRanks
    self.unknownToken = self.vocabulary["<|endoftext|>"]!
    self.startToken = self.vocabulary["<|startoftext|>"]!
    self.endToken = self.vocabulary["<|endoftext|>"]!
  }

  public func tokenize(text: String, truncation: Bool, maxLength: Int, paddingToken: Int32? = nil)
    -> [Int32]
  {
    let fixText = text.split(separator: " ").joined(separator: " ").lowercased()
    // Logic for r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"""
    // Implement this with for loop rather than regex so it is applicable with Swift 5.6.x
    var tokens = [Substring]()
    var lastIndex = fixText.startIndex
    for (i, character) in fixText.enumerated() {
      let index = fixText.index(fixText.startIndex, offsetBy: i)
      if character.isNumber {
        if lastIndex < index {
          tokens.append(fixText[lastIndex..<index])
        }
        lastIndex = fixText.index(index, offsetBy: 1)  // Skip this one.
        tokens.append(fixText[index..<lastIndex])
        continue
      }
      let pat = fixText[lastIndex...index]
      if pat.hasSuffix("'s") || pat.hasSuffix("'t") || pat.hasSuffix("'m") || pat.hasSuffix("'d") {
        let splitIndex = fixText.index(index, offsetBy: -1)
        if lastIndex < splitIndex {
          tokens.append(fixText[lastIndex..<splitIndex])
        }
        lastIndex = fixText.index(index, offsetBy: 1)
        tokens.append(fixText[splitIndex..<lastIndex])
        continue
      }
      if pat.hasSuffix("'re") || pat.hasSuffix("'ve") || pat.hasSuffix("'ll") {
        let splitIndex = fixText.index(index, offsetBy: -2)
        if lastIndex < splitIndex {
          tokens.append(fixText[lastIndex..<splitIndex])
        }
        lastIndex = fixText.index(index, offsetBy: 1)
        tokens.append(fixText[splitIndex..<lastIndex])
        continue
      }
      if pat.hasSuffix("<|startoftext|>") {
        let splitIndex = fixText.index(index, offsetBy: -14)
        if lastIndex < splitIndex {
          tokens.append(fixText[lastIndex..<splitIndex])
        }
        lastIndex = fixText.index(index, offsetBy: 1)
        tokens.append(fixText[splitIndex..<lastIndex])
        continue
      }
      if pat.hasSuffix("<|endoftext|>") {
        let splitIndex = fixText.index(index, offsetBy: -12)
        if lastIndex < splitIndex {
          tokens.append(fixText[lastIndex..<splitIndex])
        }
        lastIndex = fixText.index(index, offsetBy: 1)
        tokens.append(fixText[splitIndex..<lastIndex])
        continue
      }
      if character.isWhitespace {
        if lastIndex < index {
          tokens.append(fixText[lastIndex..<index])
        }
        lastIndex = fixText.index(index, offsetBy: 1)  // Skip this one.
        continue
      }
    }
    if lastIndex < fixText.endIndex {
      tokens.append(fixText[lastIndex...])
    }
    // Now filter token further by split if it not a number nor a letter.
    tokens = tokens.flatMap { token -> [Substring] in
      // Remove special tokens (start and end)
      guard token != "<|startoftext|>" && token != "<|endoftext|>" else {
        return []
      }
      // Skip these tokens
      guard
        token != "'s" && token != "'t" && token != "'m" && token != "'d" && token != "'re"
          && token != "'ve" && token != "'ll"
      else {
        return [token]
      }
      var tokens = [Substring]()
      var lastIndex = token.startIndex
      for (i, character) in token.enumerated() {
        let index = token.index(token.startIndex, offsetBy: i)
        // Split further if it is not a letter nor a number.
        if !character.isLetter && !character.isNumber {
          if lastIndex < index {
            tokens.append(token[lastIndex..<index])
          }
          tokens.append(token[index..<token.index(after: index)])  // Add this character.
          lastIndex = fixText.index(index, offsetBy: 1)
        }
      }
      if lastIndex < token.endIndex {
        tokens.append(token[lastIndex...])
      }
      return tokens
    }
    // token should match the token before sending to bpe mapping. Now do bpe merge.
    let bpeTokens = tokens.flatMap { token -> [String] in
      bpe(token: String(token))
    }
    // With bpeTokens, we can query vocabulary and return index now.
    var ids = [startToken]
    if truncation {
      for bpeToken in bpeTokens.prefix(maxLength - 2) {
        ids.append(vocabulary[bpeToken, default: unknownToken])
      }
    } else {
      for bpeToken in bpeTokens {
        ids.append(vocabulary[bpeToken, default: unknownToken])
      }
    }
    if ids.count < maxLength {
      ids.append(endToken)
      if ids.count < maxLength {
        let paddingToken = paddingToken ?? endToken
        for _ in ids.count..<maxLength {
          ids.append(paddingToken)
        }
      }
    } else {
      ids.append(endToken)
    }
    return ids
  }

  func getPairs(word: [String]) -> Set<Pair>? {
    guard word.count > 1 else {
      return nil
    }
    var pairs = Set<Pair>()
    var previousCharacter = word[0]
    for character in word.suffix(from: 1) {
      pairs.insert(Pair(first: previousCharacter, second: character))
      previousCharacter = character
    }
    return pairs
  }

  func bpe(token: String) -> [String] {
    var word = [String]()
    for (i, character) in token.enumerated() {
      guard i < token.count - 1 else {
        word.append(String(character) + "</w>")
        break
      }
      word.append(String(character))
    }
    guard var pairs = getPairs(word: word) else {
      return word
    }
    while true {
      var bigram: Pair? = nil
      var minRank: Int? = nil
      for pair in pairs {
        if let rank = bpeRanks[pair] {
          guard let currentMinRank = minRank else {
            bigram = pair
            minRank = rank
            continue
          }
          if rank < currentMinRank {
            bigram = pair
            minRank = rank
          }
        }
      }
      guard let bigram = bigram else {
        break
      }
      var newWord = [String]()
      var i = 0
      while i < word.count {
        guard let j = word[i...].firstIndex(of: bigram.first) else {
          newWord.append(contentsOf: word[i...])
          break
        }
        if i < j {
          newWord.append(contentsOf: word[i..<j])
        }
        i = j
        if word[i] == bigram.first && i < word.count - 1 && word[i + 1] == bigram.second {
          newWord.append(bigram.first + bigram.second)
          i += 2
        } else {
          newWord.append(word[i])
          i += 1
        }
      }
      word = newWord
      if word.count == 1 {
        break
      }
      pairs = getPairs(word: word)!  // word.count > 1, should be able to get pair.
    }
    return word
  }
}

public struct GPT2Tokenizer {
  public struct Pair: Hashable, Equatable {
    public var first: String
    public var second: String
    public init(first: String, second: String) {
      self.first = first
      self.second = second
    }
  }
  public let vocabulary: [String: Int32]
  public let decoder: [Int32: String]
  public let bpeRanks: [Pair: Int]
  public let unknownToken: Int32
  public let startToken: Int32
  public let endToken: Int32
  public let specialTokens: [String]
  public init(vocabulary: String, merges: String, specialTokens: [String: Int32] = [:]) {
    let vocabJSONData = try! Data(contentsOf: URL(fileURLWithPath: vocabulary))
    var vocabulary = try! JSONDecoder().decode([String: Int32].self, from: vocabJSONData)
    vocabulary.merge(specialTokens) { a, _ in a }
    self.vocabulary = vocabulary
    var decoder = [Int32: String]()
    for (k, v) in self.vocabulary {
      decoder[v] = k
    }
    self.decoder = decoder
    let bpeMerges = (try! String(contentsOf: URL(fileURLWithPath: merges), encoding: .utf8))
      .trimmingCharacters(in: .whitespacesAndNewlines).split(separator: "\n")[
        1...]
    var bpeRanks = [Pair: Int]()
    for (i, merge) in bpeMerges.enumerated() {
      let splits = merge.split(separator: " ", maxSplits: 2)
      bpeRanks[Pair(first: String(splits[0]), second: String(splits[1]))] = i
    }
    self.bpeRanks = bpeRanks
    self.unknownToken = self.vocabulary["<|end_of_text|>"]!
    self.startToken = self.vocabulary["<|begin_of_text|>"]!
    self.endToken = self.vocabulary["<|end_of_text|>"]!
    self.specialTokens = Array(specialTokens.keys)
  }

  private static let byteEncoder: [Int: String] = {
    /*
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    */
    var bs: [Int] = Array(
      Int("!".unicodeScalars.first!.value)...Int("~".unicodeScalars.first!.value))
    bs.append(
      contentsOf: Array(Int("¡".unicodeScalars.first!.value)...Int("¬".unicodeScalars.first!.value))
    )
    bs.append(
      contentsOf: Array(Int("®".unicodeScalars.first!.value)...Int("ÿ".unicodeScalars.first!.value))
    )
    var cs: [Int] = bs
    var n = 0
    for b in 0..<256 {
      guard !bs.contains(b) else { continue }
      bs.append(b)
      cs.append(256 + n)
      n += 1
    }
    return Dictionary(uniqueKeysWithValues: zip(bs, cs.map { String(Unicode.Scalar($0)!) }))
  }()

  private static let byteDecoder: [Int: String] = {
    let byteEncoder = Self.byteEncoder
    var byteDecoder = [Int: String]()
    for (k, v) in byteEncoder {
      byteDecoder[Int(v.unicodeScalars.first!.value)] = String(Unicode.Scalar(k)!)
    }
    return byteDecoder
  }()

  public func decode(_ tokens: [Int32]) -> String {
    tokens.map({
      let token = decoder[$0, default: ""]
      guard !token.isEmpty else { return "" }
      return token.unicodeScalars.map({
        Self.byteDecoder[Int($0.value), default: "\($0)"]
      }).joined()
    }).joined()
  }

  public func tokenize(text: String, addSpecialTokens: Bool = true)
    -> [Int32]
  {
    var fixText = text.split(separator: " ").joined(separator: " ")
    if text.hasPrefix(" ") {
      fixText = " " + fixText
    }
    if text.hasSuffix(" ") {
      fixText = fixText + " "
    }
    // Logic for r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"""
    // Implement this with for loop rather than regex so it is applicable with Swift 5.6.x
    var tokens = [Substring]()
    var lastIndex = fixText.startIndex
    var lastIsNewline = false
    for (i, character) in fixText.enumerated() {
      let index = fixText.index(fixText.startIndex, offsetBy: i)
      if character.isNumber {
        if lastIndex < index {
          tokens.append(fixText[lastIndex..<index])
        }
        lastIndex = fixText.index(index, offsetBy: 1)  // Skip this one.
        tokens.append(fixText[index..<lastIndex])
        continue
      }
      let pat = fixText[lastIndex...index]
      if pat.hasSuffix("'s") || pat.hasSuffix("'t") || pat.hasSuffix("'m") || pat.hasSuffix("'d") {
        let splitIndex = fixText.index(index, offsetBy: -1)
        if lastIndex < splitIndex {
          tokens.append(fixText[lastIndex..<splitIndex])
        }
        lastIndex = fixText.index(index, offsetBy: 1)
        tokens.append(fixText[splitIndex..<lastIndex])
        continue
      }
      if pat.hasSuffix("'re") || pat.hasSuffix("'ve") || pat.hasSuffix("'ll") {
        let splitIndex = fixText.index(index, offsetBy: -2)
        if lastIndex < splitIndex {
          tokens.append(fixText[lastIndex..<splitIndex])
        }
        lastIndex = fixText.index(index, offsetBy: 1)
        tokens.append(fixText[splitIndex..<lastIndex])
        continue
      }
      // Check if the previous one is new line but this one is not, separate up until here.
      if lastIsNewline && !character.isNewline {
        lastIsNewline = false
        if lastIndex < index {
          tokens.append(fixText[lastIndex..<index])
        }
        lastIndex = index
        continue
      }
      if character.isNewline {
        lastIsNewline = true
        continue
      }
      if let specialToken = (specialTokens.first { pat.hasSuffix($0) }) {
        let splitIndex = fixText.index(index, offsetBy: 1 - specialToken.count)
        if lastIndex < splitIndex {
          tokens.append(fixText[lastIndex..<splitIndex])
        }
        lastIndex = fixText.index(index, offsetBy: 1)
        tokens.append(fixText[splitIndex..<lastIndex])
        continue
      }
      if character.isWhitespace {
        if lastIndex < index {
          tokens.append(fixText[lastIndex..<index])
        }
        lastIndex = index
        continue
      }
    }
    if lastIndex < fixText.endIndex {
      tokens.append(fixText[lastIndex...])
    }
    // token should match the token before sending to bpe mapping. Now do bpe merge.
    let bpeTokens = tokens.flatMap { token -> [String] in
      guard !specialTokens.contains(String(token)) else {
        return [String(token)]
      }
      let token = token.unicodeScalars.map({
        Self.byteEncoder[Int($0.value), default: "\($0)"]
      }).joined()
      return bpe(token: String(token))
    }
    // With bpeTokens, we can query vocabulary and return index now.
    var ids: [Int32] = addSpecialTokens ? [startToken] : []
    for bpeToken in bpeTokens {
      ids.append(vocabulary[bpeToken, default: unknownToken])
    }
    return ids
  }

  func getPairs(word: [String]) -> Set<Pair>? {
    guard word.count > 1 else {
      return nil
    }
    var pairs = Set<Pair>()
    var previousCharacter = word[0]
    for character in word.suffix(from: 1) {
      pairs.insert(Pair(first: previousCharacter, second: character))
      previousCharacter = character
    }
    return pairs
  }

  func bpe(token: String) -> [String] {
    var word = [String]()
    for character in token {
      word.append(String(character))
    }
    guard var pairs = getPairs(word: word) else {
      return word
    }
    while true {
      var bigram: Pair? = nil
      var minRank: Int? = nil
      for pair in pairs {
        if let rank = bpeRanks[pair] {
          guard let currentMinRank = minRank else {
            bigram = pair
            minRank = rank
            continue
          }
          if rank < currentMinRank {
            bigram = pair
            minRank = rank
          }
        }
      }
      guard let bigram = bigram else {
        break
      }
      var newWord = [String]()
      var i = 0
      while i < word.count {
        guard let j = word[i...].firstIndex(of: bigram.first) else {
          newWord.append(contentsOf: word[i...])
          break
        }
        if i < j {
          newWord.append(contentsOf: word[i..<j])
        }
        i = j
        if word[i] == bigram.first && i < word.count - 1 && word[i + 1] == bigram.second {
          newWord.append(bigram.first + bigram.second)
          i += 2
        } else {
          newWord.append(word[i])
          i += 1
        }
      }
      word = newWord
      if word.count == 1 {
        break
      }
      pairs = getPairs(word: word)!  // word.count > 1, should be able to get pair.
    }
    return word
  }
}
