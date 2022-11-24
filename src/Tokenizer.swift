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
  let vocabulary: [String: Int32]
  let bpeRanks: [Pair: Int]
  let unknownToken: Int32
  let startToken: Int32
  let endToken: Int32
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
    self.unknownToken = self.vocabulary["<|endoftext|>"] ?? self.vocabulary["<end_of_text>"]!
    self.startToken = self.vocabulary["<|startoftext|>"] ?? self.vocabulary["<start_of_text>"]!
    self.endToken = self.vocabulary["<|endoftext|>"] ?? self.vocabulary["<end_of_text>"]!
  }

  public func tokenize(text: String, truncation: Bool, maxLength: Int) -> [Int32] {
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
      if pat.hasSuffix("<|startoftext|>") || pat.hasSuffix("<start_of_text>") {
        let splitIndex = fixText.index(index, offsetBy: -14)
        if lastIndex < splitIndex {
          tokens.append(fixText[lastIndex..<splitIndex])
        }
        lastIndex = fixText.index(index, offsetBy: 1)
        tokens.append(fixText[splitIndex..<lastIndex])
        continue
      }
      if pat.hasSuffix("<|endoftext|>") || pat.hasSuffix("<end_of_text>") {
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
      guard
        token != "<|startoftext|>" && token != "<|endoftext|>" && token != "<start_of_text>"
          && token != "<end_of_text>"
      else {
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
      for _ in ids.count..<maxLength {
        ids.append(endToken)
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
