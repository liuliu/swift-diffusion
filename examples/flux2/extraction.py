import json

def extract_vocab_merges(tokenizer_path):
    print(f"Loading {tokenizer_path}...")

    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. Extract Vocabulary
    try:
        vocab = data['model']['vocab']
        print(f"Found {len(vocab)} vocabulary items.")

        with open('vocab.json', 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
            print("Saved vocab.json")

    except KeyError:
        print("Error: Could not find ['model']['vocab'] in the JSON.")

    # 2. Extract Merges
    try:
        merges = data['model'].get('merges', [])

        if merges:
            print(f"Found {len(merges)} merge rules.")

            with open('merges.txt', 'w', encoding='utf-8') as f:
                f.write("#version: 0.2\n")

                # FIX: Check if the first item is a list (e.g. ["a", "b"]) or a string ("a b")
                # and format accordingly.
                formatted_merges = []
                for item in merges:
                    if isinstance(item, list):
                        # Join the pair with a space
                        formatted_merges.append(" ".join(item))
                    else:
                        formatted_merges.append(item)

                f.write("\n".join(formatted_merges))
            print("Saved merges.txt")
        else:
            print("Warning: No 'merges' field found.")

    except KeyError:
        print("Error processing merges.")

extract_vocab_merges('tokenizer.json')
