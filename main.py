import json
from pathlib import Path

from tokenizer import Tokenizer

def load_dataset(data_dir: Path) -> list[dict]:
    documents = []

    for path in sorted(data_dir.glob("*.annotations.json")):
        with path.open(encoding="utf-8", newline="") as file:
            record = json.load(file)

        documents.append({
            "id": path.name.removesuffix(".annotations.json"),
            "Text": record["Text"],
            "Annotations": record["Annotations"],
        })

    return documents


def encode_text_and_annotations(tokenizer: Tokenizer, record: dict) -> dict:
    text = record["Text"]
    input_ids = tokenizer.encode(text)
    token_bytes = [tokenizer.vocab[token_id] for token_id in input_ids]
    if b"".join(token_bytes) != text.encode("utf-8"):
        raise ValueError("BPE token bytes must reconstruct the original text exactly")

    # Compare character and token boundaries using their UTF-8 byte offsets.
    char_byte_offsets: list[int] = [0]
    for character in text:
        char_byte_offsets.append(char_byte_offsets[-1] + len(character.encode("utf-8")))

    token_by_byte_offset: dict[int, int] = {0: 0}
    byte_offset = 0
    for token_offset, raw_token in enumerate(token_bytes, start=1):
        byte_offset += len(raw_token)
        token_by_byte_offset[byte_offset] = token_offset

    targets = []
    for annotation in record["Annotations"]:
        start: int = annotation["Start"]
        end: int = annotation["End"]
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= len(text)
        ):
            raise ValueError(f"Invalid annotation offsets [{start!r}, {end!r})")
        start_token = token_by_byte_offset.get(char_byte_offsets[start])
        end_token = token_by_byte_offset.get(char_byte_offsets[end])
        if start_token is None or end_token is None:
            raise ValueError(f"Annotation [{start}, {end}) must align with BPE token boundaries")

        targets.append((start_token, end_token, annotation["Element"]))

    return {"input_ids": input_ids, "targets": targets}


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent / "Data"
    documents = load_dataset(data_dir)

    if len(documents) != 1: # TODO Remove 1 doc limit after enough data is created
        raise ValueError("This temporary split expects exactly one document")

    document = documents[0]
    text = document["Text"]
    annotations = document["Annotations"]
    cut = text.index("Tinklas aktyviai plečiamas")

    if any(a["Start"] < cut < a["End"] for a in annotations):
        raise ValueError("The split crosses an annotation")

    train = {
        "Text": text[:cut],
        "Annotations": [a for a in annotations if a["End"] <= cut],
    }
    validation = {
        "Text": text[cut:],
        "Annotations": [
            {**a, "Start": a["Start"] - cut, "End": a["End"] - cut}
            for a in annotations
            if a["Start"] >= cut
        ],
    }

    tokenizer = Tokenizer()
    tokenizer.train(train["Text"], 300)

    train_example = encode_text_and_annotations(tokenizer, train)
    validation_example = encode_text_and_annotations(tokenizer, validation)
    tokenizer.print_training_data()

    for name, example in (("Train", train_example), ("Validation", validation_example)):
        print(
            f"{name}: {len(example['input_ids'])} BPE tokens, "
            f"{len(example['targets'])} span targets"
        )

# text_path = Path(__file__).parent / "tests" / "test.txt"
# training_text = text_path.read_text(encoding="utf-8")
#
# tokenizer = Tokenizer()
# tokenizer.train(training_text)
# tokenizer.print_training_data()
