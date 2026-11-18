import json
import logging
import re
from bisect import bisect_left, bisect_right
from collections.abc import Iterable
from itertools import accumulate
from pathlib import Path

from tokenizer import Tokenizer

logger = logging.getLogger(__name__)
CONTEXT_SIZE = 384
PERIOD_ABBREVIATIONS = frozenset({
    "d", "dr", "g", "kt", "m", "min", "mln", "mlrd", "nr", "pan", "pvz",
    "proc", "prof", "sek", "t", "tūkst", "val", "y",
})


def load_dataset(data_dir: Path):
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


def token_offsets_by_character(tokenizer: Tokenizer, text: str, input_ids: list[int]):
    token_bytes = [tokenizer.vocab[token_id] for token_id in input_ids]

    token_by_byte_offset: dict[int, int] = {0: 0}
    byte_offset = 0
    for token_offset, raw_token in enumerate(token_bytes, start=1):
        byte_offset += len(raw_token)
        token_by_byte_offset[byte_offset] = token_offset

    token_by_char_offset: dict[int, int] = {0: 0}
    byte_offset = 0
    for char_offset, character in enumerate(text, start=1):
        byte_offset += len(character.encode("utf-8"))
        token_offset = token_by_byte_offset.get(byte_offset)
        if token_offset is not None:
            token_by_char_offset[char_offset] = token_offset
    return token_by_char_offset


def encode_text_and_annotations(tokenizer: Tokenizer, record: dict):
    text = record["Text"]
    input_ids = tokenizer.encode(text)
    token_by_char_offset = token_offsets_by_character(tokenizer, text, input_ids)

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
        start_token = token_by_char_offset.get(start)
        end_token = token_by_char_offset.get(end)
        if start_token is None or end_token is None:
            raise ValueError(f"Annotation [{start}, {end}) must align with BPE token boundaries")

        targets.append((start_token, end_token, annotation["Element"]))

    return {"input_ids": input_ids, "targets": targets}


def sentence_end_offsets(text: str):
    for match in re.finditer(r"""([.!?]+)["'“”’»)\]}]*(\s+|$)""", text):
        punctuation, spacing = match.groups()
        if punctuation == "." and not any(char in spacing for char in "\r\n"):
            word_start = match.start()
            while word_start > 0 and text[word_start - 1].isalpha():
                word_start -= 1
            word = text[word_start:match.start()]
            if word.casefold() in PERIOD_ABBREVIATIONS or (len(word) == 1 and word.isupper()):
                continue
        yield match.end()


def build_blocks(tokenizer: Tokenizer, text: str, example: dict, max_tokens: int, overlap_tokens: int = 32):
    if max_tokens < 1 or overlap_tokens < 0:
        raise ValueError("max_tokens must be positive and overlap_tokens must be nonnegative")

    input_ids = example["input_ids"]
    targets = example["targets"]
    total = len(input_ids)
    token_by_char = token_offsets_by_character(tokenizer, text, input_ids)

    changes = [0] * (total + 1)
    for left, right, _ in targets:
        changes[left + 1] += 1
        changes[right] -= 1
    character_cuts = sorted(token_by_char.values())
    aligned_cuts = set(character_cuts)
    safe_cuts = [
        cut for cut, crossing in enumerate(accumulate(changes))
        if crossing == 0 and cut in aligned_cuts
    ]
    safe_cut_set = set(safe_cuts)

    def safe_token_cuts(char_offsets: Iterable[int]):
        return sorted(safe_cut_set.intersection(token_by_char.get(offset) for offset in char_offsets))

    sentence_cuts = sorted({0, total, *safe_token_cuts(sentence_end_offsets(text))})
    clause_cuts = safe_token_cuts(match.end() for match in re.finditer(r"[,;:]\s+", text))
    word_cuts = sorted({0, total, *safe_token_cuts(
        offset for match in re.finditer(r"\s+", text) for offset in match.span()
    )})

    blocks = []
    start = covered_until = 0
    while covered_until < total:
        limit = min(start + max_tokens, total)
        for cuts in (sentence_cuts, clause_cuts, word_cuts, safe_cuts):
            index = bisect_right(cuts, limit) - 1
            if index >= 0 and cuts[index] > covered_until:
                end = cuts[index]
                break
        else:
            index = bisect_right(character_cuts, limit) - 1
            end = character_cuts[index]
            if end <= covered_until:
                end = limit

            crossing_targets = [target for target in targets if target[0] < end < target[1]]
            boundary = "at a character boundary" if end in aligned_cuts else "inside a UTF-8 character"
            logger.warning(
                "Forced block cut at BPE token %d (%s, limit %d). Gold spans crossing the cut: %s",
                end, boundary, max_tokens, crossing_targets,
            )

        blocks.append({
            "input_ids": input_ids[start:end],
            "targets": [
                (left - start, right - start, role)
                for left, right, role in targets
                if start <= left and right <= end
            ],
        })
        covered_until = end
        previous_start = start
        start = end

        if end < total and overlap_tokens:
            # Leave room for a whole sentence or word whenever the budget allows it.
            next_cut = sentence_cuts[bisect_right(sentence_cuts, end)]
            if next_cut - end > max_tokens:
                next_cut = word_cuts[bisect_right(word_cuts, end)]
            if next_cut - end > max_tokens:
                next_cut = safe_cuts[bisect_right(safe_cuts, end)]
            earliest_start = max(previous_start + 1, end - overlap_tokens, next_cut - max_tokens)
            last_sentence_start = sentence_cuts[bisect_left(sentence_cuts, end) - 1]
            if last_sentence_start >= earliest_start:
                start = last_sentence_start
            else:
                start = min(end, word_cuts[bisect_left(word_cuts, earliest_start)])

    return blocks


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
    train_blocks = build_blocks(tokenizer, train["Text"], train_example, CONTEXT_SIZE)
    validation_blocks = build_blocks(tokenizer, validation["Text"], validation_example, CONTEXT_SIZE)
    tokenizer.print_training_data()

    for name, example, blocks in (
        ("Train", train_example, train_blocks),
        ("Validation", validation_example, validation_blocks),
    ):
        block_count = len(blocks)
        print(
            f"{name}: {len(example['input_ids'])} BPE tokens, "
            f"{len(example['targets'])} span targets, "
            f"{block_count} {'block' if block_count == 1 else 'blocks'}"
        )

    for block in train_blocks:
        print("---> " + tokenizer.decode(block["input_ids"]))
