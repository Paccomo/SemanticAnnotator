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

    train_ids = tokenizer.encode(train["Text"])
    validation_ids = tokenizer.encode(validation["Text"])
    tokenizer.print_training_data()

# text_path = Path(__file__).parent / "tests" / "test.txt"
# training_text = text_path.read_text(encoding="utf-8")
#
# tokenizer = Tokenizer()
# tokenizer.train(training_text)
# tokenizer.print_training_data()
