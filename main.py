from pathlib import Path

from tokenizer import Tokenizer

text_path = Path(__file__).parent / "tests" / "test.txt"
training_text = text_path.read_text(encoding="utf-8")

tokenizer = Tokenizer()
tokenizer.train(training_text)
tokenizer.print_training_data()
