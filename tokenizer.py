import regex

default_text = ("Sami Pajari nuo trečiosios iš eilės pergalės WRC skiria vos penki likę greičio ruožai, o įspūdingas Adrieno Fourmaux tempas šeštadienį žada intriguojančią kovą dėl podiumo „ueno Rally del Paraguay“."
        "Temos: 2 Pasaulio ralio čempionatas (WRC) Ralis Komentarai „Toyota“ lenktynininkas iš Suomijos dieną pradėjo turėdamas 11,9 sek. persvarą ir per devynis greičio ruožus ją padidino daugiau nei dvigubai, nors teko apgadinti galinį sparną, o viename posūkyje pravažiuoti stabdymo vietą ir lenktyniauti nuolat kintančio sukibimo sąlygomis."
        "Tačiau didžiausią šuolį už S. Pajari nugaros atliko A. Fourmaux. „Hyundai“ lenktynininkas šeštadienį demonstravo įspūdingą tempą ir laimėjo penkis greičio ruožus. Dienos pradžioje nuo Elfyno Evanso jis atsiliko daugiau nei 55 sek., o vakare pasaulio čempionato lyderį nuo jo skyrė vos 3,3 sek."
        "Kadangi buvo prognozuojamas smarkus lietus, šeštadienį ekipažai tikėjosi visiškai kitokių sąlygų nei penktadienį, todėl dalis jų pasirinko šlapiai dangai labiau pritaikytus automobilių nustatymus. Tačiau pirmieji greičio ruožai daugiausia buvo sausi, slidūs, o kelio danga darėsi vis grubesnė."
        "Ryte S. Pajari nuosekliai didino persvarą prieš Haydeną Paddoną, tačiau per pirmąjį Hohenau greičio ruožo važiavimą išlėkė per plačiai, atsidūrė griovyje ir nuplėšė viršutinę savo „GR Yaris Rally1“ galinio sparno dalį."
        "Serviso zoną jis pasiekė turėdamas 22,8 sek. persvarą. Pakeitus sugadintus aerodinaminius elementus, popietę suomis ir toliau kontroliavo ralio eigą."
        "Bella Vista greičio ruože S. Pajari dar kartą teko išgyventi neramią akimirką, kai jis pravažiavo posūkį, tačiau net ir po šios klaidos suomis buvo greitesnis už H. Paddoną. Padangų tausojimui tampant vis svarbesniu veiksniu, ralio lyderis išliko ramus.")

class Tokenizer:
    def __init__(self, minimum_occurrence= 4):
        self.merges = None
        self.vocab = None
        self.minimum_occurrence = minimum_occurrence
        self.split_regex = regex.compile(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")
        #TODO simplify regex for this case

    @staticmethod
    def _get_stats(ids, counts=None):
        counts = {} if counts is None else counts
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def _merge(ids, pair, idx):
        new_ids = []
        i = 0

        while i < len(ids):
            if (
                i < len(ids) - 1
                and ids[i] == pair[0]
                and ids[i + 1] == pair[1]
            ):
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1

        return new_ids

    def _validate_applied_training(self):
        if self.merges is None or self.vocab is None:
            raise ValueError("Tokenizer has not been trained yet")

    def train(self, imported_text=None, vocab_length=10000, print_progress=True):
        if vocab_length < 256:
            raise ValueError("vocab_length must be at least 256")

        train_text = default_text if imported_text is None else imported_text
        text_pieces = self.split_regex.findall(train_text)
        ids_by_piece = [
            list(piece.encode("utf-8"))
            for piece in text_pieces
        ]
        num_merges = vocab_length - 256

        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            stats = {}
            for piece_ids in ids_by_piece:
                self._get_stats(piece_ids, stats)

            if not stats:
                break

            pair = max(stats, key=stats.get)
            if stats[pair] < self.minimum_occurrence:
                break

            idx = 256 + i

            if print_progress:
                print(f"merging {pair} into a new token {idx}")

            ids_by_piece = [
                self._merge(piece_ids, pair, idx)
                for piece_ids in ids_by_piece
            ]
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def encode(self, text):
        self._validate_applied_training()

        encoded_ids = []

        for piece in self.split_regex.findall(text):
            piece_ids = list(piece.encode("utf-8"))

            while len(piece_ids) >= 2:
                stats = self._get_stats(piece_ids)
                pair = min(
                    stats,
                    key=lambda candidate: self.merges.get(
                        candidate,
                        float("inf"),
                    ),
                )

                if pair not in self.merges:
                    break

                piece_ids = self._merge(
                    piece_ids,
                    pair,
                    self.merges[pair],
                )

            encoded_ids.extend(piece_ids)

        return encoded_ids

    def decode(self, ids):
        self._validate_applied_training()

        tokens = b"".join(self.vocab[idx] for idx in ids)
        return tokens.decode("utf-8", errors="replace")

    def print_training_data(self):
        self._validate_applied_training()

        print("Merges:")
        if not self.merges:
            print("  No merges were learned.")
        else:
            for pair, token_id in self.merges.items():
                left = self.vocab[pair[0]]
                right = self.vocab[pair[1]]
                token = self.vocab[token_id]
                print(
                    f"  {pair} -> {token_id}: "
                    f"{left!r} + {right!r} = {token!r}"
                )

        print("\nVocabulary:")
        for token_id, token in sorted(self.vocab.items()):
            print(f"  {token_id}: {token!r}")
