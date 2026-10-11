"""Train a byte-level BPE tokenizer.

    python train_tokenizer.py --data RAW --vocab-size 32000 --output tokenizer_32k.json

RAW is a .jsonl file (optionally .gz or .zst) or a directory of them. Every line
is a JSON object with a "text" field holding one document.
"""

import argparse
import gzip
import io
import json
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers


EOS = "<|endoftext|>"
# o200k_base().pat_str; alternative order is significant.
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
O200K_PATTERN = "|".join((
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"\p{N}{1,3}", r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
    r"\s*[\r\n]+", r"\s+(?!\S)", r"\s+",
))
ROUND_TRIP_TEXTS = (
    "Hello, world! I'm here. WE'RE READY.", "1234567890 １２３４５",
    "中文法律与医学，日本語です。한국어 테스트", "العربية فارسی हिन्दी বাংলা ไทย",
    "Καλημέρα Привет café cafe\u0301", "a\r\nb\n\n  \t  ",
    "👩🏽‍💻🙂🏳️‍🌈", "a\x00b", EOS, "",
)


def make_tokenizer(pretokenizer="bytelevel"):
    tokenizer = Tokenizer(models.BPE())
    if pretokenizer == "o200k":
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(O200K_PATTERN), behavior="removed", invert=True),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
    elif pretokenizer == "bytelevel":
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    else:
        raise ValueError(f"Unknown pretokenizer: {pretokenizer}")
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def validate_tokenizer(tokenizer, vocab_size, pretokenizer=None):
    if tokenizer.get_vocab_size() != vocab_size:
        raise ValueError(f"Expected {vocab_size} tokens, got {tokenizer.get_vocab_size()}")
    if tokenizer.token_to_id(EOS) != 0:
        raise ValueError("EOS must have ID 0")
    if not set(pre_tokenizers.ByteLevel.alphabet()).issubset(tokenizer.get_vocab()):
        raise ValueError("Tokenizer is missing byte symbols")
    if pretokenizer is not None:
        expected = json.loads(make_tokenizer(pretokenizer).to_str())["pre_tokenizer"]
        if json.loads(tokenizer.to_str())["pre_tokenizer"] != expected:
            raise ValueError("Saved pretokenizer differs from requested pretokenizer")
    for text in ROUND_TRIP_TEXTS:
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if tokenizer.decode(ids, skip_special_tokens=False) != text:
            raise ValueError(f"Text did not round trip: {text!r}")


def train_tokenizer(texts, output, vocab_size=32000, pretokenizer="bytelevel", overwrite=False):
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}")
    if vocab_size < 257:
        raise ValueError("Vocabulary must include 256 bytes and EOS")
    tokenizer = make_tokenizer(pretokenizer)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=[EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    validate_tokenizer(tokenizer, vocab_size, pretokenizer)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")
    tokenizer.save(str(temporary))
    validate_tokenizer(Tokenizer.from_file(str(temporary)), vocab_size, pretokenizer)
    temporary.replace(output)
    print(f"Saved {output} ({vocab_size:,} tokens)", flush=True)
    return tokenizer


def data_files(path):
    """Return the JSONL files named by a file or directory path, in sorted order."""
    path = Path(path)
    if path.is_file():
        return [path]
    files = sorted(p for p in path.rglob("*") if p.name.endswith((".jsonl", ".jsonl.gz", ".jsonl.zst")))
    if not files:
        raise FileNotFoundError(f"No .jsonl, .jsonl.gz or .jsonl.zst files under {path}")
    return files


def open_text(path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    if path.name.endswith(".zst"):
        import zstandard
        return io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(path.open("rb")), encoding="utf-8")
    return path.open(encoding="utf-8")


def iter_texts(files):
    """Yield the text of every document, in file order."""
    for path in files:
        with open_text(path) as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                text = json.loads(line).get("text")
                if not isinstance(text, str):
                    raise ValueError(f"{path}:{line_number}: expected a text string")
                yield text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="JSONL file or directory of JSONL files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--pretokenizer", choices=("bytelevel", "o200k"), default="bytelevel")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    train_tokenizer(iter_texts(data_files(args.data)), args.output, args.vocab_size,
                    args.pretokenizer, args.overwrite)


if __name__ == "__main__":
    main()
