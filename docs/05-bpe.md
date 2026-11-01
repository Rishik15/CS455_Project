# BPE tokenizer

How `tools/prep_data` turns text into token ids. Code: `tools/pretokenize.c`, `tools/bpe_train.c`, `tools/prep_data.c`.

## Why BPE

The model needs a fixed vocabulary of 4096 ids.

- **One id per word** (the old prep) runs out: TinyStories has about 16K distinct words, so every rare word becomes "unknown" and the model can never write it.
- **One id per character** never runs out, but stories become about 4x longer. The 256-token context then covers far less of a story, and every step costs more.

Byte pair encoding sits in between. It starts with one token per character and keeps gluing together the pair of tokens that appears next to each other most often. Common words end up as one token (` the`, ` happy`). Rare words are spelled with a few common chunks (` skel` + `eton`). Any text can still be written, because every single character is in the vocab.

## Training, by example

Say the text holds these pieces, with counts:

| Piece | Count | Tokens at the start |
|---|---|---|
| ` the` | 10 | `␣` `t` `h` `e` |
| ` then` | 4 | `␣` `t` `h` `e` `n` |
| ` he` | 6 | `␣` `h` `e` |
| ` to` | 3 | `␣` `t` `o` |

1. Count every adjacent pair, weighted by the piece's count: `h e` = 10 + 4 + 6 = **20**, `␣ t` = 17, `t h` = 14, ...
2. The top pair `h e` becomes new token `he`. Replace it everywhere: ` the` is now `␣` `t` `he`.
3. Recount. Now `␣ t` = 17 is the top pair, so the next token is ` t`.
4. Recount. Now ` t he` = 14 wins, so the next token is ` the`.

Repeat until there are 4096 tokens. These are the real first merges on TinyStories: `he`, ` t`, ` a`, ` s`, ` w`, `nd`, ` the`, ...

The list of merges, in order, *is* the tokenizer. Each merge makes one new id.

## Pieces: merges stay inside words

Before BPE, `piece_length` splits text into pieces: a word, a number or one punctuation mark, each keeping the single space in front of it, plus `'\n'` on its own. Pairs are only counted inside a piece, so a token never spans two words (no `.\n` or `d the` tokens).

This also makes training fast. The tokenizer learns from the first 20M characters of training stories (`TOKENIZER_CHARS`), which hold only about 15K distinct pieces. Each step counts pairs over those, weighted by count, instead of over 20 MB of text. Training takes about 3 seconds.

The tokenizer text is a fixed amount on purpose: if a later run reads more of the dataset, the vocab stays the same.

## Ids

| Id | Token |
|---|---|
| 0 | end-of-story (written `<eos>` in `vocab.bin`) |
| 1 | `'\n'` |
| 2-96 | `' '` to `'~'` (ASCII 32-126, id = char - 30) |
| 97-4095 | merges, in the order they were learned |

There is no unknown token. Stories with any other character are dropped before training.

## Encoding new text

The training pieces already hold their final ids once training ends. A piece never seen in training (in `val.bin`, or a prompt later) goes through `bpe_encode`:

1. Start with one id per character.
2. Find the adjacent pair whose merge was learned earliest (lowest id).
3. Merge it, and repeat until no learned pair is left.

This gives exactly the ids that replaying the merges in training order would. That was checked on every training piece. Decoding is just printing each id's text back to back.

## Output files

All in `data/`, written by `tools/prep_data` (run from the repo root):

| File | Contents |
|---|---|
| `train.bin` | uint16 ids `EOS story EOS story ... EOS`, whole stories until 5M + 1 ids (5,000,319) |
| `val.bin` | same, every clean story in `TinyStories-valid.txt` (4,613,560 ids) |
| `vocab.bin` | int32 vocab size, then per id: 1-byte length, then the characters |
| `merges.bin` | int32 merge count, then per merge two uint16 (left, right); merge i makes id 97 + i |

`vocab.bin` is all that decoding needs. `merges.bin` is only needed to encode new text, such as a prompt in `sample/generate.c`.

`val.bin` is the whole validation set. It is about as big as `train.bin`, so the training loop should evaluate a fixed number of validation batches rather than all of it (a full pass is too slow for the serial run). Use the same batches in every version so the losses are comparable.

## Numbers from a run

| | |
|---|---|
| training text read | first 25 MiB, 27,494 clean stories |
| tokenizer text | first 22,450 stories, 20.0M characters, 14,776 distinct pieces |
| stories in `train.bin` | 22,003 |
| stories in `val.bin` | 20,754 of 21,989 (the rest have other characters) |
| characters per token | about 3.9 |
| time | about 3.5 s |
