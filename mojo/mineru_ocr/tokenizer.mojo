"""Qwen2 byte-level BPE, checkpoint added tokens, and NFC normalization."""

from .json import JsonDocument, append_utf8
from .unicode_tables import UnicodeTables


@fieldwise_init
struct _Merge(ImplicitlyCopyable):
    var rank: Int
    var token: Int


def _read_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def _ascii_fold(cp: Int) -> Int:
    if cp >= 65 and cp <= 90:
        return cp + 32
    # Unicode simple case folding used by the regex's (?i:...) group.
    if cp == 0x17F:
        return 115
    return cp


struct Qwen2Tokenizer(Movable):
    var _vocab: Dict[String, Int]
    var _tokens: List[String]
    var _merges: Dict[Int, _Merge]
    var _byte_ids: List[Int]
    var _byte_decoder: List[Int]
    var _added: List[String]
    var _added_ids: List[Int]
    var _special: Dict[Int, Bool]
    var _unicode: UnicodeTables

    def __init__(out self) raises:
        self._vocab = Dict[String, Int]()
        self._tokens = List[String]()
        self._merges = Dict[Int, _Merge]()
        self._byte_ids = List[Int](length=256, fill=0)
        self._byte_decoder = List[Int](length=512, fill=-1)
        self._added = List[String]()
        self._added_ids = List[Int]()
        self._special = Dict[Int, Bool]()
        self._unicode = UnicodeTables()

    @staticmethod
    def load(model_dir: String) raises -> Qwen2Tokenizer:
        """Load checkpoint vocab, ranked merges, and all 31 special tokens."""
        var tokenizer = Qwen2Tokenizer()
        var doc = JsonDocument.parse(_read_text(model_dir + "/vocab.json"))
        var keys = doc.keys(doc.root())
        tokenizer._tokens.resize(len(keys), "")
        for token in keys:
            var token_id = doc.int(doc.get(doc.root(), token))
            tokenizer._vocab[token] = token_id
            tokenizer._tokens[token_id] = token
        var added = JsonDocument.parse(
            _read_text(model_dir + "/added_tokens.json")
        )
        for token in added.keys(added.root()):
            var token_id = added.int(added.get(added.root(), token))
            tokenizer._added.append(token)
            tokenizer._added_ids.append(token_id)
            tokenizer._special[token_id] = True
            tokenizer._vocab[token] = token_id
        # GPT-2 ByteLevel's reversible byte-to-Unicode alphabet.
        var extra = 256
        for byte in range(256):
            var cp = byte
            if not (
                (byte >= 33 and byte <= 126)
                or (byte >= 161 and byte <= 172)
                or byte >= 174
            ):
                cp = extra
                extra += 1
            tokenizer._byte_decoder[cp] = byte
            var encoded = List[UInt8]()
            append_utf8(encoded, cp)
            tokenizer._byte_ids[byte] = tokenizer._vocab[
                String(unsafe_from_utf8=Span(encoded))
            ]
        var merges_text = _read_text(model_dir + "/merges.txt")
        var rank = 0
        for line in merges_text.split("\n"):
            if line.byte_length() == 0 or line.startswith("#version:"):
                continue
            var parts = line.split(" ")
            if len(parts) != 2:
                raise Error("Invalid BPE merges line")
            var left = tokenizer._vocab[String(parts[0])]
            var right = tokenizer._vocab[String(parts[1])]
            var combined = tokenizer._vocab[String(parts[0]) + String(parts[1])]
            tokenizer._merges[(left << 32) | right] = _Merge(rank, combined)
            rank += 1
        return tokenizer^

    def token_id(self, token: String) raises -> Int:
        return self._vocab[token]

    def _bpe(self, text: StringSlice, mut result: List[Int]):
        var ids = List[Int](capacity=text.byte_length())
        for byte in text.as_bytes():
            ids.append(self._byte_ids[Int(byte)])
        while len(ids) > 1:
            var best = 0x7FFFFFFFFFFFFFFF
            var position = -1
            var merged_id = -1
            for i in range(len(ids) - 1):
                var merge = self._merges.get((ids[i] << 32) | ids[i + 1])
                if merge and merge.value().rank < best:
                    best = merge.value().rank
                    position = i
                    merged_id = merge.value().token
            if position == -1:
                break
            ids[position] = merged_id
            _ = ids.pop(position + 1)
        for token in ids:
            result.append(token)

    def _encode_ordinary(self, text: String, mut result: List[Int]) raises:
        if text.byte_length() == 0:
            return
        var normalized = self._unicode.normalize_nfc(text)
        var cps = List[Int]()
        var offsets = List[Int]()
        var offset = 0
        var classes = List[Int]()
        for cp_value in normalized.codepoints():
            var cp = Int(cp_value)
            cps.append(cp)
            offsets.append(offset)
            if cp < 0x80:
                offset += 1
            elif cp < 0x800:
                offset += 2
            elif cp < 0x10000:
                offset += 3
            else:
                offset += 4
            # 1: Letter, 2: Number, 3: White_Space, 0: other.
            var kind = 0
            if self._unicode.is_letter(cp):
                kind = 1
            elif self._unicode.is_number(cp):
                kind = 2
            elif self._unicode.is_space(cp):
                kind = 3
            classes.append(kind)
        offsets.append(offset)
        var i = 0
        var count = len(cps)
        while i < count:
            var end = i
            # (?i:'s|'t|'re|'ve|'m|'ll|'d)
            if cps[i] == 39 and i + 1 < count:
                var c = _ascii_fold(cps[i + 1])
                if c == 115 or c == 116 or c == 109 or c == 100:
                    end = i + 2
                elif i + 2 < count:
                    var d = _ascii_fold(cps[i + 2])
                    if (
                        (c == 114 and d == 101)
                        or (c == 118 and d == 101)
                        or (c == 108 and d == 108)
                    ):
                        end = i + 3
            # [^\r\n\p{L}\p{N}]?\p{L}+
            if end == i:
                var start = i
                if (
                    classes[i] != 1
                    and classes[i] != 2
                    and cps[i] != 10
                    and cps[i] != 13
                ):
                    start += 1
                if start < count and classes[start] == 1:
                    end = start + 1
                    while end < count and classes[end] == 1:
                        end += 1
            # \p{N}
            if end == i and classes[i] == 2:
                end = i + 1
            #  ?[^\s\p{L}\p{N}]+[\r\n]*
            if end == i:
                var start = i + 1 if cps[i] == 32 else i
                if start < count and classes[start] == 0:
                    end = start + 1
                    while end < count and classes[end] == 0:
                        end += 1
                    while end < count and (cps[end] == 10 or cps[end] == 13):
                        end += 1
            if end == i and classes[i] == 3:
                var space_end = i
                var newline_end = i
                while space_end < count and classes[space_end] == 3:
                    if cps[space_end] == 10 or cps[space_end] == 13:
                        newline_end = space_end + 1
                    space_end += 1
                # \s*[\r\n]+; otherwise \s+(?!\S), then \s+.
                if newline_end > i:
                    end = newline_end
                elif space_end < count and space_end - i > 1:
                    end = space_end - 1
                else:
                    end = space_end
            if end <= i:
                raise Error("Unmatched tokenizer codepoint")
            self._bpe(normalized[byte = offsets[i] : offsets[end]], result)
            i = end

    def encode(self, text: String) raises -> List[Int]:
        """Encode without inserting tokens; embedded checkpoint specials match.
        """
        var result = List[Int]()
        var start = 0
        var p = 0
        var bytes = text.as_bytes()
        while p < len(bytes):
            var matched = -1
            var longest = 0
            if bytes[p] == 60:
                for i in range(len(self._added)):
                    var n = self._added[i].byte_length()
                    if n > longest and p + n <= len(bytes):
                        var equal = True
                        var candidate = self._added[i].as_bytes()
                        for j in range(n):
                            if bytes[p + j] != candidate[j]:
                                equal = False
                                break
                        if equal:
                            matched = i
                            longest = n
            if matched != -1:
                self._encode_ordinary(String(text[byte=start:p]), result)
                result.append(self._added_ids[matched])
                p += longest
                start = p
            else:
                p += 1
        self._encode_ordinary(String(text[byte=start:p]), result)
        return result^

    def decode(self, ids: List[Int]) raises -> String:
        """Decode the concatenated byte stream, skipping all special tokens."""
        var bytes = List[UInt8]()
        for token in ids:
            if token < 0 or token > 0xFFFFFFFF:
                raise Error("Token ID outside unsigned 32-bit range")
            if token in self._special:
                continue
            if token >= len(self._tokens):
                # HF silently ignores IDs absent from the vocabulary.
                continue
            for cp in self._tokens[token].codepoints():
                var value = Int(cp)
                if (
                    value >= len(self._byte_decoder)
                    or self._byte_decoder[value] < 0
                ):
                    raise Error("Invalid ByteLevel token")
                bytes.append(UInt8(self._byte_decoder[value]))
        return String(from_utf8_lossy=Span(bytes))
