"""Small native JSON DOM with arena nodes and indexed object lookup."""

comptime JSON_NULL = 0
comptime JSON_BOOL = 1
comptime JSON_NUMBER = 2
comptime JSON_STRING = 3
comptime JSON_ARRAY = 4
comptime JSON_OBJECT = 5


@fieldwise_init
struct _Node(Copyable, Movable):
    var kind: Int
    var value: String
    var key: String
    var first: Int
    var next: Int
    var count: Int


def append_utf8(mut data: List[UInt8], cp: Int):
    """Append one Unicode scalar as UTF-8."""
    if cp < 0x80:
        data.append(UInt8(cp))
    elif cp < 0x800:
        data.append(UInt8(0xC0 | (cp >> 6)))
        data.append(UInt8(0x80 | (cp & 63)))
    elif cp < 0x10000:
        data.append(UInt8(0xE0 | (cp >> 12)))
        data.append(UInt8(0x80 | ((cp >> 6) & 63)))
        data.append(UInt8(0x80 | (cp & 63)))
    else:
        data.append(UInt8(0xF0 | (cp >> 18)))
        data.append(UInt8(0x80 | ((cp >> 12) & 63)))
        data.append(UInt8(0x80 | ((cp >> 6) & 63)))
        data.append(UInt8(0x80 | (cp & 63)))


struct JsonDocument(Movable):
    var _nodes: List[_Node]
    var _index: Dict[String, Int]
    var _items: List[Int]

    def __init__(out self):
        self._nodes = List[_Node]()
        self._index = Dict[String, Int]()
        self._items = List[Int]()

    @staticmethod
    def parse(text: String) raises -> JsonDocument:
        var doc = JsonDocument()
        var p = 0
        _ = _parse_value(text, p, doc, "", 0)
        _space(text, p)
        if p != text.byte_length():
            raise Error("Trailing JSON input")
        return doc^

    def root(self) -> Int:
        return 0

    def kind(self, node: Int) -> Int:
        return self._nodes[node].kind

    def find(self, node: Int, key: StringSlice) -> Optional[Int]:
        var k = String(node) + "\x00" + String(key)
        return self._index.get(k)

    def get(self, node: Int, key: StringSlice) raises -> Int:
        var value = self.find(node, key)
        if not value:
            raise Error("Missing JSON key: " + String(key))
        return value.value()

    def keys(self, node: Int) raises -> List[String]:
        if self.kind(node) != JSON_OBJECT:
            raise Error("Expected JSON object")
        var result = List[String](capacity=self._nodes[node].count)
        var n = self._nodes[node].first
        while n != -1:
            result.append(self._nodes[n].key)
            n = self._nodes[n].next
        return result^

    def len(self, node: Int) raises -> Int:
        if self.kind(node) != JSON_ARRAY and self.kind(node) != JSON_OBJECT:
            raise Error("Expected JSON array or object")
        return self._nodes[node].count

    def at(self, node: Int, i: Int) raises -> Int:
        if (
            self.kind(node) != JSON_ARRAY
            or i < 0
            or i >= self._nodes[node].count
        ):
            raise Error("JSON array index out of bounds")
        return self._items[self._nodes[node].first + i]

    def string(self, node: Int) raises -> String:
        if self.kind(node) != JSON_STRING:
            raise Error("Expected JSON string")
        return self._nodes[node].value

    def int(self, node: Int) raises -> Int:
        if self.kind(node) != JSON_NUMBER:
            raise Error("Expected JSON number")
        return Int(self._nodes[node].value)

    def float(self, node: Int) raises -> Float64:
        if self.kind(node) != JSON_NUMBER:
            raise Error("Expected JSON number")
        return Float64(self._nodes[node].value)

    def bool(self, node: Int) raises -> Bool:
        if self.kind(node) != JSON_BOOL:
            raise Error("Expected JSON boolean")
        return self._nodes[node].value == "true"


def _space(text: String, mut p: Int):
    var b = text.as_bytes()
    while p < len(b) and (b[p] == 32 or b[p] == 9 or b[p] == 10 or b[p] == 13):
        p += 1


def _hex4(text: String, mut p: Int) raises -> Int:
    if p + 4 > text.byte_length():
        raise Error("Truncated JSON escape")
    var n = 0
    for _ in range(4):
        var c = Int(text.as_bytes()[p])
        p += 1
        var digit = -1
        if c >= 48 and c <= 57:
            digit = c - 48
        elif c >= 65 and c <= 70:
            digit = c - 55
        elif c >= 97 and c <= 102:
            digit = c - 87
        if digit == -1:
            raise Error("Invalid JSON hex escape")
        n = n * 16 + digit
    return n


def _string(text: String, mut p: Int) raises -> String:
    var b = text.as_bytes()
    if p >= len(b) or b[p] != 34:
        raise Error("Expected JSON string")
    p += 1
    var start = p
    while p < len(b) and b[p] != 34 and b[p] != 92:
        if b[p] < 32:
            raise Error("Control character in JSON string")
        p += 1
    if p < len(b) and b[p] == 34:
        var value = String(text[byte=start:p])
        p += 1
        return value
    var data = List[UInt8]()
    for i in range(start, p):
        data.append(b[i])
    while p < len(b):
        var c = Int(b[p])
        p += 1
        if c == 34:
            return String(unsafe_from_utf8=Span(data))
        if c < 32:
            raise Error("Control character in JSON string")
        if c != 92:
            data.append(UInt8(c))
            continue
        if p == len(b):
            raise Error("Truncated JSON escape")
        c = Int(b[p])
        p += 1
        if c == 117:
            var cp = _hex4(text, p)
            if cp >= 0xD800 and cp <= 0xDBFF:
                if p + 2 > len(b) or b[p] != 92 or b[p + 1] != 117:
                    raise Error("Missing JSON low surrogate")
                p += 2
                var low = _hex4(text, p)
                if low < 0xDC00 or low > 0xDFFF:
                    raise Error("Invalid JSON low surrogate")
                cp = 0x10000 + ((cp - 0xD800) << 10) + low - 0xDC00
            elif cp >= 0xDC00 and cp <= 0xDFFF:
                raise Error("Unpaired JSON surrogate")
            append_utf8(data, cp)
        elif c == 34 or c == 92 or c == 47:
            data.append(UInt8(c))
        elif c == 98:
            data.append(8)
        elif c == 102:
            data.append(12)
        elif c == 110:
            data.append(10)
        elif c == 114:
            data.append(13)
        elif c == 116:
            data.append(9)
        else:
            raise Error("Invalid JSON escape")
    raise Error("Unterminated JSON string")


def _valid_number(text: String) -> Bool:
    var b = text.as_bytes()
    var p = 0
    if p < len(b) and b[p] == 45:
        p += 1
    if p == len(b):
        return False
    if b[p] == 48:
        p += 1
    elif b[p] >= 49 and b[p] <= 57:
        while p < len(b) and b[p] >= 48 and b[p] <= 57:
            p += 1
    else:
        return False
    if p < len(b) and b[p] == 46:
        p += 1
        var start = p
        while p < len(b) and b[p] >= 48 and b[p] <= 57:
            p += 1
        if start == p:
            return False
    if p < len(b) and (b[p] == 101 or b[p] == 69):
        p += 1
        if p < len(b) and (b[p] == 43 or b[p] == 45):
            p += 1
        var start = p
        while p < len(b) and b[p] >= 48 and b[p] <= 57:
            p += 1
        if start == p:
            return False
    return p == len(b)


def _parse_value(
    text: String, mut p: Int, mut doc: JsonDocument, key: String, depth: Int
) raises -> Int:
    if depth > 256:
        raise Error("JSON nesting too deep")
    _space(text, p)
    var b = text.as_bytes()
    if p >= len(b):
        raise Error("Missing JSON value")
    var n = len(doc._nodes)
    var c = Int(b[p])
    var kind: Int
    var value = String()
    if c == 123 or c == 91:
        kind = JSON_OBJECT if c == 123 else JSON_ARRAY
    elif c == 34:
        kind = JSON_STRING
        value = _string(text, p)
    else:
        var start = p
        while (
            p < len(b)
            and b[p] != 44
            and b[p] != 93
            and b[p] != 125
            and b[p] != 32
            and b[p] != 9
            and b[p] != 10
            and b[p] != 13
        ):
            p += 1
        value = String(text[byte=start:p])
        if value == "true" or value == "false":
            kind = JSON_BOOL
        elif value == "null":
            kind = JSON_NULL
        elif _valid_number(value):
            kind = JSON_NUMBER
        else:
            raise Error("Invalid JSON value")
    doc._nodes.append(_Node(kind, value, key, -1, -1, 0))
    if kind == JSON_ARRAY or kind == JSON_OBJECT:
        p += 1
        _space(text, p)
        var closer = 125 if kind == JSON_OBJECT else 93
        var last = -1
        var children = List[Int]()
        if p < len(b) and b[p] == UInt8(closer):
            p += 1
            return n
        while True:
            var child_key = String()
            if kind == JSON_OBJECT:
                child_key = _string(text, p)
                _space(text, p)
                if p >= len(b) or b[p] != 58:
                    raise Error("Missing JSON colon")
                p += 1
            var child = _parse_value(text, p, doc, child_key, depth + 1)
            if kind == JSON_ARRAY:
                children.append(child)
            if last == -1:
                doc._nodes[n].first = child
            else:
                doc._nodes[last].next = child
            last = child
            doc._nodes[n].count += 1
            if kind == JSON_OBJECT:
                doc._index[String(n) + "\x00" + child_key] = child
            _space(text, p)
            if p >= len(b):
                raise Error("Unterminated JSON container")
            if b[p] == UInt8(closer):
                p += 1
                break
            if b[p] != 44:
                raise Error("Missing JSON comma")
            p += 1
            _space(text, p)
        if kind == JSON_ARRAY:
            doc._nodes[n].first = len(doc._items)
            for child in children:
                doc._items.append(child)
    return n
