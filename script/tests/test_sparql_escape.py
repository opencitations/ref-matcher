"""Escaped literals must survive SPARQL's two decoding steps: \\uXXXX codepoint escapes first
(over the whole query text, also inside strings), then the string escapes (\\\\, \\", \\').
With the old \\\\ escaping a DOI holding a literal '\\u0026lt;' decoded to '\\&lt;' -> HTTP 400."""
import re

BS = '\\'
TRICKY = ['10.1002/(sici)x' + BS + 'u0026lt;350::aid' + BS + 'u0026gt;3.0.co;2-d',
          '10.1002/rse2.74' + BS + 'u2028', 'a "quoted" O' + "'" + 'Brien ' + BS + ' end']


def sparql_decode(lit):
    """What a SPARQL parser makes of the text between the quotes."""
    lit = re.sub(r'\\u([0-9A-Fa-f]{4})', lambda m: chr(int(m.group(1), 16)), lit)  # step 1
    out, i = [], 0
    while i < len(lit):                                                          # step 2
        if lit[i] == BS:
            nxt = lit[i + 1]
            assert nxt in BS + '"\'', f'invalid escape \\{nxt} -> HTTP 400'
            out.append(nxt); i += 2
        else:
            out.append(lit[i]); i += 1
    return ''.join(out)


def test_tool_and_bulk_escape_round_trip(rmt):
    import bulk_doi                      # on sys.path once rmt has loaded the tool
    for s in TRICKY:
        assert sparql_decode(rmt.sparql_quote_escape(s)) == s
        assert sparql_decode(bulk_doi._esc(s)) == s
        assert sparql_decode(rmt.OpenCitationsMatcherThreadSafe._esc(s)) == ' '.join(s.split())
