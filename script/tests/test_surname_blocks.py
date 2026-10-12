"""--surname-block-min: author_title candidates from a local block equal QLever's REGEX over it."""
import asyncio
import random
import re

RX = r'REGEX\(\?title, "((?:[^"\\]|\\.)*)", "i"\)'


def _rows():
    random.seed(1)
    words = ['Cancer', 'apolipoprotein', 'lipoprotein', 'Müller', 'ÉTUDE', 'étude', 'C++', 'a.b', '(2010)',
             'Analysis', 'analyses', 'of', 'the', 'İstanbul', 'Kelvin', 'kelvin', 'straße', 'STRASSE', 'x-ray', 'X-Ray']
    rows = [(f'https://w3id.org/oc/meta/br/{i}', ' '.join(random.choices(words, k=random.randint(1, 8))))
            for i in range(3000)]
    rows.append(('https://w3id.org/oc/meta/br/x', 'Analysis\nof cancer'))   # newline in the title
    return rows


def test_block_matches_regex(rmt):
    rows = _rows()
    block = rmt.SurnameBlock(rows)
    for title in ['lipoprotein Analysis', 'ÉTUDE of cancer', 'C++ x-ray', 'a.b (2010)', 'straße',
                  'analysis cancer', 'İstanbul', 'Kelvin scale', 'Analyses of the X-RAY']:
        words = [w for w in title.split()[:4] if len(w) > 3] or [w for w in title.split() if len(w) > 3][:1]
        rx = re.compile('.*'.join(rmt.escape_regex_literal(w) for w in words), re.I)
        assert block.match(words, rx) == [br for br, t in rows if rx.search(t)], title


def test_candidates_from_real_query(rmt):
    rows = _rows()
    block = rmt.SurnameBlock(rows)
    cfg = rmt.MatcherConfig()
    cfg.author_title_view, cfg.surname_block_min = 'autori-titoli', 1
    m = rmt.OpenCitationsMatcherThreadSafe(config=cfg, query_cache=None)
    ref = rmt.Reference(first_author_lastname='Müller', article_title='Analysis of the lipoprotein x-ray', year='2010')
    q = m.build_sparql_query(rmt.normalize_reference_safe(ref), 'author_title')
    rx = re.compile(rmt._sparql_unquote(re.search(RX, q).group(1)), re.I)
    undated = {next(br for br, t in rows if rx.search(t))}     # a match without prism:publicationDate
    asked = []

    async def size(view, name):
        return 10

    async def get(view, name):
        asked.append(name)
        return block

    async def run(query, kind):
        assert kind == 'author_title:dates'
        return [{'br': {'value': b}} for b in re.findall(r'<([^>]+)>', query.split('VALUES')[1]) if b not in undated]
    m._block_size, m._get_block, m._execute_query = size, get, run
    out = asyncio.run(m._block_candidates('SELECT DISTINCT ?br WHERE' + q.partition('WHERE')[2]))
    expected = [br for br, t in rows if rx.search(t) and br not in undated]
    assert out == expected and len(expected) > 1
    assert asked == rmt.surname_variants(ref.first_author_lastname)
