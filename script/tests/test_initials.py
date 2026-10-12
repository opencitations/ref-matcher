"""--skip-initial-surnames: no author query when the surname is just an initial; nothing else changes."""


def _matcher(rmt, skip):
    cfg = rmt.MatcherConfig()
    cfg.skip_initial_surnames = skip
    return rmt.OpenCitationsMatcherThreadSafe(query_cache=None, config=cfg)


def test_initial_detection(rmt):
    for s in ('A.', 'A', 'C.S.', 'J.-P.'):
        assert rmt.is_initial_surname(s), s
    for s in ('Li', 'Ng', 'NG', 'Wang', 'li', ''):
        assert not rmt.is_initial_surname(s), s


def test_author_queries_skipped_only_when_asked(rmt):
    ref = rmt.Reference(first_author_lastname='A.', article_title='Analysis of lipoproteins',
                        year='2010', volume='12', first_page='34')
    author = ('author_title', 'year_author_page', 'year_author_volume')
    off, on = _matcher(rmt, False), _matcher(rmt, True)
    for q in author:
        assert off.build_sparql_query(ref, q) and on.build_sparql_query(ref, q) is None, q
    # the query without the author is untouched
    assert on.build_sparql_query(ref, 'year_volume_page') == off.build_sparql_query(ref, 'year_volume_page')
    # a real surname is never affected
    ref.first_author_lastname = 'Li'
    for q in author:
        assert on.build_sparql_query(ref, q) == off.build_sparql_query(ref, q), q
