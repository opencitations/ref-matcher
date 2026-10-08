"""--author-view: the author queries read the (name, br) view instead of the
familyName/isHeldBy/isDocumentContextFor chain; everything else stays the same."""
import re


def _queries(T, view):
    cfg = T.MatcherConfig()
    cfg.author_view = view
    m = T.OpenCitationsMatcherThreadSafe(query_cache=None, config=cfg)
    ref = T.Reference(first_author_lastname='Wang', year='2015', volume='12', first_page='34',
                      article_title='Catalytic oxidation of benzene')
    return {q: m.build_sparql_query(ref, q, use_doi=False)
            for q in ('author_title', 'year_author_page', 'year_author_volume')}


def test_author_view_replaces_only_the_author_chain(rmt):
    plain, view = _queries(rmt, ''), _queries(rmt, 'autori')
    chain = re.compile(r'\?first_author foaf:familyName \?author_name \.\s*\?role pro:isHeldBy \?first_author \.'
                       r'\s*\?br pro:isDocumentContextFor \?role \.')
    service = 'SERVICE view:autori { [ view:column-name ?author_name ; view:column-br ?br ] }'
    norm = lambda s: ' '.join(s.split())
    for q in plain:
        assert chain.search(plain[q]) and 'SERVICE' not in plain[q]
        assert service in view[q] and not chain.search(view[q])
        assert 'PREFIX view: <https://qlever.cs.uni-freiburg.de/materializedView/>' in view[q]
        # same query once the chain is swapped for the view (and the prefix dropped)
        back = chain.sub(service, plain[q])
        assert norm(back) == norm(view[q].replace('PREFIX view: <https://qlever.cs.uni-freiburg.de/materializedView/>', ''))
