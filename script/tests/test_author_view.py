"""--author-view / --author-title-view: the author queries read QLever materialized views instead
of the familyName/isHeldBy/isDocumentContextFor chain (and the title join); nothing else changes."""
import re

PREFIX = 'PREFIX view: <https://qlever.cs.uni-freiburg.de/materializedView/>'
CHAIN = (r'\?first_author foaf:familyName \?author_name \.\s*\?role pro:isHeldBy \?first_author \.'
         r'\s*\?br pro:isDocumentContextFor \?role \.')
SERVICE = 'SERVICE view:autori { [ view:column-name ?author_name ; view:column-br ?br ] }'
SERVICE_T = ('SERVICE view:autori-titoli { [ view:column-name ?author_name ; view:column-br ?br ; '
             'view:column-title ?title ] }')
norm = lambda s: ' '.join(s.replace(PREFIX, '').split())


def _queries(T, view='', title_view=''):
    cfg = T.MatcherConfig()
    cfg.author_view, cfg.author_title_view = view, title_view
    m = T.OpenCitationsMatcherThreadSafe(query_cache=None, config=cfg)
    ref = T.Reference(first_author_lastname='Wang', year='2015', volume='12', first_page='34',
                      article_title='Catalytic oxidation of benzene')
    return {q: m.build_sparql_query(ref, q, use_doi=False)
            for q in ('author_title', 'year_author_page', 'year_author_volume')}


def test_author_view_replaces_only_the_author_chain(rmt):
    plain, view = _queries(rmt), _queries(rmt, view='autori')
    for q in plain:
        assert re.search(CHAIN, plain[q]) and 'SERVICE' not in plain[q] and PREFIX not in plain[q]
        assert PREFIX in view[q] and not re.search(CHAIN, view[q])
        assert norm(re.sub(CHAIN, SERVICE, plain[q])) == norm(view[q])


def test_volume_views_replace_only_the_volume_chains(rmt):
    T = rmt
    xs = r'\^\^<http://www\.w3\.org/2001/XMLSchema#string>'
    vol = r'\?volume fabio:hasSequenceIdentifier "12"' + xs + r' \.\s*\?issue frbr:partOf \?volume \.\s*\?br frbr:partOf \?issue \.'
    page = r'\s*\?br frbr:embodiment \?embodiment \.\s*\?embodiment prism:startingPage "34"' + xs + r' \.'
    X = '^^<http://www.w3.org/2001/XMLSchema#string>'
    cfg = T.MatcherConfig()
    cfg.volume_view, cfg.volume_page_view = 'volumi', 'volumi-pagine'
    m = T.OpenCitationsMatcherThreadSafe(query_cache=None, config=cfg)
    plain = _queries(T)
    ref = T.Reference(first_author_lastname='Wang', year='2015', volume='12', first_page='34',
                      article_title='Catalytic oxidation of benzene')
    vp = m.build_sparql_query(ref, 'year_volume_page', use_doi=False)
    pv = T.OpenCitationsMatcherThreadSafe(query_cache=None).build_sparql_query(ref, 'year_volume_page', use_doi=False)
    s_vp = (f'SERVICE view:volumi-pagine {{ [ view:column-vol "12"{X} ; view:column-page "34"{X} ; '
            f'view:column-br ?br ; view:column-embodiment ?embodiment ] }}')
    assert norm(re.sub(vol + page, lambda _: s_vp, pv)) == norm(vp)
    s_v = f'SERVICE view:volumi {{ [ view:column-vol "12"{X} ; view:column-br ?br ] }}'
    av = m.build_sparql_query(ref, 'year_author_volume', use_doi=False)
    assert norm(re.sub(vol, lambda _: s_v, plain['year_author_volume'])) == norm(av)
    assert av.count('PREFIX view:') == 1


def test_author_title_view_replaces_chain_and_title_join(rmt):
    plain, tv = _queries(rmt), _queries(rmt, title_view='autori-titoli')
    title = r'\s*\?br dcterms:title \?title \.'
    assert norm(re.sub(CHAIN + title, SERVICE_T, plain['author_title'])) == norm(tv['author_title'])
    assert 'FILTER(REGEX(?title' in tv['author_title']
    for q in ('year_author_page', 'year_author_volume'):     # untouched by the title view
        assert norm(tv[q]) == norm(plain[q])
