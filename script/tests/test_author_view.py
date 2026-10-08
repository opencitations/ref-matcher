"""Materialized views (--author-view, --author-title-view, --author-page-view, --volume-view,
--volume-page-view): each replaces exactly the triples it was built from; nothing else changes.
Surname views get the surname as a constant, one UNION branch per variant (QLever jumps to a
block only when the fixed columns are a prefix)."""
import re

PREFIX = 'PREFIX view: <https://qlever.cs.uni-freiburg.de/materializedView/>'
X = '^^<http://www.w3.org/2001/XMLSchema#string>'
CHAIN = (r'\?first_author foaf:familyName \?author_name \.\s*\?role pro:isHeldBy \?first_author \.'
         r'\s*\?br pro:isDocumentContextFor \?role \.')
norm = lambda s: ' '.join(s.replace(PREFIX, '').split())


def by_name(view, cols, names=('Wang',)):
    return ' UNION '.join(f'{{ SERVICE view:{view} {{ [ view:column-name "{n}"{X} ; {cols} ] }} '
                          f'BIND("{n}"{X} AS ?author_name) }}' for n in names)


def _matcher(T, **views):
    cfg = T.MatcherConfig()
    for k, v in views.items():
        setattr(cfg, k, v)
    return T.OpenCitationsMatcherThreadSafe(query_cache=None, config=cfg)


def _q(T, tipo, surname='Wang', **views):
    ref = T.Reference(first_author_lastname=surname, year='2015', volume='12', first_page='34',
                      article_title='Catalytic oxidation of benzene')
    return _matcher(T, **views).build_sparql_query(ref, tipo, use_doi=False)


TIPI = ('author_title', 'year_author_page', 'year_author_volume')


def test_author_view_replaces_only_the_author_chain(rmt):
    for t in TIPI:
        plain, view = _q(rmt, t), _q(rmt, t, author_view='autori')
        assert re.search(CHAIN, plain) and 'SERVICE' not in plain and PREFIX not in plain
        assert view.count(PREFIX) == 1 and not re.search(CHAIN, view)
        assert norm(re.sub(CHAIN, lambda _: by_name('autori', 'view:column-br ?br'), plain)) == norm(view)


def test_surname_variants_become_one_branch_each(rmt):
    q = _q(rmt, 'author_title', surname='Müller', author_view='autori')
    names = rmt.surname_variants('Müller')
    assert len(names) == 2 and q.count('SERVICE view:autori') == 2
    assert norm(by_name('autori', 'view:column-br ?br', names)) in norm(q)


def test_author_title_view_replaces_chain_and_title_join(rmt):
    plain, tv = _q(rmt, 'author_title'), _q(rmt, 'author_title', author_title_view='autori-titoli')
    title = r'\s*\?br dcterms:title \?title \.'
    new = by_name('autori-titoli', 'view:column-br ?br ; view:column-title ?title')
    assert norm(re.sub(CHAIN + title, lambda _: new, plain)) == norm(tv)
    for t in ('year_author_page', 'year_author_volume'):     # untouched by the title view
        assert norm(_q(rmt, t, author_title_view='autori-titoli')) == norm(_q(rmt, t))


def test_author_page_view_replaces_author_embodiment_page(rmt):
    plain, q = _q(rmt, 'year_author_page'), _q(rmt, 'year_author_page', author_page_view='autori-pagine')
    # VALUES ?author_name goes too: each branch fixes the surname itself
    emb = r'VALUES \?author_name \{[^}]*\}\s*' + CHAIN + r'\s*\?br frbr:embodiment \?embodiment \.\s*\?embodiment prism:startingPage '
    svc = lambda page: by_name('autori-pagine', f'view:column-page {page} ; view:column-br ?br ; '
                                                f'view:column-embodiment ?embodiment')
    back = re.sub(emb + re.escape(f'"34"{X}') + r' \.', lambda _: svc(f'"34"{X}'), plain)
    back = re.sub(emb + r'\?start_page \.', lambda _: svc('?start_page'), back)
    assert norm(back) == norm(q) and q.count(PREFIX) == 1


def test_volume_views_replace_only_the_volume_chains(rmt):
    T = rmt
    xs = re.escape(X)
    vol = r'\?volume fabio:hasSequenceIdentifier "12"' + xs + r' \.\s*\?issue frbr:partOf \?volume \.\s*\?br frbr:partOf \?issue \.'
    page = r'\s*\?br frbr:embodiment \?embodiment \.\s*\?embodiment prism:startingPage "34"' + xs + r' \.'
    views = dict(volume_view='volumi', volume_page_view='volumi-pagine')
    s_vp = (f'SERVICE view:volumi-pagine {{ [ view:column-vol "12"{X} ; view:column-page "34"{X} ; '
            f'view:column-br ?br ; view:column-embodiment ?embodiment ] }}')
    assert norm(re.sub(vol + page, lambda _: s_vp, _q(T, 'year_volume_page'))) == norm(_q(T, 'year_volume_page', **views))
    s_v = f'SERVICE view:volumi {{ [ view:column-vol "12"{X} ; view:column-br ?br ] }}'
    av = _q(T, 'year_author_volume', **views)
    assert norm(re.sub(vol, lambda _: s_v, _q(T, 'year_author_volume'))) == norm(av)
    assert _q(T, 'year_author_volume', author_view='autori', **views).count(PREFIX) == 1
