"""--grobid-page: GROBID's page fills an empty Crossref page, never replaces one."""


def _merged(rmt, on, crossref_page):
    cfg = rmt.MatcherConfig()
    cfg.grobid_page = on
    p = rmt.ReferenceProcessor(use_grobid=False, config=cfg)
    ref = rmt.Reference(first_page=crossref_page)
    p._merge_reference_data(ref, rmt.Reference(first_page='34'))
    return ref.first_page


def test_grobid_page(rmt):
    assert _merged(rmt, False, '') == ''
    assert _merged(rmt, True, '') == '34'
    assert _merged(rmt, True, '12') == '12'
