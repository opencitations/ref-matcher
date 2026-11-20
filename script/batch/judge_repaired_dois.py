"""Judges the repaired DOIs citation by citation, for merge_citations.py --judgement.

For each citation whose repaired DOI is on Meta: the points of the data written in the reference (surname,
title, year, volume, page) against the Meta work of that DOI, field by field with the tool's
calculate_matching_score and without the DOI's 15 points. It replaces the bulk's surname check, which is made
once per DOI: one citation with the right surname makes all the citations of that DOI agree, also those with
only the DOI.

    uv run --with duckdb --with pyarrow python script/batch/judge_repaired_dois.py \
        <merge output> dois_to_verify.jsonl <meta tables> repaired_judgement.parquet [--limit 100000]

Inputs: a merge already made (merge_with_doi.parquet: the citations whose source starts with "repaired DOI";
merge_con_doi.parquet of the Italian version is read too), the whole references (extract_dois_to_verify.py's
.jsonl) and Meta's tables (export_meta_tables.sh). Output: citing_work, ref_id, repaired_doi, ref_points (empty
if the reference has nothing to compare with the Meta work) and ref_fields (the fields that agree, for whoever
checks by hand). Equal (DOI, reference) pairs are computed once. The repaired DOIs come from the DOI run, so
this is run once. --limit N takes N citations (always the same ones) to measure the time.
6,033,909 citations of the June 2026 dump took 3 minutes with 4 processes.
"""
import argparse, json, os, sys, time
from multiprocessing import Pool
import duckdb
import pyarrow as pa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import match_bulk as B                    # the tool's functions (score, Reference, Meta views), no network

FIELDS = {'surname': 'author_name', 'title': 'title', 'year': 'pub_date', 'volume': 'volume_num',
          'page': 'start_page'}
META = (('title', 'SELECT br, title FROM m_titles'), ('pub_date', 'SELECT br, date FROM m_dates'),
        ('volume_num', 'SELECT br, vol FROM m_volumes'),
        ('start_page', 'SELECT br, start FROM m_pages WHERE start IS NOT NULL'),
        ('author_name', f"SELECT br, name FROM m_authors WHERE role = '{B.AUTHOR}'"))


def reference(r):
    """The matcher's Reference from a Crossref reference (like read_json_folder)."""
    y = B.P._extract_year_from_data(r)
    return B.T.Reference(year=str(y) if y else '', first_author_lastname=B.P._extract_author(r),
                         article_title=B.P._extract_title(r), volume=r.get('volume', '') or '',
                         first_page=B.P._extract_page(r), unstructured=r.get('unstructured', '') or '')


def points(ref, values):
    """{field: points, or None if the reference or Meta lack that field}; values: {matcher field: [values]}."""
    has = {'surname': ref.first_author_lastname, 'title': ref.get_main_title(), 'year': ref.year,
           'volume': ref.volume, 'page': ref.first_page}
    return {c: (max(B.M.calculate_matching_score(ref, {f: {'value': x}}) for x in values[f])
                if has[c] and values.get(f) else None) for c, f in FIELDS.items()}


def judge(rows):
    """[(id, reference JSON, titles, dates, volumes, pages, surnames)] -> [(id, ref_points, ref_fields)]."""
    out = []
    for i, ref, *vals in rows:
        p = points(reference(json.loads(ref)), {f: v or [] for (f, _), v in zip(META, vals)})
        v = [x for x in p.values() if x is not None]
        out.append((i, sum(v) if v else None, ', '.join(c for c, x in p.items() if x)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('merge', help='folder of a merge (merge_with_doi.parquet)')
    ap.add_argument('jsonl', help="extract_dois_to_verify.py's .jsonl (the whole references)")
    ap.add_argument('tables', help="Meta's tables")
    ap.add_argument('output', help='.parquet')
    ap.add_argument('--limit', type=int, help='only N citations, to measure the time')
    ap.add_argument('--processes', type=int, default=4)
    ap.add_argument('--temp', default=None, help="DuckDB's temporary files")
    ap.add_argument('--memory', default='20GB')
    a = ap.parse_args()
    t0 = time.time()
    con = duckdb.connect()
    for s in (f"memory_limit='{a.memory}'", f'threads={a.processes}', 'preserve_insertion_order=false'):
        con.execute(f'SET {s}')
    if a.temp:
        con.execute(f"SET temp_directory='{a.temp}'")
    B.meta_views(con, a.tables)
    m = os.path.join(a.merge, 'merge_with_doi.parquet')
    src, cond = (m, "source LIKE 'repaired DOI%'") if os.path.exists(m) else \
        (os.path.join(a.merge, 'merge_con_doi.parquet'), "fonte LIKE 'DOI riparato%'")
    col = 'source' if src == m else 'fonte AS source'
    limit = f'ORDER BY hash(citing_work, ref_id) LIMIT {a.limit}' if a.limit else ''
    con.execute(f"""CREATE TEMP TABLE occ AS SELECT citing_work, ref_id, cited_doi AS doi, {col}
                    FROM '{src}' WHERE {cond} {limit}""")
    con.execute(f"""CREATE TEMP TABLE o2 AS SELECT o.*, r.reference FROM occ o JOIN read_json('{a.jsonl}',
        format='newline_delimited', columns={{'citing_work': 'VARCHAR', 'ref_id': 'VARCHAR', 'reference': 'JSON'}}) r
        USING (citing_work, ref_id)""")
    n_occ, n_ref = (con.execute(f'SELECT count(*) FROM {t}').fetchone()[0] for t in ('occ', 'o2'))
    print(f'citations with a repaired DOI: {n_occ:,}, found in the .jsonl: {n_ref:,}, {time.time() - t0:.0f} s',
          flush=True)
    con.execute('CREATE TEMP TABLE uq AS SELECT row_number() OVER () AS id, * FROM (SELECT DISTINCT doi, reference FROM o2)')
    con.execute('CREATE TEMP TABLE md AS SELECT DISTINCT lower(doi) AS doi, br FROM m_doi '
                'WHERE lower(doi) IN (SELECT doi FROM uq)')
    for f, sql in META:
        con.execute(f"""CREATE TEMP TABLE f_{f} AS SELECT md.doi, list(DISTINCT CAST(s.x AS VARCHAR)) AS v
                        FROM ({sql}) AS s(br, x) JOIN md USING (br) GROUP BY md.doi""")
    n_uq = con.execute('SELECT count(*) FROM uq').fetchone()[0]
    print(f'distinct (DOI, reference) pairs: {n_uq:,}, {time.time() - t0:.0f} s', flush=True)

    sel = ', '.join(f'f_{f}.v AS {f}' for f, _ in META)          # distinct names: to_pydict is a dict
    join = ' '.join(f'LEFT JOIN f_{f} USING (doi)' for f, _ in META)
    q = con.execute(f'SELECT uq.id, CAST(uq.reference AS VARCHAR) AS ref, {sel} FROM uq {join}')
    reader = q.to_arrow_reader(20000) if hasattr(q, 'to_arrow_reader') else q.fetch_record_batch(20000)

    def chunks():
        for b in reader:
            d = b.to_pydict()
            rows = list(zip(d['id'], d['ref'], *(d[f] for f, _ in META)))
            for k in range(0, len(rows), 2000):
                yield rows[k:k + 2000]

    res = []
    with Pool(a.processes) as pool:
        for r in pool.imap_unordered(judge, chunks()):
            res.extend(r)
            if len(res) % 500000 < len(r):
                print(f'  {len(res):,}/{n_uq:,}, {time.time() - t0:.0f} s', flush=True)
    con.register('pt', pa.table({'id': [x[0] for x in res], 'ref_points': pa.array([x[1] for x in res], pa.int64()),
                                 'ref_fields': [x[2] for x in res]}))
    con.execute(f"""COPY (SELECT o.citing_work, o.ref_id, o.doi AS repaired_doi, pt.ref_points, pt.ref_fields, o.source
                    FROM o2 o JOIN uq USING (doi, reference) JOIN pt USING (id)) TO '{a.output}' (FORMAT parquet)""")
    print(f'written {a.output}, {time.time() - t0:.0f} s\n')
    print('By current source (rule: high from 10 points, medium 1-9, to check 0, empty: nothing to compare):')
    print(f"{'current source':<46}{'10+':>12}{'1-9':>12}{'0':>12}{'nothing':>12}")
    for s, *k in con.execute(f"""SELECT source, count(*) FILTER (WHERE ref_points >= 10),
            count(*) FILTER (WHERE ref_points BETWEEN 1 AND 9), count(*) FILTER (WHERE ref_points = 0),
            count(*) FILTER (WHERE ref_points IS NULL) FROM '{a.output}' GROUP BY 1 ORDER BY count(*) DESC""").fetchall():
        print(f'{s:<46}' + ''.join(f'{x:>12,}' for x in k))


if __name__ == '__main__':
    if sys.argv[1:2] == ['test']:                 # the field comparison, without Meta
        ref = B.T.Reference(year='2019', first_author_lastname='Rossi', article_title='A study of things',
                            volume='12', first_page='34')
        p = points(ref, {'author_name': ['Bianchi', 'Rossi'], 'title': ['A Study Of Things'], 'pub_date': ['2018-05-01'],
                         'volume_num': ['13'], 'start_page': []})
        assert p == {'surname': 7, 'title': 14, 'year': 0, 'volume': 0, 'page': None}, p
        assert judge([(1, '{"author": "Verdi", "year": "2020", "volume": "5"}', None, ['2020'], ['5'], None,
                       ['Verdi'])]) == [(1, 11, 'surname, year, volume')]
        assert judge([(2, '{"DOI": "10.1/x"}', ['T'], ['2020'], None, None, ['Verdi'])]) == [(2, None, '')]
        print('ok')
    else:
        main()
