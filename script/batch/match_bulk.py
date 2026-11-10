"""Batch matching: the tool's rules without one query per reference.

The candidates of each of the tool's queries come from joins on Meta's tables exported from QLever
(export_meta_tables.sh), the score from the tool's calculate_matching_score, the choice as in the
tool's loop (query order, stop at 26, threshold 23.4). Then GROBID on the unmatched ones with text.
Writes <file>_matches.csv and <file>_matches_unmatched.csv like the tool.

    uv run --with duckdb --with pyarrow python script/batch/match_bulk.py \
        <folder of Crossref JSON> <output> --tables <meta tables> --grobid-config data/grobid/config.json

With --batches the input is the folder of prepare_dump.py (deduplicated dump references) and every batch
writes batch_KKKK_matches.csv and _unmatched.csv, one row per occurrence; batches that already have their
CSVs are skipped, so after an interruption the same command resumes.

Like the tool, the candidates of a query are looked at in IRI order (the order in which QLever returns
them): the first one reaching 26 stops the search, on a tie the smaller IRI wins. With --best-candidate
there is no stop: all the candidates of all the queries are scored and the one with the most points wins
(on a tie the smaller IRI); on the held-out set this gives exactly the paper's held-out table, while the
early stop sometimes picks a Meta duplicate of the right work (10 of 1,498 at 26), at about +25% score time.

Known differences from the tool: no "without year" pass (it only redoes author_title, which does not
filter on the year: same candidates, same score minus the year, same threshold); for page ranges
("123-130") each part is compared instead of the substring ("123-" and "-123"); for each candidate work
the best row is taken (the tool takes the first reaching 26). On the paper's dataset the metrics and
Tables 4-7 are identical to the tool's, with and without DOI.
"""
import argparse, csv, gc, glob, io, json, logging, multiprocessing as mp, os, shutil, sys, tempfile, time
import urllib.error, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # script/
logging.disable(logging.CRITICAL)
import ReferenceMatchingTool as T

# in the tool's order; the first two only with a DOI (in the dump: found by GROBID in the text)
QT = ('year_and_doi', 'doi_title', 'author_title', 'year_author_page', 'year_volume_page', 'year_author_volume')
DATE_REQUIRED = set(QT) - {'doi_title'}          # ?br prism:publicationDate outside the OPTIONALs
YEAR_FILTERED = DATE_REQUIRED - {'author_title'}  # FILTER(STRSTARTS(?pub_date, year ±1))
AUTHORS_ONLY = {'year_and_doi', 'doi_title', 'year_volume_page'}   # OPTIONAL_AUTHOR: pro:withRole pro:author
THRESHOLD = 26
MIN_ACCEPT = THRESHOLD * 0.9    # the lowest threshold a pass accepts (threshold_adjustment)
USE_DOI = True                  # False: no DOI queries (like the tool's use_doi=False); the 15 points stay
AUTHOR = 'http://purl.org/spar/pro/author'
MIN_SPLIT = 2000                # below this many references a full memory is not split any more
BEST = False                    # --best-candidate: no early stop, the candidate with the most points wins
# Meta's tables (export_meta_tables.sh); tables exported before the English names are read too
TABLES = {'authors': 'autori', 'titles': 'titoli', 'dates': 'date', 'volumes': 'volumi', 'pages': 'pagine',
          'doi': 'doi'}

cfg = T.MatcherConfig()
cfg.skip_initial_surnames = cfg.grobid_page = True      # as in the dump run
M = T.OpenCitationsMatcherThreadSafe(config=cfg)          # only for the score: no network
P = T.ReferenceProcessor(use_grobid=False, config=cfg)


def meta_views(con, tab):
    """Views m_<table> on Meta's tables in `tab`: <English name>.parquet, or the old name (autori.parquet,
    whose column ruolo becomes role)."""
    for new, old in TABLES.items():
        p = os.path.join(tab, f'{new}.parquet')
        if os.path.exists(p):
            con.execute(f"CREATE OR REPLACE TEMP VIEW m_{new} AS SELECT * FROM '{p}'")
        else:
            p = os.path.join(tab, f'{old}.parquet')
            cols = 'name, br, ruolo AS role' if new == 'authors' else '*'
            con.execute(f"CREATE OR REPLACE TEMP VIEW m_{new} AS SELECT {cols} FROM '{p}'")


def read_json_folder(folder):
    """[(file, ref_id, Reference)] like _process_crossref_file."""
    out = []
    for p in sorted(glob.glob(os.path.join(folder, '*.json'))):
        f = os.path.basename(p)[:-5]
        for i, r in enumerate(json.load(open(p, encoding='utf-8'))['message']['reference'], 1):
            y = P._extract_year_from_data(r)
            out.append((f, f'ref_{i}', T.Reference(
                year=str(y) if y else '', first_author_lastname=P._extract_author(r),
                article_title=P._extract_title(r), volume=r.get('volume', ''),
                first_page=P._extract_page(r), doi=r.get('DOI', ''), unstructured=r.get('unstructured', ''))))
    return out


def keys(ref):
    """The fields the tool's queries use, computed like build_sparql_query."""
    y = M._extract_year(ref.year)
    title = (ref.get_main_title() or '')[:500]
    words = [w for w in title.split()[:4] if len(w) > 3] or [w for w in title.split() if len(w) > 3][:1]
    years = sorted({max(y - 1, T.DEFAULT_YEAR_RANGE[0]), y, min(y + 1, T.DEFAULT_YEAR_RANGE[1])}) if y else []
    return dict(year=y, years=[str(a) for a in years], ya=years[0] if y else None, yb=years[-1] if y else None,
                vol=ref.volume or None, page=ref.first_page or None, doi=(ref.doi or None) if USE_DOI else None,
                pattern='.*'.join(T.escape_regex_literal(w) for w in words) if title and words else None,
                # the longest word, lower case: a title without it does not pass the REGEX
                # (like SurnameBlock.match), and the filter costs much less than the REGEX
                word=T._fold(max(words, key=len)) if title and words else None,
                variants=T.surname_variants(ref.first_author_lastname),
                initial=cfg.skip_initial_surnames and T.is_initial_surname(ref.first_author_lastname))


def table(con, name, rows, empty=None):
    """Creates the temporary table `name` from a list of dicts (Arrow: no row-by-row inserts).
    `empty`: the columns (DDL) if there may be no rows, e.g. no surname in the batch."""
    if not rows and empty:
        con.execute(f'CREATE OR REPLACE TEMP TABLE {name} ({empty})')
        return
    t = pa.Table.from_pylist(rows)
    # a column that is all empty (doi in pass 1, years in the pass without year) would have a null type:
    # years are numbers, the rest text
    t = t.cast(pa.schema([pa.field(f.name, pa.int64() if f.name in ('yr', 'ya', 'yb') else pa.string())
                          if pa.types.is_null(f.type) else f for f in t.schema]))
    con.register('_rows', t)
    con.execute(f'CREATE OR REPLACE TEMP TABLE {name} AS SELECT * FROM _rows')
    con.unregister('_rows')


def candidates(con, refs):
    """{i: {query: [(br, author_name, start_page, title)]}} for the references `refs` (index -> Reference).
    Each query as in the tool, year included (FILTER on ?pub_date: year ±1)."""
    K = {i: keys(r) for i, r in refs.items()}
    table(con, 'rif', [dict(rid=i, yr=k['year'], ya=k['ya'], yb=k['yb'], vol=k['vol'], page=k['page'],
                            doi=k['doi'], pattern=k['pattern'], word=k['word'], initial=k['initial'])
                       for i, k in K.items()])
    table(con, 'var', [dict(rid=i, name=v) for i, k in K.items() for v in k['variants']], 'rid BIGINT, name VARCHAR')
    # the rows of Meta that may be needed, one read per table
    con.execute("CREATE OR REPLACE TEMP TABLE a AS SELECT name, br, role FROM m_authors "
                "WHERE name IN (SELECT name FROM var)")
    con.execute("CREATE OR REPLACE TEMP TABLE v AS SELECT vol, br FROM m_volumes "
                "WHERE vol IN (SELECT vol FROM rif)")
    con.execute("""CREATE OR REPLACE TEMP TABLE pg AS SELECT br, start FROM m_pages
        WHERE start IN (SELECT page FROM rif) OR (contains(start, '-') AND
              (split_part(start, '-', 1) IN (SELECT page FROM rif) OR split_part(start, '-', 2) IN (SELECT page FROM rif)))""")
    con.execute("CREATE OR REPLACE TEMP TABLE pgi AS SELECT br, start FROM pg WHERE contains(start, '-')")   # ranges
    con.execute("CREATE OR REPLACE TEMP TABLE t AS SELECT br, title, lower(title) AS low "
                "FROM m_titles WHERE br IN (SELECT br FROM a)")
    con.execute("CREATE OR REPLACE TEMP TABLE dd AS SELECT br, doi FROM m_doi "
                "WHERE doi IN (SELECT doi FROM rif)")
    con.execute("CREATE OR REPLACE TEMP TABLE td AS SELECT br, title FROM m_titles "
                "WHERE br IN (SELECT br FROM dd)")
    # publication year of the works that can be candidates
    con.execute("""CREATE OR REPLACE TEMP TABLE dy AS SELECT DISTINCT br, TRY_CAST(substr(date, 1, 4) AS INT) AS y
        FROM m_dates WHERE br IN (SELECT br FROM a UNION SELECT br FROM v UNION SELECT br FROM dd)""")
    year = 'JOIN dy ON dy.br = {b}.br AND dy.y BETWEEN r.ya AND r.yb'
    # author_title without comparing each reference with all the titles of its surname (13 billion
    # pairs on 200,000 references): an index (surname, first 4 letters, word, work) of the title words;
    # the longest alphanumeric part of the reference's longest word (key) must START a word of the
    # title. Then the tool's REGEX.
    # ponytail: misses the key inside a word ("Thermo Solutal" / "Thermosolutal": 1 in 1,397);
    # the few references with a key under 4 letters (short) do the full scan.
    W = r"'[\pL\pN_]+'"
    con.execute(f"""CREATE OR REPLACE TEMP TABLE rk AS SELECT rid, arg_max(w, length(w)) AS wkey
        FROM (SELECT rid, unnest(regexp_extract_all(word, {W})) AS w FROM rif
              WHERE word IS NOT NULL AND pattern IS NOT NULL AND NOT initial) GROUP BY rid""")
    con.execute("""CREATE OR REPLACE TEMP TABLE short_refs AS SELECT rid FROM rif
        WHERE NOT initial AND pattern IS NOT NULL AND rid NOT IN (SELECT rid FROM rk WHERE length(wkey) >= 4)""")
    # only the words that start like the key of some reference of the same surname
    # (filtered while extracting, without DISTINCT: duplicates go away later, on the pairs)
    con.execute("""CREATE OR REPLACE TEMP TABLE kp AS SELECT DISTINCT var.name, left(rk.wkey, 4) AS p
        FROM rk JOIN var USING (rid) WHERE length(rk.wkey) >= 4""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE tk AS SELECT y.name, y.p, y.w, y.br FROM (
            SELECT name, br, w, left(w, 4) AS p FROM (
                SELECT a.name, a.br, unnest(regexp_extract_all(t.low, {W})) AS w FROM a JOIN t ON t.br = a.br
                WHERE a.name IN (SELECT name FROM kp))) y
        SEMI JOIN kp ON kp.name = y.name AND kp.p = y.p""")
    Q = {
        'year_and_doi': f"""SELECT r.rid, dd.br, NULL, NULL, NULL FROM rif r
            JOIN dd ON dd.doi = r.doi {year.format(b='dd')}""",
        'doi_title': """SELECT r.rid, dd.br, NULL, NULL, td.title FROM rif r
            JOIN dd ON dd.doi = r.doi JOIN td ON td.br = dd.br
            WHERE r.pattern IS NOT NULL AND regexp_matches(td.title, r.pattern, 'i')""",
        'author_title': """SELECT r.rid, x.br, x.name, NULL, t.title FROM rif r JOIN (
                SELECT DISTINCT rk.rid, var.name, tk.br FROM rk JOIN var USING (rid)
                    JOIN tk ON tk.name = var.name AND tk.p = left(rk.wkey, 4) AND starts_with(tk.w, rk.wkey)
                    WHERE length(rk.wkey) >= 4
                UNION
                -- first the few references with a short key, then the titles of their surname
                SELECT DISTINCT c.rid, var.name, a.br FROM short_refs c JOIN rif r2 ON r2.rid = c.rid
                    JOIN var ON var.rid = c.rid JOIN a ON a.name = var.name JOIN t ON t.br = a.br
                    WHERE contains(t.low, r2.word)
            ) x ON x.rid = r.rid JOIN t ON t.br = x.br
            WHERE regexp_matches(t.title, r.pattern, 'i') AND x.br IN (SELECT br FROM dy)""",
        # the tool's condition (same page, or contained in a range "123-130") in two branches: with the
        # OR in a single branch DuckDB joined each reference with all the works of the surname (full
        # memory on the dump's batch 1); the exact page becomes a join key
        'year_author_page': f"""SELECT r.rid, a.br, var.name, pg.start, NULL FROM rif r JOIN var USING (rid)
            JOIN a ON a.name = var.name JOIN pg ON pg.br = a.br AND pg.start = r.page {year.format(b='a')}
            WHERE NOT r.initial AND r.page IS NOT NULL
            UNION ALL
            SELECT r.rid, a.br, var.name, pgi.start, NULL FROM rif r JOIN var USING (rid)
            JOIN a ON a.name = var.name JOIN pgi ON pgi.br = a.br {year.format(b='a')}
            WHERE NOT r.initial AND r.page IS NOT NULL AND
                  (contains(pgi.start, r.page || '-') OR contains(pgi.start, '-' || r.page))""",
        'year_volume_page': f"""SELECT r.rid, v.br, NULL, r.page, NULL FROM rif r
            JOIN v ON v.vol = r.vol JOIN pg ON pg.br = v.br AND pg.start = r.page {year.format(b='v')}""",
        'year_author_volume': f"""SELECT r.rid, a.br, var.name, NULL, NULL FROM rif r JOIN var USING (rid)
            JOIN a ON a.name = var.name JOIN v ON v.br = a.br AND v.vol = r.vol {year.format(b='a')}
            WHERE NOT r.initial""",
    }
    n = 0
    for q, sql in Q.items():     # the candidates stay in DuckDB (c_<query>): to Python in chunks
        t = time.time()
        con.execute(f'CREATE OR REPLACE TEMP TABLE c_{q} AS SELECT DISTINCT * FROM ({sql}) AS s(rid, br, name, start, title)')
        m = con.execute(f'SELECT count(*) FROM c_{q}').fetchone()[0]
        n += m
        print(f'    {q}: {m} candidates, {time.time() - t:.0f} s', flush=True)
    con.execute('CREATE OR REPLACE TEMP TABLE cand AS ' +
                ' UNION '.join(f'SELECT br FROM c_{q}' for q in Q))
    return K, n


DETAILS = ('titles', 'dates', 'volumes', 'pages', 'authors', 'doi')


def details(con):
    """For each candidate work (table cand), in DuckDB tables d_<name>: titles, dates, volumes, starting
    pages, surnames with the author role, DOIs. One read of each Meta table per pass."""
    for name, sql in zip(DETAILS, ("SELECT br, title FROM m_titles",
                                   "SELECT br, date FROM m_dates",
                                   "SELECT br, vol FROM m_volumes",
                                   "SELECT br, start FROM m_pages WHERE start IS NOT NULL",
                                   f"SELECT br, name FROM m_authors WHERE role = '{AUTHOR}'",
                                   "SELECT br, doi FROM m_doi")):
        con.execute(f'CREATE OR REPLACE TEMP TABLE d_{name} AS SELECT DISTINCT * FROM ({sql}) AS s(br, x) '
                    'WHERE br IN (SELECT br FROM cand)')


def write_chunks(con, chunks, folder):
    """Candidates and details split by chunk (chunks: lists of indices), written by DuckDB in one go as
    <folder>/cand/chunk=K and <folder>/det/chunk=K (Parquet): each child process reads its own, the
    parent builds no Python objects."""
    table(con, 'pz', [dict(rid=i, chunk=k) for k, ids in enumerate(chunks) for i in ids])
    con.execute('CREATE OR REPLACE TEMP TABLE cu AS ' + ' UNION ALL '.join(
        f"""SELECT pz.chunk, '{q}' AS q, c.rid, c.br, CAST(c.name AS VARCHAR) AS name,
                   CAST(c.start AS VARCHAR) AS start, CAST(c.title AS VARCHAR) AS title
            FROM c_{q} c JOIN pz USING (rid)""" for q in QT))
    # writing many files with fewer threads: DuckDB keeps a buffer per thread and per open file
    # (with 8 threads, 400 chunks and the DISTINCT it went over 20 GB on 200,000 references)
    thr = con.execute("SELECT current_setting('threads')").fetchone()[0]
    con.execute('SET threads=4')
    try:
        con.execute(f"COPY cu TO '{folder}/cand' (FORMAT parquet, PARTITION_BY (chunk))")
        # no DISTINCT: the d_<name> tables have no duplicates already, and so do the (chunk, work) pairs
        con.execute(f"""COPY (SELECT cb.chunk, d.name, d.br, d.x
                FROM (SELECT DISTINCT chunk, br FROM cu) cb JOIN (""" + ' UNION ALL '.join(
            f"SELECT '{name}' AS name, br, CAST(x AS VARCHAR) AS x FROM d_{name}" for name in DETAILS) +
            f""") d USING (br)) TO '{folder}/det' (FORMAT parquet, PARTITION_BY (chunk))""")
    finally:
        con.execute(f'SET threads={thr}')


def read_chunk(folder, k):
    """Candidates {i: {query: [(br, author_name, start_page, title)]}} and details {name: {br: [values]}}
    of chunk k, from the files written by write_chunks (then deleted)."""
    C, D = {}, {name: defaultdict(list) for name in DETAILS}
    c, d = f'{folder}/cand/chunk={k}', f'{folder}/det/chunk={k}'
    if os.path.isdir(c):
        t = pq.read_table(c).to_pydict()
        for q, rid, br, name, start, title in zip(t['q'], t['rid'], t['br'], t['name'], t['start'], t['title']):
            C.setdefault(rid, {}).setdefault(q, []).append((br, name, start, title))
        shutil.rmtree(c)
    if os.path.isdir(d):
        t = pq.read_table(d).to_pydict()
        for name, br, x in zip(t['name'], t['br'], t['x']):
            D[name][br].append(x)
        shutil.rmtree(d)
    return C, D


def best_value(ref, values, field):
    """The value of `field` that gives the most points on its own (the tool's score is a sum per field)."""
    return max(values, key=lambda x: M.calculate_matching_score(ref, {field: {'value': x}}))


def row(ref, k, q, br, name, start, title, D):
    """The best row of that query for that work, as QLever would give it, or None."""
    dates = D['dates'].get(br, [])
    if q in YEAR_FILTERED:
        dates = [d for d in dates if d[:4] in k['years']]
    if q in DATE_REQUIRED and not dates:
        return None
    r = {'br': {'value': br}}
    if dates:
        r['pub_date'] = {'value': best_value(ref, dates, 'pub_date')}
    if q in AUTHORS_ONLY:
        name = next((n for n in D['authors'].get(br, []) if n in k['variants']), None)
    if name:
        r['author_name'] = {'value': name}
    dois = [ref.doi] if q in ('year_and_doi', 'doi_title') else D['doi'].get(br, [])   # BIND or OPTIONAL_DOI
    if dois:
        r['doi'] = {'value': best_value(ref, dois, 'doi')}
        r['all_dois'] = sorted(set(dois))       # for the output: a work can have several DOIs in Meta
                                                # (separated by spaces: SICI DOIs contain ';')
    vol = ref.volume if q in ('year_volume_page', 'year_author_volume') else None
    vols = [vol] if vol else D['volumes'].get(br, [])
    if vols:
        r['volume_num'] = {'value': best_value(ref, vols, 'volume_num')}
    starts = [start] if start else D['pages'].get(br, [])
    if starts:
        r['start_page'] = {'value': best_value(ref, starts, 'start_page')}
    # the title last (the fuzzy comparison is the expensive part): if not even a perfect title brings
    # the row to the lowest threshold (23.4), it cannot be chosen (skip_unreachable)
    titles = [title] if title else D['titles'].get(br, [])
    if titles and M.calculate_matching_score(ref, r) + cfg.title_exact_score >= MIN_ACCEPT:
        r['title'] = {'value': best_value(ref, titles, 'title')}
    r['score'] = M.calculate_matching_score(ref, r)
    r['query_type'] = q
    return r


def choose(ref, k, cand, D):
    """The tool's loop on one reference: queries in order, candidates in IRI order, the best replaced
    only by a strictly higher one, stop at the first >= 26."""
    b = None
    for q in QT:
        rows = [x for c in cand.get(q, []) if (x := row(ref, k, q, *c, D))]
        for x in sorted(rows, key=lambda x: x['br']['value']):   # QLever's order
            if x['score'] > (b['score'] if b else 0):
                b = x
            if b and b['score'] >= THRESHOLD and not BEST:       # early stop (like the tool)
                return b
    return b


_STATE = {}   # references and keys of the pass, inherited by the children (fork): only the chunk's are read


def _chunk(job):
    """In a child process: reads its own chunk from disk and chooses the best of each reference."""
    k, ids = job
    refs, K, folder = _STATE['refs'], _STATE['K'], _STATE['folder']
    C, D = read_chunk(folder, k)
    return [(i, choose(refs[i], K[i], C.get(i, {}), D)) for i in ids]


def match(con, refs, processes=1, chunk=2000, temp='.'):
    """_match, but if DuckDB runs out of memory ONLY this pass is split in half and redone (recursively):
    the previous pass and GROBID stay. Each reference is independent, so the outcome is the same."""
    try:
        return _match(con, refs, processes, chunk, temp)
    except duckdb.OutOfMemoryException as e:
        if len(refs) < MIN_SPLIT:
            raise
        ids = list(refs)
        print(f'  memory full on {len(ids)} refs ({str(e).splitlines()[0][:200]}): splitting in half', flush=True)
        return {**match(con, {i: refs[i] for i in ids[:len(ids) // 2]}, processes, chunk, temp),
                **match(con, {i: refs[i] for i in ids[len(ids) // 2:]}, processes, chunk, temp)}


def _match(con, refs, processes=1, chunk=2000, temp='.'):
    """{index: best row or None} for one pass of the tool (refs: index -> normalized Reference).
    Candidates and details are computed on the whole batch in DuckDB; the scores in chunks of `chunk`
    references, handed out to the processes as they go (whoever finishes takes the next)."""
    if not refs:
        return {}
    t0 = time.time()
    K, n = candidates(con, refs)
    t1 = time.time()
    details(con)
    ids = list(refs)
    chunks = [ids[s:s + chunk] for s in range(0, len(ids), chunk)]
    folder = tempfile.mkdtemp(prefix='chunks_', dir=temp)
    try:
        write_chunks(con, chunks, folder)
        t2 = time.time()
        _STATE.update(refs=refs, K=K, folder=folder)
        jobs = list(enumerate(chunks))
        best = {}
        if processes > 1:
            gc.freeze()   # the inherited objects are not touched by the children's GC (no copies)
            with mp.get_context('fork').Pool(processes) as pool:
                for p in pool.imap_unordered(_chunk, jobs):
                    best.update(p)
        else:
            for job in jobs:
                best.update(_chunk(job))
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    print(f'  {len(refs)} refs: candidates {t1 - t0:.0f} s ({n}), '
          f'details and chunks {t2 - t1:.0f} s, scores {time.time() - t2:.0f} s', flush=True)
    return best


TEI = '{http://www.tei-c.org/ns/1.0}'


def grobid_block(G, texts):
    """--grobid-batch: like G.process_unstructured_reference on each text, but with ONE
    processCitationList request for the whole block (one citation a line), in memory. Each biblStruct of
    the answer is read on its own with G._parse_tei_xml, like the answer to one citation; texts without
    fields go to the single processCitation, as in the tool. If the request fails or the answers are not
    one per text, the whole block goes back to the single calls."""
    clean = [' '.join(t.split()) for t in texts]          # like process_unstructured_reference
    empty = [not p for p in clean]
    lines = [p for p in clean if p]
    bibl = None
    if lines:
        data = urllib.parse.urlencode({'citations': lines}, doseq=True).encode('utf-8')
        for _ in range(3):
            try:
                req = urllib.request.Request(f'{G.server}/api/processCitationList', data=data,
                                             headers={'Accept': 'application/xml'})
                with urllib.request.urlopen(req, timeout=600) as resp:
                    root = ET.fromstring(resp.read())
                bibl = root.findall(f'.//{TEI}biblStruct')
                break
            except urllib.error.HTTPError as e:
                if e.code != 503:                            # 503: GROBID busy, retry
                    break
                time.sleep(G.client.config.get('sleep_time', 5))
            except Exception:
                break
    if bibl is None or len(bibl) != len(lines):
        return [G.process_unstructured_reference(t) for t in texts]
    out, it = [], iter(zip(lines, bibl))
    for e in empty:
        if e:
            out.append(None)
            continue
        text, b = next(it)
        tei = (b'<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><back><listBibl>' + ET.tostring(b)
               + b'</listBibl></back></text></TEI>')
        ref = G._parse_tei_xml(io.BytesIO(tei))
        if not G._has_fields(ref):
            single = G._process_citation_single(text)
            if G._has_fields(single):
                ref = single
        out.append(ref)
    return out


def grobid(G, texts, threads, block=0):
    """GROBID's results for `texts`, in the same order: one call a text (like the tool) or, with
    block > 0, one request per block of `block` texts."""
    with ThreadPoolExecutor(threads) as ex:
        if not block:
            return list(ex.map(G.process_unstructured_reference, texts))
        parts = [texts[s:s + block] for s in range(0, len(texts), block)]
        return [r for p in ex.map(lambda p: grobid_block(G, p), parts) for r in p]


def accepted(b):
    if not b:
        return False
    return b['score'] >= T.apply_threshold_adjustment(b['score'], THRESHOLD, cfg.threshold_adjustment,
                                                      cfg.enable_threshold_adjustment)


def process(con, a, G, part, t0):
    """Pass 1, GROBID and pass 2 on a batch (index -> normalized Reference): (best1, best2)."""
    b1 = match(con, part, a.processes, a.chunk, a.temp)
    b2 = {}
    if G:
        todo = [i for i, r in part.items() if not accepted(b1[i]) and r.unstructured]
        t = time.time()
        gr = dict(zip(todo, grobid(G, [part[i].unstructured for i in todo], a.grobid_threads, a.grobid_batch)))
        merged = {}
        for i, g in gr.items():
            if g:
                r = T.Reference(**vars(part[i]))
                P._merge_reference_data(r, g)
                if a.drop_grobid_doi and not part[i].doi:
                    r.doi = ''     # a DOI removed on purpose (discordant) must not come back from the text
                merged[i] = T.normalize_reference_safe(r)   # like the tool: GROBID's fields (DOI included)
                                                            # are normalized at matching time
        print(f'GROBID: {len(todo)} refs, {len(merged)} with fields, {time.time() - t:.0f} s', flush=True)
        print('pass 2 (GROBID):', flush=True)
        b2 = match(con, merged, a.processes, a.chunk, a.temp)
    print(f'  end of batch: {time.time() - t0:.0f} s from the start', flush=True)
    return b1, b2


COLUMNS_M = ['reference_id', 'article_title', 'matched_title', 'score', 'matched_doi', 'meta_id', 'query_type']
COLUMNS_U = ['reference_id', 'best_score', 'score_original', 'score_after_grobid', 'score_without_year',
             'candidate_meta_id', 'candidate_doi', 'candidate_title', 'candidate_query_type']


def write_csv(base, rows, refs, best1, best2, first=()):
    """<base>_matches.csv and <base>_matches_unmatched.csv like the tool. rows: [(index, {columns})],
    `first`: extra columns before reference_id. Written under a temporary name and then renamed, the
    matches file last: if it exists, the batch is complete. Returns the matches written."""
    n = 0
    with open(base + '_matches.csv.tmp', 'w', newline='', encoding='utf-8') as m, \
         open(base + '_matches_unmatched.csv.tmp', 'w', newline='', encoding='utf-8') as u:
        wm = csv.DictWriter(m, [*first, *COLUMNS_M])
        wu = csv.DictWriter(u, [*first, *COLUMNS_U])
        wm.writeheader()
        wu.writeheader()
        for i, col in rows:
            b = best1[i] if accepted(best1[i]) else best2.get(i) if accepted(best2.get(i)) else None
            if b:
                n += 1
                wm.writerow({**col, 'article_title': refs[i].get_main_title() or 'N/A',
                             'matched_title': b.get('title', {}).get('value', 'N/A'), 'score': b['score'],
                             'matched_doi': ' '.join(b.get('all_dois', [])) or 'N/A', 'meta_id': b['br']['value'],
                             'query_type': b['query_type']})
            else:
                # the best candidate under the threshold, to apply a lower threshold later (--min-candidate-score)
                c = max((x for x in (best1[i], best2.get(i)) if x and x['score'] > 0),
                        key=lambda x: x['score'], default=None)
                s = c['score'] if c else 'N/A'
                wu.writerow({**col, 'best_score': s, 'score_original': s,
                             'score_after_grobid': 'N/A', 'score_without_year': 'N/A',
                             'candidate_meta_id': c['br']['value'] if c else '',
                             'candidate_doi': ' '.join(c.get('all_dois', [])) if c else '',
                             'candidate_title': c.get('title', {}).get('value', '') if c else '',
                             'candidate_query_type': c['query_type'] if c else ''})
    os.replace(base + '_matches_unmatched.csv.tmp', base + '_matches_unmatched.csv')
    os.replace(base + '_matches.csv.tmp', base + '_matches.csv')
    return n


FIELDS = tuple(vars(T.Reference()))


def process_halving(con, a, G, refs, t0):
    """process(), but if DuckDB runs out of memory the batch is split in half and redone (recursively):
    each reference is independent of the others, so the outcome is the same."""
    try:
        return process(con, a, G, refs, t0)
    except duckdb.OutOfMemoryException as e:     # usually match() handles it already, per pass
        if len(refs) < MIN_SPLIT:
            raise
        ids = list(refs)
        print(f'  memory full on {len(ids)} refs, whole batch ({str(e).splitlines()[0][:200]}): splitting in half',
              flush=True)
        b1, b2 = process_halving(con, a, G, {i: refs[i] for i in ids[:len(ids) // 2]}, t0)
        c1, c2 = process_halving(con, a, G, {i: refs[i] for i in ids[len(ids) // 2:]}, t0)
        return {**b1, **c1}, {**b2, **c2}


def main_batches(con, a, G, t0):
    """--batches: the batches of prepare_dump.py (<input>/batch=K), one at a time. Each batch writes
    batch_KKKK_matches.csv and _unmatched.csv at once, one row per occurrence (citing work, ref_id);
    batches that already have their CSVs are skipped, so after an interruption the same command resumes.
    Batches and CSVs with the old names (lotto=K, lotto_KKKK) are read and skipped too."""
    os.makedirs(a.output, exist_ok=True)
    batches = sorted((int(d.rsplit('=', 1)[1]), d) for d in glob.glob(os.path.join(a.input, 'batch=*'))
                     + glob.glob(os.path.join(a.input, 'lotto=*')))
    done = times = 0
    for pos, (k, d) in enumerate(batches):
        base = os.path.join(a.output, f'batch_{k:04d}')
        if os.path.exists(base + '_matches.csv') or os.path.exists(os.path.join(a.output, f'lotto_{k:04d}_matches.csv')):
            continue
        t = time.time()
        rows = pq.read_table(d).to_pylist()
        refs = {i: T.Reference(**{c: r[c] for c in FIELDS}) for i, r in enumerate(rows)}   # already normalized
        print(f'batch {k} ({len(refs)} distinct refs), pass 1:', flush=True)
        best1, best2 = process_halving(con, a, G, refs, t0)
        occ = [(i, {'citing_work': o['citing_work'], 'citing_work_doi': o['citing_work_doi'],
                    'reference_id': o['ref_id']}) for i, r in enumerate(rows) for o in r['occ']]
        n = write_csv(base, occ, refs, best1, best2, first=('citing_work', 'citing_work_doi'))
        done, times = done + 1, times + time.time() - t
        left = len(batches) - pos - 1
        print(f'batch {k}: {n} matches out of {len(occ)} occurrences, {time.time() - t:.0f} s | '
              f'{left} batches left, about {times / done * left / 86400:.1f} days', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input', help='folder of Crossref JSON, or with --batches the folder of prepare_dump.py')
    ap.add_argument('output')
    ap.add_argument('--batches', action='store_true', help='input: the deduplicated batches of the dump (prepare_dump.py)')
    ap.add_argument('--tables', required=True, help="folder of Meta's tables (export_meta_tables.sh)")
    ap.add_argument('--grobid-config')
    ap.add_argument('--grobid-threads', type=int, default=8)
    ap.add_argument('--grobid-batch', type=int, default=0,
                    help='citations per GROBID request (0 = one a request, like the tool; 100 on the dump)')
    ap.add_argument('--processes', type=int, default=8, help='processes for the scores (1 = no fork)')
    ap.add_argument('--threads', type=int, default=8, help='DuckDB threads')
    ap.add_argument('--temp', default=tempfile.gettempdir(),
                    help="DuckDB's temporary files; two DuckDB jobs running together need two folders")
    ap.add_argument('--temp-max', default='100GB', help='maximum space of the temporary files')
    ap.add_argument('--memory', default='20GB', help="DuckDB's maximum memory")
    ap.add_argument('--drop-grobid-doi', action='store_true',
                    help='drop the DOI extracted by GROBID if the reference had no DOI: for checking the discordant '
                         'DOIs (removed on purpose), NOT for the run on references without DOI, where the tool uses it')
    ap.add_argument('--min-candidate-score', type=float, default=None,
                    help="minimum score for which the candidates' title is computed (default 0.9 x 26 = 23.4); "
                         'e.g. 21.6 to be able to apply threshold 24 later. The accepted matches do not change')
    ap.add_argument('--best-candidate', action='store_true',
                    help='no stop at the first candidate reaching 26: the one with the most points over all the '
                         'queries wins (like the threshold-1000 runs behind the paper\'s held-out table). '
                         'Without it the choice is the tool\'s')
    ap.add_argument('--batch-size', type=int, default=20000,
                    help='JSON input only: references per batch (candidates and queries)')
    ap.add_argument('--chunk', type=int, default=2000, help='references per chunk of the scores (handed to the processes)')
    a = ap.parse_args()
    if a.min_candidate_score is not None:
        global MIN_ACCEPT
        MIN_ACCEPT = a.min_candidate_score
    global BEST
    BEST = a.best_candidate
    t0 = time.time()
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{a.memory}'")
    con.execute(f'SET threads={a.threads}')
    con.execute('SET preserve_insertion_order=false')   # the order of the intermediate rows is not needed
    # temporary files in one folder and with a cap: without them DuckDB fills a shared disk
    con.execute(f"SET temp_directory='{a.temp}'")
    con.execute(f"SET max_temp_directory_size='{a.temp_max}'")
    meta_views(con, a.tables)
    G = T.GrobidProcessor(a.grobid_config) if a.grobid_config else None
    if a.batches:
        return main_batches(con, a, G, t0)
    everything = read_json_folder(a.input)
    refs = {i: T.normalize_reference_safe(T.Reference(**vars(r))) for i, (_, _, r) in enumerate(everything)}
    print(f'{len(refs)} references read', flush=True)
    best1, best2 = {}, {}
    ids = list(refs)
    for k in range(0, len(ids), a.batch_size):   # independent batches: the author_title pairs stay in memory
        part = {i: refs[i] for i in ids[k:k + a.batch_size]}
        print(f'batch {k // a.batch_size + 1} ({len(part)} refs), pass 1:', flush=True)
        b1, b2 = process(con, a, G, part, t0)
        best1.update(b1)
        best2.update(b2)
    # writing, one output file per input file, like the tool
    os.makedirs(a.output, exist_ok=True)
    per_file = defaultdict(list)
    for i, (f, rid, _) in enumerate(everything):
        per_file[f].append((i, {'reference_id': rid}))
    n = sum(write_csv(os.path.join(a.output, f), rows, refs, best1, best2) for f, rows in per_file.items())
    print(f'matches: {n} out of {len(everything)} | total time {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
