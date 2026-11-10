"""Merge of the results: for each occurrence (citing work + ref_id) the cited work, where it comes from (source)
and how reliable it is (level).

    uv run --with duckdb --with pyarrow python script/batch/merge_citations.py \
        dois_to_verify.jsonl <match_bulk output of the DOI run> <output> [--recheck <recheck_absent_dois output>] \
        [--no-doi <match_bulk output of the main run>] [--verified doi_occurrences.parquet --bulk <bulk output>] \
        [--threshold 24] [--corrections corrections.csv]

Inputs: the keys of the references with a DOI to verify (extract_dois_to_verify.py's .jsonl, or a Parquet with
its columns citing_work, ref_id, group, original_doi), match_bulk.py's CSVs on them, and optionally the
repaired DOIs (--recheck), the matcher on the references without DOI (--no-doi) and the DOIs the bulk found
on Meta (--verified with --bulk). Files written before the English names (gruppo, doi_originale, assente,
discorde, errore, mappa.tsv) are read too.

Rules: a DOI (original or repaired) wins over the matcher, and a disagreement is flagged; a discordant DOI
that the matcher does not find keeps the DOI; the matcher's threshold is not fixed in the matching (the score
is a column) but chosen in the merge with --threshold.
Levels:
    high      two independent signals agree: DOI + matcher (same work), or DOI + agreeing surname
    medium    one signal only: the DOI exists (even with a disagreeing surname, never contradicted by the
              matcher), or only the matcher
    to check  the DOI leads to one work, the matcher to another: the DOI wins, the matcher's work stays in
              the matcher_* columns; these rows also go to to_check.csv
    none      no link (goes to the triage for ingestion)
--threshold T (default 26) counts as a match every candidate from 0.9 x T points up, like the tool, both in
the _matches and in the _unmatched CSVs. The level depends on the threshold, so to compare two thresholds
make two runs in two folders. A threshold is exact only if match_bulk.py ran with --min-candidate-score
0.9 x T or lower (21.6: from 24 up; 16.2: from 18 up; 14.4: from 16 up); under that it gives a lower bound,
because under that score the title is not always compared.
--verified adds the DOIs the bulk found on Meta without disagreement (category "verified": high if the
surname agrees, medium if it cannot be checked).
--corrections adds human judgements, applied last, one row per occurrence (citing_work, ref_id, verdict,
cited_doi, cited_omid, who, date, note; only the first five are used by the merge):
    right        level high, source + " (confirmed by hand)" (only if the row has a cited work)
    wrong        level none, no cited work, source + " (rejected by hand)"
    other work   the cited work becomes the file's cited_doi / cited_omid, level high, source + " (corrected by hand)"
The evidence stays in its columns anyway (original_doi, repaired_doi, authors, matcher_*, score).
"""
import argparse, json, os, tempfile, time
import duckdb
import pyarrow as pa

ap = argparse.ArgumentParser()
ap.add_argument('keys', help="extract_dois_to_verify.py's .jsonl, or a Parquet of its keys")
ap.add_argument('results', help="match_bulk.py's CSVs on the DOI references (the DOI run)")
ap.add_argument('output')
ap.add_argument('--recheck', help='folder of recheck_absent_dois.py: the repaired DOIs, optional')
ap.add_argument('--no-doi', help="match_bulk.py's CSVs on the references without DOI (main run), optional")
ap.add_argument('--verified', help='doi_occurrences.parquet (extract_doi_occurrences.py), optional: with --bulk')
ap.add_argument('--bulk', help='folder of the bulk run: matched_dois.txt and corroborated_dois.txt')
ap.add_argument('--threshold', type=float, default=26, help="matcher's threshold: a match from 0.9 x threshold up")
ap.add_argument('--corrections', help='CSV of human judgements (see above), optional')
ap.add_argument('--temp', default=tempfile.gettempdir(), help="DuckDB's temporary files")
ap.add_argument('--threads', type=int, default=8)
ap.add_argument('--memory', default='20GB', help="DuckDB's maximum memory")
a = ap.parse_args()
t0 = time.time()
minimum = round(0.9 * a.threshold, 6)            # minimum score of a match, like the tool (0.9 x 26 = 23.4)
if a.threshold < 24:
    print(f'warning: threshold {a.threshold:g}, exact only if match_bulk.py ran with --min-candidate-score '
          f'{minimum:g} or lower')


def lines(name, *alternatives):
    """The tab-separated fields of each line of a recheck file (or of the first alternative name that exists)."""
    p = next((p for p in (os.path.join(a.recheck, n) for n in (name, *alternatives)) if os.path.exists(p)),
             os.path.join(a.recheck, name))
    if not os.path.exists(p):
        return
    with open(p, 'rb') as f:     # binary: only '\n' ends a line
        for l in f:
            yield l.rstrip(b'\n').decode('utf-8', 'replace').split('\t')


old, new, on_meta, authors = [], [], [], []
if a.recheck:
    present = {r[0] for r in lines('matched_dois.txt')}
    agree = {r[0] for r in lines('corroborated_dois.txt')}
    disagree = {r[0] for r in lines('discordant_authors.tsv')}
    mapping = {r[0]: r[1] for r in lines('map.tsv', 'mappa.tsv') if len(r) >= 2}
    old = list(mapping)
    new = [mapping[v] for v in old]
    on_meta = [n in present for n in new]
    authors = ['agrees' if n in agree else 'disagrees' if n in disagree else '' for n in new]
    del present, agree, disagree, mapping
t = pa.table({'old': pa.array(old, pa.string()), 'new': pa.array(new, pa.string()),
              'on_meta': pa.array(on_meta, pa.bool_()), 'authors': pa.array(authors, pa.string())})
print(f'recheck: {t.num_rows:,} DOIs, {time.time() - t0:.0f} s', flush=True)

os.makedirs(a.output, exist_ok=True)
os.makedirs(a.temp, exist_ok=True)
con = duckdb.connect()
for s in (f"memory_limit='{a.memory}'", f'threads={a.threads}', f"temp_directory='{a.temp}'",
          'preserve_insertion_order=false'):     # the order of the rows is not needed and costs memory
    con.execute(f'SET {s}')

# the keys, as a Parquet: the .jsonl has the whole references and would be parsed twice
keys = a.keys
if keys.endswith('.jsonl'):
    first = json.loads(open(keys, encoding='utf-8').readline())
    g, d = ('gruppo', 'doi_originale') if 'gruppo' in first else ('group', 'original_doi')
    keys = os.path.join(a.output, 'keys.parquet')
    con.execute(f"""COPY (SELECT citing_work, ref_id, "{g}" AS "{g}", "{d}" AS "{d}"
        FROM read_json('{a.keys}', format='newline_delimited', columns={{'citing_work': 'VARCHAR',
             'ref_id': 'VARCHAR', '{g}': 'VARCHAR', '{d}': 'VARCHAR'}})) TO '{keys}' (FORMAT parquet)""")
    print(f'keys: {keys}, {time.time() - t0:.0f} s', flush=True)
cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM '{keys}'").fetchall()]
g, d = ('gruppo', 'doi_originale') if 'gruppo' in cols else ('group', 'original_doi')
con.execute(f"""CREATE TEMP VIEW k AS SELECT citing_work, ref_id,
    CASE "{g}" WHEN 'assente' THEN 'absent' WHEN 'discorde' THEN 'discordant' WHEN 'errore' THEN 'error'
         ELSE "{g}" END AS category, "{d}" AS original_doi FROM '{keys}'""")
read = "read_csv('{}', all_varchar=true, header=true, union_by_name=true)"


def matcher(folder):
    """The matcher's outcome per occurrence: the match or the best candidate; found if it has 0.9 x threshold points."""
    return f"""SELECT *, coalesce(score >= {minimum} AND coalesce(matcher_omid, '') <> '', false) AS found FROM (
               SELECT coalesce(m.citing_work, u.citing_work) AS citing_work,
                      coalesce(m.reference_id, u.reference_id) AS ref_id,
                      coalesce(m.citing_work_doi, u.citing_work_doi) AS citing_work_doi,
                      coalesce(nullif(CASE WHEN m.reference_id IS NOT NULL THEN m.matched_doi
                                           ELSE u.candidate_doi END, ''), 'N/A') AS matcher_doi,
                      CASE WHEN m.reference_id IS NOT NULL THEN m.meta_id ELSE u.candidate_meta_id END AS matcher_omid,
                      nullif(CASE WHEN m.reference_id IS NOT NULL THEN m.query_type
                                  ELSE u.candidate_query_type END, '') AS matcher_query,
                      TRY_CAST(CASE WHEN m.reference_id IS NOT NULL THEN m.score ELSE u.best_score END AS DOUBLE) AS score
               FROM {read.format(os.path.join(folder, '*_matches.csv'))} m
               FULL JOIN {read.format(os.path.join(folder, '*_matches_unmatched.csv'))} u
                      ON u.citing_work = m.citing_work AND u.reference_id = m.reference_id)"""


# one part at a time, each in its merge_<part>.parquet: all together in one query they do not fit in memory
con.register('t', t)
# references with a DOI: category and original DOI from the keys, repaired DOI from the recheck, matcher
# from the DOI run
parts = {'with_doi': f"""SELECT k.citing_work, k.ref_id, x.citing_work_doi, k.category, k.original_doi,
           CASE WHEN p.new <> k.original_doi THEN p.new END AS repaired_doi,
           CASE WHEN k.category = 'discordant' THEN k.original_doi WHEN p.on_meta THEN p.new END AS doi,
           coalesce(p.authors, '') AS authors,
           coalesce(x.found, false) AS found, x.matcher_doi, x.matcher_omid, x.matcher_query, x.score
    FROM k
    LEFT JOIN t p ON p.old = k.original_doi
    LEFT JOIN ({matcher(a.results)}) x ON x.citing_work = k.citing_work AND x.ref_id = k.ref_id"""}
if a.no_doi:
    parts['no_doi'] = f"""SELECT citing_work, ref_id, citing_work_doi, 'no DOI' AS category,
                     NULL::VARCHAR AS original_doi, NULL::VARCHAR AS repaired_doi, NULL::VARCHAR AS doi, '' AS authors,
                     coalesce(found, false) AS found, matcher_doi, matcher_omid, matcher_query, score
                     FROM ({matcher(a.no_doi)})"""
if a.verified:
    # all the occurrences with a DOI, minus those of absent, discordant and error references (the keys): the
    # bulk found the rest on Meta; agreeing surname -> high. A DOI that is not in matched_dois.txt should not
    # be there (different normalization): it stays without a link and shows in the summary.
    def bulk_dois(name):
        return (f"read_csv('{os.path.join(a.bulk, name)}', columns={{'doi': 'VARCHAR'}}, "
                "header=false, delim='\\t', quote='', escape='', ignore_errors=true)")
    con.execute(f"""CREATE TEMP TABLE bk AS SELECT doi, bool_or(c) AS agrees
        FROM (SELECT doi, false AS c FROM {bulk_dois('matched_dois.txt')}
              UNION ALL SELECT doi, true FROM {bulk_dois('corroborated_dois.txt')}) GROUP BY doi""")
    print(f'bulk DOIs: {con.execute("SELECT count(*) FROM bk").fetchone()[0]:,}, {time.time() - t0:.0f} s', flush=True)
    parts['verified'] = f"""SELECT o.citing_work, o.ref_id, o.citing_work_doi, 'verified' AS category,
               o.doi AS original_doi, NULL::VARCHAR AS repaired_doi,
               CASE WHEN bk.doi IS NOT NULL THEN o.doi END AS doi,
               CASE WHEN bk.agrees THEN 'agrees' ELSE '' END AS authors,
               false AS found, NULL::VARCHAR AS matcher_doi, NULL::VARCHAR AS matcher_omid,
               NULL::VARCHAR AS matcher_query, NULL::DOUBLE AS score
        FROM '{a.verified}' o
        ANTI JOIN k ON k.citing_work = o.citing_work AND k.ref_id = o.ref_id
        LEFT JOIN bk ON bk.doi = o.doi"""

CLASSIFY = """
    SELECT *,
           CASE WHEN doi IS NOT NULL AND same THEN 'high'
                WHEN doi IS NOT NULL AND found THEN 'to check'
                WHEN doi IS NOT NULL AND authors = 'agrees' THEN 'high'
                WHEN doi IS NOT NULL OR found THEN 'medium'
                ELSE 'none' END AS level,
           CASE WHEN doi IS NOT NULL THEN
                     CASE WHEN repaired_doi IS NOT NULL THEN 'repaired DOI' ELSE 'DOI' END ||
                     CASE WHEN same THEN ' + matcher'
                          WHEN found THEN ', disagrees with the matcher'
                          WHEN category = 'discordant' OR authors = 'disagrees' THEN ', surname disagrees'
                          WHEN authors = 'agrees' THEN ' + surname'
                          ELSE ', existence only' END
                WHEN found AND matcher_doi = 'N/A' THEN 'matcher, work without DOI'
                WHEN found THEN 'matcher'
                ELSE 'no link' END AS source,
           CASE WHEN doi IS NOT NULL THEN doi
                WHEN found AND matcher_doi <> 'N/A' THEN split_part(lower(matcher_doi), ' ', 1) END AS cited_doi,
           CASE WHEN found AND (doi IS NULL OR same) THEN matcher_omid END AS cited_omid
    FROM (SELECT *, found AND (list_contains(string_split(lower(matcher_doi), ' '), doi)
                               OR list_contains(string_split(lower(matcher_doi), ' '), original_doi)) AS same
          FROM (SRC))"""

# human judgements, applied last (one row per occurrence: with two rows the join would double the citation)
VERDICTS = ('right', 'wrong', 'other work')
if a.corrections:
    con.execute(f"""CREATE TEMP TABLE corr AS SELECT citing_work, ref_id, lower(trim(verdict)) AS verdict,
                    nullif(lower(trim(cited_doi)), '') AS c_doi, nullif(trim(cited_omid), '') AS c_omid
                    FROM {read.format(a.corrections)}""")
    errors = con.execute(f"""SELECT count(*) FILTER (WHERE verdict NOT IN {VERDICTS}),
            count(*) FILTER (WHERE verdict = 'other work' AND c_doi IS NULL AND c_omid IS NULL),
            count(*) - count(DISTINCT (citing_work, ref_id)) FROM corr""").fetchone()
    if any(errors):
        raise SystemExit(f'corrections: {errors[0]} unknown verdicts (valid: {", ".join(VERDICTS)}), '
                         f'{errors[1]} "other work" without cited_doi/cited_omid, {errors[2]} repeated occurrences')
    print(f'corrections: {con.execute("SELECT count(*) FROM corr").fetchone()[0]:,}', flush=True)
else:
    con.execute('CREATE TEMP TABLE corr (citing_work VARCHAR, ref_id VARCHAR, verdict VARCHAR, c_doi VARCHAR, '
                'c_omid VARCHAR)')

CORRECT = """
    SELECT x.* REPLACE (
           CASE WHEN c.verdict = 'right' AND (cited_doi IS NOT NULL OR cited_omid IS NOT NULL) THEN 'high'
                WHEN c.verdict = 'wrong' THEN 'none'
                WHEN c.verdict = 'other work' THEN 'high' ELSE level END AS level,
           source || CASE WHEN c.verdict = 'right' AND (cited_doi IS NOT NULL OR cited_omid IS NOT NULL)
                               THEN ' (confirmed by hand)'
                          WHEN c.verdict = 'wrong' THEN ' (rejected by hand)'
                          WHEN c.verdict = 'other work' THEN ' (corrected by hand)' ELSE '' END AS source,
           CASE WHEN c.verdict = 'wrong' THEN NULL WHEN c.verdict = 'other work' THEN c.c_doi
                ELSE cited_doi END AS cited_doi,
           CASE WHEN c.verdict = 'wrong' THEN NULL WHEN c.verdict = 'other work' THEN c.c_omid
                ELSE cited_omid END AS cited_omid)
    FROM (CLASSIFIED) x LEFT JOIN corr c USING (citing_work, ref_id)"""

COLS = ('citing_work, citing_work_doi, ref_id, category, original_doi, repaired_doi, cited_doi, cited_omid, '
        'level, source, score, matcher_doi, matcher_omid, matcher_query, '
        "CASE WHEN category = 'discordant' THEN 'disagrees' ELSE authors END AS authors")
for name, src in parts.items():
    con.execute(f"""COPY (SELECT {COLS} FROM ({CORRECT.replace('CLASSIFIED', CLASSIFY.replace('SRC', src))}))
                    TO '{os.path.join(a.output, f'merge_{name}.parquet')}' (FORMAT parquet)""")
    print(f'  written merge_{name}.parquet, {time.time() - t0:.0f} s', flush=True)
con.execute(f"CREATE TEMP VIEW r AS SELECT * FROM '{os.path.join(a.output, 'merge_*.parquet')}'")   # the summaries
print(f'occurrences: {con.execute("SELECT count(*) FROM r").fetchone()[0]:,}, {time.time() - t0:.0f} s', flush=True)
con.execute(f"""COPY (SELECT {COLS} FROM r WHERE level = 'to check')
                TO '{os.path.join(a.output, 'to_check.csv')}' (HEADER)""")

LEVELS = "['high', 'medium', 'to check', 'none']"
print(f'\nMerge (threshold {a.threshold:g}, match from {minimum:g} points): occurrences by level and source, '
      f'{time.time() - t0:.0f} s')
print(f"{'category':<12}{'level':<10}{'source':<44}{'occurrences':>14}{'%':>7}")
for c, l, s, n, p in con.execute(f"""SELECT category, level, source, count(*),
        100.0 * count(*) / sum(count(*)) OVER (PARTITION BY category) FROM r GROUP BY 1, 2, 3
        ORDER BY 1, array_position({LEVELS}, level), 4 DESC""").fetchall():
    print(f'{c:<12}{l:<10}{s:<44}{n:>14,}{p:>6.1f}%')
print('\nTotal by level:')
for l, n, p in con.execute(f"""SELECT level, count(*), 100.0 * count(*) / sum(count(*)) OVER () FROM r
        GROUP BY 1 ORDER BY array_position({LEVELS}, level)""").fetchall():
    print(f'  {l:<10}{n:>14,}{p:>6.1f}%')
# triage for the ingestion: "no candidate" joins "no query could be formulated" and "queries made, no
# candidate" (the CSVs do not say whether the queries were sent)
print('\nNot linked, by best candidate (triage for the ingestion):')
for c, f, n in con.execute("""SELECT category, CASE WHEN score IS NULL OR score <= 0 THEN 'no candidate'
                                                    WHEN score >= 21.6 THEN 'candidate from 21.6 up (probable duplicate)'
                                                    ELSE 'candidate under 21.6 (uncertain)' END, count(*)
        FROM r WHERE level = 'none' GROUP BY 1, 2 ORDER BY 1, 3 DESC""").fetchall():
    print(f'  {c:<12}{f:<44}{n:>14,}')
print(f'\nwritten merge_*.parquet and to_check.csv in {a.output}, {time.time() - t0:.0f} s')
