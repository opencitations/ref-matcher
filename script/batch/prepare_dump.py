"""Deduplication of the dump's references without DOI (the bulk's no_doi_references.jsonl) and split into
batches for match_bulk.py --batches.

Two references are the same if the matcher sees them as the same: same fields after the tool's
extraction and normalization (match_bulk.read_json_folder + normalize_reference_safe), free text included
(GROBID reads it). Same input -> same outcome, so each distinct reference is looked up once and the
outcome holds for all its occurrences.

    uv run --with duckdb --with pyarrow python script/batch/prepare_dump.py \
        <bulk output>/no_doi_references.jsonl <batches folder>

1) N processes each read a stretch of the file and write to <output>/_fragments/ the normalized
   fields, the hash (blake2b, 128 bits) and the occurrence (citing work, ref_id).
2) DuckDB groups by hash, a group of batches at a time, and writes <output>/batch=K/*.parquet:
   one distinct reference a row, with the list of its occurrences. The batch comes from the hash,
   so every batch is a random sample.
At the end the fragments are deleted (--keep-fragments to keep them).
"""
import argparse, hashlib, json, os, shutil, sys, tempfile, time
import multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import match_bulk as B
T, P = B.T, B.P

FIELDS = tuple(vars(T.Reference()))          # year, volume, first_page, ... unstructured
FRAGMENT_ROWS = 200_000


def normalized(rd):
    """The Reference that match_bulk would give the matcher (read_json_folder + normalize_reference_safe)."""
    y = P._extract_year_from_data(rd)
    return T.normalize_reference_safe(T.Reference(
        year=str(y) if y else '', first_author_lastname=P._extract_author(rd),
        article_title=P._extract_title(rd), volume=rd.get('volume', ''),
        first_page=P._extract_page(rd), doi=rd.get('DOI', ''), unstructured=rd.get('unstructured', '')))


def write_fragment(rows, output, w, n):
    t = pa.Table.from_pydict({k: [r[k] for r in rows] for k in rows[0]}).sort_by('batch')
    # sorted by batch, small row groups: DuckDB skips the groups outside the requested batches
    pq.write_table(t, os.path.join(output, f'w{w:02d}_{n:05d}.parquet'), row_group_size=10_000,
                   compression='zstd')


def work(args):
    """One process: the lines that start in the stretch [start, end) of the file."""
    w, src, start, end, output, batches = args
    rows, n, read = [], 0, 0
    with open(src, 'rb') as f:
        if start:
            f.seek(start - 1)
            f.readline()                     # up to the first newline: a line cut in half belongs to the previous stretch
        while f.tell() < end:
            l = f.readline()
            if not l:
                break
            d = json.loads(l)
            r = normalized(d['reference'])
            v = {k: str(getattr(r, k) or '') for k in FIELDS}
            h = hashlib.blake2b('\x1f'.join(v[k] for k in FIELDS).encode('utf-8'), digest_size=16).hexdigest()
            rows.append({'batch': int(h[:12], 16) % batches, 'h': h, **v,
                         'citing_work': d.get('citing_work') or '', 'citing_work_doi': d.get('citing_work_doi') or '',
                         'ref_id': d.get('ref_id') or ''})
            read += 1
            if len(rows) == FRAGMENT_ROWS:
                write_fragment(rows, output, w, n)
                rows, n = [], n + 1
                # a file of hundreds of GB read only once: out of the disk cache
                if hasattr(os, 'posix_fadvise'):
                    os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    if rows:
        write_fragment(rows, output, w, n)
    return read


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input', help='no_doi_references.jsonl (or extract_dois_to_verify.py\'s .jsonl)')
    ap.add_argument('output', help='folder of the batches')
    ap.add_argument('--processes', type=int, default=16)
    ap.add_argument('--batches', type=int, default=2300,
                    help='number of batches (2,300 for the 459.5 million distinct references of the June 2026 dump, '
                         'about 200,000 each)')
    ap.add_argument('--group', type=int, default=50, help='batches grouped by DuckDB in one pass')
    ap.add_argument('--limit-bytes', type=int, default=0, help='TEST: only the first N bytes of the file')
    ap.add_argument('--temp', default=tempfile.gettempdir(), help="DuckDB's temporary files")
    ap.add_argument('--memory', default='40GB', help="DuckDB's maximum memory")
    ap.add_argument('--keep-fragments', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    fr = os.path.join(a.output, '_fragments')
    if os.path.isdir(fr):
        sys.exit(f'{fr} exists already: delete it or use another output')
    os.makedirs(fr)

    # 1) fragments
    size = a.limit_bytes or os.path.getsize(a.input)
    stretches = [(w, a.input, size * w // a.processes, size * (w + 1) // a.processes, fr, a.batches)
                 for w in range(a.processes)]
    with mp.Pool(a.processes) as pool:
        read = sum(pool.map(work, stretches))
    print(f'1) {read:,} lines read in {time.time() - t0:.0f} s', flush=True)

    # 2) grouping by hash, a group of batches at a time
    con = duckdb.connect()
    for s in (f"memory_limit='{a.memory}'", 'threads=8', 'preserve_insertion_order=false',
              f"temp_directory='{a.temp}'", "max_temp_directory_size='100GB'"):
        con.execute(f'SET {s}')
    values = ', '.join(f'any_value({k}) AS {k}' for k in FIELDS)
    for g in range(0, a.batches, a.group):
        con.execute(f"""COPY (SELECT batch, h, {values},
                    list({{'citing_work': citing_work, 'citing_work_doi': citing_work_doi, 'ref_id': ref_id}}) AS occ
                FROM read_parquet('{fr}/*.parquet') WHERE batch BETWEEN {g} AND {g + a.group - 1}
                GROUP BY batch, h)
            TO '{a.output}' (FORMAT parquet, PARTITION_BY (batch), OVERWRITE_OR_IGNORE, COMPRESSION zstd)""")
        print(f'2) batches {g}-{min(g + a.group, a.batches) - 1}: {time.time() - t0:.0f} s', flush=True)
    distinct = con.execute(f"SELECT count(*) FROM read_parquet('{a.output}/batch=*/*.parquet')").fetchone()[0]
    print(f'{read:,} lines, {distinct:,} distinct ({distinct / max(read, 1):.1%}), {a.batches} batches '
          f'(~{distinct // a.batches:,} each), {time.time() - t0:.0f} s')
    if not a.keep_fragments:
        shutil.rmtree(fr)


if __name__ == '__main__':
    main()
