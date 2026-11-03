"""All the dump's occurrences with a DOI (citing work, ref_id, DOI) in a Parquet file, for the merge: the bulk
saves only the unique DOIs, so to know WHO cites a checked DOI the dump has to be read again.

    uv run --with pyarrow python script/batch/extract_doi_occurrences.py <dump .tar.gz> doi_occurrences.parquet

Same reading and numbering as extract_dois_to_verify.py (citing_work <base>_w<i>, ref_<n>), so the merge
removes absent, discordant and error references with the keys (citing_work, ref_id) and what is left is
"verified". The DOI is normalized like the bulk run it is merged with: normalize_doi, or with
--legacy-normalization as before 20/10/2026 (lower case, prefixes, NFC), for a bulk run made then.
"""
import argparse, os, sys, time, unicodedata
import pyarrow as pa
import pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # script/
from bulk_doi import _iter_members, _DOI_PREFIXES, normalize_doi


def legacy_normalize_doi(raw):
    """normalize_doi as it was before 20/10/2026."""
    doi = str(raw or '').strip().lower()
    for p in _DOI_PREFIXES:
        if doi.startswith(p):
            doi = doi[len(p):].strip()
            break
    doi = unicodedata.normalize('NFC', doi.replace('\\/', '/'))
    return doi if doi.startswith('10.') else None


ap = argparse.ArgumentParser()
ap.add_argument('dump', help='Crossref dump (.tar.gz)')
ap.add_argument('output', help='.parquet')
ap.add_argument('--legacy-normalization', action='store_true',
                help='normalize the DOIs as before 20/10/2026, for a bulk run made with that version')
ap.add_argument('--limit', type=int, default=0, help='TEST: only the first N works')
a = ap.parse_args()
norm = legacy_normalize_doi if a.legacy_normalization else normalize_doi
t0 = time.time()
schema = pa.schema([(c, pa.string()) for c in ('citing_work', 'citing_work_doi', 'ref_id', 'doi')])
w = pq.ParquetWriter(a.output + '.tmp', schema, compression='zstd')
buf = {c: [] for c in schema.names}
works = tot = 0
for base, data in _iter_members(a.dump):
    items = data.get('items', []) if isinstance(data, dict) else (data or [])
    for i, work in enumerate(items):
        works += 1
        for n, ref in enumerate(work.get('reference', []) or [], 1):
            d = norm(ref.get('DOI'))
            if d:
                for c, v in zip(schema.names, (f'{base}_w{i}', work.get('DOI'), f'ref_{n}', d)):
                    buf[c].append(v)
        if a.limit and works >= a.limit:
            break
    if len(buf['doi']) >= 2_000_000 or (a.limit and works >= a.limit):
        tot += len(buf['doi'])
        w.write_table(pa.table(buf, schema=schema))
        buf = {c: [] for c in schema.names}
        print(f'  {works:,} works, {tot:,} occurrences with a DOI, {time.time() - t0:.0f} s', flush=True)
    if a.limit and works >= a.limit:
        break
tot += len(buf['doi'])
w.write_table(pa.table(buf, schema=schema))
w.close()
os.replace(a.output + '.tmp', a.output)
print(f'{works:,} works, {tot:,} occurrences with a DOI in {a.output}, {time.time() - t0:.0f} s')
