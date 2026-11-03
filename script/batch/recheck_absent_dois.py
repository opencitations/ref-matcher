"""A new bulk check (phase 2, with the authors) on the DOIs that a bulk run did not settle: the absent DOIs
that the current normalize_doi changes (repaired: Unicode dashes, %2f, trailing punctuation, ...) and all
the check errors. Needed when the bulk ran with an older normalize_doi (before 20/10/2026) or had failed
chunks. Does not read the dump again: it reuses the bulk's files.

    uv run python script/batch/recheck_absent_dois.py <bulk output> <recheck output>

Writes in <output> the same files as the bulk (matched_dois.txt, unmatched_dois.txt, corroborated_dois.txt,
discordant_authors.tsv, check_errors.txt), on the NEW DOIs, plus map.tsv (old DOI, new DOI, group) to
find the references again (merge_citations.py --recheck). Resumable: the preparation is skipped if
dois_unique.txt exists, the check resumes from the bulk's checkpoint.
"""
import argparse, asyncio, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # script/
from bulk_doi import normalize_doi, _sort_unique, check


def prepare(bulk, output):
    """map.tsv, dois_raw.txt and doi_author_raw.tsv on the new DOIs, then sorted as the bulk does."""
    new, count = {}, {'absent': 0, 'absent changed': 0, 'error': 0}
    with open(os.path.join(output, 'map.tsv'), 'w', encoding='utf-8') as m:
        for group, file in (('absent', 'unmatched_dois.txt'), ('error', 'check_errors.txt')):
            with open(os.path.join(bulk, file), encoding='utf-8') as f:
                for line in f:
                    d = line.split('\t', 1)[0].strip()
                    if not d:
                        continue
                    n = normalize_doi(d)
                    count[group] += 1
                    if not n or (group == 'absent' and n == d):
                        continue        # absent and unchanged: it would stay absent
                    count['absent changed'] += group == 'absent'
                    new[d] = n
                    m.write(f'{d}\t{n}\t{group}\n')
    with open(os.path.join(output, 'dois_raw.txt'), 'w', encoding='utf-8') as f:
        f.writelines(n + '\n' for n in new.values())
    pairs = 0
    with open(os.path.join(bulk, 'doi_author_unique.tsv'), encoding='utf-8') as f, \
         open(os.path.join(output, 'doi_author_raw.tsv'), 'w', encoding='utf-8') as out:
        for line in f:
            d, _, fam = line.rstrip('\n').partition('\t')
            if d in new:
                out.write(f'{new[d]}\t{fam}\n')
                pairs += 1
    _sort_unique(os.path.join(output, 'dois_raw.txt'), os.path.join(output, 'dois_unique.txt'))
    _sort_unique(os.path.join(output, 'doi_author_raw.tsv'), os.path.join(output, 'doi_author_unique.tsv'))
    print(f"absent {count['absent']:,} (changed {count['absent changed']:,}), errors {count['error']:,}, "
          f"DOI-surname pairs {pairs:,}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bulk', help="folder of the dump's bulk run")
    ap.add_argument('output')
    ap.add_argument('--endpoint', default='http://localhost:7011', help="QLever: the same Meta as the matcher's")
    ap.add_argument('--chunk', type=int, default=2000)
    ap.add_argument('--per-minute', type=int, default=300, help='requests a minute')
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.output, exist_ok=True)
    if not os.path.exists(os.path.join(a.output, 'dois_unique.txt')):
        prepare(a.bulk, a.output)
    n = sum(1 for _ in open(os.path.join(a.output, 'dois_unique.txt'), encoding='utf-8'))
    print(f'new DOIs to check: {n:,}, {time.time() - t0:.0f} s', flush=True)
    ck = asyncio.run(check(os.path.join(a.output, 'dois_unique.txt'), a.output, endpoint=a.endpoint,
                           chunk_size=a.chunk, max_per_min=a.per_minute, authors=True))
    ck['seconds'] = round(time.time() - t0)
    json.dump(ck, open(os.path.join(a.output, 'summary.json'), 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(json.dumps(ck, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
