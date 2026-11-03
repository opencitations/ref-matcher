"""The references WITH a DOI that the bulk DOI check could not settle, to be passed to the full matcher:

    absent      DOI not found on Meta (unmatched_dois.txt): maybe wrong, or the work is on Meta with
                another DOI or without one. The DOI stays: the matcher also tries it with its DOI queries.
    error       DOI whose check failed (check_errors.txt): the DOI stays, the matcher's DOI queries
                also act as a check.
    discordant  DOI on Meta but with a surname different from the reference's (discordant_authors.tsv):
                the DOI is REMOVED, so the matcher judges only on author, title, year, volume and page
                (with the DOI it would find that DOI's work at once, with a 15-point head start).

Reads the dump like phase 1 of the bulk (bulk_doi._iter_members, normalize_doi, same numbering
ref_<n>) and writes a .jsonl in the format of no_doi_references.jsonl, plus 'group' and 'original_doi'
(prepare_dump.py ignores them; merge_citations.py finds them again from citing_work + ref_id).

    uv run python script/batch/extract_dois_to_verify.py <dump .tar.gz> <bulk output> dois_to_verify.jsonl
Then prepare_dump.py and match_bulk.py --batches --drop-grobid-doi, in separate folders.
"""
import argparse, json, os, sys, time
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # script/
from bulk_doi import _iter_members, normalize_doi


def read_dois(p):
    """The normalized DOI of the first column of each line (txt or tsv)."""
    if not os.path.exists(p):
        print(f'  {p} missing: empty group', flush=True)
        return set()
    with open(p, encoding='utf-8') as f:
        return {d for d in (normalize_doi(l.split('\t', 1)[0]) for l in f) if d}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dump', help='Crossref dump (.tar.gz)')
    ap.add_argument('bulk', help='folder of the bulk results (python bulk_doi.py ... --authors)')
    ap.add_argument('output', help='output .jsonl')
    ap.add_argument('--limit', type=int, default=0, help='TEST: only the first N works')
    a = ap.parse_args()
    t0 = time.time()
    # the order matters only if a DOI were in two lists: discordant wins (DOI removed)
    groups = {}
    for name, file in (('absent', 'unmatched_dois.txt'), ('error', 'check_errors.txt'),
                       ('discordant', 'discordant_authors.tsv')):
        s = read_dois(os.path.join(a.bulk, file))
        print(f'{name}: {len(s):,} DOIs', flush=True)
        groups.update(dict.fromkeys(s, name))
    count, works = Counter(), 0
    with open(a.output + '.tmp', 'w', encoding='utf-8') as out:
        for base, data in _iter_members(a.dump):
            items = data.get('items', []) if isinstance(data, dict) else (data or [])
            for i, work in enumerate(items):
                works += 1
                for n, ref in enumerate(work.get('reference', []) or [], 1):   # ref_<n> like the bulk
                    d = normalize_doi(ref.get('DOI'))
                    g = groups.get(d) if d else None
                    if not g:
                        continue
                    count[g] += 1
                    if g == 'discordant':
                        ref = {k: v for k, v in ref.items() if k != 'DOI'}
                    out.write(json.dumps({'citing_work': f'{base}_w{i}', 'citing_work_doi': work.get('DOI'),
                                          'ref_id': f'ref_{n}', 'group': g, 'original_doi': d,
                                          'reference': ref}, ensure_ascii=False) + '\n')
                if a.limit and works >= a.limit:
                    break
            if a.limit and works >= a.limit:
                break
            if works % 1_000_000 < len(items):
                print(f'  {works:,} works, {sum(count.values()):,} references '
                      f'({", ".join(f"{k} {v:,}" for k, v in count.items())}), {time.time() - t0:.0f} s', flush=True)
    os.replace(a.output + '.tmp', a.output)
    print(f'{works:,} works, {sum(count.values()):,} references written to {a.output}: '
          f'{dict(count)}, {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
