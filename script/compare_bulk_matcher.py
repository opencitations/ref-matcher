#!/usr/bin/env python3
"""
compare_bulk_matcher.py — confronto tra il matcher (--dump) e la modalità bulk
(--bulk --bulk-same-as) sullo STESSO insieme di work, riferimento per riferimento.

Uso:
  1) il matcher ha già prodotto una cartella, es. /srv/fs/MATTEO/dumptime5000
  2) si lancia il bulk sugli stessi work:
       uv run script/ReferenceMatchingTool.py DUMP.tar.gz --bulk \
           --bulk-same-as /srv/fs/MATTEO/dumptime5000 -o /srv/fs/MATTEO/bulk_same
  3) si confrontano:
       uv run script/compare_bulk_matcher.py \
           --matcher-dir /srv/fs/MATTEO/dumptime5000 --bulk-dir /srv/fs/MATTEO/bulk_same

Output (in --out, di default la cartella del bulk):
  confronto_per_riferimento.csv.gz  una riga per riferimento con i due esiti
  confronto_riepilogo.txt           riepilogo leggibile: tempi, match, campi
  confronto_riepilogo.json          stesso contenuto, per elaborazioni successive

Tempo del matcher: il matcher scrive `<work>_matches_stats.txt` per ultimo, quindi
la sua data di modifica è l'istante in cui il work è stato completato. Con
--batch-size 1 i work sono sequenziali: la differenza tra due work consecutivi è
il tempo speso sul secondo. Si sommano le differenze sotto una soglia (--gap-cap,
default 2 ore) per escludere le pause tra un run e l'altro. Se si conosce il
tempo reale (somma dei `real` dei run), lo si può passare con --matcher-seconds.
"""

import argparse
import csv
import gzip
import json
import os
from collections import Counter, defaultdict

MATCHES_SUFFIX = '_matches.csv'
UNMATCHED_SUFFIX = '_matches_unmatched.csv'
DONE_SUFFIX = '_matches_stats.txt'

# campi Crossref che il matcher usa davvero per le query con DOI
YEAR_KEYS = ('year', 'issued')
TITLE_KEYS = ('article-title', 'title')

CATEGORIES = [
    ('A_entrambi_presente',        'con DOI — trovato da entrambi'),
    ('B_solo_matcher_trova',       'con DOI — trovato dal matcher, DOI assente su Meta'),
    ('C_falso_negativo_matcher',   'con DOI — DOI su Meta, ma il matcher non lo trova'),
    ('D_entrambi_assente',         'con DOI — non trovato da nessuno dei due'),
    ('E_senza_doi_trovato',        'senza DOI — trovato dal matcher (il bulk non può)'),
    ('F_senza_doi_non_trovato',    'senza DOI — non trovato'),
    ('G_matcher_senza_esito',      'il matcher non ha un esito (errore / non elaborato)'),
    ('H_bulk_senza_esito',         'con DOI — il bulk non ha un esito (richiesta fallita)'),
]
CAT_LABEL = dict(CATEGORIES)

PROFILES = {
    (1, 0, 0): 'solo DOI',
    (1, 1, 0): 'DOI + campi strutturati',
    (1, 0, 1): 'DOI + unstructured',
    (1, 1, 1): 'DOI + strutturati + unstructured',
    (0, 1, 0): 'strutturato senza DOI',
    (0, 1, 1): 'strutturato senza DOI + unstructured',
    (0, 0, 1): 'solo unstructured',
    (0, 0, 0): 'vuoto',
}


def read_set(path):
    if not os.path.exists(path):
        return set()
    with open(path, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f if line.strip()}


def load_matcher_work(matcher_dir, work):
    """ref_id -> (esito, query_type, score, grobid_attempted) per un work."""
    res = {}
    p = os.path.join(matcher_dir, work + MATCHES_SUFFIX)
    if os.path.exists(p):
        with open(p, newline='', encoding='utf-8', errors='replace') as f:
            for row in csv.DictReader(f):
                res[row.get('reference_id')] = (
                    'matchato', row.get('query_type', ''), row.get('score', ''), '')
    p = os.path.join(matcher_dir, work + UNMATCHED_SUFFIX)
    if os.path.exists(p):
        with open(p, newline='', encoding='utf-8', errors='replace') as f:
            for row in csv.DictReader(f):
                rid = row.get('reference_id')
                if rid not in res:
                    res[rid] = ('non_trovato', '', row.get('best_score', ''),
                                row.get('grobid_attempted', ''))
    return res


def categorize(has_doi, matcher, bulk):
    if matcher == 'nessun_esito':
        return 'G_matcher_senza_esito'
    if has_doi:
        if bulk not in ('presente', 'assente'):
            return 'H_bulk_senza_esito'
        if matcher == 'matchato':
            return 'A_entrambi_presente' if bulk == 'presente' else 'B_solo_matcher_trova'
        return 'C_falso_negativo_matcher' if bulk == 'presente' else 'D_entrambi_assente'
    return 'E_senza_doi_trovato' if matcher == 'matchato' else 'F_senza_doi_non_trovato'


def matcher_active_seconds(matcher_dir, works, gap_cap):
    """Stima del tempo attivo del matcher sui work dati (vedi docstring)."""
    mtimes = []
    with os.scandir(matcher_dir) as it:
        for e in it:
            if e.name.endswith(DONE_SUFFIX) and e.name[:-len(DONE_SUFFIX)] in works:
                mtimes.append(e.stat().st_mtime)
    mtimes.sort()
    active = excluded = 0.0
    n_excluded = 0
    for a, b in zip(mtimes, mtimes[1:]):
        gap = b - a
        if gap <= gap_cap:
            active += gap
        else:
            excluded += gap
            n_excluded += 1
    return {
        'works_with_timestamp': len(mtimes),
        'active_seconds': round(active, 1),
        'excluded_gaps': n_excluded,
        'excluded_seconds': round(excluded, 1),
        'first_completion': mtimes[0] if mtimes else None,
        'last_completion': mtimes[-1] if mtimes else None,
    }


def pct(n, d):
    return round(100.0 * n / d, 2) if d else 0.0


def hms(seconds):
    s = int(round(seconds or 0))
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return (f"{d}g " if d else "") + f"{h:02d}h {m:02d}m {s:02d}s"


def main():
    ap = argparse.ArgumentParser(description="Confronto matcher vs bulk sugli stessi work.")
    ap.add_argument('--matcher-dir', required=True, help="Cartella di output del matcher (--dump).")
    ap.add_argument('--bulk-dir', required=True, help="Cartella di output di --bulk --bulk-same-as.")
    ap.add_argument('--out', help="Dove scrivere il confronto (default: --bulk-dir).")
    ap.add_argument('--matcher-seconds', type=float,
                    help="Tempo reale del matcher, se noto (somma dei `real`); "
                         "altrimenti viene stimato dalle date dei file.")
    ap.add_argument('--gap-cap', type=float, default=7200,
                    help="Pausa massima (s) fra due work considerata lavoro e non "
                         "interruzione, per la stima del tempo (default 7200).")
    args = ap.parse_args()

    out_dir = args.out or args.bulk_dir
    os.makedirs(out_dir, exist_ok=True)

    index_path = os.path.join(args.bulk_dir, 'references_index.csv')
    if not os.path.exists(index_path):
        raise SystemExit(f"Manca {index_path}: lancia il bulk con --bulk-same-as {args.matcher_dir}")
    with open(os.path.join(args.bulk_dir, 'bulk_summary.json'), encoding='utf-8') as f:
        bulk_summary = json.load(f)

    present = read_set(os.path.join(args.bulk_dir, 'matched_dois.txt'))
    absent = read_set(os.path.join(args.bulk_dir, 'unmatched_dois.txt'))
    bulk_err = read_set(os.path.join(args.bulk_dir, 'check_errors.txt'))

    cat_counts = Counter()
    cat_fields = defaultdict(Counter)        # categoria -> campo -> n
    cat_profiles = defaultdict(Counter)      # categoria -> profilo -> n
    cat_combos = defaultdict(Counter)        # categoria -> combinazione di campi -> n
    cat_qtypes = defaultdict(Counter)        # categoria -> query_type -> n
    c_score = Counter()                      # falsi negativi: il matcher aveva un candidato?
    c_diag = Counter()                       # falsi negativi: anno / titolo usati dal matcher
    profile_all = Counter()
    profile_by_matcher = defaultdict(Counter)
    field_all = Counter()
    works_seen = set()
    total = with_doi = invalid_doi = 0

    per_ref_path = os.path.join(out_dir, 'confronto_per_riferimento.csv.gz')
    current_work, matcher_map = None, {}
    with open(index_path, newline='', encoding='utf-8') as fin, \
         gzip.open(per_ref_path, 'wt', newline='', encoding='utf-8') as fout:
        w = csv.writer(fout)
        w.writerow(['work', 'ref_id', 'doi', 'profilo', 'campi', 'esito_matcher',
                    'query_type', 'score_matcher', 'grobid_tentato', 'esito_bulk', 'categoria'])
        for row in csv.DictReader(fin):
            work, rid, doi = row['work'], row['ref_id'], row['doi']
            if work != current_work:
                current_work = work
                matcher_map = load_matcher_work(args.matcher_dir, work)
                works_seen.add(work)
            fields = [x for x in row['fields'].split('|') if x]
            fset = set(fields)
            has_doi = 1 if doi else 0
            other_struct = 1 if (fset - {'DOI', 'unstructured'}) else 0
            has_unstr = 1 if 'unstructured' in fset else 0
            profile = PROFILES[(has_doi, other_struct, has_unstr)]

            m_verdict, qtype, score, grobid = matcher_map.get(rid, ('nessun_esito', '', '', ''))
            if has_doi:
                bulk = ('presente' if doi in present else 'assente' if doi in absent
                        else 'errore' if doi in bulk_err else 'sconosciuto')
            else:
                bulk = 'senza_doi'
            cat = categorize(has_doi, m_verdict, bulk)

            total += 1
            with_doi += has_doi
            if not has_doi and row['doi_raw_present'] == '1':
                invalid_doi += 1
            cat_counts[cat] += 1
            profile_all[profile] += 1
            profile_by_matcher[m_verdict][profile] += 1
            for fld in fields:
                cat_fields[cat][fld] += 1
                field_all[fld] += 1
            cat_profiles[cat][profile] += 1
            cat_combos[cat][row['fields'] or '(nessun campo)'] += 1
            if qtype:
                cat_qtypes[cat][qtype] += 1
            if cat == 'C_falso_negativo_matcher':
                c_score['nessun candidato' if score in ('', 'N/A', 'None') else 'candidato sotto soglia'] += 1
                y = any(k in fset for k in YEAR_KEYS)
                t = any(k in fset for k in TITLE_KEYS)
                c_diag['anno e titolo' if y and t else 'solo anno' if y
                       else 'solo titolo' if t else 'né anno né titolo'] += 1

            w.writerow([work, rid, doi, profile, row['fields'], m_verdict, qtype, score,
                        grobid, bulk, cat])

    # ------------------------------------------------------------------ tempi
    # stesso insieme di work del bulk (inclusi quelli con 0 riferimenti)
    works_for_time = read_set(os.path.join(args.bulk_dir, 'works_compared.txt')) or works_seen
    est = matcher_active_seconds(args.matcher_dir, works_for_time, args.gap_cap)
    matcher_seconds = args.matcher_seconds if args.matcher_seconds else est['active_seconds']
    bulk_seconds = bulk_summary.get('total_seconds', 0)
    n_works = bulk_summary.get('works', len(works_seen))

    # --------------------------------------------------------------- match
    A, B, C, D = (cat_counts[k] for k in ('A_entrambi_presente', 'B_solo_matcher_trova',
                                            'C_falso_negativo_matcher', 'D_entrambi_assente'))
    E, F, G, H = (cat_counts[k] for k in ('E_senza_doi_trovato', 'F_senza_doi_non_trovato',
                                            'G_matcher_senza_esito', 'H_bulk_senza_esito'))
    matcher_found = A + B + E
    bulk_found = A + C
    hybrid_found = A + B + C + E          # bulk sui DOI + matcher su assenti e senza DOI
    residual = B + D + E + F              # ciò che la pipeline ibrida passa al matcher
    hybrid_matcher_est = matcher_seconds * pct(residual, total) / 100.0

    summary = {
        'insieme_confrontato': {
            'works': n_works, 'riferimenti': total, 'con_doi_valido': with_doi,
            'senza_doi': total - with_doi, 'doi_grezzo_non_valido': invalid_doi,
            'work_del_matcher_non_trovati_nel_dump': bulk_summary.get('works_not_found_in_dump', 0),
        },
        'tempi': {
            'matcher_secondi': round(matcher_seconds, 1),
            'matcher_fonte': 'passato con --matcher-seconds' if args.matcher_seconds
                             else 'stimato dalle date dei file',
            'matcher_stima_dalle_date': est,
            'bulk_secondi': bulk_seconds,
            'bulk_fasi': {k: bulk_summary.get(k) for k in
                          ('extract_seconds', 'dedup_seconds', 'check_seconds')},
            'bulk_richieste_sparql': bulk_summary.get('requests'),
            'bulk_doi_unici': bulk_summary.get('unique_dois'),
            'rapporto_matcher_su_bulk': round(matcher_seconds / bulk_seconds, 1) if bulk_seconds else None,
            'matcher_work_per_ora': round(n_works / (matcher_seconds / 3600), 1) if matcher_seconds else None,
            'bulk_work_per_ora': round(n_works / (bulk_seconds / 3600), 1) if bulk_seconds else None,
            'ibrido_stima_secondi': round(bulk_seconds + hybrid_matcher_est, 1),
        },
        'match': {
            'matcher_trovati': matcher_found, 'matcher_pct': pct(matcher_found, total),
            'bulk_trovati': bulk_found, 'bulk_pct_su_tutti': pct(bulk_found, total),
            'bulk_pct_su_con_doi': pct(bulk_found, with_doi),
            'ibrido_trovati': hybrid_found, 'ibrido_pct': pct(hybrid_found, total),
            'residuo_al_matcher_nell_ibrido': residual,
            'residuo_pct': pct(residual, total),
        },
        'categorie': {k: {'descrizione': CAT_LABEL[k], 'n': cat_counts[k],
                          'pct': pct(cat_counts[k], total)} for k, _ in CATEGORIES},
        'falsi_negativi_diagnosi': {'candidato_del_matcher': dict(c_score),
                                    'campi_usati_dal_matcher': dict(c_diag)},
        'profili_tutti': dict(profile_all.most_common()),
        'profili_per_esito_matcher': {k: dict(v.most_common()) for k, v in profile_by_matcher.items()},
        'campi_tutti': dict(field_all.most_common()),
        'campi_per_categoria': {k: dict(cat_fields[k].most_common()) for k, _ in CATEGORIES if cat_counts[k]},
        'profili_per_categoria': {k: dict(cat_profiles[k].most_common()) for k, _ in CATEGORIES if cat_counts[k]},
        'combinazioni_top_per_categoria': {k: cat_combos[k].most_common(10) for k, _ in CATEGORIES if cat_counts[k]},
        'query_type_per_categoria': {k: dict(v.most_common()) for k, v in cat_qtypes.items()},
    }
    with open(os.path.join(out_dir, 'confronto_riepilogo.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # --------------------------------------------------------- riepilogo txt
    L = []
    s, t, m = summary['insieme_confrontato'], summary['tempi'], summary['match']
    L.append("=" * 72)
    L.append("CONFRONTO MATCHER vs BULK — stesso insieme di work")
    L.append("=" * 72)
    L.append(f"work: {s['works']:,} | riferimenti: {total:,} | con DOI: {with_doi:,} "
             f"({pct(with_doi, total)}%) | senza DOI: {total - with_doi:,}")
    if invalid_doi:
        L.append(f"  (di cui {invalid_doi:,} con un campo DOI non valido, trattati come senza DOI)")
    if s['work_del_matcher_non_trovati_nel_dump']:
        L.append(f"  ATTENZIONE: {s['work_del_matcher_non_trovati_nel_dump']:,} work del matcher non trovati nel dump")

    L.append("\n--- TEMPI " + "-" * 62)
    L.append(f"matcher : {hms(matcher_seconds)}  ({t['matcher_fonte']})")
    if not args.matcher_seconds:
        L.append(f"          escluse {est['excluded_gaps']} pause > {int(args.gap_cap)} s "
                 f"(totale {hms(est['excluded_seconds'])}) tra un run e l'altro")
    L.append(f"bulk    : {hms(bulk_seconds)}  (estrazione {bulk_summary.get('extract_seconds')} s, "
             f"dedup {bulk_summary.get('dedup_seconds')} s, verifica {bulk_summary.get('check_seconds')} s)")
    L.append(f"          {t['bulk_richieste_sparql'] or 0:,} richieste SPARQL per "
             f"{t['bulk_doi_unici'] or 0:,} DOI unici")
    if t['rapporto_matcher_su_bulk']:
        L.append(f"il bulk è {t['rapporto_matcher_su_bulk']:,}× più veloce "
                 f"({t['matcher_work_per_ora']:,} vs {t['bulk_work_per_ora']:,} work/ora)")
    L.append(f"pipeline ibrida (bulk + matcher sul residuo): ~{hms(t['ibrido_stima_secondi'])}")
    L.append("          stima indicativa e probabilmente per difetto: il residuo sono i")
    L.append("          riferimenti che al matcher costano di più (cascata intera + GROBID)")

    L.append("\n--- MATCH " + "-" * 62)
    L.append(f"matcher          : {matcher_found:,} trovati ({m['matcher_pct']}%)")
    L.append(f"bulk (solo DOI)  : {bulk_found:,} trovati ({m['bulk_pct_su_tutti']}% di tutti, "
             f"{m['bulk_pct_su_con_doi']}% di quelli con DOI)")
    L.append(f"ibrido           : {hybrid_found:,} trovati ({m['ibrido_pct']}%) — residuo al matcher: "
             f"{residual:,} ({m['residuo_pct']}%)")

    L.append("\nRiferimenti con DOI:          bulk: su Meta   bulk: assente")
    L.append(f"  matcher: trovato        {A:>14,}   {B:>14,}")
    L.append(f"  matcher: non trovato    {C:>14,}   {D:>14,}")
    L.append(f"Riferimenti senza DOI:   matcher trovati {E:,} | non trovati {F:,}")
    if G or H:
        L.append(f"Senza esito: matcher {G:,} | bulk {H:,}")

    L.append("\n--- CATEGORIE " + "-" * 58)
    for k, desc in CATEGORIES:
        L.append(f"  {k:<27} {cat_counts[k]:>10,}  {pct(cat_counts[k], total):>6}%  {desc}")

    if C:
        L.append("\n--- FALSI NEGATIVI DEL MATCHER (categoria C) " + "-" * 27)
        L.append("  il matcher aveva un candidato?  " +
                 " | ".join(f"{k}: {v:,} ({pct(v, C)}%)" for k, v in c_score.most_common()))
        L.append("  campi che il matcher usa con il DOI (anno=year/issued, titolo=article-title/title):")
        for k, v in c_diag.most_common():
            L.append(f"    {k:<20} {v:>10,}  ({pct(v, C)}%)")

    L.append("\n--- PROFILO DEI CAMPI (tutti i riferimenti) " + "-" * 28)
    for k, v in profile_all.most_common():
        L.append(f"  {k:<38} {v:>10,}  ({pct(v, total)}%)")

    L.append("\n--- PROFILO DEI CAMPI PER CATEGORIA " + "-" * 36)
    for k, _ in CATEGORIES:
        n = cat_counts[k]
        if not n:
            continue
        L.append(f"  {k} ({n:,})")
        for p, v in cat_profiles[k].most_common():
            L.append(f"    {p:<36} {v:>10,}  ({pct(v, n)}%)")

    L.append("\n--- CAMPI POPOLATI PER CATEGORIA (% dei riferimenti della categoria) " + "-" * 3)
    top_fields = [f for f, _ in field_all.most_common(14)]
    for k, _ in CATEGORIES:
        n = cat_counts[k]
        if not n:
            continue
        L.append(f"  {k} ({n:,})")
        L.append("    " + ", ".join(f"{f} {pct(cat_fields[k][f], n)}%" for f in top_fields))

    L.append("\n--- COMBINAZIONI DI CAMPI PIÙ FREQUENTI PER CATEGORIA " + "-" * 19)
    for k, _ in CATEGORIES:
        n = cat_counts[k]
        if not n:
            continue
        L.append(f"  {k} ({n:,})")
        for combo, v in cat_combos[k].most_common(5):
            L.append(f"    {v:>9,} ({pct(v, n):>5}%)  {combo}")

    if cat_qtypes:
        L.append("\n--- QUERY DEL MATCHER CHE HANNO TROVATO IL MATCH " + "-" * 23)
        for k, v in cat_qtypes.items():
            L.append(f"  {k}: " + ", ".join(f"{q} {c:,}" for q, c in v.most_common()))

    L.append("\nDettaglio per riferimento: " + per_ref_path)
    text = "\n".join(L) + "\n"
    with open(os.path.join(out_dir, 'confronto_riepilogo.txt'), 'w', encoding='utf-8') as f:
        f.write(text)
    print(text)


if __name__ == '__main__':
    main()
