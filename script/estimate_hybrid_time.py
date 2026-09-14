#!/usr/bin/env python3
"""
estimate_hybrid_time.py — quanto costa al matcher ogni tipo di riferimento, e
quindi quanto durerebbe la modalità ibrida (bulk + matcher sul residuo).

Usa solo dati già prodotti, nessun nuovo run:
  - dalla cartella del matcher: la data di completamento di ogni work
    (`<work>_matches_stats.txt`, scritto per ultimo). Con --batch-size 1 i work
    sono in sequenza, quindi la differenza tra due completamenti consecutivi è
    il tempo speso sul secondo work (pausa tra i work inclusa);
  - dalla cartella del bulk (--bulk-same-as + compare_bulk_matcher.py):
    `works_compared.txt` e `confronto_per_riferimento.csv.gz`, cioè quanti
    riferimenti di ciascun tipo contiene ogni work.

Con questi dati stima, per regressione lineare,
    durata del work = costo fisso per work + somma(costo del tipo × n. riferimenti del tipo)
e da lì il tempo del matcher, dell'ibrido e di alcune varianti (senza pausa tra
i work, rate più alto), sul campione e proiettato sull'intero dump.

Uso:
  uv run script/estimate_hybrid_time.py \
      --matcher-dir /srv/fs/MATTEO/dumptime5000 --bulk-dir /srv/fs/MATTEO/bulk_same
"""

import argparse
import csv
import gzip
import json
import os
from collections import Counter

DONE_SUFFIX = '_matches_stats.txt'

# tipi di riferimento usati come variabili della regressione
FEATURES = [
    ('A_ydoi',         'con DOI su Meta, trovato con DOI+anno'),
    ('A_altro',        'con DOI su Meta, trovato con altre query'),
    ('C_nessun_cand',  'con DOI su Meta, non trovato, nessun candidato'),
    ('C_sotto_soglia', 'con DOI su Meta, non trovato, candidato sotto soglia'),
    ('B',              'con DOI assente, trovato dal matcher'),
    ('D',              'con DOI assente, non trovato'),
    ('E',              'senza DOI, trovato'),
    ('F_solo_unstr',   'senza DOI, non trovato, solo stringa libera'),
    ('F_altro',        'senza DOI, non trovato, con campi strutturati'),
]
FEAT_LABEL = dict(FEATURES)
# ciò che nell'ibrido risolve il bulk (DOI presente su Meta): il matcher non lo vede più
REMOVED_BY_BULK = {'A_ydoi', 'A_altro', 'C_nessun_cand', 'C_sotto_soglia'}


def feature_of(cat, qtype, score, profile):
    if cat == 'A_entrambi_presente':
        return 'A_ydoi' if qtype == 'year_and_doi' else 'A_altro'
    if cat == 'C_falso_negativo_matcher':
        return 'C_nessun_cand' if score in ('', 'N/A', 'None') else 'C_sotto_soglia'
    if cat == 'B_solo_matcher_trova':
        return 'B'
    if cat == 'D_entrambi_assente':
        return 'D'
    if cat == 'E_senza_doi_trovato':
        return 'E'
    if cat == 'F_senza_doi_non_trovato':
        return 'F_solo_unstr' if profile == 'solo unstructured' else 'F_altro'
    return None


def solve(A, b):
    """Risolve A x = b (eliminazione di Gauss con pivot parziale)."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            raise ValueError("matrice singolare")
        M[col], M[piv] = M[piv], M[col]
        for r in range(n):
            if r != col:
                f = M[r][col] / M[col][col]
                if f:
                    for c in range(col, n + 1):
                        M[r][c] -= f * M[col][c]
    return [M[i][n] / M[i][i] for i in range(n)]


def ols(rows, y, names):
    """Minimi quadrati con intercetta. rows: liste di conteggi allineate a names."""
    k = len(names) + 1
    XtX = [[0.0] * k for _ in range(k)]
    Xty = [0.0] * k
    for x, t in zip(rows, y):
        v = [1.0] + x
        for i in range(k):
            vi = v[i]
            if vi:
                Xty[i] += vi * t
                row = XtX[i]
                for j in range(k):
                    if v[j]:
                        row[j] += vi * v[j]
    beta = solve(XtX, Xty)
    return beta[0], dict(zip(names, beta[1:]))


def fit_nonneg(rows_full, y, names_full):
    """OLS; se un costo per tipo esce negativo lo si fissa a 0 e si rifà il fit."""
    names = list(names_full)
    forced_zero = []
    while True:
        idx = [names_full.index(n) for n in names]
        rows = [[r[i] for i in idx] for r in rows_full]
        b0, betas = ols(rows, y, names)
        neg = [n for n in names if betas[n] < 0]
        if not neg:
            break
        worst = min(neg, key=lambda n: betas[n])
        names.remove(worst)
        forced_zero.append(worst)
    for n in names_full:
        betas.setdefault(n, 0.0)
    return b0, betas, forced_zero


def pctl(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, max(0, int(round(p / 100 * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


def fmt_dur(sec):
    sec = max(0.0, sec)
    days = sec / 86400
    if days >= 365:
        return f"{days / 365:.1f} anni"
    if days >= 60:
        return f"{days / 30.44:.1f} mesi"
    if days >= 1:
        return f"{days:.1f} giorni"
    return f"{sec / 3600:.1f} ore"


def main():
    ap = argparse.ArgumentParser(description="Stima del tempo del matcher e della modalità ibrida.")
    ap.add_argument('--matcher-dir', required=True)
    ap.add_argument('--bulk-dir', required=True)
    ap.add_argument('--out', help="Dove scrivere la stima (default: --bulk-dir).")
    ap.add_argument('--gap-cap', type=float, default=7200,
                    help="Oltre questa distanza (s) fra due work è un'interruzione, non lavoro.")
    ap.add_argument('--dump-works', type=float, default=160_000_000,
                    help="Work stimati nell'intero dump (default 160 milioni).")
    ap.add_argument('--pause-used', type=float, default=10,
                    help="Pausa tra i work usata nei run (--pause-duration; default del tool 10 s).")
    ap.add_argument('--rate-used', type=float, default=2.0, help="--rate-limit usato nei run.")
    ap.add_argument('--rate-new', type=float, default=2.8, help="--rate-limit ipotetico più alto.")
    ap.add_argument('--max-distinct-dois', type=float, default=200_000_000,
                    help="Tetto ai DOI diversi nell'intero dump, per la stima del bulk.")
    args = ap.parse_args()
    out_dir = args.out or args.bulk_dir

    # ---------------------------------------------- work e date di completamento
    works_path = os.path.join(args.bulk_dir, 'works_compared.txt')
    with open(works_path, encoding='utf-8') as f:
        works = [l.strip() for l in f if l.strip()]
    wset = set(works)
    mtime = {}
    with os.scandir(args.matcher_dir) as it:
        for e in it:
            if e.name.endswith(DONE_SUFFIX):
                w = e.name[:-len(DONE_SUFFIX)]
                if w in wset:
                    mtime[w] = e.stat().st_mtime
    order = sorted(mtime, key=mtime.get)
    duration, n_restart = {}, 0
    for prev, cur in zip(order, order[1:]):
        gap = mtime[cur] - mtime[prev]
        if gap <= args.gap_cap:
            duration[cur] = gap
        else:
            n_restart += 1

    # ------------------------------------- composizione dei riferimenti per work
    counts = {w: Counter() for w in works}
    totals = Counter()
    per_ref = os.path.join(args.bulk_dir, 'confronto_per_riferimento.csv.gz')
    with gzip.open(per_ref, 'rt', newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            feat = feature_of(r['categoria'], r['query_type'], r['score_matcher'], r['profilo'])
            if feat and r['work'] in counts:
                counts[r['work']][feat] += 1
                totals[feat] += 1
    names = [n for n, _ in FEATURES if totals[n] > 0]

    # ---------------------------------------------------------- regressione
    fit_works = [w for w in works if w in duration]
    rows = [[counts[w][n] for n in names] for w in fit_works]
    y = [duration[w] for w in fit_works]
    b0, beta, forced_zero = fit_nonneg(rows, y, names)
    pred = [b0 + sum(beta[n] * x for n, x in zip(names, row)) for row in rows]
    mean_y = sum(y) / len(y)
    ss_tot = sum((t - mean_y) ** 2 for t in y)
    ss_res = sum((t - p) ** 2 for t, p in zip(y, pred))
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0

    # diagnostica: durata dei work con 0 riferimenti = costo fisso (pausa inclusa)
    zero_ref = sorted(duration[w] for w in fit_works if not counts[w])
    all_d = sorted(y)
    n_300 = sum(1 for d in y if 280 <= d <= 340)

    # ------------------------------------------------------------- scenari
    W = len(works)
    fixed = b0 * W
    var_removed = sum(beta[n] * totals[n] for n in names if n in REMOVED_BY_BULK)
    var_resid = sum(beta[n] * totals[n] for n in names if n not in REMOVED_BY_BULK)
    fixed_nopause = max(b0 - args.pause_used, 0.0) * W
    rate_k = args.rate_used / args.rate_new
    measured = sum(y)
    model_total = fixed + var_removed + var_resid

    bs = json.load(open(os.path.join(args.bulk_dir, 'bulk_summary.json'), encoding='utf-8'))
    bulk_sample = bs.get('total_seconds', 0.0)
    factor = args.dump_works / W
    req = bs.get('requests') or 0
    sec_req = (bs.get('check_seconds', 0.0) / req) if req else 0.0
    chunk = bs.get('chunk_size') or 2000
    req_lin = req * factor
    req_cap = min(req_lin, args.max_distinct_dois / chunk)
    bulk_dump_lo = bs.get('extract_seconds', 0.0) * factor + req_cap * sec_req
    bulk_dump_hi = bs.get('extract_seconds', 0.0) * factor + req_lin * sec_req

    scen = [
        ('Matcher come nei tuoi run', fixed + var_removed + var_resid, 0.0),
        ('Matcher senza pausa tra i work', fixed_nopause + var_removed + var_resid, 0.0),
        ('Ibrido (bulk + matcher sul residuo)', fixed + var_resid, bulk_sample),
        ('Ibrido senza pausa tra i work', fixed_nopause + var_resid, bulk_sample),
        (f'Ibrido senza pausa, rate {args.rate_new:g} req/s',
         fixed_nopause + var_resid * rate_k, bulk_sample),
    ]

    # ------------------------------------------------------------- output
    L = []
    L.append("=" * 72)
    L.append("STIMA DEI TEMPI — costo per tipo di riferimento e modalità ibrida")
    L.append("=" * 72)
    L.append(f"work nel campione: {W:,} | con durata misurabile: {len(fit_works):,} "
             f"| interruzioni escluse (> {int(args.gap_cap)} s): {n_restart}")
    L.append(f"durata di un work — mediana {pctl(all_d, 50):.1f} s, 90° percentile "
             f"{pctl(all_d, 90):.1f} s, 99° percentile {pctl(all_d, 99):.1f} s")
    L.append(f"tempo misurato sui work con durata: {fmt_dur(measured)}; "
             f"tempo ricostruito dal modello su tutti i work: {fmt_dur(model_total)}")

    L.append("\n--- COSTO FISSO PER WORK " + "-" * 47)
    L.append(f"stima del modello: {b0:.1f} s per work (su {W:,} work: {fmt_dur(fixed)})")
    if zero_ref:
        L.append(f"controllo diretto: i {len(zero_ref):,} work con 0 riferimenti (nessuna query) "
                 f"durano in mediana {pctl(zero_ref, 50):.1f} s")
    L.append(f"la pausa tra i work usata nei run è {args.pause_used:g} s (--pause-used): se il "
             f"costo fisso è di poco superiore, quasi tutto è attesa")
    if n_300:
        L.append(f"work con durata 280-340 s: {n_300:,} (possibili pause di 5 minuti dopo errori ripetuti)")

    L.append("\n--- COSTO MEDIO PER TIPO DI RIFERIMENTO " + "-" * 33)
    L.append(f"  {'tipo':<16}{'riferimenti':>13}{'s per rif.':>12}{'tempo totale':>16}  descrizione")
    for n, desc in FEATURES:
        if totals[n] == 0:
            continue
        tot = beta[n] * totals[n]
        mark = ' (fissato a 0)' if n in forced_zero else ''
        L.append(f"  {n:<16}{totals[n]:>13,}{beta[n]:>12.2f}{fmt_dur(tot):>16}  {desc}{mark}")
    L.append(f"  qualità del modello: R² = {r2:.2f} (quota della variabilità delle durate spiegata)")
    L.append(f"  quota del tempo variabile che il bulk toglierebbe al matcher: "
             f"{100 * var_removed / (var_removed + var_resid):.1f}%"
             if (var_removed + var_resid) else "")

    L.append("\n--- SCENARI " + "-" * 60)
    L.append(f"  (proiezione sul dump: {args.dump_works:,.0f} work, cioè {factor:,.0f} volte il campione)")
    L.append(f"  {'scenario':<44}{'campione':>14}{'intero dump':>16}")
    rows_out = []
    for name, matcher_sec, bulk_sec in scen:
        sample_tot = matcher_sec + bulk_sec
        dump_tot = matcher_sec * factor
        L.append(f"  {name:<44}{fmt_dur(sample_tot):>14}{fmt_dur(dump_tot):>16}")
        rows_out.append({'scenario': name, 'campione_s': round(sample_tot, 1),
                         'dump_matcher_s': round(dump_tot, 1)})
    L.append(f"  negli scenari ibridi va aggiunto il bulk sull'intero dump: "
             f"tra {fmt_dur(bulk_dump_lo)} e {fmt_dur(bulk_dump_hi)}")
    L.append("\nNOTE")
    L.append("  - La proiezione assume che il resto del dump abbia la stessa composizione del")
    L.append("    campione (i primi file del dump): è un'ipotesi, non una misura.")
    L.append(f"  - Lo scenario con rate più alto riduce solo la parte variabile, in proporzione")
    L.append(f"    {args.rate_used:g}/{args.rate_new:g}: vale se il limite di richieste era il vero collo di bottiglia.")
    L.append("  - Il costo per tipo è una media: comprende attese per 429, GROBID, tentativi ripetuti.")

    text = "\n".join(L) + "\n"
    with open(os.path.join(out_dir, 'stima_tempi.txt'), 'w', encoding='utf-8') as f:
        f.write(text)
    with open(os.path.join(out_dir, 'stima_tempi.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'work': W, 'work_con_durata': len(fit_works), 'interruzioni_escluse': n_restart,
            'costo_fisso_s': b0, 'costo_per_tipo_s': beta, 'fissati_a_zero': forced_zero,
            'riferimenti_per_tipo': dict(totals), 'r2': r2,
            'mediana_work_0_riferimenti_s': pctl(zero_ref, 50) if zero_ref else None,
            'scenari': rows_out, 'fattore_dump': factor,
            'bulk_dump_s': [round(bulk_dump_lo, 1), round(bulk_dump_hi, 1)],
        }, f, indent=2, ensure_ascii=False)
    print(text)


if __name__ == '__main__':
    main()
