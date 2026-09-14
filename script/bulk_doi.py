"""
bulk_doi.py — modalità "bulk" del Reference Matching Tool (flag --bulk).

Modulo SEPARATO, importato dal tool principale, così le due logiche restano
distinte. Risponde alla domanda di ingestione ("questo riferimento è già su
OpenCitations Meta?") per i riferimenti CON DOI, in modo veloce e scalabile.

ITER LOGICO
  Fase 1   scorre il dump un membro alla volta (RAM costante) e divide i
           riferimenti in: CON DOI  vs  SENZA DOI.                    [locale]
  Fase 1.5 deduplica i DOI (su disco, con `sort -u`; fallback in memoria).
  Fase 2   verifica in blocco i DOI unici contro Meta: una query VALUES con
           qualche migliaio di DOI per richiesta.            [rete, rate-limited]
  Output   matched_dois.txt   -> DOI già presenti su Meta
           unmatched_dois.txt -> DOI assenti = CANDIDATI all'integrazione
           no_doi_references.jsonl -> residuo da passare al matcher fuzzy
           bulk_report.txt / bulk_summary.json -> conteggi e tempi per fase
           references_index.csv -> (solo con --bulk-same-as) un riferimento per
                                   riga, per il confronto con il matcher

NOTA IMPORTANTE (per il punto sollevato: "il DOI è meno univoco di quanto si
pensi"). Questo modulo NON promuove mai un DOI a "opera da integrare" sulla sola
base del DOI. L'esistenza del DOI è usata SOLO come filtro grezzo per scartare
in fretta ciò che è palesemente già presente. I `unmatched_dois` sono CANDIDATI
che vanno poi validati dal matcher completo (metadati: titolo/autore/anno),
proprio per intercettare le opere presenti su Meta sotto un altro identificatore.

La query di esistenza usa la tipizzazione ^^xsd:string, come il tool principale:
Meta memorizza gli identificatori come letterali tipizzati e Virtuoso NON unifica
un letterale semplice con uno tipizzato (era il bug degli 0 match). Non toglierla.
"""

import asyncio
import csv
import json
import os
import subprocess
import tarfile
import time
from typing import Dict, List, Optional, Set

import aiohttp


DEFAULT_ENDPOINT = "https://sparql.opencitations.net/meta"
XSD_STRING = "http://www.w3.org/2001/XMLSchema#string"

_PREFIXES = (
    "PREFIX datacite: <http://purl.org/spar/datacite/>\n"
    "PREFIX literal: <http://www.essepuntato.it/2010/06/literalreification/>\n"
)

_DOI_PREFIXES = (
    'doi:',
    'https://doi.org/', 'http://doi.org/',
    'https://dx.doi.org/', 'http://dx.doi.org/',
)


# ---------------------------------------------------------------------------
# DOI normalisation — tenuta compatibile con ReferenceMatchingTool
# ---------------------------------------------------------------------------

def normalize_doi(raw) -> Optional[str]:
    if not raw:
        return None
    doi = str(raw).strip().lower()
    if not doi:
        return None
    for p in _DOI_PREFIXES:
        if doi.startswith(p):
            doi = doi[len(p):].strip()
            break
    doi = doi.replace('\\/', '/')
    # un DOI reale inizia sempre per "10." — scarta il resto per non mandare
    # spazzatura all'endpoint
    return doi if doi.startswith('10.') else None


# ---------------------------------------------------------------------------
# Fase 1 — streaming del dump (identico alla logica --dump del tool)
# ---------------------------------------------------------------------------

def _iter_members(dump_path: str):
    """Restituisce (nome_base_membro, JSON) per ogni membro, in streaming.

    Il nome base ("0" per "0.json") è lo stesso che usa il matcher in --dump per
    chiamare i work "<base>_w<i>", così i due metodi sono allineabili.
    """
    if dump_path.endswith('.tar.gz') or dump_path.endswith('.tgz'):
        with tarfile.open(dump_path, 'r:gz') as tar:
            for m in tar:
                if not (m.isfile() and m.name.endswith('.json')):
                    continue
                fh = tar.extractfile(m)
                if fh is None:
                    continue
                base = os.path.splitext(os.path.basename(m.name))[0]
                try:
                    yield base, json.load(fh)
                finally:
                    fh.close()
    else:
        with open(dump_path, 'r', encoding='utf-8') as f:
            yield os.path.splitext(os.path.basename(dump_path))[0], json.load(f)


_MATCHER_DONE_SUFFIX = '_matches_stats.txt'


def load_works_from_matcher_dir(matcher_dir: str) -> Set[str]:
    """Insieme dei work che il matcher ha COMPLETATO in una sua cartella di output.

    Il matcher scrive `<work>_matches_stats.txt` per ultimo, quindi la sua
    presenza indica un work finito (un work ancora in corso viene escluso).
    """
    works: Set[str] = set()
    with os.scandir(matcher_dir) as it:
        for entry in it:
            if entry.name.endswith(_MATCHER_DONE_SUFFIX):
                works.add(entry.name[:-len(_MATCHER_DONE_SUFFIX)])
    return works


# campi di servizio di Crossref che non descrivono l'opera citata
_NON_DESCRIPTIVE_KEYS = {'key', 'doi-asserted-by'}


def _populated_fields(ref: dict) -> List[str]:
    return sorted(k for k, v in ref.items()
                  if k not in _NON_DESCRIPTIVE_KEYS and v not in (None, '', [], {}))


def extract(dump_path: str, out_dir: str, limit: int = 0,
            works_filter: Optional[Set[str]] = None,
            ref_index: bool = False) -> Dict:
    """Fase 1: separa i riferimenti CON DOI da quelli SENZA DOI.

    Scrive:
      <out_dir>/dois_raw.txt            un DOI per riga (con duplicati)
      <out_dir>/no_doi_references.jsonl un riferimento (senza DOI) per riga
      <out_dir>/references_index.csv    (solo se ref_index) una riga per
                                        riferimento: work, ref_N, DOI, campi
    works_filter: se dato, elabora SOLO i work con quel nome ("<base>_w<i>"),
    ad es. quelli già elaborati dal matcher, per un confronto sullo stesso set.
    Restituisce i conteggi.
    """
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, 'dois_raw.txt')
    nodoi_path = os.path.join(out_dir, 'no_doi_references.jsonl')
    index_path = os.path.join(out_dir, 'references_index.csv')

    works = refs_total = refs_with_doi = refs_no_doi = 0
    remaining = set(works_filter) if works_filter is not None else None
    processed_works: List[str] = []
    found_any = False
    stop = False
    t0 = time.time()

    index_out = open(index_path, 'w', newline='', encoding='utf-8') if ref_index else None
    index_writer = None
    if index_out is not None:
        index_writer = csv.writer(index_out)
        index_writer.writerow(['work', 'ref_id', 'doi', 'doi_raw_present', 'fields'])

    try:
        with open(raw_path, 'w', encoding='utf-8') as raw_out, \
             open(nodoi_path, 'w', encoding='utf-8') as nodoi_out:
            for base, data in _iter_members(dump_path):
                if stop:
                    break
                items = data.get('items', []) if isinstance(data, dict) else (data or [])
                in_this_member = 0
                for i, work in enumerate(items):
                    work_name = f"{base}_w{i}"
                    if remaining is not None:
                        if work_name not in remaining:
                            continue
                        remaining.discard(work_name)
                        processed_works.append(work_name)
                        in_this_member += 1
                    works += 1
                    parent_doi = work.get('DOI')
                    # ref_<n> con n da 1: stessa numerazione del matcher
                    for n, ref in enumerate(work.get('reference', []) or [], 1):
                        refs_total += 1
                        d = normalize_doi(ref.get('DOI'))
                        if index_writer is not None:
                            index_writer.writerow([
                                work_name, f'ref_{n}', d or '',
                                1 if str(ref.get('DOI') or '').strip() else 0,
                                '|'.join(_populated_fields(ref)),
                            ])
                        if d is not None:
                            refs_with_doi += 1
                            raw_out.write(d + '\n')
                        else:
                            refs_no_doi += 1
                            # residuo per il matcher: il riferimento grezzo + il DOI
                            # dell'opera che lo cita (contesto)
                            nodoi_out.write(json.dumps(
                                {'citing_work': work_name, 'citing_work_doi': parent_doi,
                                 'ref_id': f'ref_{n}', 'reference': ref},
                                ensure_ascii=False) + '\n')
                    if limit and works >= limit:
                        stop = True
                        break
                if remaining is not None:
                    if in_this_member:
                        found_any = True
                    # il matcher elabora il dump in ordine: un membro (non vuoto) senza
                    # work del set, dopo averne trovati, significa che il set è finito
                    if not remaining or (found_any and items and in_this_member == 0):
                        stop = True
    finally:
        if index_out is not None:
            index_out.close()

    stats = {
        'works': works,
        'references_total': refs_total,
        'references_with_doi': refs_with_doi,
        'references_without_doi': refs_no_doi,
        'pct_refs_with_doi': round(100 * refs_with_doi / refs_total, 2) if refs_total else 0.0,
        'extract_seconds': round(time.time() - t0, 1),
        'dois_raw_path': raw_path,
        'no_doi_references_path': nodoi_path,
    }
    if ref_index:
        stats['references_index_path'] = index_path
    if works_filter is not None:
        # elenco esatto dei work confrontati (anche quelli con 0 riferimenti,
        # che non compaiono nell'indice): serve alla stima dei tempi del matcher
        works_list_path = os.path.join(out_dir, 'works_compared.txt')
        with open(works_list_path, 'w', encoding='utf-8') as f:
            for wn in processed_works:
                f.write(wn + '\n')
        stats['works_requested'] = len(works_filter)
        stats['works_not_found_in_dump'] = len(remaining)
        stats['works_list_path'] = works_list_path
    return stats


def dedup(out_dir: str) -> str:
    """Fase 1.5: DOI unici. Prova `sort -u` (scala su disco), fallback in memoria."""
    raw_path = os.path.join(out_dir, 'dois_raw.txt')
    uniq_path = os.path.join(out_dir, 'dois_unique.txt')
    try:
        env = dict(os.environ, LC_ALL='C')
        subprocess.run(['sort', '-u', raw_path, '-o', uniq_path], check=True, env=env)
    except Exception:
        # fallback: dedup in memoria (attenzione: usa RAM proporzionale ai DOI unici)
        seen: Set[str] = set()
        with open(raw_path, 'r', encoding='utf-8') as f:
            for line in f:
                d = line.strip()
                if d:
                    seen.add(d)
        with open(uniq_path, 'w', encoding='utf-8') as f:
            for d in sorted(seen):
                f.write(d + '\n')
    return uniq_path


# ---------------------------------------------------------------------------
# Fase 2 — verifica di esistenza in blocco contro Meta (async, rate-limited)
# ---------------------------------------------------------------------------

def _esc(doi: str) -> str:
    return doi.replace('\\', '\\\\').replace('"', '\\"')


def _build_query(dois: List[str]) -> str:
    values = " ".join(f'"{_esc(d)}"^^<{XSD_STRING}>' for d in dois)
    return (
        _PREFIXES +
        "SELECT DISTINCT ?doi WHERE {\n"
        f"  VALUES ?doi {{ {values} }}\n"
        "  ?id datacite:usesIdentifierScheme datacite:doi ;\n"
        "      literal:hasLiteralValue ?doi .\n"
        "}"
    )


class _RateLimiter:
    """Mantiene un intervallo minimo fra richieste (per stare sotto max_per_min)."""
    def __init__(self, max_per_min: int):
        self.min_interval = 60.0 / max(max_per_min, 1)
        self._last = 0.0

    async def wait(self):
        gap = time.monotonic() - self._last
        if gap < self.min_interval:
            await asyncio.sleep(self.min_interval - gap)
        self._last = time.monotonic()


def _read_checkpoint(path: str) -> int:
    if os.path.exists(path):
        try:
            return int(open(path).read().strip() or 0)
        except ValueError:
            return 0
    return 0


def _write_checkpoint(path: str, n: int):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write(str(n))
    os.replace(tmp, path)


def _iter_chunks(uniq_path: str, chunk_size: int, skip: int):
    buf: List[str] = []
    with open(uniq_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < skip:
                continue
            d = line.strip()
            if not d:
                continue
            buf.append(d)
            if len(buf) >= chunk_size:
                yield buf
                buf = []
    if buf:
        yield buf


async def _query_chunk(session, endpoint, query, limiter,
                       max_retries=6, timeout=180) -> Optional[Set[str]]:
    for attempt in range(max_retries):
        await limiter.wait()
        try:
            async with session.post(
                endpoint,
                data={'query': query, 'format': 'json'},
                headers={'Accept': 'application/sparql-results+json'},
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status == 200:
                    js = await resp.json(content_type=None)
                    bindings = js.get('results', {}).get('bindings', [])
                    return {b['doi']['value'] for b in bindings if 'doi' in b}
                if resp.status == 429:
                    ra = resp.headers.get('Retry-After')
                    wait = int(ra) if (ra and ra.isdigit()) else min(5 * (attempt + 1), 60)
                    await asyncio.sleep(wait)
                    continue
                await asyncio.sleep(min(5 * (attempt + 1), 60))
        except Exception:
            await asyncio.sleep(min(5 * (attempt + 1), 60))
    return None


async def check(uniq_path: str, out_dir: str, endpoint: str = DEFAULT_ENDPOINT,
                chunk_size: int = 2000, max_per_min: int = 170) -> Dict:
    """Fase 2: quali DOI unici esistono su Meta. Ripartibile via checkpoint."""
    matched_path = os.path.join(out_dir, 'matched_dois.txt')
    unmatched_path = os.path.join(out_dir, 'unmatched_dois.txt')
    err_path = os.path.join(out_dir, 'check_errors.txt')
    ckpt_path = os.path.join(out_dir, 'bulk_checkpoint.txt')

    total = sum(1 for _ in open(uniq_path, 'r', encoding='utf-8'))
    done = _read_checkpoint(ckpt_path)
    limiter = _RateLimiter(max_per_min)

    n_matched = n_unmatched = n_error = n_requests = 0
    async with aiohttp.ClientSession() as session:
        with open(matched_path, 'a', encoding='utf-8') as m_out, \
             open(unmatched_path, 'a', encoding='utf-8') as u_out, \
             open(err_path, 'a', encoding='utf-8') as e_out:
            for chunk in _iter_chunks(uniq_path, chunk_size, done):
                present = await _query_chunk(session, endpoint, _build_query(chunk), limiter)
                n_requests += 1
                if present is None:
                    for d in chunk:
                        e_out.write(d + '\n')
                    n_error += len(chunk)
                else:
                    for d in chunk:
                        if d in present:
                            m_out.write(d + '\n'); n_matched += 1
                        else:
                            u_out.write(d + '\n'); n_unmatched += 1
                m_out.flush(); u_out.flush(); e_out.flush()
                done += len(chunk)
                _write_checkpoint(ckpt_path, done)
                pct = 100 * done / total if total else 100
                print(f"  bulk: {done:,}/{total:,} ({pct:.1f}%) | "
                      f"presenti {n_matched:,} assenti {n_unmatched:,} err {n_error:,}",
                      flush=True)

    return {
        'unique_dois': total,
        'matched_in_meta': n_matched,
        'unmatched_candidates': n_unmatched,
        'errored': n_error,
        'requests': n_requests,
        'matched_path': matched_path,
        'unmatched_path': unmatched_path,
    }


# ---------------------------------------------------------------------------
# Orchestrazione — è ciò che il tool principale chiama con --bulk
# ---------------------------------------------------------------------------

async def run_bulk(dump_path: str, out_dir: str, endpoint: str = DEFAULT_ENDPOINT,
                   limit: int = 0, chunk_size: int = 2000,
                   max_per_min: int = 170,
                   works_filter: Optional[Set[str]] = None,
                   ref_index: bool = False) -> Dict:
    """Esegue l'intera pipeline bulk e scrive un report. Restituisce il riepilogo.

    works_filter / ref_index servono per il confronto con il matcher sullo stesso
    insieme di work (vedi --bulk-same-as e compare_bulk_matcher.py).
    """
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()

    if works_filter is not None:
        print(f"\n[bulk] Limitato a {len(works_filter):,} work (stesso set del matcher)")
    print(f"\n[bulk] Fase 1 — estrazione DOI dal dump: {dump_path}")
    ex = extract(dump_path, out_dir, limit=limit,
                 works_filter=works_filter, ref_index=ref_index)
    print(f"[bulk]   works={ex['works']:,} | riferimenti={ex['references_total']:,} "
          f"| con DOI={ex['references_with_doi']:,} ({ex['pct_refs_with_doi']}%) "
          f"| senza DOI={ex['references_without_doi']:,}")
    if ex.get('works_not_found_in_dump'):
        print(f"[bulk]   ATTENZIONE: {ex['works_not_found_in_dump']:,} work del set "
              f"non trovati nel dump")

    print("[bulk] Fase 1.5 — deduplica DOI")
    t = time.time()
    uniq_path = dedup(out_dir)
    dedup_seconds = round(time.time() - t, 1)
    unique_count = sum(1 for _ in open(uniq_path, 'r', encoding='utf-8'))
    print(f"[bulk]   DOI unici da verificare: {unique_count:,}")

    print(f"[bulk] Fase 2 — verifica esistenza su Meta "
          f"(chunk={chunk_size}, ~{max_per_min}/min)")
    t = time.time()
    ck = await check(uniq_path, out_dir, endpoint=endpoint,
                     chunk_size=chunk_size, max_per_min=max_per_min)
    check_seconds = round(time.time() - t, 1)
    total_seconds = round(time.time() - t_start, 1)

    summary = {**ex, **ck, 'unique_dois': unique_count,
               'dedup_seconds': dedup_seconds, 'check_seconds': check_seconds,
               'total_seconds': total_seconds, 'chunk_size': chunk_size,
               'max_per_min': max_per_min, 'dump': dump_path}

    with open(os.path.join(out_dir, 'bulk_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    report_path = os.path.join(out_dir, 'bulk_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== BULK DOI CHECK — REPORT ===\n")
        f.write(f"dump: {dump_path}\n")
        f.write(f"works                    : {ex['works']:,}\n")
        f.write(f"riferimenti totali       : {ex['references_total']:,}\n")
        f.write(f"  con DOI                : {ex['references_with_doi']:,} ({ex['pct_refs_with_doi']}%)\n")
        f.write(f"  senza DOI (al matcher) : {ex['references_without_doi']:,}\n")
        f.write(f"DOI unici verificati     : {ck['unique_dois']:,}\n")
        f.write(f"  gia su Meta            : {ck['matched_in_meta']:,}\n")
        f.write(f"  ASSENTI (candidati)    : {ck['unmatched_candidates']:,}\n")
        f.write(f"  in errore (da rifare)  : {ck['errored']:,}\n")
        f.write(f"richieste SPARQL         : {ck['requests']:,}\n")
        f.write(f"tempo estrazione         : {ex['extract_seconds']:,} s\n")
        f.write(f"tempo deduplica          : {dedup_seconds:,} s\n")
        f.write(f"tempo verifica su Meta   : {check_seconds:,} s\n")
        f.write(f"tempo totale             : {total_seconds:,} s\n")
        f.write("\nCandidati e residuo senza DOI vanno validati col matcher completo.\n")

    print("\n[bulk] COMPLETATO")
    print(f"[bulk]   gia su Meta : {ck['matched_in_meta']:,}")
    print(f"[bulk]   candidati   : {ck['unmatched_candidates']:,}  -> {ck['unmatched_path']}")
    print(f"[bulk]   senza DOI   : {ex['references_without_doi']:,}  -> {ex['no_doi_references_path']}")
    print(f"[bulk]   tempo totale: {total_seconds:,} s ({ck['requests']:,} richieste SPARQL)")
    print(f"[bulk]   report      : {report_path}")
    return summary


# Esecuzione stand-alone opzionale (per test rapidi senza il tool principale)
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description="Bulk DOI existence check (stand-alone).")
    ap.add_argument('dump')
    ap.add_argument('-o', '--output-dir', required=True)
    ap.add_argument('--endpoint', default=DEFAULT_ENDPOINT)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--chunk-size', type=int, default=2000)
    ap.add_argument('--max-per-min', type=int, default=170)
    ap.add_argument('--same-as', metavar='MATCHER_DIR',
                    help='Elabora solo i work già completati dal matcher in MATCHER_DIR '
                         'e scrive references_index.csv per il confronto.')
    a = ap.parse_args()
    wf = load_works_from_matcher_dir(a.same_as) if a.same_as else None
    asyncio.run(run_bulk(a.dump, a.output_dir, endpoint=a.endpoint, limit=a.limit,
                         chunk_size=a.chunk_size, max_per_min=a.max_per_min,
                         works_filter=wf, ref_index=bool(a.same_as)))
