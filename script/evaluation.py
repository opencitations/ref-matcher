import os
import csv
import json
import re
from typing import Dict, List, Tuple, Optional, Set
from SPARQLWrapper import SPARQLWrapper, JSON
from collections import defaultdict
import time
import argparse

# Same production endpoint the main matcher targets (kept consistent on purpose).
DEFAULT_SPARQL_ENDPOINT = "https://sparql.opencitations.net/meta"


class OpenCitationsDOIMatcher:
    def __init__(self, endpoint_url=DEFAULT_SPARQL_ENDPOINT):
        self.sparql = SPARQLWrapper(endpoint_url)
        self.sparql.setReturnFormat(JSON)

    def extract_dois_from_json(self, json_data: Dict) -> List[Tuple[str, str]]:
        """
        Estrae (reference_id, doi) dal JSON Crossref:
        - reference_id = b{idx}
        - doi normalizzato in minuscolo e con '/' non escaped
        """
        dois = []
        if 'message' in json_data and 'reference' in json_data['message']:
            for idx, ref in enumerate(json_data['message']['reference']):
                if 'DOI' in ref and ref['DOI'] not in ['.', '']:
                    doi = ref['DOI'].replace('\\/', '/').lower()
                    dois.append((f"b{idx}", doi))
        return dois

    def generate_sparql_query(self, doi: str) -> str:
        # Enumerate every predicate/object of the bibliographic resource carrying
        # this DOI. (The previous version had two dead BINDs: ?publicationDate was
        # never bound, and ?id was never selected.)
        safe_doi = doi.replace('\\', '\\\\').replace('"', '\\"')
        return f"""PREFIX datacite: <http://purl.org/spar/datacite/>
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX literal: <http://www.essepuntato.it/2010/06/literalreification/>
PREFIX prism: <http://prismstandard.org/namespaces/basic/2.0/>

SELECT ?predicate ?object {{
    ?identifier literal:hasLiteralValue "{safe_doi}" .
    ?br datacite:hasIdentifier ?identifier ;
        ?predicate ?object .
}}"""

    def execute_sparql_query(self, query: str, max_retries=3, retry_delay=5) -> Optional[Dict]:
        self.sparql.setQuery(query)
        for attempt in range(max_retries):
            try:
                return self.sparql.query().convert()
            except Exception as e:
                print(f"Query attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
        return None

    def process_json_file(self, json_file_path: str) -> Tuple[Dict, set, int, int, List, List]:
        """
        Ritorna:
          - results_dict: ref_id -> dict con campi aggregati (incluso DOI)
          - all_predicates: insieme dei predicati visti
          - successful_queries: numero DOI con almeno 1 binding
          - total_queries: DOI totali interrogati
          - dois_with_results: lista (ref_id, doi) con risultati
          - dois_without_results: lista (ref_id, doi) senza risultati
        """
        # lettura JSON robusta (utf-8, poi fallback)
        data = None
        for enc in ('utf-8', 'utf-8-sig', 'latin-1'):
            try:
                with open(json_file_path, 'r', encoding=enc) as f:
                    data = json.load(f)
                    break
            except Exception:
                data = None
        if data is None:
            raise ValueError(f"Impossibile leggere il file JSON: {json_file_path}")

        dois = self.extract_dois_from_json(data)
        total_queries = len(dois)
        successful_queries = 0

        dois_with_results = []
        dois_without_results = []
        all_predicates = set()
        results_dict = {}

        for ref_id, doi in dois:
            query = self.generate_sparql_query(doi)
            results = self.execute_sparql_query(query)

            if results and 'results' in results and results['results']['bindings']:
                successful_queries += 1
                dois_with_results.append((ref_id, doi))

                doi_data = defaultdict(list)
                doi_data['reference_id'] = ref_id
                doi_data['DOI'] = doi

                for binding in results['results']['bindings']:
                    predicate = binding.get('predicate', {}).get('value', '')
                    object_value = binding.get('object', {}).get('value', '')
                    if predicate:
                        all_predicates.add(predicate)
                        doi_data[predicate] = object_value

                results_dict[ref_id] = doi_data
            else:
                dois_without_results.append((ref_id, doi))

        return results_dict, all_predicates, successful_queries, total_queries, dois_with_results, dois_without_results

    def save_results(self, results_dict: Dict, all_predicates: set, output_file: str):
        headers = ['reference_id', 'DOI'] + sorted(list(all_predicates))
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for ref_id in sorted(results_dict.keys(), key=lambda x: int(x[1:])):
                doi_data = results_dict[ref_id]
                row = {'reference_id': doi_data['reference_id'], 'DOI': doi_data['DOI']}
                for predicate in all_predicates:
                    row[predicate] = doi_data.get(predicate, '')
                writer.writerow(row)

    def save_statistics(self, successful_queries: int, total_queries: int, stats_file: str):
        with open(stats_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Total queries', total_queries])
            writer.writerow(['Successful queries', successful_queries])
            if total_queries > 0:
                success_rate = (successful_queries/total_queries)*100
                writer.writerow(['Success rate', f'{success_rate:.2f}%'])
            else:
                writer.writerow(['Success rate', 'N/A'])

    def save_unmatched_dois(self, dois_without_results: List, unmatched_file: str):
        with open(unmatched_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['reference_id', 'DOI'])
            for ref_id, doi in sorted(dois_without_results, key=lambda x: int(x[0][1:])):
                writer.writerow([ref_id, doi])


def _norm_doi(s: str) -> str:
    """Normalizzazione globale (usata nel blocco metrics per filtered_matches)."""
    if not s:
        return ""
    s = s.strip().lower().replace('\\/', '/')
    if s.startswith('doi:'):
        s = s[4:].strip()
    for pref in ('https://doi.org/', 'http://doi.org/', 'https://dx.doi.org/', 'http://dx.doi.org/'):
        if s.startswith(pref):
            s = s[len(pref):].strip()
            break
    return s


def _find_matches_file(matches_dir: str, base: str) -> str:
    """
    Trova il file dei match per 'base' dentro matches_dir (anche ricorsivo).
    Ordine di preferenza:
      1) <base>_matches.csv
      2) <base>_matches_GS.csv
      3) qualunque CSV che contenga 'matches' nel nome e anche 'base' (case-insensitive)
    Ritorna percorso assoluto o stringa vuota se non trovato.
    """
    base_lower = base.lower()

    exact1 = os.path.join(matches_dir, f"{base}_matches.csv")
    if os.path.exists(exact1):
        return exact1
    exact2 = os.path.join(matches_dir, f"{base}_matches_GS.csv")
    if os.path.exists(exact2):
        return exact2


    for root, _, files in os.walk(matches_dir):
        for fn in files:
            if not fn.lower().endswith(".csv"):
                continue
            name = fn.lower()
            if "matches" not in name:
                continue
            if base_lower in name:
                return os.path.join(root, fn)

    return ""


class MatchComparator:
    # ---------- helpers comuni ----------
    @staticmethod
    def _norm_doi(s: str) -> str:
        """Normalizza il DOI per confronti robusti (delega a _norm_doi di modulo)."""
        return _norm_doi(s)

    @staticmethod
    def _read_csv_rows(path: str) -> List[Dict[str, str]]:
        with open(path, 'r', encoding='utf-8') as f:
            return list(csv.DictReader(f))

    @staticmethod
    def _first_col(fieldnames, candidates) -> Optional[str]:
        """Trova la prima colonna presente (case-insensitive) tra i candidates."""
        if not fieldnames:
            return None
        lower = {c.lower(): c for c in fieldnames}
        for cand in candidates:
            if cand.lower() in lower:
                return lower[cand.lower()]
        return None

    # ---------- compare ----------
    @staticmethod
    def compare_dois(file1_path: str, file2_path: str) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
        """
        Confronta DOIs:
        - file1_path: <base>_doi_results.csv  (ground truth positivi)
        - file2_path: <base>_matches*.csv     (predetti positivi)
        Ritorna liste di (reference_id, doi_normalizzato, basename) per missed/earned.
        """
        filename = os.path.basename(file1_path)
        basename = filename.replace('_doi_results.csv', '')

        rows1 = MatchComparator._read_csv_rows(file1_path)
        rows2 = MatchComparator._read_csv_rows(file2_path)

        # colonne flessibili
        doi_col_1 = MatchComparator._first_col(rows1[0].keys() if rows1 else [], ['DOI', 'doi'])
        ref_col_1 = MatchComparator._first_col(rows1[0].keys() if rows1 else [], ['reference_id', 'ref_id', 'id'])

        doi_col_2 = MatchComparator._first_col(rows2[0].keys() if rows2 else [], ['matched_doi', 'doi'])
        ref_col_2 = MatchComparator._first_col(rows2[0].keys() if rows2 else [], ['reference_id', 'ref_id', 'id'])

        dois_file1: Dict[str, Tuple[str, str]] = {}
        if doi_col_1:
            for row in rows1:
                d = MatchComparator._norm_doi(row.get(doi_col_1, ''))
                if not d:
                    continue
                refid = (row.get(ref_col_1) if ref_col_1 else '') or ''
                dois_file1[d] = (refid, basename)

        dois_file2: Dict[str, Tuple[str, str]] = {}
        if doi_col_2:
            for row in rows2:
                d = MatchComparator._norm_doi(row.get(doi_col_2, ''))
                if not d:
                    continue
                refid = (row.get(ref_col_2) if ref_col_2 else '') or ''
                dois_file2[d] = (refid, basename)

        missed_set = set(dois_file1) - set(dois_file2)
        earned_set = set(dois_file2) - set(dois_file1)

        missed_matches = [(dois_file1[doi][0], doi, dois_file1[doi][1]) for doi in missed_set]
        earned_matches = [(dois_file2[doi][0], doi, dois_file2[doi][1]) for doi in earned_set]

        return missed_matches, earned_matches

    # ---------- loader insiemi per metrics ----------
    @staticmethod
    def _load_pos_dois(doi_results_path: str) -> Set[str]:
        """DOI positivi (OpenCitations ha restituito risultati)."""
        rows = MatchComparator._read_csv_rows(doi_results_path)
        if not rows:
            return set()
        doi_col = MatchComparator._first_col(rows[0].keys(), ['DOI', 'doi'])
        if not doi_col:
            return set()
        return { MatchComparator._norm_doi(r.get(doi_col, '')) for r in rows
                 if MatchComparator._norm_doi(r.get(doi_col, '')) }

    @staticmethod
    def _load_neg_dois(unmatched_path: str) -> Set[str]:
        """DOI negativi (presenti nel JSON ma senza risultati da OpenCitations)."""
        if not os.path.exists(unmatched_path):
            return set()
        rows = MatchComparator._read_csv_rows(unmatched_path)
        if not rows:
            return set()
        doi_col = MatchComparator._first_col(rows[0].keys(), ['DOI', 'doi'])
        if not doi_col:
            return set()
        return { MatchComparator._norm_doi(r.get(doi_col, '')) for r in rows
                 if MatchComparator._norm_doi(r.get(doi_col, '')) }

    @staticmethod
    def _load_predicted_dois(matches_path: str) -> Set[str]:
        """DOI predetti dal matcher (preferisce 'matched_doi', fallback 'doi')."""
        rows = MatchComparator._read_csv_rows(matches_path)
        if not rows:
            return set()
        doi_col = MatchComparator._first_col(rows[0].keys(), ['matched_doi', 'doi'])
        if not doi_col:
            return set()
        return { MatchComparator._norm_doi(r.get(doi_col, '')) for r in rows
                 if MatchComparator._norm_doi(r.get(doi_col, '')) }

    # NOTE: the per-<base> TP/FP/FN/TN aggregation lives in main()'s ``metrics``
    # action (it also emits per-base debug rows). A second, never-called copy used
    # to live here and was removed to keep a single implementation.


def _read_json_robust(path: str) -> Optional[Dict]:
    """Read a JSON file trying a few encodings (Crossref dumps vary)."""
    for enc in ('utf-8', 'utf-8-sig', 'latin-1'):
        try:
            with open(path, 'r', encoding=enc) as f:
                return json.load(f)
        except Exception:
            continue
    return None


def _load_crossref_oracle(json_path: str) -> Dict[int, str]:
    """Map reference index (0-based) -> normalised DOI, for references that carry
    a DOI in Crossref. This is the (silver) per-reference oracle used by
    per_ref_metrics. Index i corresponds to the matcher's reference_id 'ref_{i+1}'.
    """
    data = _read_json_robust(json_path)
    oracle: Dict[int, str] = {}
    if data and 'message' in data and 'reference' in data['message']:
        for idx, ref in enumerate(data['message']['reference']):
            raw = ref.get('DOI')
            if raw and raw not in ('.', ''):
                d = _norm_doi(raw)
                if d:
                    oracle[idx] = d
    return oracle


def _load_matches_by_refid(matches_path: str) -> Dict[str, Dict[str, str]]:
    """reference_id -> full match row (so we can read matched_doi, title, score)."""
    out: Dict[str, Dict[str, str]] = {}
    for r in MatchComparator._read_csv_rows(matches_path):
        rid = (r.get('reference_id') or '').strip()
        if rid:
            out[rid] = r
    return out


def main():
    parser = argparse.ArgumentParser(description='Process references and compare matches')
    parser.add_argument('action',
                        choices=['check_doi', 'compare', 'metrics',
                                 'per_ref_metrics', 'check_presence'],
                        help='Action to perform')
    parser.add_argument('input_path', help='Input directory or file (placeholder for some actions)')
    parser.add_argument('--output_dir', help='Output directory', default='.')
    parser.add_argument('--check_doi_dir', help='Check DOI results directory')
    parser.add_argument('--matches_dir', help='Matches directory')
    # per_ref_metrics: the Crossref JSON directory is the per-reference DOI oracle.
    parser.add_argument('--crossref_dir',
                        help='Directory of Crossref JSON files (oracle for per_ref_metrics)')
    # check_presence: directory of *_unmatched.csv to test for metadata presence.
    parser.add_argument('--unmatched_dir',
                        help='Directory of *_unmatched.csv files (for check_presence)')
    parser.add_argument('--limit', type=int, default=0,
                        help='Cap the number of references checked (check_presence; 0 = no cap)')
    parser.add_argument('--endpoint', default='https://sparql.opencitations.net/meta',
                        help='SPARQL endpoint (check_presence)')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    matcher = OpenCitationsDOIMatcher()
    comparator = MatchComparator()

    if args.action == 'check_doi':

        for filename in os.listdir(args.input_path):
            if filename.lower().endswith('.json'):
                basename = os.path.splitext(filename)[0]
                json_file_path = os.path.join(args.input_path, filename)

                results = matcher.process_json_file(json_file_path)

                results_file = os.path.join(args.output_dir, f"{basename}_doi_results.csv")
                stats_file = os.path.join(args.output_dir, f"{basename}_statistics.csv")
                unmatched_file = os.path.join(args.output_dir, f"{basename}_unmatched_dois.csv")

                matcher.save_results(results[0], results[1], results_file)
                matcher.save_statistics(results[2], results[3], stats_file)
                matcher.save_unmatched_dois(results[5], unmatched_file)

    elif args.action == 'compare':
        compare_results = {
            'missed_matches': [],
            'earned_matches': [],
            'total_missed': 0,
            'total_earned': 0
        }

        for check_doi_file in os.listdir(args.check_doi_dir):
            if check_doi_file.lower().endswith('_doi_results.csv'):
                base_name = check_doi_file.replace('_doi_results.csv', '')

                file1_path = os.path.join(args.check_doi_dir, check_doi_file)

                # Trova il file dei match corrispondente (flessibile/ricorsivo)
                file2_path = _find_matches_file(args.matches_dir, base_name)
                if not file2_path:
                    print(f"[compare] Nessun file di match trovato per '{base_name}' "
                          f"in '{args.matches_dir}' (attesi: <base>_matches*.csv).")
                    continue

                missed, earned = comparator.compare_dois(file1_path, file2_path)
                compare_results['missed_matches'].extend(missed)
                compare_results['earned_matches'].extend(earned)
                compare_results['total_missed'] += len(missed)
                compare_results['total_earned'] += len(earned)

        comparison_file = os.path.join(args.output_dir, "comparison_results.csv")
        with open(comparison_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Type', 'Reference ID', 'DOI', 'File'])
            writer.writerow(['Missed Matches:'])
            for ref_id, doi, basename in compare_results['missed_matches']:
                writer.writerow(['missed', ref_id, doi, basename])
            writer.writerow([])
            writer.writerow(['Earned Matches:'])
            for ref_id, doi, basename in compare_results['earned_matches']:
                writer.writerow(['earned', ref_id, doi, basename])
            writer.writerow([])
            writer.writerow(['Summary:'])
            writer.writerow(['Total Missed', compare_results['total_missed']])
            writer.writerow(['Total Earned', compare_results['total_earned']])

    elif args.action == 'metrics':
        print("Calculating overall evaluation metrics...")

        filtered_matches_dir = os.path.join(args.output_dir, "filtered_matches")
        os.makedirs(filtered_matches_dir, exist_ok=True)

        # DEBUG: contatore basi elaborate
        bases_seen = 0
        debug_rows = []

        total_TP = total_FP = total_FN = total_TN = 0
        # "Invisible"/unverifiable predictions: DOIs the matcher predicted that are
        # in NEITHER the positive nor the negative ground-truth set (i.e. not one of
        # the paper's cited-and-known DOIs). These are the evaluation's blind spot —
        # a mix of enrichment of DOI-less references and possible mismatches. We
        # surface them so they can be inspected/verified rather than silently dropped.
        total_invisible = 0
        invisible_rows = []

        for check_doi_file in os.listdir(args.check_doi_dir):
            if not check_doi_file.lower().endswith('_doi_results.csv'):
                continue

            file_base_name = check_doi_file[:-len('_doi_results.csv')]
            doi_results_path = os.path.join(args.check_doi_dir, check_doi_file)

            # Trova il file dei match corrispondente (flessibile/ricorsivo)
            matches_path = _find_matches_file(args.matches_dir, file_base_name)
            if not matches_path:
                print(f"[metrics] Nessun matches per base '{file_base_name}' "
                      f"in '{args.matches_dir}'. Calcolo comunque FN/TN con PRED vuoto.")
                POS = MatchComparator._load_pos_dois(doi_results_path)
                NEG = MatchComparator._load_neg_dois(os.path.join(args.check_doi_dir, file_base_name + '_unmatched_dois.csv'))
                PRED = set()
                TP = 0
                FP = 0
                FN = len(POS - PRED)
                TN = len(NEG - PRED)
                total_TP += TP; total_FP += FP; total_FN += FN; total_TN += TN
                bases_seen += 1
                debug_rows.append({
                    'base': file_base_name, 'POS': len(POS), 'NEG': len(NEG), 'PRED': len(PRED),
                    'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN, 'INVISIBLE': 0,
                    'matches_file': '(none)'
                })
                continue



            # TP = PRED ∩ POS → predetti e presenti in Meta.

            # FP = PRED ∩ NEG → predetti ma “non trovati in Meta”.

            # FN = POS - PRED → presenti in Meta ma non predetti.

            # TN = NEG - PRED → “non trovati in Meta” e non predetti.
            
            
            POS = MatchComparator._load_pos_dois(doi_results_path)
            NEG = MatchComparator._load_neg_dois(os.path.join(args.check_doi_dir, file_base_name + '_unmatched_dois.csv'))
            PRED = MatchComparator._load_predicted_dois(matches_path)

            # Salva i TP dettagliati (filtered_matches)
            if POS:
                filtered_matches = []
                doi_to_row = {}
                for row in MatchComparator._read_csv_rows(doi_results_path):
                    d = _norm_doi(row.get('DOI') or row.get('doi') or '')
                    if d:
                        doi_to_row[d] = row
                for row in MatchComparator._read_csv_rows(matches_path):
                    d = _norm_doi(row.get('matched_doi') or row.get('doi') or '')
                    if d and d in doi_to_row:
                        combined = {**doi_to_row[d], **row}
                        filtered_matches.append(combined)
                if filtered_matches:
                    out_path = os.path.join(filtered_matches_dir, file_base_name + '_filtered_matches.csv')
                    with open(out_path, 'w', encoding='utf-8', newline='') as f_out:
                        writer = csv.DictWriter(f_out, fieldnames=filtered_matches[0].keys())
                        writer.writeheader()
                        writer.writerows(filtered_matches)

            TP = len(PRED & POS)
            FP = len(PRED & NEG)
            FN = len(POS - PRED)
            TN = len(NEG - PRED)
            # Predicted DOIs the ground truth has no opinion on.
            invisible_set = PRED - POS - NEG
            INVISIBLE = len(invisible_set)

            # Collect the full matcher rows for those predictions so they can be
            # inspected/verified (reference_id, matched_doi, score, title, ...).
            if invisible_set:
                for row in MatchComparator._read_csv_rows(matches_path):
                    d = _norm_doi(row.get('matched_doi') or row.get('doi') or '')
                    if d and d in invisible_set:
                        invisible_rows.append({'base': file_base_name, **row})

            total_TP += TP
            total_FP += FP
            total_FN += FN
            total_TN += TN
            total_invisible += INVISIBLE
            bases_seen += 1

            print(f"[metrics] base={file_base_name} POS={len(POS)} NEG={len(NEG)} PRED={len(PRED)} -> "
                  f"TP={TP} FP={FP} FN={FN} TN={TN} INVISIBLE={INVISIBLE} (using {os.path.basename(matches_path)})")
            debug_rows.append({
                'base': file_base_name, 'POS': len(POS), 'NEG': len(NEG), 'PRED': len(PRED),
                'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN, 'INVISIBLE': INVISIBLE,
                'matches_file': os.path.basename(matches_path)
            })

        # metriche aggregate
        precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
        recall    = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
        f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        denom_acc = (total_TP + total_FP + total_FN + total_TN)
        accuracy  = (total_TP + total_TN) / denom_acc if denom_acc > 0 else 0.0

        # Worst-case precision: if EVERY unverifiable prediction were actually wrong.
        # Real precision lies between this and the reported precision. A big gap
        # means the headline precision is largely unproven, not confirmed.
        denom_worst = total_TP + total_FP + total_invisible
        precision_worst = total_TP / denom_worst if denom_worst > 0 else 0.0

        metrics = {
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100,
            'accuracy': accuracy * 100,
            'TP': total_TP, 'FP': total_FP, 'FN': total_FN, 'TN': total_TN,
            'invisible': total_invisible,
            'precision_worst': precision_worst * 100,
        }

        # CSV finale
        metrics_file = os.path.join(args.output_dir, "overall_evaluation_metrics.csv")
        with open(metrics_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['True Positives', metrics['TP']])
            writer.writerow(['False Positives', metrics['FP']])
            writer.writerow(['False Negatives', metrics['FN']])
            writer.writerow(['True Negatives', metrics['TN']])
            writer.writerow(['Unverifiable predictions (not in ground truth)', metrics['invisible']])
            writer.writerow(['Precision', f"{metrics['precision']:.2f}%"])
            writer.writerow(['Precision (worst case, all unverifiable wrong)', f"{metrics['precision_worst']:.2f}%"])
            writer.writerow(['Recall', f"{metrics['recall']:.2f}%"])
            writer.writerow(['F1 Score', f"{metrics['f1_score']:.2f}%"])
            writer.writerow(['Accuracy', f"{metrics['accuracy']:.2f}%"])

        # List every unverifiable prediction so they can be inspected / verified.
        if invisible_rows:
            inv_path = os.path.join(args.output_dir, "unverifiable_predictions.csv")
            # Union of keys keeps whatever columns the matches CSV provided.
            fieldnames = ['base']
            for r in invisible_rows:
                for k in r:
                    if k not in fieldnames:
                        fieldnames.append(k)
            with open(inv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(invisible_rows)
            print(f"Unverifiable predictions ({len(invisible_rows)}) listed in: {inv_path}")

        #debug per-base
        if debug_rows:
            dbg = os.path.join(args.output_dir, "metrics_debug_per_base.csv")
            with open(dbg, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['base','POS','NEG','PRED','TP','FP','FN','TN','INVISIBLE','matches_file'])
                writer.writeheader()
                writer.writerows(debug_rows)

        print(f"Processed bases: {bases_seen}")
        print(f"Filtered match files saved in: {filtered_matches_dir}")
        print(f"Unverifiable (invisible) predictions total: {total_invisible}")

    elif args.action == 'per_ref_metrics':
        # (a) PER-REFERENCE CORRECTNESS. Uses Crossref's own asserted reference
        # DOIs as a *silver* oracle. Intended to be run against a --no-doi matches
        # directory so it measures the NON-DOI matching logic (author/title/etc.):
        # for each reference that has a Crossref DOI, did the tool -- which was NOT
        # allowed to use that DOI -- independently predict the SAME DOI?
        #   predicted == oracle -> correct        (TP)
        #   predicted != oracle -> WRONG match    (FP)   <-- finally measurable
        #   not matched         -> miss           (FN, a recall floor)
        # This replaces the structural precision of `metrics` with a real one.
        crossref_dir = args.crossref_dir or args.input_path
        if not crossref_dir or not os.path.isdir(crossref_dir):
            parser.error("per_ref_metrics needs --crossref_dir (Crossref JSON dir)")
        if not args.matches_dir:
            parser.error("per_ref_metrics needs --matches_dir (ideally a --no-doi run)")

        print("Computing per-reference correctness (silver oracle = Crossref DOIs)...")
        TP = FP = FN = matched_no_doi = oracle_total = 0
        fp_rows = []
        per_base = []

        for jf in sorted(os.listdir(crossref_dir)):
            if not jf.lower().endswith('.json'):
                continue
            base = os.path.splitext(jf)[0]
            oracle = _load_crossref_oracle(os.path.join(crossref_dir, jf))
            if not oracle:
                continue
            mpath = _find_matches_file(args.matches_dir, base)
            matches = _load_matches_by_refid(mpath) if mpath else {}

            b_tp = b_fp = b_fn = b_nodoi = 0
            for idx, true_doi in oracle.items():
                oracle_total += 1
                rid = f"ref_{idx + 1}"
                row = matches.get(rid)
                if row is None:
                    b_fn += 1; FN += 1
                    continue
                pred = _norm_doi(row.get('matched_doi') or row.get('doi') or '')
                if not pred:
                    b_nodoi += 1; matched_no_doi += 1          # matched, but record had no DOI
                elif pred == true_doi:
                    b_tp += 1; TP += 1
                else:
                    b_fp += 1; FP += 1
                    fp_rows.append({
                        'base': base, 'reference_id': rid,
                        'predicted_doi': pred, 'crossref_doi': true_doi,
                        'score': row.get('score', ''),
                        'query_type': row.get('query_type', ''),
                        'reference_title': row.get('article_title', ''),
                        'matched_title': row.get('matched_title', ''),
                    })
            per_base.append({
                'base': base, 'oracle_refs': len(oracle),
                'TP': b_tp, 'FP': b_fp, 'FN': b_fn, 'matched_no_doi': b_nodoi,
                'matches_file': os.path.basename(mpath) if mpath else '(none)',
            })

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0          # real correctness
        resolution_rate = TP / oracle_total if oracle_total > 0 else 0.0  # recall floor

        metrics_path = os.path.join(args.output_dir, "per_reference_metrics.csv")
        with open(metrics_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['Metric', 'Value'])
            w.writerow(['References with a Crossref DOI (oracle)', oracle_total])
            w.writerow(['Correct matches (TP: predicted == Crossref DOI)', TP])
            w.writerow(['Wrong matches (FP: predicted != Crossref DOI)', FP])
            w.writerow(['Not matched (FN)', FN])
            w.writerow(['Matched but record had no DOI (excluded from precision)', matched_no_doi])
            w.writerow(['Per-reference precision (silver)', f"{precision * 100:.2f}%"])
            w.writerow(['Correct-resolution rate (recall floor)', f"{resolution_rate * 100:.2f}%"])

        if fp_rows:
            fp_path = os.path.join(args.output_dir, "per_reference_false_positives.csv")
            fields = ['base', 'reference_id', 'predicted_doi', 'crossref_doi',
                      'score', 'query_type', 'reference_title', 'matched_title']
            with open(fp_path, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader(); w.writerows(fp_rows)

        if per_base:
            pb_path = os.path.join(args.output_dir, "per_reference_debug_per_base.csv")
            with open(pb_path, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=['base', 'oracle_refs', 'TP', 'FP',
                                                  'FN', 'matched_no_doi', 'matches_file'])
                w.writeheader(); w.writerows(per_base)

        print(f"  Oracle references (with Crossref DOI): {oracle_total}")
        print(f"  TP={TP} FP={FP} FN={FN} matched_no_doi={matched_no_doi}")
        print(f"  Per-reference precision (silver): {precision * 100:.2f}%")
        print(f"  Correct-resolution rate (recall floor): {resolution_rate * 100:.2f}%")
        print(f"  Wrong matches to hand-check: per_reference_false_positives.csv ({len(fp_rows)})")
        print("  NOTE: silver oracle = Crossref's own (imperfect) DOIs; hand-check "
              "the FP file to separate real tool errors from oracle errors.")

    elif args.action == 'check_presence':
        # (b) INDEPENDENT PRESENCE CHECK. For references the matcher did NOT match,
        # ask Meta -- via a robust title search, independent of the matcher's
        # brittle exact-string cascade -- whether a plausibly-matching entity
        # exists. This separates "present in Meta but the matcher missed it"
        # (false negatives) from "genuinely absent" (ingestion candidates).
        try:
            from rapidfuzz import fuzz
        except ImportError:  # pragma: no cover
            from fuzzywuzzy import fuzz

        unmatched_dir = args.unmatched_dir or args.input_path
        if not unmatched_dir or not os.path.isdir(unmatched_dir):
            parser.error("check_presence needs --unmatched_dir (dir of *_unmatched.csv)")

        probe = SPARQLWrapper(args.endpoint)
        probe.setReturnFormat(JSON)
        probe.setTimeout(25)  # cap each query; unindexed title REGEX is slow

        print(f"Checking Meta presence of unmatched references via title search "
              f"(endpoint={args.endpoint})...")
        print("NOTE: the title REGEX is UNINDEXED and slow on the public endpoint "
              "(~seconds to tens of seconds each). Use a small --limit; for the full "
              "set a LOCAL Meta dump is the realistic approach.")

        checked = likely_present = not_found = unknown = skipped = 0
        rows = []
        for uf in sorted(os.listdir(unmatched_dir)):
            if not uf.lower().endswith('_unmatched.csv'):
                continue
            base = uf[:-len('_unmatched.csv')]
            for r in MatchComparator._read_csv_rows(os.path.join(unmatched_dir, uf)):
                if args.limit and checked >= args.limit:
                    break
                title = (r.get('article_title') or r.get('volume_title')
                         or r.get('journal_title') or '').strip()
                words = [w for w in re.sub(r'[^a-z0-9\s]', ' ', title.lower()).split()
                         if len(w) > 3][:4]
                if not words:
                    skipped += 1
                    continue
                pattern = '.*'.join(re.escape(w) for w in words)
                query = (
                    'PREFIX dcterms: <http://purl.org/dc/terms/>\n'
                    'SELECT DISTINCT ?br ?title WHERE {\n'
                    '  ?br dcterms:title ?title .\n'
                    f'  FILTER(REGEX(?title, "{pattern}", "i"))\n'
                    '} LIMIT 10'
                )
                probe.setQuery(query)
                best = 0.0
                cand_id = cand_title = ''
                query_ok = True
                try:
                    res = probe.query().convert()
                    for b in res.get('results', {}).get('bindings', []):
                        t = b.get('title', {}).get('value', '')
                        s = fuzz.token_set_ratio(title.lower(), t.lower())
                        if s > best:
                            best = s; cand_title = t; cand_id = b.get('br', {}).get('value', '')
                except Exception as e:
                    query_ok = False  # timeout / endpoint error -> "unknown", not "absent"
                    print(f"  [warn] query failed for {base}/{r.get('reference_id','?')}: {e}")

                checked += 1
                if not query_ok:
                    status = 'unknown'; unknown += 1
                elif best >= 85:
                    status = 'yes'; likely_present += 1
                else:
                    status = 'no'; not_found += 1
                rows.append({
                    'base': base,
                    'reference_id': r.get('reference_id', ''),
                    'article_title': title[:150],
                    'year': r.get('year', ''),
                    'first_author_lastname': r.get('first_author_lastname', ''),
                    'in_meta_likely': status,
                    'best_title_similarity': round(best, 1),
                    'candidate_meta_id': cand_id,
                    'candidate_title': cand_title[:150],
                })
                time.sleep(0.4)  # ~2.5 req/s, polite to the endpoint

        out_path = os.path.join(args.output_dir, "presence_check.csv")
        if rows:
            with open(out_path, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)

        print(f"  Checked: {checked} (skipped {skipped} with no usable title)")
        print(f"  Likely IN Meta (matcher false negatives): {likely_present}")
        print(f"  NOT found (likely absent -> ingestion candidates): {not_found}")
        print(f"  Unknown (query timed out/failed): {unknown}")
        print(f"  Details: {out_path}")


if __name__ == "__main__":
    main()
