import os
import csv
import json
from typing import Dict, List, Tuple, Optional, Set
from SPARQLWrapper import SPARQLWrapper, JSON
from collections import defaultdict
import time
import argparse

class OpenCitationsDOIMatcher:
    def __init__(self, endpoint_url="https://opencitations.net/meta/sparql"):
        # def __init__(self, endpoint_url="https://sparql.opencitations.net/meta"):
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
        return f"""PREFIX datacite: <http://purl.org/spar/datacite/>
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX literal: <http://www.essepuntato.it/2010/06/literalreification/>
PREFIX prism: <http://prismstandard.org/namespaces/basic/2.0/>

SELECT ?predicate ?object {{
    ?identifier literal:hasLiteralValue "{doi}".
    ?br datacite:hasIdentifier ?identifier;
        ?predicate ?object.
    BIND(STR(?publicationDate) AS ?pub_date)
    BIND((CONCAT("doi:", "{doi}")) AS ?id)
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
        """Normalizza il DOI per confronti robusti."""
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

    @staticmethod
    def calculate_overall_metrics(check_doi_dir: str, matches_dir: str, output_dir: str = "filtered_matches") -> Dict[str, float]:
        """
        Calcola TP/FP/FN/TN a livello DOI per ogni <base> e aggrega:
          TP = |PRED ∩ POS|
          FP = |PRED ∩ NEG|
          FN = |POS - PRED|
          TN = |NEG - PRED|
        """
        total_TP = total_FP = total_FN = total_TN = 0
        os.makedirs(output_dir, exist_ok=True)

        for check_doi_file in os.listdir(check_doi_dir):
            if not check_doi_file.lower().endswith('_doi_results.csv'):
                continue

            base = check_doi_file[:-len('_doi_results.csv')]
            doi_results_path = os.path.join(check_doi_dir, check_doi_file)

            # trova il file dei match in modo flessibile
            matches_path = _find_matches_file(matches_dir, base)
            if not matches_path:
                # nessun matches, computiamo comunque FN/TN con PRED vuoto
                POS = MatchComparator._load_pos_dois(doi_results_path)
                NEG = MatchComparator._load_neg_dois(os.path.join(check_doi_dir, base + '_unmatched_dois.csv'))
                PRED = set()
                total_TP += 0
                total_FP += 0
                total_FN += len(POS - PRED)
                total_TN += len(NEG - PRED)
                continue

            POS = MatchComparator._load_pos_dois(doi_results_path)
            NEG = MatchComparator._load_neg_dois(os.path.join(check_doi_dir, base + '_unmatched_dois.csv'))
            PRED = MatchComparator._load_predicted_dois(matches_path)

            # salva TP dettagliati (filtered_matches)
            if POS:
                filtered = []
                # mappa doi -> riga dai doi_results
                doi_to_row = {}
                for r in MatchComparator._read_csv_rows(doi_results_path):
                    d = MatchComparator._norm_doi(r.get('DOI') or r.get('doi') or '')
                    if d:
                        doi_to_row[d] = r
                for r in MatchComparator._read_csv_rows(matches_path):
                    d = MatchComparator._norm_doi(r.get('matched_doi') or r.get('doi') or '')
                    if d and d in doi_to_row:
                        filtered.append({**doi_to_row[d], **r})
                if filtered:
                    outp = os.path.join(output_dir, base + '_filtered_matches.csv')
                    with open(outp, 'w', encoding='utf-8', newline='') as f_out:
                        writer = csv.DictWriter(f_out, fieldnames=filtered[0].keys())
                        writer.writeheader()
                        writer.writerows(filtered)

            TP = len(PRED & POS)
            FP = len(PRED & NEG)
            FN = len(POS - PRED)
            TN = len(NEG - PRED)

            total_TP += TP
            total_FP += FP
            total_FN += FN
            total_TN += TN

        precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
        recall    = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
        f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        denom_acc = (total_TP + total_FP + total_FN + total_TN)
        accuracy  = (total_TP + total_TN) / denom_acc if denom_acc > 0 else 0.0

        return {
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100,
            'accuracy': accuracy * 100,
            'TP': total_TP,
            'FP': total_FP,
            'FN': total_FN,
            'TN': total_TN
        }


def main():
    parser = argparse.ArgumentParser(description='Process references and compare matches')
    parser.add_argument('action', choices=['check_doi', 'compare', 'metrics'],
                        help='Action to perform')
    parser.add_argument('input_path', help='Input directory or file')
    parser.add_argument('--output_dir', help='Output directory', default='.')
    parser.add_argument('--check_doi_dir', help='Check DOI results directory')
    parser.add_argument('--matches_dir', help='Matches directory')

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
                    'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN, 'matches_file': '(none)'
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

            total_TP += TP
            total_FP += FP
            total_FN += FN
            total_TN += TN
            bases_seen += 1

            print(f"[metrics] base={file_base_name} POS={len(POS)} NEG={len(NEG)} PRED={len(PRED)} -> "
                  f"TP={TP} FP={FP} FN={FN} TN={TN} (using {os.path.basename(matches_path)})")
            debug_rows.append({
                'base': file_base_name, 'POS': len(POS), 'NEG': len(NEG), 'PRED': len(PRED),
                'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN, 'matches_file': os.path.basename(matches_path)
            })

        # metriche aggregate
        precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
        recall    = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
        f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        denom_acc = (total_TP + total_FP + total_FN + total_TN)
        accuracy  = (total_TP + total_TN) / denom_acc if denom_acc > 0 else 0.0

        metrics = {
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100,
            'accuracy': accuracy * 100,
            'TP': total_TP, 'FP': total_FP, 'FN': total_FN, 'TN': total_TN
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
            writer.writerow(['Precision', f"{metrics['precision']:.2f}%"])
            writer.writerow(['Recall', f"{metrics['recall']:.2f}%"])
            writer.writerow(['F1 Score', f"{metrics['f1_score']:.2f}%"])
            writer.writerow(['Accuracy', f"{metrics['accuracy']:.2f}%"])

        #debug per-base
        if debug_rows:
            dbg = os.path.join(args.output_dir, "metrics_debug_per_base.csv")
            with open(dbg, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['base','POS','NEG','PRED','TP','FP','FN','TN','matches_file'])
                writer.writeheader()
                writer.writerows(debug_rows)

        print(f"Processed bases: {bases_seen}")
        print(f"Filtered match files saved in: {filtered_matches_dir}")


if __name__ == "__main__":
    main()
