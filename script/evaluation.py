import os
import csv
import json
from typing import Dict, List, Tuple
from SPARQLWrapper import SPARQLWrapper, JSON
from collections import defaultdict
import time
import argparse

class OpenCitationsDOIMatcher:
    def __init__(self, endpoint_url="https://opencitations.net/meta/sparql"):
        self.sparql = SPARQLWrapper(endpoint_url)
        self.sparql.setReturnFormat(JSON)

    def extract_dois_from_json(self, json_data: Dict) -> List[Tuple[str, str]]:
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

    def execute_sparql_query(self, query: str, max_retries=3, retry_delay=5) -> Dict:
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
        with open(json_file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)

        dois = self.extract_dois_from_json(json_data)
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

class MatchComparator:
    @staticmethod
    def compare_dois(file1_path: str, file2_path: str) -> Tuple[List, List]:
        
        dois_file1 = {}
        filename = os.path.basename(file1_path)
        basename = filename.replace('_doi_results.csv', '')
        
        with open(file1_path, 'r', encoding='utf-8') as f1:
            reader = csv.DictReader(f1)
            for row in reader:
                if not row.get('DOI'):  
                    continue
                dois_file1[row['DOI']] = (row['reference_id'], basename)

        dois_file2 = {}
        with open(file2_path, 'r', encoding='utf-8') as f2:
            reader = csv.DictReader(f2)
            for row in reader:
                if row.get('matched_doi'):
                    dois_file2[row['matched_doi']] = (row['reference_id'], basename)

        
        missed_matches = [(dois_file1[doi][0], doi, dois_file1[doi][1]) for doi in (set(dois_file1) - set(dois_file2))]
        earned_matches = [(dois_file2[doi][0], doi, dois_file2[doi][1]) for doi in (set(dois_file2) - set(dois_file1))]
        
        return missed_matches, earned_matches

    @staticmethod
    def calculate_overall_metrics(check_doi_dir: str, matches_dir: str, output_dir: str = "filtered_matches") -> Dict[str, float]:
        total_TP, total_FP, total_FN = 0, 0, 0  
        
        
        os.makedirs(output_dir, exist_ok=True)

        
        for check_doi_file in os.listdir(check_doi_dir):
            if check_doi_file.endswith('_doi_results.csv'):
                file_base_name = check_doi_file.replace('_doi_results.csv', '')

                
                matches_file = file_base_name + '_matches.csv'
                matches_path = os.path.join(matches_dir, matches_file)

                
                if not os.path.exists(matches_path):
                    continue

                
                doi_from_opencitations = {}
                with open(os.path.join(check_doi_dir, check_doi_file), 'r', encoding='utf-8') as f1:
                    reader = csv.DictReader(f1)
                    for row in reader:
                        if row.get('DOI'):
                            doi_from_opencitations[row['DOI']] = row  

                
                filtered_matches = []
                with open(matches_path, 'r', encoding='utf-8') as f2:
                    reader = csv.DictReader(f2)
                    match_headers = reader.fieldnames  
                    for row in reader:
                        if row.get('matched_doi') and row['matched_doi'] in doi_from_opencitations:
                            combined_data = {**doi_from_opencitations[row['matched_doi']], **row}  
                            filtered_matches.append(combined_data)

                
                if filtered_matches:
                    output_file_path = os.path.join(output_dir, file_base_name + '_filtered_matches.csv')
                    with open(output_file_path, 'w', encoding='utf-8', newline='') as f_out:
                        writer = csv.DictWriter(f_out, fieldnames=filtered_matches[0].keys())
                        writer.writeheader()
                        writer.writerows(filtered_matches)

                
                TP = len(filtered_matches)
                FP = len(filtered_matches) - len(doi_from_opencitations)
                FN = len(doi_from_opencitations) - len(filtered_matches)

                
                total_TP += TP
                total_FP += max(FP, 0)
                total_FN += max(FN, 0)

        
        precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
        recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100,
            'TP': total_TP,
            'FP': total_FP,
            'FN': total_FN
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
            if filename.endswith('.json'):
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
            if check_doi_file.endswith('_doi_results.csv'):
                base_name = check_doi_file.replace('_doi_results.csv', '')
                matches_file = f"{base_name}_matches.csv"
                
                file1_path = os.path.join(args.check_doi_dir, check_doi_file)
                file2_path = os.path.join(args.matches_dir, matches_file)
                
                if os.path.exists(file2_path):
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

        metrics = comparator.calculate_overall_metrics(
            check_doi_dir=args.check_doi_dir,
            matches_dir=args.matches_dir,
            output_dir=filtered_matches_dir  
        )
        
        metrics_file = os.path.join(args.output_dir, "overall_evaluation_metrics.csv")
        with open(metrics_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['True Positives', metrics['TP']])
            writer.writerow(['False Positives', metrics['FP']])
            writer.writerow(['False Negatives', metrics['FN']])
            writer.writerow(['Precision', f"{metrics['precision']:.2f}%"])
            writer.writerow(['Recall', f"{metrics['recall']:.2f}%"])
            writer.writerow(['F1 Score', f"{metrics['f1_score']:.2f}%"])

        print("Overall metrics calculation complete")
        print(f"Filtered match files saved in: {filtered_matches_dir}")


if __name__ == "__main__":
    main()
