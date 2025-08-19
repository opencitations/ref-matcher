import json
import csv
from typing import Dict, List, Optional
from dataclasses import dataclass
from SPARQLWrapper import SPARQLWrapper, JSON
from fuzzywuzzy import fuzz
import xml.etree.ElementTree as ET
from grobid_client.grobid_client import GrobidClient
import tempfile
import os
import argparse
from unidecode import unidecode
import re
import time
from random import uniform
import pickle
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Set, List

@dataclass
class Reference:
    """Class to store reference metadata"""
    year: str = ""
    volume: str = ""
    first_page: str = ""
    first_author_lastname: str = ""
    article_title: str = "" 
    volume_title: str = ""  
    journal_title: str = ""  
    doi: str = ""
    unstructured: str = ""
    
    def get_main_title(self) -> str:
        return self.article_title or self.volume_title or self.journal_title or ""

class GrobidProcessor:
    """Class to handle Grobid processing of unstructured references"""
    def __init__(self, config_path: str = "/home/chiara/TIROCINIO/tirocinio_OC/config.json"):
        self.client = GrobidClient(config_path=config_path)
        
    def process_unstructured_reference(self, unstructured_text: str) -> Optional[Reference]:
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "input")
            output_path = os.path.join(temp_dir, "output")
            os.makedirs(input_path)
            os.makedirs(output_path)
            
            
            with open(os.path.join(input_path, "ref.txt"), "w") as f:
                f.write(unstructured_text)
            
            
            self.client.process("processCitationList", input_path, output=output_path, 
                              n=1, verbose=False)
            
            
            xml_file = os.path.join(output_path, "ref.tei.xml")
            if not os.path.exists(xml_file):
                return None
                
            return self._parse_tei_xml(xml_file)
    
    def _parse_tei_xml(self, xml_file: str) -> Optional[Reference]:
        """Parse TEI XML output from Grobid into Reference object"""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
        
        
        ref = Reference()
        
        
        date = root.find('.//tei:date', ns)
        if date is not None:
            ref.year = date.get('when', '')[:4]  
            
        
        author = root.find('.//tei:author', ns)
        if author is not None:
            surname = author.find('.//tei:surname', ns)
            if surname is not None:
                ref.first_author_lastname = surname.text
                
        
        title = root.find('.//tei:title', ns)
        if title is not None:
            ref.article_title = title.text
            
        
        biblScope = root.findall('.//tei:biblScope', ns)
        for scope in biblScope:
            unit = scope.get('unit', '')
            if unit == 'volume':
                ref.volume = scope.text
            elif unit == 'page':
                ref.first_page = scope.text.split('-')[0]  
                
        return ref

class OpenCitationsMatcher:
    """Class to handle queries in OpenCitations and matching"""
    
    def __init__(self, endpoint: str = "https://opencitations.net/meta/sparql", max_retries: int = 5):
        self.sparql = SPARQLWrapper(endpoint)
        self.sparql.setReturnFormat(JSON)
        self.sparql.setTimeout(30)  
        self.max_retries = max_retries  

    def query_opencitations(self, sparql_query: str) -> Optional[List[Dict]]:
        """Exectues a SPARQL with retry mechanism in case of error 503"""
        self.sparql.setQuery(sparql_query)

        for attempt in range(self.max_retries):
            try:
                results = self.sparql.query().convert()
                return results['results']['bindings']
            except Exception as e:
                error_message = str(e)
                
                if "503" in error_message:
                    wait_time = 2 ** attempt + uniform(0, 1)  
                    print(f"Errore 503: tentativo {attempt + 1}/{self.max_retries}. Riprovo tra {wait_time:.2f} secondi...")
                    time.sleep(wait_time)
                else:
                    print(f"Errore durante la query SPARQL: {e}")
                    break  
        
        print("Numero massimo di tentativi raggiunto. Impossibile completare la query.")
        return None

    def _extract_year(self, year_str: str) -> Optional[int]:
        if not year_str:
            return None

        months = {
            'january': '01', 'february': '02', 'march': '03', 'april': '04',
            'may': '05', 'june': '06', 'july': '07', 'august': '08',
            'september': '09', 'october': '10', 'november': '11', 'december': '12',
            'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 'jun': '06',
            'jul': '07', 'aug': '08', 'sep': '09', 'oct': '10', 'nov': '11',
            'dec': '12'
        }
        
        try:
            return int(year_str)
        except ValueError:
            year_str = year_str.lower().strip()
            
            import re
            year_match = re.search(r'\b(19|20)\d{2}\b', year_str)
            if year_match:
                return int(year_match.group())
                
            if year_str in months:
                return None
            
            parts = re.split(r'[-/\s]', year_str)
            for part in parts:
                if part.isdigit() and len(part) == 4:
                    year = int(part)
                    if 1700 <= year <= 2025: 
                        return year
                        
            return None
    
    def _normalize_title(self, title: str) -> str:
        """Normalizes title fore a more robus matching"""
        
        if not title:
            return ""
            
        
        text = title.lower()
        
        
        text = text.rstrip('.!?')
        
        
        replacements = {
            'ω': 'omega',
            'α': 'alpha',
            'β': 'beta',
            'γ': 'gamma',
            'δ': 'delta',
            '(1)h': '1h',  
            'ω3': 'omega3',
            'ω-3': 'omega-3',
            ''': "'",      
            ''': "'",
            '"': '"',
            '"': '"',
            '–': '-',      
            '—': '-',
            'l': 'i',      
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        
        text = re.sub(r'[^a-z0-9\s\-]', '', text)
        
        
        text = ' '.join(text.split())
        
        return text

    def build_sparql_query(self, reference: Reference, query_type: str) -> Optional[str]:
        """Builds the appropriate SPARQL query based on available metadata"""
        year_int = self._extract_year(reference.year)
        if not year_int:
            return self._build_alternative_query(reference, query_type)
        
        title = reference.get_main_title()
        
        if query_type == "year_and_doi" and reference.doi and reference.year:
            
            year_int = int(reference.year)
            return f"""
            PREFIX datacite: <http://purl.org/spar/datacite/>
            PREFIX dcterms: <http://purl.org/dc/terms/>
            PREFIX literal: <http://www.essepuntato.it/2010/06/literalreification/>
            PREFIX prism: <http://prismstandard.org/namespaces/basic/2.0/>
            SELECT DISTINCT ?br ?title ?pub_date ?doi {{
                ?identifier literal:hasLiteralValue "{reference.doi}".
                ?br datacite:hasIdentifier ?identifier;
                dcterms:title ?title;
                prism:publicationDate ?publicationDate;
                datacite:hasIdentifier ?doi_id.
                
                ?doi_id datacite:usesIdentifierScheme datacite:doi;
                        literal:hasLiteralValue ?doi.

                BIND(STR(?publicationDate) AS ?pub_date)
                FILTER(STRSTARTS(?pub_date, "{year_int}") || 
                    STRSTARTS(?pub_date, "{year_int-1}") || 
                    STRSTARTS(?pub_date, "{year_int+1}"))
            }}
            """

        elif query_type == "author_title" and title:
            return f"""
            PREFIX dcterms: <http://purl.org/dc/terms/>
            PREFIX pro: <http://purl.org/spar/pro/>
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            PREFIX datacite: <http://purl.org/spar/datacite/>
            PREFIX literal: <http://www.essepuntato.it/2010/06/literalreification/>
            PREFIX prism: <http://prismstandard.org/namespaces/basic/2.0/>
            SELECT DISTINCT ?br ?title ?pub_date ?first_author ?doi {{
                ?first_author foaf:familyName "{reference.first_author_lastname}".
                ?role pro:isHeldBy ?first_author.
                ?br pro:isDocumentContextFor ?role;
                dcterms:title ?title;
                prism:publicationDate ?publicationDate;
                datacite:hasIdentifier ?doi_id.

                ?doi_id datacite:usesIdentifierScheme datacite:doi;
                literal:hasLiteralValue ?doi.

                BIND(STR(?publicationDate) AS ?pub_date)
                FILTER(CONTAINS(LCASE(?title), LCASE("{title}")))
            }}
            """
        
        elif (query_type == "year_volume_page" and reference.year 
              and reference.volume and reference.first_page):
            year_int = int(reference.year)
            return f"""
            PREFIX datacite: <http://purl.org/spar/datacite/>
            PREFIX dcterms: <http://purl.org/dc/terms/>
            PREFIX literal: <http://www.essepuntato.it/2010/06/literalreification/>
            PREFIX prism: <http://prismstandard.org/namespaces/basic/2.0/>
            PREFIX frbr: <http://purl.org/vocab/frbr/core
            PREFIX fabio: <http://purl.org/spar/fabio/>
            SELECT DISTINCT ?br ?title ?pub_date ?volume_num ?start_page ?doi {{
                ?br dcterms:title ?title;
                    prism:publicationDate ?publicationDate;
                    frbr:embodiment ?embodiment;
                    frbr:partOf ?issue;
                    datacite:hasIdentifier ?doi_id.

                ?doi_id datacite:usesIdentifierScheme datacite:doi;
                        literal:hasLiteralValue ?doi.

                BIND(STR(?publicationDate) AS ?pub_date)
                FILTER(STRSTARTS(?pub_date, "{year_int}") || 
                    STRSTARTS(?pub_date, "{year_int-1}") || 
                    STRSTARTS(?pub_date, "{year_int+1}"))
                ?embodiment prism:startingPage "{reference.first_page}".
                ?issue frbr:partOf ?volume.
                ?volume fabio:hasSequenceIdentifier "{reference.volume}".
                BIND(STR(?volume) AS ?volume_num)
            }}
            """
        elif (query_type == "year_author_page" and reference.year 
              and reference.first_author_lastname and reference.first_page):
            year_int = int(reference.year)
            return f"""
            PREFIX datacite: <http://purl.org/spar/datacite/>
            PREFIX dcterms: <http://purl.org/dc/terms/>
            PREFIX literal: <http://www.essepuntato.it/2010/06/literalreification/>
            PREFIX prism: <http://prismstandard.org/namespaces/basic/2.0/>
            PREFIX frbr: <http://purl.org/vocab/frbr/core
            PREFIX pro: <http://purl.org/spar/pro/>
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            SELECT DISTINCT ?br ?title ?pub_date ?first_author ?start_page ?doi {{
                ?first_author foaf:familyName "{reference.first_author_lastname}".
                ?role pro:isHeldBy ?first_author.
                ?br pro:isDocumentContextFor ?role;
                    dcterms:title ?title;
                    prism:publicationDate ?publicationDate;
                    frbr:embodiment ?embodiment;
                    datacite:hasIdentifier ?doi_id.

                ?doi_id datacite:usesIdentifierScheme datacite:doi;
                        literal:hasLiteralValue ?doi.

                BIND(STR(?publicationDate) AS ?pub_date)
                FILTER(STRSTARTS(?pub_date, "{year_int}") || 
                    STRSTARTS(?pub_date, "{year_int-1}") || 
                    STRSTARTS(?pub_date, "{year_int+1}"))
                ?embodiment prism:startingPage "{reference.first_page}".
            }}
            """
        return None
    
    def _build_alternative_query(self, reference: Reference, query_type: str) -> Optional[str]:
        """Builds fallback query when the year is not available"""
        if reference.doi:
            return f"""
            PREFIX datacite: <http://purl.org/spar/datacite/>
            PREFIX dcterms: <http://purl.org/dc/terms/>
            PREFIX literal: <http://www.essepuntato.it/2010/06/literalreification/>
            PREFIX prism: <http://prismstandard.org/namespaces/basic/2.0/>
            SELECT DISTINCT ?br ?title ?pub_date ?doi {{
                ?identifier literal:hasLiteralValue "{reference.doi}".
                ?br datacite:hasIdentifier ?identifier;
                dcterms:title ?title;
                prism:publicationDate ?publicationDate;
                datacite:hasIdentifier ?doi_id.
                
                ?doi_id datacite:usesIdentifierScheme datacite:doi;
                        literal:hasLiteralValue ?doi.

                BIND(STR(?publicationDate) AS ?pub_date)
            }}
            """
        return None

    def calculate_matching_score(self, reference: Reference, result: Dict) -> int:
        """Calculates matching score with flexible criteria"""
        score = 0
        
        
        if reference.year and 'pub_date' in result:
            record_year = int(reference.year)
            result_year = int(result['pub_date']['value'][:4])
            if record_year == result_year:
                score += 20
            elif abs(record_year - result_year) == 1:
                score += 15
        
        
        if 'title' in result:
            result_title = self._normalize_title(result['title']['value'])
            
            
            titles_to_check = [
                reference.article_title,
                reference.volume_title,
                reference.journal_title
            ]
            
            best_title_score = 0
            for title in titles_to_check:
                if title:
                    record_title = self._normalize_title(title)
                    
                    
                    exact_match = 100 if record_title == result_title else 0
                    ratio = fuzz.ratio(record_title, result_title)
                    partial_ratio = fuzz.partial_ratio(record_title, result_title)
                    token_sort_ratio = fuzz.token_sort_ratio(record_title, result_title)
                    token_set_ratio = fuzz.token_set_ratio(record_title, result_title)
                    
                    
                    title_score = max(exact_match, ratio, partial_ratio, 
                                    token_sort_ratio, token_set_ratio)
                    best_title_score = max(best_title_score, title_score)
            
            
            if best_title_score == 100:
                score += 50
            elif best_title_score > 95:
                score += 45
            elif best_title_score > 90:
                score += 40
            elif best_title_score > 85:
                score += 35
            elif best_title_score > 80:
                score += 30
            elif best_title_score > 75:
                score += 25
        
        
        if reference.volume and 'volume_num' in result:
            if reference.volume == result['volume_num']['value']:
                score += 15
        
        if reference.first_page and 'start_page' in result:
            ref_page = reference.first_page.lstrip('0')
            result_page = result['start_page']['value'].lstrip('0')
            if ref_page == result_page:
                score += 15
        
        return score
    
class ReferenceProcessor:
    """Main class to coordinate reference processing and matching"""
    def __init__(self, use_grobid: bool = False):
        self.matcher = OpenCitationsMatcher()
        self.use_grobid = use_grobid
        
    def process_reference(self, ref: Reference, threshold: int = 50, use_doi: bool = False) -> Optional[Dict]:
        """Process a reference with optional DOI matching"""
        
        processed_ref = Reference(**{k: v for k, v in ref.__dict__.items()})
        
        if not use_doi:
            
            processed_ref.doi = ''
        
        best_score = 0
        best_match = None
        
        
        query_types = [
            "year_and_doi",
            "year_volume_page",
            "year_author_page",
            "author_title" 
        ]
        
        for query_type in query_types:
            query = self.matcher.build_sparql_query(processed_ref, query_type)
            if query:
                results = self.matcher.query_opencitations(query)
                if results:
                    for result in results:
                        score = self.matcher.calculate_matching_score(processed_ref, result)
                        if score > best_score:
                            best_score = score
                            best_match = result
                            best_match['score'] = score
                            best_match['query_type'] = query_type
        
        
        adjusted_threshold = threshold * 0.9 if best_score > (threshold * 0.8) else threshold
        
        if best_score >= adjusted_threshold:
            return best_match
        return None

    def _merge_reference_data(self, original_ref: Reference, grobid_ref: Reference):
        """Combines data from Grobid with existing metadata, mantaining original metadata if available"""
        if not original_ref.doi:
            original_ref.doi = grobid_ref.doi
        if not original_ref.year:
            original_ref.year = grobid_ref.year
        if not original_ref.first_author_lastname:
            original_ref.first_author_lastname = grobid_ref.first_author_lastname
        if not original_ref.article_title:
            original_ref.article_title = grobid_ref.article_title
        if not original_ref.volume:
            original_ref.volume = grobid_ref.volume
        if not original_ref.first_page:
            original_ref.first_page = grobid_ref.first_page
    
    def process_file(self, input_file: str, output_file: str, threshold: int = 50):
        """Process input file based on its extension"""
        _, extension = os.path.splitext(input_file)
        
        if extension.lower() == '.json':
            self._process_crossref_file(input_file, output_file, threshold)
        elif extension.lower() == '.xml':
            self._process_tei_file(input_file, output_file, threshold)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
    def _process_crossref_file(self, input_file: str, output_file: str, threshold: int):
        """Process Crossref JSON file"""
        stats = {'total_references': 0, 'matches_found': 0}
        
        with open(input_file, 'r', encoding='utf-8', errors='replace') as f, \
            open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            data = json.load(f)
            writer = csv.DictWriter(csvfile, 
                                fieldnames=['reference_id', 'article_title', 
                                            'matched_title', 'score', 'matched_doi',
                                            'meta_id', 'query_type'])
            writer.writeheader()
            
            if 'message' in data and 'reference' in data['message']:
                for i, ref_data in enumerate(data['message']['reference'], 1):
                    stats['total_references'] += 1
                    
                    
                    ref = Reference(
                        year=ref_data.get('year', ''),
                        volume=ref_data.get('volume', ''),
                        first_page=ref_data.get('first-page', ''),
                        first_author_lastname=ref_data.get('author', '').split()[-1] 
                            if ref_data.get('author') else '',
                        article_title=ref_data.get('article-title', ''),
                        volume_title=ref_data.get('volume-title', ''),
                        journal_title=ref_data.get('journal-title', ''),
                        doi=ref_data.get('DOI', '').lower(),
                        unstructured=ref_data.get('unstructured', '')  
                    )
                    
                    match = self.process_reference(ref, threshold)
                    if match:
                        stats['matches_found'] += 1
                        writer.writerow({
                            'reference_id': f"ref_{i}",
                            'article_title': ref.get_main_title(),
                            'matched_title': match['title']['value'],
                            'score': match.get('score', 0),
                            'matched_doi': match['doi']['value'] if 'doi' in match else 'N/A',
                            'meta_id': match['br']['value'] if 'br' in match else 'N/A',
                            'query_type': match.get('query_type', 'N/A')
                        })
            
            self._print_stats(stats)
            stats_file = os.path.splitext(output_file)[0] + '_stats.txt'
            self._print_stats(stats, stats_file)

    def _process_tei_file(self, input_file: str, output_file: str, threshold: int):
        """Process TEI XML file"""
        stats = {'total_references': 0, 'matches_found': 0}
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
        
        tree = ET.parse(input_file)
        root = tree.getroot()
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, 
                                fieldnames=['reference_id', 'article_title', 
                                            'matched_title', 'score', 'matched_doi',
                                            'meta_id', 'query_type'])
            writer.writeheader()
            
            for bibl in root.findall('.//tei:biblStruct', ns):
                stats['total_references'] += 1
                ref = self._parse_bibl_struct(bibl, ns)
                
                if ref:
                    match = self.process_reference(ref, threshold)
                    if match:
                        stats['matches_found'] += 1
                        writer.writerow({
                            'reference_id': bibl.get('{http://www.w3.org/XML/1998/namespace}id', 'unknown'),
                            'article_title': ref.article_title or 'N/A',
                            'matched_title': match['title']['value'],
                            'score': match.get('score', 0),
                            'matched_doi': match['doi']['value'] if 'doi' in match else 'N/A',
                            'meta_id': match['br']['value'] if 'br' in match else 'N/A',
                            'query_type': match.get('query_type', 'N/A')
                        })
        
        self._print_stats(stats)
        stats_file = os.path.splitext(output_file)[0] + '_stats.txt'
        self._print_stats(stats, stats_file)
            
    @staticmethod
    def _parse_bibl_struct(bibl: ET.Element, ns: Dict) -> Optional[Reference]:
        """Parse a single biblStruct element into a Reference object"""
        ref = Reference()
        
        
        date = bibl.find('.//tei:date[@when]', ns)
        if date is not None:
            ref.year = date.get('when', '')[:4]
        
        
        authors = bibl.findall('.//tei:author/tei:persName/tei:surname', ns)
        if authors:
            ref.first_author_lastname = authors[0].text
            
        
        
        analytic = bibl.find('tei:analytic', ns)
        if analytic is not None:
            title = analytic.find('tei:title', ns)
            if title is not None:
                ref.article_title = title.text
        else:
            
            monogr = bibl.find('tei:monogr', ns)
            if monogr is not None:
                
                m_title = monogr.find('tei:title[@level="m"]', ns)
                if m_title is not None:
                    ref.volume_title = m_title.text
                
                
                j_title = monogr.find('tei:title[@level="j"]', ns)
                if j_title is not None:
                    ref.journal_title = j_title.text
        
        
        volume = bibl.find('.//tei:biblScope[@unit="volume"]', ns)
        if volume is not None:
            ref.volume = volume.text
            
        
        page = bibl.find('.//tei:biblScope[@unit="page"]', ns)
        if page is not None:
            if page.get('from'):
                ref.first_page = page.get('from')
            else:
                ref.first_page = page.text.split('-')[0].strip() if page.text else ''
                
        return ref

    @staticmethod
    def _print_stats(stats: Dict, output_file: str = None):
        """Print processing statistics and optionally save to file"""
        total = stats['total_references']
        matches = stats['matches_found']
        percentage = (matches / total) * 100 if total > 0 else 0
        
        
        print(f"Total references processed: {total}")
        print(f"Total matches found: {matches}")
        print(f"Match percentage: {percentage:.2f}%")
        
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(f"Total references processed: {total}\n")
                f.write(f"Total matches found: {matches}\n")
                f.write(f"Match percentage: {percentage:.2f}%\n")

class BatchProcessor:
    """Handles batch processing with parallelization and checkpoints and deals with errors 500."""
    
    def __init__(self, reference_processor: 'ReferenceProcessor', max_workers: int = 4, batch_size: int = 3, pause_duration: int = 300, error_threshold: int = 3):
        self.reference_processor = reference_processor
        self.max_workers = max_workers
        self.batch_size = batch_size  
        self.pause_duration = pause_duration  
        self.error_threshold = error_threshold  
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('batch_processing.log'),
                logging.StreamHandler()
            ]
        )

    def process_file(self, input_path: str, output_path: str, threshold: int) -> str:
        """Processes a single input file."""
        try:
            self.reference_processor.process_file(input_path, output_path, threshold)
            return f"Success: {os.path.basename(input_path)}"
        except Exception as e:
            error_message = str(e)
            if "500" in error_message or "EndPointInternalError" in error_message:
                return f"ServerError: {os.path.basename(input_path)}"
            return f"Error processing {os.path.basename(input_path)}: {error_message}"

    def process_directory(self, input_dir: str, output_dir: str, threshold: int = 50, checkpoint_file: str = 'processing_checkpoint.pkl'):
        """Processes all files in a directory in smaller batches, handling errors 500."""
        os.makedirs(output_dir, exist_ok=True)
        processed_files = self.load_checkpoint(checkpoint_file)

        input_files = [f for f in os.listdir(input_dir) if f.endswith(('.xml', '.json')) and f not in processed_files]
        total_files = len(input_files)
        logging.info(f"Found {total_files} new files to process.")

        if not input_files:
            logging.info("No new files to process. Exiting.")
            return

        error_500_count = 0

        for i in range(0, total_files, self.batch_size):
            batch_files = input_files[i:i + self.batch_size]
            logging.info(f"Processing batch {i//self.batch_size + 1}/{(total_files // self.batch_size) + 1} ({len(batch_files)} files).")

            tasks = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for filename in batch_files:
                    input_path = os.path.join(input_dir, filename)
                    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_matches.csv")
                    future = executor.submit(self.process_file, input_path, output_path, threshold)
                    tasks.append((future, filename))

                for future, filename in tasks:
                    result = future.result()
                    logging.info(result)

                    if result.startswith("Success"):
                        processed_files.add(filename)
                        self.save_checkpoint(checkpoint_file, processed_files)
                    elif result.startswith("ServerError"):
                        error_500_count += 1

            if error_500_count >= self.error_threshold:
                logging.warning(f"Too many 500 errors ({error_500_count}). Stopping processing. Restart manually.")
                return

            logging.info(f"Batch {i//self.batch_size + 1} completed. Pausing for {self.pause_duration} seconds.")
            time.sleep(self.pause_duration)

        logging.info("Batch processing completed!")

    @staticmethod
    def load_checkpoint(checkpoint_file: str) -> Set[str]:
        """Loads the list of already processed files."""
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logging.error(f"Error loading checkpoint: {str(e)}")
        return set()

    @staticmethod
    def save_checkpoint(checkpoint_file: str, processed_files: Set[str]):
        """Saves the already processed files list in the checkpoint."""
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(processed_files, f)
        except Exception as e:
            logging.error(f"Error saving checkpoint: {str(e)}")

def process_single(processor: ReferenceProcessor, input_file: str, output_file: str = None, threshold: int = 50):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' does not exist.")
    
    if output_file is None:
        output_file = f"{os.path.splitext(os.path.basename(input_file))[0]}_matches_GS.csv"
    
    print(f"Processing file: {input_file}")
    processor.process_file(input_file, output_file, threshold)
    print("Processing completed successfully!")

def main():
    parser = argparse.ArgumentParser(description='Process references from Crossref JSON or TEI XML files')
    parser.add_argument('input', help='Path to input file or directory')
    parser.add_argument('--batch', '-b', action='store_true', help='Process all files in the input directory')
    parser.add_argument('--output', '-o', help='Output file (single) or directory (batch)')
    parser.add_argument('--threshold', '-t', type=int, default=50, help='Matching score threshold (default: 50)')
    parser.add_argument('--use-grobid', action='store_true', help='Enable Grobid for unstructured references')
    parser.add_argument('--workers', '-w', type=int, default=4, help='Number of parallel workers (default: 4)')
    
    args = parser.parse_args()
    processor = ReferenceProcessor(use_grobid=args.use_grobid)
    
    if args.batch:
        batch_processor = BatchProcessor(processor, max_workers=args.workers)
        batch_processor.process_directory(args.input, args.output or args.input, args.threshold)
    else:
        process_single(processor, args.input, args.output, args.threshold)

if __name__ == "__main__":
    main()













        










        







        








        


        



        


        




                



            



                



                

                




        

    










    













    




    


    







    




    




    

    





    


    



    








    


        




        


        



        



            




            

                





            



            


        



    








    



    









    














    

    


        




            






