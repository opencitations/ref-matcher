import json
import csv
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from SPARQLWrapper import SPARQLWrapper, JSON
from fuzzywuzzy import fuzz
import xml.etree.ElementTree as ET
from grobid_client_python.grobid_client.grobid_client import GrobidClient
import tempfile
import os
import argparse
from unidecode import unidecode
import re
from glob import glob
import time
from random import uniform
import pickle
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import logging
logging.basicConfig(level=logging.INFO)
# Debug print to confirm this file is loaded:
print(">>> Loaded ReferenceMatchingTool2.py at", __file__)


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

    def __init__(self, config_path: str = r"C:\Users\fabio\Desktop\Matteo\DHDK\Borsa_di_ricerca_ASMR-E\structureCitations\tirocinio_OC\data\grobid\config.json"):
        self.client = GrobidClient(config_path=config_path)

    def process_unstructured_reference(self, unstructured_text: str) -> Optional[Reference]:
        """
        Send a single raw reference string to Grobid (processCitationList),
        parse the resulting TEI-XML, and return a Reference object with whatever
        fields Grobid extracted. Returns None if Grobid did not produce TEI-XML.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "input")
            output_path = os.path.join(temp_dir, "output")
            os.makedirs(input_path)
            os.makedirs(output_path)

            # Write the unstructured citation string into a file named "ref.txt"
            with open(os.path.join(input_path, "ref.txt"), "w", encoding="utf-8") as f:
                f.write(unstructured_text)

            # Call Grobid's processCitationList endpoint on that single file
            self.client.process("processCitationList", input_path, output=output_path, n=1, verbose=False)

            # DEBUG: Log what files were generated
            logging.debug("Grobid output files: %s", os.listdir(output_path))
            
            # Pick up any .tei.xml file that was generated
            tei_files = glob(os.path.join(output_path, "*.tei.xml"))
            if not tei_files:
                logging.warning("[GROBID] No TEI XML file found in output directory.")
                return None

            return self._parse_tei_xml(tei_files[0])

    def _parse_tei_xml(self, xml_file: str) -> Optional[Reference]:
        """
        Parse a Grobid TEI-XML file into a Reference. Extracts:
          - year  (from <date when="YYYY">)
          - first_author_lastname (from <author><surname>)
          - article_title         (from first <title> under TEI root)
          - volume                (from <biblScope unit="volume">)
          - first_page            (from <biblScope unit="page">)
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

        ref = Reference()

        # Extract year (first <date when="YYYY-MM-DD"> or similar)
        date = root.find('.//tei:date', ns)
        if date is not None:
            ref.year = date.get('when', '')[:4]

        # Extract first author’s surname
        author = root.find('.//tei:author', ns)
        if author is not None:
            surname = author.find('.//tei:surname', ns)
            if surname is not None:
                ref.first_author_lastname = surname.text or ""

        # Extract first <title> (which often is the article title in Grobid’s citation TEI)
        title = root.find('.//tei:title', ns)
        if title is not None:
            ref.article_title = title.text or ""

        # Extract volume and first page from <biblScope> elements
        biblScope = root.findall('.//tei:biblScope', ns)
        for scope in biblScope:
            unit = scope.get('unit', '')
            if unit == 'volume':
                ref.volume = scope.text or ""
            elif unit == 'page':
                ref.first_page = (scope.text or "").split('-')[0].strip()

        return ref


class OpenCitationsMatcher:
    """Class to handle queries in OpenCitations and matching"""

    # def __init__(self, endpoint: str = "https://opencitations.net/meta/sparql", max_retries: int = 5):
    def __init__(self, endpoint: str = "https://sparql.opencitations.net/meta", max_retries: int = 5):
        self.sparql = SPARQLWrapper(endpoint)
        self.sparql.setReturnFormat(JSON)
        self.sparql.setTimeout(30)
        self.max_retries = max_retries

    def query_opencitations(self, sparql_query: str) -> Optional[List[Dict]]:
        """Executes a SPARQL query with retry on HTTP 503 errors."""
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
        """Given a year string, try to parse out a four-digit year."""
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
        """Normalize a title string for fuzzy matching."""
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
            "'": "'",
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
        # Try to extract a single four-digit year (handles things like "2006–2007").
        year_int = self._extract_year(reference.year)
        if year_int is None:
            return self._build_alternative_query(reference, query_type)

        title = reference.get_main_title()

        if query_type == "year_and_doi" and reference.doi and reference.year:
            # year_int is already an integer
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

        elif (query_type == "year_volume_page"
              and reference.year and reference.volume and reference.first_page):
            return f"""
            PREFIX datacite: <http://purl.org/spar/datacite/>
            PREFIX dcterms: <http://purl.org/dc/terms/>
            PREFIX literal: <http://www.essepuntato.it/2010/06/literalreification/>
            PREFIX prism: <http://prismstandard.org/namespaces/basic/2.0/>
            PREFIX frbr: <http://purl.org/vocab/frbr/core/>
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

        elif (query_type == "year_author_page"
              and reference.year
              and reference.first_author_lastname
              and reference.first_page):
            return f"""
            PREFIX datacite: <http://purl.org/spar/datacite/>
            PREFIX dcterms: <http://purl.org/dc/terms/>
            PREFIX literal: <http://www.essepuntato.it/2010/06/literalreification/>
            PREFIX prism: <http://prismstandard.org/namespaces/basic/2.0/>
            PREFIX frbr: <http://purl.org/vocab/frbr/core/>
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
        """Fallback SPARQL query when year is missing (use DOI if available)."""
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
        """Calculates a flexible matching score between our Reference and a SPARQL result."""
        score = 0

        # ─── Year‐based scoring using _extract_year ─────────────────────────────────
        if reference.year and 'pub_date' in result:
            record_year_int = self._extract_year(reference.year)
            if record_year_int is not None:
                try:
                    result_year = int(result['pub_date']['value'][:4])
                    if record_year_int == result_year:
                        score += 20
                    elif abs(record_year_int - result_year) == 1:
                        score += 15
                except ValueError:
                    pass
        # ────────────────────────────────────────────────────────────────────

        # Title‐based scoring (fuzzy)
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
                    title_score = max(exact_match, ratio, partial_ratio, token_sort_ratio, token_set_ratio)
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

        # Volume‐match scoring
        if reference.volume and 'volume_num' in result:
            if reference.volume == result['volume_num']['value']:
                score += 15

        # First‐page scoring
        if reference.first_page and 'start_page' in result:
            ref_page = reference.first_page.lstrip('0')
            result_page = result['start_page']['value'].lstrip('0')
            if ref_page == result_page:
                score += 15

        return score


class ReferenceProcessor:
    """Main class to coordinate reference processing and matching, with optional Grobid fallback."""

    def __init__(self, use_grobid: bool = False):
        self.matcher = OpenCitationsMatcher()
        self.use_grobid = use_grobid
        if self.use_grobid:
            self.grobid_processor = GrobidProcessor()
        # Optional counter if you want to track total fallbacks
        self.grobid_fallback_count = 0

    def process_reference(self, ref: Reference, threshold: int = 50, use_doi: bool = True) -> Optional[Dict]:
        """
        Attempt to match a Reference via SPARQL queries. If no match exceeds threshold,
        and use_grobid=True, then parse the unstructured text with Grobid, merge fields,
        and retry SPARQL one more time.
        """
        # Make a shallow copy so we don’t modify the caller’s Reference
        processed_ref = Reference(**{k: v for k, v in ref.__dict__.items()})
        if not use_doi:
            processed_ref.doi = ""

        # ───── Helper: run through all SPARQL query types once ─────
        def run_sparql_matching_loop(reference_obj: Reference) -> Tuple[Optional[Dict], int]:
            best_score = 0
            best_match = None
            query_types = [
                "year_and_doi",
                "year_volume_page",
                "year_author_page",
                "author_title",
            ]
            for query_type in query_types:
                query = self.matcher.build_sparql_query(reference_obj, query_type)
                if not query:
                    continue
                results = self.matcher.query_opencitations(query)
                if not results:
                    continue
                for result in results:
                    score = self.matcher.calculate_matching_score(reference_obj, result)
                    if score > best_score:
                        best_score = score
                        best_match = result
                        best_match["score"] = score
                        best_match["query_type"] = query_type
            return best_match, best_score

        # ───────────────────────────────────────────────────────

        # 1) First SPARQL-only pass
        best_match, best_score = run_sparql_matching_loop(processed_ref)
        adjusted_threshold = threshold * 0.9 if (best_score > threshold * 0.8) else threshold
        if best_score >= adjusted_threshold:
            logging.info(f"Match found with SPARQL only (score: {best_score})")
            return best_match
        else:
            logging.info(f"SPARQL match below threshold ({best_score} < {adjusted_threshold})")
  

        # 2) If SPARQL-only failed and Grobid is enabled (and we have unstructured text), do a Grobid fallback
        if self.use_grobid and processed_ref.unstructured:
            logging.info("Attempting Grobid fallback...")
            logging.info(f"[GROBID] Processing unstructured text: {processed_ref.unstructured[:100]}...")

            grobid_ref = self.grobid_processor.process_unstructured_reference(processed_ref.unstructured)
            if grobid_ref:
                # Log and increment fallback counter
                self.grobid_fallback_count += 1
                logging.info(f"[GROBID] Fallback #{self.grobid_fallback_count} used for unstructured reference: \"{processed_ref.unstructured[:50]}…\"")
                
                # Log which fields Grobid found
                fields_found = []
                if grobid_ref.doi: fields_found.append(f"DOI: {grobid_ref.doi}")
                if grobid_ref.year: fields_found.append(f"year: {grobid_ref.year}")
                if grobid_ref.first_author_lastname: fields_found.append(f"author: {grobid_ref.first_author_lastname}")
                if grobid_ref.article_title: fields_found.append(f"title: {grobid_ref.article_title}")
                if grobid_ref.volume: fields_found.append(f"volume: {grobid_ref.volume}")
                if grobid_ref.first_page: fields_found.append(f"page: {grobid_ref.first_page}")
                logging.info(f"[GROBID] Fields extracted: {', '.join(fields_found)}")

                # Merge and log which fields were added
                fields_merged = []
                if not processed_ref.doi and grobid_ref.doi:
                    processed_ref.doi = grobid_ref.doi
                    fields_merged.append("DOI")
                if not processed_ref.year and grobid_ref.year:
                    processed_ref.year = grobid_ref.year
                    fields_merged.append("year")
                if not processed_ref.first_author_lastname and grobid_ref.first_author_lastname:
                    processed_ref.first_author_lastname = grobid_ref.first_author_lastname
                    fields_merged.append("author")
                if not processed_ref.article_title and grobid_ref.article_title:
                    processed_ref.article_title = grobid_ref.article_title
                    fields_merged.append("title")
                if not processed_ref.volume and grobid_ref.volume:
                    processed_ref.volume = grobid_ref.volume
                    fields_merged.append("volume")
                if not processed_ref.first_page and grobid_ref.first_page:
                    processed_ref.first_page = grobid_ref.first_page
                    fields_merged.append("page")
                
                if fields_merged:
                    logging.info(f"[GROBID] Fields merged into reference: {', '.join(fields_merged)}")
                else:
                    logging.info("[GROBID] No new fields were merged (all fields already present)")

                # Run SPARQL again with merged data
                logging.info("[GROBID] Attempting second SPARQL query with enhanced data")
                best_match2, best_score2 = run_sparql_matching_loop(processed_ref)
                adjusted_threshold2 = threshold * 0.9 if (best_score2 > threshold * 0.8) else threshold
                
                if best_score2 >= adjusted_threshold2:
                    logging.info(f"[GROBID] Second SPARQL query successful (score: {best_score2} >= {adjusted_threshold2})")
                    return best_match2
                else:
                    logging.info(f"[GROBID] Second SPARQL query below threshold ({best_score2} < {adjusted_threshold2})")
            else:
                logging.info("[GROBID] Failed to extract any structured data from the reference")
        else:
            if not self.use_grobid:
                logging.info("[GROBID] Fallback not enabled (--use-grobid flag not set)")
            if not processed_ref.unstructured:
                logging.info("[GROBID] No unstructured text available in reference")

        # No match above threshold even after Grobid fallback
        logging.info("No match found after all attempts")
        return None

    def _merge_reference_data(self, original_ref: Reference, grobid_ref: Reference):
        """
        Combine data from Grobid with existing metadata, preserving original fields when they exist.
        """
        if not original_ref.doi and grobid_ref.doi:
            original_ref.doi = grobid_ref.doi
        if not original_ref.year and grobid_ref.year:
            original_ref.year = grobid_ref.year
        if not original_ref.first_author_lastname and grobid_ref.first_author_lastname:
            original_ref.first_author_lastname = grobid_ref.first_author_lastname
        if not original_ref.article_title and grobid_ref.article_title:
            original_ref.article_title = grobid_ref.article_title
        if not original_ref.volume and grobid_ref.volume:
            original_ref.volume = grobid_ref.volume
        if not original_ref.first_page and grobid_ref.first_page:
            original_ref.first_page = grobid_ref.first_page

    def process_file(self, input_file: str, output_file: str, threshold: int = 50):
        """Process an input file (JSON or XML) and write matches to CSV."""
        _, extension = os.path.splitext(input_file)

        if extension.lower() == '.json':
            self._process_crossref_file(input_file, output_file, threshold)
        elif extension.lower() == '.xml':
            self._process_tei_file(input_file, output_file, threshold)
        else:
            raise ValueError(f"Unsupported file format: {extension}")

    def _process_crossref_file(self, input_file: str, output_file: str, threshold: int):
        """Process a Crossref JSON file, row by row, writing matches to CSV."""
        stats = {'total_references': 0, 'matches_found': 0}

        try:
            # Try reading with UTF-8 encoding first
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except UnicodeDecodeError:
            # If UTF-8 fails, try with utf-8-sig (handles BOM)
            try:
                with open(input_file, 'r', encoding='utf-8-sig') as f:
                    data = json.load(f)
            except UnicodeDecodeError:
                # Last resort: read as latin-1
                with open(input_file, 'r', encoding='latin-1') as f:
                    data = json.load(f)

        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=[
                    'reference_id',
                    'article_title',
                    'matched_title',
                    'score',
                    'matched_doi',
                    'meta_id',
                    'query_type'
                ]
            )
            writer.writeheader()

            if 'message' in data and 'reference' in data['message']:
                for i, ref_data in enumerate(data['message']['reference'], 1):
                    stats['total_references'] += 1

                    ref = Reference(
                        year=ref_data.get('year', ''),
                        volume=ref_data.get('volume', ''),
                        first_page=ref_data.get('first-page', ''),
                        first_author_lastname=(
                            ref_data.get('author', '').split()[-1]
                            if ref_data.get('author') else ''
                        ),
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

        # Print and save the per-file stats
        self._print_stats(stats)
        stats_file = os.path.splitext(output_file)[0] + '_stats.txt'
        self._print_stats(stats, stats_file)

    def _process_tei_file(self, input_file: str, output_file: str, threshold: int):
        """Process a TEI XML file, extract each <biblStruct>, and write matches to CSV."""
        stats = {'total_references': 0, 'matches_found': 0}
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

        tree = ET.parse(input_file)
        root = tree.getroot()

        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=[
                    'reference_id',
                    'article_title',
                    'matched_title',
                    'score',
                    'matched_doi',
                    'meta_id',
                    'query_type'
                ]
            )
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

        # Print and save the per-file stats
        self._print_stats(stats)
        stats_file = os.path.splitext(output_file)[0] + '_stats.txt'
        self._print_stats(stats, stats_file)

    @staticmethod
    def _parse_bibl_struct(bibl: ET.Element, ns: Dict) -> Optional[Reference]:
        """Parse a single <biblStruct> element into a Reference object."""
        ref = Reference()

        date = bibl.find('.//tei:date[@when]', ns)
        if date is not None:
            ref.year = date.get('when', '')[:4]

        authors = bibl.findall('.//tei:author/tei:persName/tei:surname', ns)
        if authors:
            ref.first_author_lastname = authors[0].text or ""

        analytic = bibl.find('tei:analytic', ns)
        if analytic is not None:
            title = analytic.find('tei:title', ns)
            if title is not None:
                ref.article_title = title.text or ""
        else:
            monogr = bibl.find('tei:monogr', ns)
            if monogr is not None:
                m_title = monogr.find('tei:title[@level="m"]', ns)
                if m_title is not None:
                    ref.volume_title = m_title.text or ""
                j_title = monogr.find('tei:title[@level="j"]', ns)
                if j_title is not None:
                    ref.journal_title = j_title.text or ""

        volume = bibl.find('.//tei:biblScope[@unit="volume"]', ns)
        if volume is not None and volume.text:
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
        total = stats['total_references']
        matches = stats['matches_found']
        percentage = (matches / total * 100) if total > 0 else 0.0

        print(f"Total references processed: {total}")
        print(f"Total matches found: {matches}")
        print(f"Match percentage: {percentage:.2f}%")

        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"Total references processed: {total}\n")
                    f.write(f"Total matches found: {matches}\n")
                    f.write(f"Match percentage: {percentage:.2f}%\n")
            except Exception as e:
                logging.error(f"Error writing stats file {output_file}: {e}")
class BatchProcessor:
    """Handles batch processing with parallelization and checkpoints and deals with errors 500."""

    def __init__(self, reference_processor: 'ReferenceProcessor', max_workers: int = 4,
                 batch_size: int = 3, pause_duration: int = 300, error_threshold: int = 3):
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

        # ────────────────────────────────────────────────────────────────────────────────

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
        # After batch, print how many times Grobid was invoked:
        if args.use_grobid:
            print(f"Total Grobid fallbacks performed: {processor.grobid_fallback_count}")
    else:
        process_single(processor, args.input, args.output, args.threshold)
        if args.use_grobid:
            print(f"Total Grobid fallbacks performed: {processor.grobid_fallback_count}")


if __name__ == "__main__":
    main()
