import json
import csv
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from fuzzywuzzy import fuzz
import xml.etree.ElementTree as ET
from data.grobid.grobid_client.grobid_client import GrobidClient
from tqdm import tqdm
import time
import tempfile
import os
import argparse
from unidecode import unidecode
import unicodedata
import re
from glob import glob
from random import uniform
import pickle
from datetime import datetime
import logging
import threading
from pathlib import Path
import asyncio
import aiohttp
from logging.handlers import RotatingFileHandler
import sys

# LOGGING SETUP 
class MessageFilter(logging.Filter):
    """Custom filter for log messages based on content"""
    def __init__(self, filter_func):
        super().__init__()
        self.filter_func = filter_func
    
    def filter(self, record):
        if self.filter_func is None:
            return True
        try:
            return self.filter_func(record)
        except Exception:
            return True 

def setup_logging(log_level=logging.INFO):
    """
    Setup multi-file logging system with proper filters
    """
    
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    root_logger = logging.getLogger()
    
    
    file_handlers = [
        ('reference_matching_main.log', logging.DEBUG, None),
        ('reference_matching_authors.log', logging.DEBUG, 
         lambda r: 'AUTHOR' in r.getMessage() or '👤' in r.getMessage()),
        ('reference_matching_queries.log', logging.DEBUG,
         lambda r: 'SPARQL' in r.getMessage() or 'QUERY' in r.getMessage() or '🔍' in r.getMessage() or '🔨' in r.getMessage()),
        ('reference_matching_scores.log', logging.DEBUG,
         lambda r: 'SCORE' in r.getMessage() or 'MATCH' in r.getMessage() or '🎯' in r.getMessage()),
        ('reference_matching_errors.log', logging.WARNING, None)
    ]
    
    for filename, level, filter_func in file_handlers:
        try:
            handler = RotatingFileHandler(
                filename,
                maxBytes=10*1024*1024,
                backupCount=5,
                encoding='utf-8'
            )
            handler.setLevel(level)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            
            
            if filter_func:
                handler.addFilter(MessageFilter(filter_func))
            
            root_logger.addHandler(handler)
            
        except Exception as e:
            print(f"Warning: Could not create log file {filename}: {e}")
    
    return root_logger

# Call setup
# setup_logging(log_level=logging.INFO)
# print(">>> Loaded ReferenceMatchingTool.py at", __file__)

class ReferenceMatchingError(Exception):
    """Base exception for reference matching errors"""
    pass


class QueryExecutionError(ReferenceMatchingError):
    """Error executing SPARQL query"""
    def __init__(self, message: str, query: str = None, attempt: int = None):
        self.query = query
        self.attempt = attempt
        super().__init__(message)

class RateLimitError(QueryExecutionError):
    """Rate limit exceeded"""
    pass

class ServerError(QueryExecutionError):
    """Server error (5xx)"""
    pass

@dataclass
class MatcherConfig:
    """Configuration for reference matching"""
    # Timeouts and retries
    default_timeout: int = 600
    max_retries: int = 3
    
    # Year validation
    min_year: int = 1700
    max_year: int = datetime.now().year + 1
    
    # Metadata Scoring - Max Scoring == 48
    doi_exact_score: int = 15     
    author_exact_match_score: int = 7  
    title_exact_score: int = 14     
    title_95_score: int = 13
    title_90_score: int = 13
    title_85_score: int = 12
    title_80_score: int = 11
    title_75_score: int = 10
    year_exact_score: int = 1      
    year_adjacent_score: int = 0 # Kept for statistical purposes
    volume_match_score: int = 3 
    page_match_score: int = 8 
    
    matching_threshold: int = 26 # Approximately 54.5 % of the max scoring
    threshold_adjustment: float = 0.9 
    
    # Rate limiting
    requests_per_second: float = 2.5
    burst_size: int = 10
    
    # Batch processing
    default_batch_size: int = 3
    default_pause_duration: int = 10
    default_error_threshold: int = 5
    checkpoint_interval: int = 10

    @property
    def year_range(self) -> Tuple[int, int]:
        return (self.min_year, self.max_year)

# Default config instance
DEFAULT_CONFIG = MatcherConfig()
DEFAULT_GROBID_CONFIG = "grobid_config.json"
DEFAULT_YEAR_RANGE = DEFAULT_CONFIG.year_range
DEFAULT_TIMEOUT = DEFAULT_CONFIG.default_timeout
DEFAULT_MAX_RETRIES = DEFAULT_CONFIG.max_retries

# SCORING_CONFIG for backward compatibility
SCORING_CONFIG = {
    'doi_exact': DEFAULT_CONFIG.doi_exact_score,  
    'year_exact': DEFAULT_CONFIG.year_exact_score,
    'year_adjacent': DEFAULT_CONFIG.year_adjacent_score,
    'title_exact': DEFAULT_CONFIG.title_exact_score,
    'title_95': DEFAULT_CONFIG.title_95_score,
    'title_90': DEFAULT_CONFIG.title_90_score,
    'title_85': DEFAULT_CONFIG.title_85_score,
    'title_80': DEFAULT_CONFIG.title_80_score,
    'title_75': DEFAULT_CONFIG.title_75_score,
    'volume_match': DEFAULT_CONFIG.volume_match_score,
    'author_exact_match': DEFAULT_CONFIG.author_exact_match_score,
    'page_match': DEFAULT_CONFIG.page_match_score,
    'threshold_adjustment': DEFAULT_CONFIG.threshold_adjustment
}

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

def normalize_reference_safe(ref: Reference) -> Reference:
    """Safely normalize reference fields with comprehensive error handling."""
    try:
        # DOI normalization with type checking
        if ref.doi:
            doi = str(ref.doi).strip().lower() if ref.doi else ""
            
            if doi:
                # Remove common prefixes
                for prefix in ['doi:', 'https://doi.org/', 'http://doi.org/', 
                              'https://dx.doi.org/', 'http://dx.doi.org/']:
                    if doi.startswith(prefix.lower()):
                        doi = doi[len(prefix):].strip()
                        break
                
                # Only normalize if we have a valid string
                if doi:
                    try:
                        doi = doi.replace('\\/', '/')
                        ref.doi = unicodedata.normalize("NFC", doi)
                    # Keep original if normalization fails
                    except (TypeError, ValueError) as e:
                        logging.info(f"Error normalizing DOI '{doi}': {e}")
                        ref.doi = doi  
                else:
                    ref.doi = ""

        # Text field normalization with type checking
        for attr in ["first_author_lastname", "article_title", "volume_title", 
                    "journal_title", "year", "volume", "first_page"]:
            try:
                value = getattr(ref, attr, None)
                
                if value is None:
                    setattr(ref, attr, "")
                    continue
                
                # Ensure it's a string
                value = str(value).strip() if value else ""
                
                if value:
                    # Apply unicode normalization
                    try:
                        value = unicodedata.normalize("NFC", value)
                    except (TypeError, ValueError) as e:
                        logging.info(f"Error normalizing {attr} '{value[:50]}': {e}")
                        # Keep original value if normalization fails
                
                setattr(ref, attr, value)
                
            except Exception as e:
                logging.info(f"Error processing field {attr}: {e}")
                setattr(ref, attr, "")

        return ref
        
    except Exception as e:
        logging.error(f"Critical error normalizing reference: {e}")
        # Return reference as-is rather than failing completely
        return ref

def normalize_for_fuzzy_title(s: str) -> str:
    """Normalize title for fuzzy matching with specific error handling"""
    if not s:
        return ""
    
    try:
        # Correct character mappings
        repl = {
            # Dashes and hyphens
            '\u2013': '-', '\u2014': '-', '\u2212': '-',  # en-dash, em-dash, minus
            # Quotation marks
            '\u201c': '"', '\u201d': '"', '\u2018': "'", '\u2019': "'",
            '\u00b4': "'", '\u00a8': '"',  # acute accent, diaeresis
            # Ellipsis
            '\u2026': '...',
            # Greek letters (common in scientific texts)
            '\u03b1': 'alpha', '\u03b2': 'beta', '\u03b3': 'gamma', '\u03b4': 'delta',
            '\u03b5': 'epsilon', '\u03b6': 'zeta', '\u03b7': 'eta', '\u03b8': 'theta',
            '\u03c9': 'omega', '\u03a9': 'Omega',
            # Special patterns
            '\u03c93': 'omega3', '\u03c9-3': 'omega-3',
            '(1)H': '1H', '(13)C': '13C',
            'â€"': '-', 'â€"': '-'
        }
        
        for old, new in repl.items():
            s = s.replace(old, new)

        # Transliterate and normalize
        t = unidecode(s).lower().strip()

        # Refinements
        t = t.rstrip('.!?;:')  # ← AGGIUNGI: ; e :
        # Keep only letters, numbers, spaces, hyphens
        t = re.sub(r'[^a-z0-9\s\-]', '', t)
        # Compact whitespace
        t = ' '.join(t.split())
        return t
        
    except UnicodeDecodeError as e:
        logging.info(f"Unicode decode error in title normalization: {e}")
        return str(s).lower().strip()
    except AttributeError as e:
        logging.info(f"AttributeError in title normalization (input type: {type(s)}): {e}")
        return str(s).lower().strip() if s else ""
    except Exception as e:
        logging.error(f"Unexpected error normalizing title '{str(s)[:50]}...': {e}")
        return str(s).lower().strip() if s else ""

def apply_threshold_adjustment(best_score: int, threshold: int, 
                               adjustment_factor: float = None) -> int:
    """
    Apply dynamic threshold adjustment based on config

    """
    # Use config value if not provided
    if adjustment_factor is None:
        adjustment_factor = SCORING_CONFIG.get('threshold_adjustment', 0.9)
    
    # Calculate trigger point (when to apply adjustment)
    adjustment_trigger = threshold * adjustment_factor
    
    if best_score >= adjustment_trigger:
        adjusted = int(adjustment_trigger)
        logging.debug(
            f"Threshold adjustment: {threshold} → {adjusted} "
            f"(score={best_score} >= {adjustment_trigger:.1f}, factor={adjustment_factor})"
        )
        return adjusted
    
    return threshold

class GrobidProcessor:
    """Improved Grobid processor with robust path resolution"""
    
    @staticmethod
    def find_config_file(custom_path: Optional[str] = None) -> str:
        """Find Grobid config with multiple fallback strategies"""
        
        # Priority 1: Custom path provided
        if custom_path and os.path.exists(custom_path):
            return custom_path
        
        # Priority 2: Environment variable
        env_config = os.environ.get('GROBID_CONFIG_PATH')
        if env_config and os.path.exists(env_config):
            return env_config
        
        # Priority 3: Current directory and common locations
        search_paths = [
            # Current directory
            DEFAULT_GROBID_CONFIG,
            'grobid.json',
            'config.json',
            
            # User home directory
            os.path.join(os.path.expanduser("~"), ".grobid", "config.json"),
            
            # Script directory
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "grobid_config.json"),
            
            # Common config locations
            os.path.join("config", "grobid.json"),
            os.path.join("conf", "grobid.json"),
            
            # Project root (if running from src/)
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        "data", "grobid", "config.json"),
            
            # Common project structures
            os.path.join(os.getcwd(), "data", "grobid", "config.json"),
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                logging.debug(f"Found Grobid config at: {path}")  # Changed from INFO
                return path
        
        # Priority 4: Search in parent directories (up to 3 levels)
        current = os.path.abspath(__file__)
        for _ in range(3):
            current = os.path.dirname(current)
            potential = os.path.join(current, "data", "grobid", "config.json")
            if os.path.exists(potential):
                logging.debug(f"Found Grobid config at: {potential}")
                return potential
        
        # Not found anywhere
        raise FileNotFoundError(
            f"Grobid config file not found. Searched locations:\n" + 
            "\n".join(f"  - {p}" for p in search_paths) +
            f"\n\nYou can:\n"
            f"1. Create a config file at any of the above locations\n"
            f"2. Set GROBID_CONFIG_PATH environment variable\n"
            f"3. Pass config_path parameter explicitly"
        )

    def __init__(self, config_path: Optional[str] = None):
        """Initialize config finding"""
        try:
            config_path = self.find_config_file(config_path)
            self.client = GrobidClient(config_path=config_path)
            
            logging.debug(f"Grobid client initialized for thread {threading.get_ident()}")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(str(e))
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Grobid client: {e}")

    def process_unstructured_reference(self, unstructured_text: str) -> Optional[Reference]:
        """
        Comprehensive error handling for Grobid processing
        """
        if not unstructured_text or not unstructured_text.strip():
            return None

        try:
            # Clean the text to remove problematic characters
            try:
                # Ensure text is properly decoded/encoded
                if isinstance(unstructured_text, bytes):
                    unstructured_text = unstructured_text.decode('utf-8', errors='ignore')
                else:
                    # Re-encode to UTF-8 to clean up any encoding issues
                    unstructured_text = unstructured_text.encode('utf-8', errors='ignore').decode('utf-8')
            except Exception as e:
                logging.info(f"Text encoding cleanup failed: {e}")
                # Continue with original text
            
            with tempfile.TemporaryDirectory() as temp_dir:
                input_path = os.path.join(temp_dir, "input")
                output_path = os.path.join(temp_dir, "output")
                os.makedirs(input_path, exist_ok=True)
                os.makedirs(output_path, exist_ok=True)

                # Write the unstructured citation string
                input_file = os.path.join(input_path, "ref.txt")
                with open(input_file, "w", encoding="utf-8") as f:
                    f.write(unstructured_text)

                # Call Grobid with timeout
                try:
                    self.client.process(
                        "processCitationList", 
                        input_path, 
                        output=output_path, 
                        n=1, 
                        verbose=False
                    )
                except UnicodeDecodeError as e:
                    logging.info(f"Grobid unicode error: {e}")
                    return None
                except Exception as e:
                    logging.info(f"Grobid processing failed: {e}")
                    return None

                # Find TEI XML output
                tei_files = list(Path(output_path).glob("*.tei.xml"))
                if not tei_files:
                    logging.debug("No TEI XML file generated by Grobid")
                    return None

                return self._parse_tei_xml(str(tei_files[0]))

        except Exception as e:
            logging.error(f"Error in Grobid processing: {e}")
            return None

    def _parse_tei_xml(self, xml_file: str) -> Optional[Reference]:
        """Enhanced TEI XML parsing with comprehensive error handling"""
        try:
            # Parse XML with explicit encoding handling
            try:
                # Try UTF-8 first
                tree = ET.parse(xml_file)
                root = tree.getroot()
            except (ET.ParseError, UnicodeDecodeError):
                # If UTF-8 fails, read as binary and decode manually
                try:
                    with open(xml_file, 'rb') as f:
                        content = f.read()
                        # Try to decode, replacing problematic bytes
                        try:
                            text = content.decode('utf-8', errors='replace')
                        except:
                            text = content.decode('latin-1', errors='replace')
                        root = ET.fromstring(text.encode('utf-8'))
                except Exception as e:
                    logging.error(f"Could not parse TEI XML {xml_file}: {e}")
                    return None
            except FileNotFoundError:
                logging.error(f"TEI file not found: {xml_file}")
                return None
            except Exception as e:
                logging.error(f"Unexpected error reading TEI file {xml_file}: {e}")
                return None
            
            ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
            ref = Reference()
            try:
                # Extract year
                date = root.find('.//tei:date', ns)
                if date is not None:
                    when = date.get('when', '')
                    if when and len(when) >= 4:
                        ref.year = when[:4]
                # Extract first author's surname
                author = root.find('.//tei:author', ns)
                if author is not None:
                    surname = author.find('.//tei:surname', ns)
                    if surname is not None and surname.text:
                        ref.first_author_lastname = surname.text.strip()
                # Extract title
                title = root.find('.//tei:title', ns)
                if title is not None and title.text:
                    ref.article_title = title.text.strip()
                # Extract volume and page
                for scope in root.findall('.//tei:biblScope', ns):
                    unit = scope.get('unit', '')
                    if unit == 'volume' and scope.text:
                        ref.volume = scope.text.strip()
                    elif unit == 'page' and scope.text:
                        ref.first_page = scope.text.split('-')[0].strip()
                
                # Extract DOI - try multiple common locations
                doi_elem = root.find('.//tei:idno[@type="DOI"]', ns)
                if doi_elem is not None and doi_elem.text:
                    ref.doi = doi_elem.text.strip()
                else:
                    # Try lowercase variant
                    doi_elem = root.find('.//tei:idno[@type="doi"]', ns)
                    if doi_elem is not None and doi_elem.text:
                        ref.doi = doi_elem.text.strip()
                    else:
                        # Alternative: try ptr element with DOI target
                        ptr_elem = root.find('.//tei:ptr[@type="DOI"]', ns)
                        if ptr_elem is not None:
                            doi_target = ptr_elem.get('target', '')
                            if doi_target:
                                # Remove common DOI URL prefixes if present
                                doi_target = doi_target.replace('https://doi.org/', '')
                                doi_target = doi_target.replace('http://dx.doi.org/', '')
                                ref.doi = doi_target.strip() 
                return ref
                
            except AttributeError as e:
                logging.error(f"Missing expected XML structure in {xml_file}: {e}")
                return None
            except Exception as e:
                logging.error(f"Error extracting data from TEI XML {xml_file}: {e}")
                return None
        except Exception as e:
            logging.error(f"Unexpected error in _parse_tei_xml for {xml_file}: {e}")
            return None


class ImprovedRateLimiter:
    """
    Token bucket rate limiter for async operations
    Allows burst traffic while maintaining average rate
    """
    
    def __init__(self, requests_per_second: float = 2.5, burst_size: int = 10):
        """
        Args:
            requests_per_second: Average rate (e.g., 2.5 = one request every 0.4s)
            burst_size: Maximum concurrent requests allowed
        """
        self.rate = requests_per_second
        self.burst_size = burst_size
        self.tokens = float(burst_size)
        self.last_update = time.time()
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(burst_size)
        self._429_count = 0
        self._last_429_time = 0.0
        
        logging.info(f"Rate limiter initialized: {requests_per_second} req/s, burst={burst_size}")
    
    async def acquire(self):
        """Acquire permission to make a request"""
        # Limit concurrent requests
        async with self._semaphore:  
            async with self._lock:
                now = time.time()
                
                # Refill tokens based on time elapsed
                elapsed = now - self.last_update
                self.tokens = min(self.burst_size, self.tokens + elapsed * self.rate)
                self.last_update = now
                
                # Wait if no tokens available
                if self.tokens < 1.0:
                    wait_time = (1.0 - self.tokens) / self.rate
                    logging.debug(f"Rate limit: waiting {wait_time:.2f}s for token")
                    await asyncio.sleep(wait_time)
                    self.tokens = 0.0
                else:
                    self.tokens -= 1.0
                    
                logging.debug(f"Token acquired. Remaining: {self.tokens:.2f}")

    
    async def handle_429(self, attempt: int = 0) -> float:
        """Handle 429 rate limit response"""
        async with self._lock:
            self._429_count += 1
            self._last_429_time = time.time()
            
            # Exponential backoff
            wait_time = min(60, (2 ** attempt) * 5)
            
            # Reduce rate temporarily
            self.tokens = 0.0
            
            logging.warning(f"⚠️ 429 Rate Limit Hit! (count: {self._429_count})")
            logging.warning(f"   Backing off for {wait_time:.1f}s")
            
            return wait_time
        
    def get_stats(self) -> dict:
        """Get rate limiter statistics"""
        return {
            'tokens': self.tokens,
            'rate': self.rate,
            'burst_size': self.burst_size,
            '429_count': self._429_count
        }

class OpenCitationsMatcherThreadSafe:
    """Async matcher with rate limiting"""
    
    def __init__(self, endpoint: str = "https://sparql-stg.opencitations.net/meta" ,
             max_retries: int = None, timeout: int = None, config: MatcherConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.endpoint = endpoint
        self.max_retries = max_retries or self.config.max_retries
        self.timeout = timeout or self.config.default_timeout
        self.rate_limiter = ImprovedRateLimiter(
            requests_per_second=self.config.requests_per_second,
            burst_size=self.config.burst_size
        )
        self.session = None
    
    async def __aenter__(self):
        """Async context manager with optimized connection pooling"""
        timeout = aiohttp.ClientTimeout(
                total=self.timeout,
                sock_connect=10  
            )        
        # Optimized connector
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            ttl_dns_cache=300,
            force_close=False,
            enable_cleanup_closed=True,
            ssl=False
        )
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def query_opencitations(self, sparql_query: str, query_type: str = "unknown") -> List[Dict]:  # Changed to async
        """Execute query with async operations and explicit error handling

        Args:
            sparql_query: The SPARQL query string to execute
            query_type: The type of query being executed (for logging/stats)
        """

        if not sparql_query or not sparql_query.strip():
            raise QueryExecutionError("Empty SPARQL query provided")

        query_preview = sparql_query[:300] + "..." if len(sparql_query) > 300 else sparql_query
        
        logging.info(f"\n🔍 EXECUTING SPARQL QUERY ({query_type})")
        logging.info(f"{'─'*60}")
        logging.debug(f"Query:\n{sparql_query}")

        for attempt in range(self.max_retries):
            try:
                await self.rate_limiter.acquire()
                
                logging.info(f"⏳ Attempt {attempt + 1}/{self.max_retries}...")
                
                # Use aiohttp instead of SPARQLWrapper
                # I changed it from first version due to performance issues
                params = {'query': sparql_query, 'format': 'json'}
                headers = {'Accept': 'application/sparql-results+json'}
                
                async with self.session.get(self.endpoint, params=params, headers=headers) as response:
                    if response.status == 200:
                        results = await response.json()
                        bindings = results.get('results', {}).get('bindings', [])
                        
                        logging.info(f"✅ Query returned {len(bindings)} results")
                        
                        # Log first result details if any
                        if bindings:
                            first_result = bindings[0]
                            logging.info(f"\n📊 First Result Details:")
                            for key, value in first_result.items():
                                val_str = str(value.get('value', ''))
                                if len(val_str) > 80:
                                    val_str = val_str[:80] + "..."
                                logging.info(f"  {key}: {val_str}")
                        
                        return bindings
                    
                    elif response.status == 429:
                        # Rate limit error
                        if attempt < self.max_retries - 1:
                            wait_time = await self.rate_limiter.handle_429(attempt)  
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            raise RateLimitError(
                                f"Rate limit exceeded after {self.max_retries} attempts",
                                query=query_preview,
                                attempt=attempt
                            )
                    
                    elif response.status in (500, 502, 503, 504):
                        # Server error
                        if attempt < self.max_retries - 1:
                            wait_time = (2 ** attempt) + uniform(0, 1)
                            logging.warning(f"⚠️ Server error {response.status}, retrying in {wait_time:.2f}s")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            raise ServerError(
                                f"Server error {response.status} after {self.max_retries} attempts",
                                query=query_preview,
                                attempt=attempt
                            )
                    
                    else:
                        # Other error
                        error_text = await response.text()
                        raise QueryExecutionError(f"Query failed with status {response.status}: {error_text[:200]}")

            except asyncio.TimeoutError:
                if attempt < self.max_retries - 1:
                    logging.info(f"⌛ Timeout at attempt {attempt + 1}, retrying...")
                    await asyncio.sleep(2) 
                    continue
                else:
                    raise QueryExecutionError(f"Query timed out after {self.max_retries} attempts")
            
            except aiohttp.ClientError as e:
                logging.error(f"Network error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  
                    continue
                else:
                    raise QueryExecutionError(f"Network error after {self.max_retries} attempts: {e}")
            
            except Exception as e:
                error_message = str(e)
                error_type = type(e).__name__
                
                logging.error(
                    f"SPARQL query attempt {attempt + 1}/{self.max_retries} failed: "
                    f"Type={error_type}, Message={error_message}"
                )
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  
                    continue
                else:
                    raise QueryExecutionError(f"Query failed after {self.max_retries} attempts: {e}")
        
        raise QueryExecutionError(f"Query failed after {self.max_retries} attempts")


    @staticmethod
    def _esc(s: str) -> str:
        """Comprehensive SPARQL string escaping with validation"""
        if not s:
            return ""
        
        try:
            # Ensure we have a string
            if not isinstance(s, str):
                s = str(s)
            
            # First normalize unicode
            s = unicodedata.normalize("NFC", s)
            
            # SPARQL-specific escaping - ORDER MATTERS
            # 1. Escape backslashes first (most important)
            s = s.replace('\\', '\\\\')
            
            # 2. Escape quotes
            s = s.replace('"', '\\"')
            s = s.replace("'", "\\'")
            
            # 3. Replace control characters with space
            s = s.replace('\n', ' ')
            s = s.replace('\r', ' ')
            s = s.replace('\t', ' ')
            
            # 4. Remove other control characters (0x00-0x1F except those already handled)
            s = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', ' ', s)
            
            # 5. Remove SPARQL problematic characters
            for char in ['<', '>', '{', '}', '|', '^', '`', '\x7f']:
                s = s.replace(char, ' ')
            
            # 6. Normalize whitespace
            s = ' '.join(s.split())
            
            # 7. Validate length (SPARQL has practical limits)
            if len(s) > 1000:
                logging.info(f"Escaped string very long ({len(s)} chars), truncating")
                s = s[:1000]
            
            return s
            
        except UnicodeError as e:
            logging.error(f"Unicode error escaping string: {e}")
            # Fallback: aggressive cleaning
            return ''.join(c for c in str(s) if c.isprintable() and c.isascii())[:1000]
        except Exception as e:
            logging.error(f"Error escaping string '{str(s)[:50]}...': {e}")
            # Last resort fallback
            try:
                return str(s).replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'")
            except:
                return ""
    
    def _extract_year(self, year_str: str) -> Optional[int]:
        """Extract year from string with validation"""
        if not year_str:
            return None

        # Try direct integer conversion first
        try:
            year = int(year_str)
            if DEFAULT_YEAR_RANGE[0] <= year <= DEFAULT_YEAR_RANGE[1]:
                return year
            else:
                logging.debug(f"Year {year} outside valid range {DEFAULT_YEAR_RANGE}")
                return None
        except ValueError:
            pass

        # Try regex extraction as fallback
        year_match = re.search(r'\b(17|18|19|20)\d{2}\b', str(year_str))
        if year_match:
            year = int(year_match.group())
            if DEFAULT_YEAR_RANGE[0] <= year <= DEFAULT_YEAR_RANGE[1]:
                return year
            else:
                logging.debug(f"Extracted year {year} outside valid range {DEFAULT_YEAR_RANGE}")
                return None

        logging.debug(f"Could not extract valid year from: {year_str}")
        return None

    def _normalize_title(self, title: str) -> str:
        """Normalize title for fuzzy matching"""
        return normalize_for_fuzzy_title(title)

    def build_sparql_query(self, reference: Reference, query_type: str, use_doi: bool = True) -> Optional[str]:
            """Build SPARQL query with comprehensive input validation AND full optional data retrieval for scoring"""
            
            logging.info(f"\n🔨 BUILDING SPARQL QUERY: {query_type}")
            logging.info(f"{'─'*60}")
            
            # Shared SPARQL prefixes
            PREFIXES = """
                PREFIX datacite: <http://purl.org/spar/datacite/>
                PREFIX dcterms: <http://purl.org/dc/terms/>
                PREFIX literal: <http://www.essepuntato.it/2010/06/literalreification/>
                PREFIX prism: <http://prismstandard.org/namespaces/basic/2.0/>
                PREFIX pro: <http://purl.org/spar/pro/>
                PREFIX foaf: <http://xmlns.com/foaf/0.1/>
                PREFIX frbr: <http://purl.org/vocab/frbr/core#>
                PREFIX fabio: <http://purl.org/spar/fabio/>
            """
            
            # Standard SELECT for all queries
            SELECT_ALL = "SELECT DISTINCT ?br ?title ?pub_date ?doi ?author_name ?volume_num ?start_page ?end_page"

            # Standard OPTIONAL blocks
            OPTIONAL_TITLE = "OPTIONAL { ?br dcterms:title ?title . }"
            OPTIONAL_PUB_DATE = "OPTIONAL { ?br prism:publicationDate ?publicationDate . BIND(STR(?publicationDate) AS ?pub_date) }"
            OPTIONAL_DOI = """
                OPTIONAL {
                    ?br datacite:hasIdentifier ?doi_id .
                    ?doi_id datacite:usesIdentifierScheme datacite:doi ;
                            literal:hasLiteralValue ?doi .
                }"""
            OPTIONAL_AUTHOR = """
                OPTIONAL {
                    ?role pro:isDocumentContextFor ?br ;
                        pro:isHeldBy ?author .
                    ?author foaf:familyName ?author_name .
                }"""
            OPTIONAL_VOLUME = """
                OPTIONAL {
                    ?br frbr:partOf ?issue .
                    ?issue frbr:partOf ?volume .
                    ?volume fabio:hasSequenceIdentifier ?volume_num .
                }"""
            OPTIONAL_PAGES = """
                OPTIONAL {
                    ?br frbr:embodiment ?embodiment .
                    OPTIONAL { ?embodiment prism:startingPage ?start_page . }
                    OPTIONAL { ?embodiment prism:endingPage ?end_page . }
                }"""

            # STEP 1: Year validation
            year_int = self._extract_year(reference.year)
            
            # STEP 2: Mandatory fields validation
            if query_type == "year_and_doi":
                if not reference.doi or not use_doi:
                    logging.info(f"❌ SKIP: Missing DOI for year_and_doi")
                    return None
                if not year_int:
                    logging.info(f"❌ SKIP: Missing/invalid year for year_and_doi")
                    return None
                    
            elif query_type == "doi_title": 
                if not reference.doi or not use_doi:
                    logging.info(f"❌ SKIP: Missing DOI for doi_title")
                    return None
                if not reference.get_main_title():
                    logging.info(f"❌ SKIP: Missing title for doi_title")
                    return None
                    
            elif query_type == "author_title":
                if not reference.get_main_title():
                    logging.info(f"❌ SKIP: Missing title for author_title")
                    return None
                if not reference.first_author_lastname:
                    logging.info(f"❌ SKIP: Missing author for author_title")
                    return None
                    
            elif query_type == "year_volume_page":
                if not year_int:
                    logging.info(f"❌ SKIP: Missing/invalid year for year_volume_page")
                    return None
                if not reference.volume:
                    logging.info(f"❌ SKIP: Missing volume for year_volume_page")
                    return None
                if not reference.first_page:
                    logging.info(f"❌ SKIP: Missing page for year_volume_page")
                    return None
                
            elif query_type == "year_author_page":
                if not year_int:
                    logging.info(f"❌ SKIP: Missing/invalid year for year_author_page")
                    return None
                if not reference.first_author_lastname:
                    logging.info(f"❌ SKIP: Missing author for year_author_page")
                    return None
                if not reference.first_page:
                    logging.info(f"❌ SKIP: Missing page for year_author_page")
                    return None
                
            elif query_type == "year_author_volume":
                if not year_int:
                    logging.info(f"❌ SKIP: Missing/invalid year for year_author_volume")
                    return None
                if not reference.first_author_lastname:
                    logging.info(f"❌ SKIP: Missing author for year_author_volume")
                    return None
                if not reference.volume:
                    logging.info(f"❌ SKIP: Missing volume for year_author_volume")
                    return None            
            else:
                logging.info(f"❌ Unknown query_type: {query_type}")
                return None
            
            # STEP 3: Escaping
            e = self._esc
            
            title = reference.get_main_title()
            if title and len(title) > 500:
                logging.info(f"Title too long ({len(title)} chars), truncating")
                title = title[:500]
            
            title_esc = e(title)
            fam_esc = e(reference.first_author_lastname)
            vol_esc = e(reference.volume)
            page_esc = e(reference.first_page)
            doi_esc = e(reference.doi) if use_doi else ""

            # Title pattern's logic
            title_words = title_esc.split()[:4]  # First 4 words
            title_pattern = ".*".join(word for word in title_words if len(word) > 3)

            # Query building

            if query_type == "year_and_doi":
                year_prev = max(year_int - 1, DEFAULT_YEAR_RANGE[0])
                year_next = min(year_int + 1, DEFAULT_YEAR_RANGE[1])
                
                return f"""
                {PREFIXES}
                {SELECT_ALL}
                WHERE {{
                    
                    ?doi_id literal:hasLiteralValue "{doi_esc}" .
                    ?doi_id datacite:usesIdentifierScheme datacite:doi .
                    ?br datacite:hasIdentifier ?doi_id .
                    BIND("{doi_esc}" AS ?doi)
                    
                    ?br prism:publicationDate ?publicationDate .
                    BIND(STR(?publicationDate) AS ?pub_date)
                    FILTER(
                        STRSTARTS(?pub_date, "{year_int}") ||
                        STRSTARTS(?pub_date, "{year_prev}") ||
                        STRSTARTS(?pub_date, "{year_next}")
                    )
                    
                    {OPTIONAL_TITLE}
                    {OPTIONAL_AUTHOR}
                    {OPTIONAL_VOLUME}
                    {OPTIONAL_PAGES}
                }}
                """

            elif query_type == "doi_title":
                return f"""
                {PREFIXES}
                {SELECT_ALL}
                WHERE {{
                    ?doi_id literal:hasLiteralValue "{doi_esc}" .
                    ?doi_id datacite:usesIdentifierScheme datacite:doi .
                    ?br datacite:hasIdentifier ?doi_id .
                    BIND("{doi_esc}" AS ?doi)

                    ?br dcterms:title ?title .
                    FILTER(REGEX(?title, "{title_pattern}", "i"))
                    
                    {OPTIONAL_PUB_DATE}
                    {OPTIONAL_AUTHOR}
                    {OPTIONAL_VOLUME}
                    {OPTIONAL_PAGES}
                }}
                """

            elif query_type == "author_title":
                return f"""
                {PREFIXES}
                {SELECT_ALL}
                WHERE {{
                    
                    ?first_author foaf:familyName "{fam_esc}" .
                    BIND("{fam_esc}" AS ?author_name)
                    
                    ?role pro:isHeldBy ?first_author .
                    ?br pro:isDocumentContextFor ?role .
                    
                    ?br dcterms:title ?title ;
                        prism:publicationDate ?publicationDate .
                    BIND(STR(?publicationDate) AS ?pub_date)

                    FILTER(REGEX(?title, "{title_pattern}", "i"))
                    
                    
                    {OPTIONAL_DOI}
                    {OPTIONAL_VOLUME}
                    {OPTIONAL_PAGES}
                }}
                """

            elif query_type == "year_volume_page":
                year_prev = max(year_int - 1, DEFAULT_YEAR_RANGE[0])
                year_next = min(year_int + 1, DEFAULT_YEAR_RANGE[1])
                
                return f"""
                {PREFIXES}
                {SELECT_ALL}
                WHERE {{
                    
                    ?volume fabio:hasSequenceIdentifier "{vol_esc}" .
                    BIND("{vol_esc}" AS ?volume_num)
                    ?issue frbr:partOf ?volume .
                    ?br frbr:partOf ?issue .
                    
                    ?br frbr:embodiment ?embodiment .
                    ?embodiment prism:startingPage ?start_page .
                    FILTER(STR(?start_page) = "{page_esc}") # Filtro pagina esatto
                    
                    ?br prism:publicationDate ?publicationDate .
                    BIND(STR(?publicationDate) AS ?pub_date)
                    FILTER(
                        STRSTARTS(?pub_date, "{year_int}") ||
                        STRSTARTS(?pub_date, "{year_prev}") ||
                        STRSTARTS(?pub_date, "{year_next}")
                    )

                    OPTIONAL {{ ?br dcterms:title ?title . }}
                    OPTIONAL {{ ?embodiment prism:endingPage ?end_page . }} # end_page aggiunto
                    {OPTIONAL_DOI}
                    {OPTIONAL_AUTHOR}
                }}
                """

            elif query_type == "year_author_page":
                year_prev = max(year_int - 1, DEFAULT_YEAR_RANGE[0])
                year_next = min(year_int + 1, DEFAULT_YEAR_RANGE[1])
                
                return f"""
                {PREFIXES}
                {SELECT_ALL}
                WHERE {{
                    
                    ?first_author foaf:familyName "{fam_esc}" .
                    BIND("{fam_esc}" AS ?author_name)
                    
                    ?role pro:isHeldBy ?first_author .
                    ?br pro:isDocumentContextFor ?role .
                    
                    ?br frbr:embodiment ?embodiment .
                    ?embodiment prism:startingPage ?start_page .
                    FILTER(
                        STR(?start_page) = "{page_esc}" ||
                        CONTAINS(STR(?start_page), "{page_esc}-") ||
                        CONTAINS(STR(?start_page), "-{page_esc}")
                    )
                    
                    ?br prism:publicationDate ?publicationDate .
                    BIND(STR(?publicationDate) AS ?pub_date)
                    FILTER(
                        STRSTARTS(?pub_date, "{year_int}") ||
                        STRSTARTS(?pub_date, "{year_prev}") ||
                        STRSTARTS(?pub_date, "{year_next}")
                    )
                                        
                    OPTIONAL {{ ?br dcterms:title ?title . }}
                    OPTIONAL {{ ?embodiment prism:endingPage ?end_page . }} # end_page aggiunto
                    {OPTIONAL_DOI}
                    {OPTIONAL_VOLUME}
                }}
                """
            elif query_type == "year_author_volume":                
                year_prev = max(year_int - 1, DEFAULT_YEAR_RANGE[0])
                year_next = min(year_int + 1, DEFAULT_YEAR_RANGE[1])
                
                return f"""
                {PREFIXES}
                {SELECT_ALL}
                WHERE {{
                    ?first_author foaf:familyName "{fam_esc}" .
                    BIND("{fam_esc}" AS ?author_name)
                    
                    ?role pro:isHeldBy ?first_author .
                    ?br pro:isDocumentContextFor ?role .
                    
                    ?volume fabio:hasSequenceIdentifier "{vol_esc}" .
                    BIND("{vol_esc}" AS ?volume_num)
                    ?issue frbr:partOf ?volume .
                    ?br frbr:partOf ?issue .
                    
                    ?br prism:publicationDate ?publicationDate .
                    BIND(STR(?publicationDate) AS ?pub_date)
                    FILTER(
                        STRSTARTS(?pub_date, "{year_int}") ||
                        STRSTARTS(?pub_date, "{year_prev}") ||
                        STRSTARTS(?pub_date, "{year_next}")
                    )
                    
                    {OPTIONAL_TITLE}
                    {OPTIONAL_DOI}
                    {OPTIONAL_PAGES}
                }}
                """
            
            logging.info(f"Unknown or unsupported query_type: {query_type}")
            return None

    def calculate_matching_score(self, reference: Reference, result: Dict) -> int:
        """
        Calculate matching score with configurable weights and optional stats tracking
        """
        
        logging.info(f"\n🎯 CALCULATING MATCH SCORE")
        logging.info(f"{'─'*60}")
        
        score = 0
        score_breakdown = []

        try:
            # DOI SCORING
            if reference.doi and 'doi' in result:
                ref_doi = reference.doi.lower().strip()
                result_doi = result['doi']['value'].lower().strip()

                if ref_doi == result_doi:
                    doi_score = SCORING_CONFIG['doi_exact']  # 15
                    score += doi_score
                    score_breakdown.append(f"DOI match: +{doi_score}")
                    logging.info(f"📎 DOI: EXACT MATCH → +{doi_score} points")  
            # AUTHOR SCORING
            if reference.first_author_lastname and 'author_name' in result:
                try:
                    result_author = self._normalize_author_name(result['author_name']['value'])
                    ref_author = self._normalize_author_name(reference.first_author_lastname)
                    
                    logging.info(f"👤 AUTHOR COMPARISON:")
                    logging.info(f"  Reference: '{reference.first_author_lastname}' → normalized: '{ref_author}'")
                    logging.info(f"  Result:    '{result['author_name']['value']}' → normalized: '{result_author}'")
                    
                    # EXACT MATCH ONLY
                    if ref_author == result_author:
                        author_score = SCORING_CONFIG['author_exact_match']
                        score += author_score
                        score_breakdown.append(f"Author exact match: +{author_score}")
                        logging.info(f"  ✅ EXACT MATCH → +{author_score} points")
                    else:
                        score_breakdown.append(f"Author mismatch: +0")
                        logging.info(f"  ❌ NO MATCH → +0 points")
                except Exception as e:
                    logging.info(f"  ⚠️ Error in author matching: {e}")
            else:
                if not reference.first_author_lastname:
                    logging.debug(f"👤 Author scoring skipped: No author in reference")
                elif 'author_name' not in result:
                    logging.debug(f"👤 Author scoring skipped: No author_name in result")

            # YEAR SCORING
            if reference.year and 'pub_date' in result:
                record_year_int = self._extract_year(reference.year)
                if record_year_int is not None:
                    try:
                        result_year = int(result['pub_date']['value'][:4])
                        
                        logging.info(f"📅 YEAR COMPARISON:")
                        logging.info(f"  Reference: {record_year_int}")
                        logging.info(f"  Result:    {result_year}")
                        
                        if record_year_int == result_year:
                            year_score = SCORING_CONFIG['year_exact']
                            score += year_score
                            score_breakdown.append(f"Year exact: +{year_score}")
                            logging.info(f"  ✅ EXACT → +{year_score} points")
                            

                        elif abs(record_year_int - result_year) == 1:
                            year_score = SCORING_CONFIG['year_adjacent']
                            score += year_score
                            score_breakdown.append(f"Year adjacent: +{year_score}")
                            logging.info(f"  ⚠️ ADJACENT (±1) → +{year_score} points")
                            
                        else:
                            score_breakdown.append(f"Year mismatch: +0")
                            logging.info(f"  ❌ MISMATCH → +0 points")
                    except (ValueError, IndexError) as e:
                        logging.info(f"  ⚠️ Year parsing error: {e}")

            # TITLE SCORING
            if 'title' in result:
                result_title = self._normalize_title(result['title']['value'])
                titles_to_check = [
                    reference.article_title,
                    reference.volume_title,
                    reference.journal_title
                ]
                
                logging.info(f"📰 TITLE COMPARISON:")
                logging.info(f"  Result title: '{result['title']['value'][:60]}...'")
                logging.info(f"  Result (normalized): '{result_title[:60]}...'")
                
                logging.info(f"\n  📚 Titles to check:")
                logging.info(f"    1. article_title: '{reference.article_title[:60] if reference.article_title else '(empty)'}...'")
                logging.info(f"    2. volume_title: '{reference.volume_title[:60] if reference.volume_title else '(empty)'}...'")
                logging.info(f"    3. journal_title: '{reference.journal_title[:60] if reference.journal_title else '(empty)'}...'")
                
                best_title_score = 0
                best_title_match = None
                
                for i, title in enumerate(titles_to_check):
                    if title:
                        record_title = self._normalize_title(title)
                        
                        logging.info(f"\n  🔍 Checking title #{i+1}:")
                        logging.info(f"    Raw: '{title[:50]}...'")
                        logging.info(f"    Normalized: '{record_title[:50]}...'")
                        
                        if record_title == result_title:
                            title_score = 100
                            logging.info(f"    ✅ EXACT STRING MATCH!")
                        else:
                            ratio = fuzz.ratio(record_title, result_title)
                            partial = fuzz.partial_ratio(record_title, result_title)
                            token_sort = fuzz.token_sort_ratio(record_title, result_title)
                            token_set = fuzz.token_set_ratio(record_title, result_title)
                            
                            title_score = max(ratio, partial, token_sort, token_set)
                            
                            logging.info(f"    Fuzzy scores:")
                            logging.info(f"      - ratio: {ratio}")
                            logging.info(f"      - partial_ratio: {partial}")
                            logging.info(f"      - token_sort_ratio: {token_sort}")
                            logging.info(f"      - token_set_ratio: {token_set}")
                            logging.info(f"      → MAX: {title_score}")
                        
                        if title_score > best_title_score:
                            best_title_score = title_score
                            best_title_match = title
                            logging.info(f"    🆕 NEW BEST SCORE: {title_score}")
                        else:
                            logging.info(f"    📊 Score {title_score} ≤ current best {best_title_score}")
                
                logging.info(f"  Best title score: {best_title_score} from '{best_title_match[:40] if best_title_match else 'N/A'}...'")
                
                # ASSIGN POINTS AND TRACK STATS
                if best_title_score == 100:
                    title_points = SCORING_CONFIG['title_exact']
                    score += title_points
                    score_breakdown.append(f"Title exact (100): +{title_points}")
                    logging.info(f"  ✅ EXACT (100) → +{title_points} points")
                    

                elif best_title_score > 95:
                    title_points = SCORING_CONFIG['title_95']
                    score += title_points
                    score_breakdown.append(f"Title 95+ ({best_title_score}): +{title_points}")
                    logging.info(f"  ✅ 95+ ({best_title_score}) → +{title_points} points")

                elif best_title_score > 90:
                    title_points = SCORING_CONFIG['title_90']
                    score += title_points
                    score_breakdown.append(f"Title 90+ ({best_title_score}): +{title_points}")
                    logging.info(f"  ✅ 90+ ({best_title_score}) → +{title_points} points")
                    
      
                elif best_title_score > 85:
                    title_points = SCORING_CONFIG['title_85']
                    score += title_points
                    score_breakdown.append(f"Title 85+ ({best_title_score}): +{title_points}")
                    logging.info(f"  ⚠️ 85+ ({best_title_score}) → +{title_points} points")
                    
   
                elif best_title_score > 80:
                    title_points = SCORING_CONFIG['title_80']
                    score += title_points
                    score_breakdown.append(f"Title 80+ ({best_title_score}): +{title_points}")
                    logging.info(f"  ⚠️ 80+ ({best_title_score}) → +{title_points} points")
                    
    
                elif best_title_score > 75:
                    title_points = SCORING_CONFIG['title_75']
                    score += title_points
                    score_breakdown.append(f"Title 75+ ({best_title_score}): +{title_points}")
                    logging.info(f"  ⚠️ 75+ ({best_title_score}) → +{title_points} points")
    
                else:
                    score_breakdown.append(f"Title too low ({best_title_score}): +0")
                    logging.info(f"  ❌ TOO LOW ({best_title_score}) → +0 points")

            # VOLUME SCORING
            if reference.volume and 'volume_num' in result:
                logging.info(f"📚 VOLUME COMPARISON:")
                logging.info(f"  Reference: '{reference.volume}'")
                logging.info(f"  Result:    '{result['volume_num']['value']}'")
                
                if reference.volume == result['volume_num']['value']:
                    vol_score = SCORING_CONFIG['volume_match']
                    score += vol_score
                    score_breakdown.append(f"Volume match: +{vol_score}")
                    logging.info(f"  ✅ MATCH → +{vol_score} points")
                    
                else:
                    score_breakdown.append(f"Volume mismatch: +0")
                    logging.info(f"  ❌ MISMATCH → +0 points")

            # PAGE SCORING
            if reference.first_page and 'start_page' in result:
                ref_page = reference.first_page.lstrip('0')
                result_page = result['start_page']['value'].lstrip('0')
                
                logging.info(f"📄 PAGE COMPARISON:")
                logging.info(f"  Reference: '{reference.first_page}' → normalized: '{ref_page}'")
                logging.info(f"  Result:    '{result['start_page']['value']}' → normalized: '{result_page}'")
                
                if ref_page == result_page:
                    page_score = SCORING_CONFIG['page_match']
                    score += page_score
                    score_breakdown.append(f"Page match: +{page_score}")
                    logging.info(f"  ✅ MATCH → +{page_score} points")
                    
                else:
                    score_breakdown.append(f"Page mismatch: +0")
                    logging.info(f"  ❌ MISMATCH → +0 points")

            # Final score summary
            logging.info(f"\n📊 SCORE BREAKDOWN:")
            for item in score_breakdown:
                logging.info(f"  • {item}")
            logging.info(f"{'─'*60}")
            logging.info(f"  🎯 TOTAL SCORE: {score}")
            logging.info(f"{'─'*60}\n")

        except Exception as e:
            logging.error(f"⚠️ Error calculating matching score: {e}")
            import traceback
            logging.error(traceback.format_exc())

        return score
    def _normalize_author_name(self, name: str) -> str:
        """Normalize author name for exact matching only"""
        if not name:
            return ""
        
        # Minimal normalization: lowercase, strip, remove extra spaces
        normalized = name.lower().strip()
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())  
        
        return normalized

class ReferenceProcessor:
    """Processor with async matcher handling"""

    def __init__(self, use_grobid: bool = False, grobid_config: Optional[str] = None,
                 endpoint: str = None, config: MatcherConfig = None):
        self.matcher_endpoint = endpoint or "https://sparql-stg.opencitations.net/meta"
        self.matcher_config = config or DEFAULT_CONFIG
        
        self.use_grobid = use_grobid
        self.grobid_config = grobid_config
        
        # Validate Grobid availability at init time
        if self.use_grobid:
            try:
                # Test initialization
                _ = GrobidProcessor(grobid_config)  
            except (FileNotFoundError, RuntimeError) as e:
                logging.error(f"Failed to initialize Grobid: {e}")
                self.use_grobid = False
        self.grobid_fallback_count = 0
        self._count_lock = asyncio.Lock()  
        
    def _extract_author(self, ref_data: dict) -> str:
        """Extract first author lastname"""
        author_lastname = ""
        if ref_data.get('author'):
            authors = ref_data['author']
            
            if isinstance(authors, str):
                author_str = authors.strip()
                author_lastname = author_str.split(',')[0].strip() if ',' in author_str else \
                                author_str.split()[-1].strip() if author_str else ""
            
            elif isinstance(authors, list) and authors:
                first = authors[0]
                if isinstance(first, dict):
                    author_lastname = (first.get('family') or 
                                    first.get('family-name') or 
                                    first.get('name', '').split(',')[0].strip())
                elif isinstance(first, str):
                    author_lastname = first.split(',')[0].strip()
        
        return author_lastname.strip()

    def _extract_title(self, ref_data: dict) -> str:
        """Extract article title"""
        title = ref_data.get('article-title', '')
        if not title and 'title' in ref_data:
            title_data = ref_data['title']
            title = title_data[0] if isinstance(title_data, list) and title_data else \
                    title_data if isinstance(title_data, str) else ""
        return title

    def _extract_year_from_data(self, ref_data: dict) -> str:
        """Extract year"""
        year = ref_data.get('year', '')
        if not year and 'issued' in ref_data:
            try:
                date_parts = ref_data['issued'].get('date-parts', [])
                year = str(date_parts[0][0]) if date_parts and date_parts[0] else ""
            except (KeyError, IndexError, TypeError):
                pass
        return year

    def _extract_page(self, ref_data: dict) -> str:
        """Extract first page"""
        page = ref_data.get('first-page', '')
        if not page and 'page' in ref_data:
            page_str = ref_data['page']
            page = page_str.split('-')[0].strip() if isinstance(page_str, str) else ""
        return page
    @staticmethod
    def _normalize_author_name(name: str) -> str:
        """Normalize author name for exact matching only"""
        if not name:
            return ""
        normalized = name.lower().strip()
        normalized = ' '.join(normalized.split())
        return normalized

    @staticmethod
    def _normalize_title(title: str) -> str:
        """Normalize title for fuzzy matching"""
        return normalize_for_fuzzy_title(title)

    @staticmethod
    def _extract_year(year_str: str) -> Optional[int]:
        """Extract year from string with validation"""
        if not year_str:
            return None
        try:
            year = int(year_str)
            if DEFAULT_YEAR_RANGE[0] <= year <= DEFAULT_YEAR_RANGE[1]:
                return year
            return None
        except ValueError:
            year_match = re.search(r'\b(17|18|19|20)\d{2}\b', str(year_str))
            if year_match:
                year = int(year_match.group())
                if DEFAULT_YEAR_RANGE[0] <= year <= DEFAULT_YEAR_RANGE[1]:
                    return year
            return None

    @property
    def grobid_processor(self):
        """Get Grobid processor instance"""
        if not self.use_grobid:
            return None
        
        if not hasattr(self, '_grobid_instance'):
            try:
                self._grobid_instance = GrobidProcessor(self.grobid_config)
            except (FileNotFoundError, RuntimeError) as e:
                logging.error(f"Failed to initialize Grobid: {e}")
                return None
        
        return self._grobid_instance
    
    async def process_reference(self, ref: Reference, threshold: int = 26, use_doi: bool = True, stats: Dict = None, stats_lock: asyncio.Lock = None) -> Optional[Dict]:
        """
        Process reference with async operations and optional stats tracking
        
        Args:
            ref: Reference object to process
            threshold: Minimum score threshold for a match
            use_doi: Whether to use DOI in queries
            stats: Optional stats dict for tracking field contributions
        
        Returns:
            Optional[Dict]: Match result or None
        """
        async with OpenCitationsMatcherThreadSafe(
            endpoint=self.matcher_endpoint,
            config=self.matcher_config
        ) as matcher:
            try:
                # Create working copy
                processed_ref = Reference(**{k: v for k, v in ref.__dict__.items()})
                
                # Normalize (DOI will be handled in query building)
                processed_ref = normalize_reference_safe(processed_ref)

                async def run_sparql_matching_loop(reference_obj: Reference, stop_threshold: int, use_doi: bool = True) -> Tuple[Optional[Dict], int]:


                    best_score = 0
                    best_match = None
                    query_types = []
                        
                    logging.info(f"\n{'='*70}")
                    logging.info(f"🔄 STARTING SPARQL MATCHING LOOP")
                    logging.info(f"{'='*70}")
                    logging.info(f"Stop threshold: {stop_threshold}")
                    logging.info(f"🏴 DEBUG: use_doi parameter = {use_doi}")
                    logging.info(f"🏴 DEBUG: reference has DOI = {bool(reference_obj.doi)}")
                    logging.info(f"🏴 DEBUG: DOI value = {reference_obj.doi if reference_obj.doi else 'None'}")
                    if reference_obj.doi and use_doi:
                        logging.info(f"✅ Adding DOI-based queries (use_doi=True, DOI exists)")
                        query_types.extend(["year_and_doi", "doi_title"])
                    else:
                        logging.info(f"❌ Skipping DOI-based queries (use_doi={use_doi}, has_doi={bool(reference_obj.doi)})")

                    query_types.extend([
                        "author_title",
                        "year_author_page",
                        "year_volume_page",
                        "year_author_volume"
                    ])
                    
                    logging.info(f"📋 Query sequence: {' → '.join(query_types)}")
                    logging.info(f"{'='*70}\n")

                    for idx, query_type in enumerate(query_types, 1):
                        logging.info(f"\n{'▼'*70}")
                        logging.info(f"Query {idx}/{len(query_types)}: {query_type.upper()}")
                        logging.info(f"{'▼'*70}")
                        
                        try:
                            query = matcher.build_sparql_query(reference_obj, query_type, use_doi)  
                            if not query:
                                logging.info(f"⚠️ Query {query_type} SKIPPED (missing required fields)")
                                continue

                            results = await matcher.query_opencitations(query, query_type)  
                                
                            if not results:
                                logging.info(f"ℹ️ Query {query_type} returned 0 results")
                                continue
                            
                            logging.info(f"✅ Processing {len(results)} results from {query_type}...")

                            for res_idx, result in enumerate(results, 1):
                                logging.info(f"\n  → Evaluating result {res_idx}/{len(results)}...")
                                
                                score = matcher.calculate_matching_score(reference_obj, result)                                
                                if score > best_score:
                                    logging.info(f"  🆕 NEW BEST SCORE: {score} (previous: {best_score})")
                                    best_score = score
                                    best_match = result
                                    best_match["score"] = score
                                    best_match["query_type"] = query_type
                                else:
                                    logging.info(f"  📊 Score {score} ≤ current best {best_score} (skipped)")

                                # Early stop
                                if best_score >= stop_threshold:
                                    logging.info(f"\n{'✓'*70}")
                                    logging.info(f"🎉 EARLY STOP: Score {best_score} ≥ threshold {stop_threshold}")
                                    logging.info(f"   Query type: {query_type}")
                                    logging.info(f"   Matched DOI: {result.get('doi', {}).get('value', 'N/A')}")
                                    logging.info(f"{'✓'*70}\n")
                                    return best_match, best_score    

                        except (RateLimitError, ServerError) as e:
                            logging.error(f"❌ FATAL ERROR in {query_type}: {e}")
                            raise
                        except QueryExecutionError as e:
                            logging.info(f"⚠️ Query execution error in {query_type}: {e}")
                            continue
                        except Exception as e:
                            logging.info(f"⚠️ Unexpected error in {query_type}: {e}")
                            import traceback
                            logging.debug(traceback.format_exc())
                            continue

                    logging.info(f"\n{'='*70}")
                    logging.info(f"🏁 LOOP COMPLETE")
                    logging.info(f"  Best score: {best_score}")
                    logging.info(f"  Best match: {'Found' if best_match else 'None'}")
                    if best_match:
                        logging.info(f"  Query type: {best_match.get('query_type', 'N/A')}")
                    logging.info(f"{'='*70}\n")
                    
                    return best_match, best_score
            
                # Initialize ALL variables at the start
                best_match2, best_score2 = None, 0
                best_match3, best_score3 = None, 0
                
                # First attempt
                best_match, best_score = await run_sparql_matching_loop(processed_ref, threshold, use_doi=use_doi)

                # Apply threshold adjustment if score is close (>= 90%)
                adjusted_threshold = apply_threshold_adjustment(best_score, threshold)

                if best_score >= adjusted_threshold:
                    logging.info(
                        f"Match found with SPARQL (score: {best_score}, "
                        f"threshold: {adjusted_threshold}{' (adjusted)' if adjusted_threshold < threshold else ''})"
                    )
                    return best_match

                # Grobid fallback (if enabled and unstructured text available)
                if self.use_grobid and self.grobid_processor and processed_ref.unstructured:
                    logging.info("\n" + "="*70)
                    logging.info(f"🔧 GROBID FALLBACK ATTEMPT")
                    logging.info("="*70)
                    logging.info(f"📝 Unstructured text:")
                    logging.info(f"   '{processed_ref.unstructured[:150]}...'")
                    logging.info(f"📊 Current reference state BEFORE Grobid:")
                    logging.info(f"   Year: {processed_ref.year or '(empty)'}")
                    logging.info(f"   Author: {processed_ref.first_author_lastname or '(empty)'}")
                    logging.info(f"   Title: {processed_ref.get_main_title()[:50] or '(empty)'}...")
                    logging.info(f"   Volume: {processed_ref.volume or '(empty)'}")
                    logging.info(f"   Page: {processed_ref.first_page or '(empty)'}")

                    try:
                        grobid_ref = self.grobid_processor.process_unstructured_reference(processed_ref.unstructured)
                        
                        if grobid_ref:
                            # Increment global counter
                            async with self._count_lock:
                                self.grobid_fallback_count += 1

                            if stats and stats_lock:
                                async with stats_lock:
                                    stats['grobid_fallbacks'] += 1

                            logging.info(f"✅ Grobid extracted fields:")
                            logging.info(f"   Year: {grobid_ref.year or '(empty)'}")
                            logging.info(f"   Author: {grobid_ref.first_author_lastname or '(empty)'}")
                            logging.info(f"   Title: {grobid_ref.article_title[:50] or '(empty)'}...")
                            logging.info(f"   Volume: {grobid_ref.volume or '(empty)'}")
                            logging.info(f"   Page: {grobid_ref.first_page or '(empty)'}")

                            # Validate year before merging
                            suspicious_year = False
                            if grobid_ref.year:
                                try:
                                    year_int = int(grobid_ref.year)
                                    current_year = datetime.now().year
                                    if year_int < 1700 or year_int > current_year + 1:
                                        logging.info(f"⚠️ Grobid extracted suspicious year: {year_int}")
                                        logging.info(f"   Expected range: 1700-{current_year + 1}")
                                        suspicious_year = True
                                except ValueError:
                                    logging.info(f"⚠️ Grobid extracted invalid year format: '{grobid_ref.year}'")
                                    suspicious_year = True

                            self._merge_reference_data(processed_ref, grobid_ref, use_doi)
                            
                            logging.info(f"📊 Reference state AFTER merge:")
                            logging.info(f"   Year: {processed_ref.year or '(empty)'}")
                            logging.info(f"   Author: {processed_ref.first_author_lastname or '(empty)'}")
                            logging.info(f"   Title: {processed_ref.get_main_title()[:50] or '(empty)'}...")
                            logging.info(f"   Volume: {processed_ref.volume or '(empty)'}")
                            logging.info(f"   Page: {processed_ref.first_page or '(empty)'}")
                            
                            best_match2, best_score2 = await run_sparql_matching_loop(processed_ref, threshold, use_doi=use_doi)
                            
                            adjusted_threshold2 = apply_threshold_adjustment(best_score2, threshold)

                            if best_score2 >= adjusted_threshold2:
                                if stats and stats_lock:
                                    async with stats_lock:
                                        stats['grobid_successes'] += 1
                                logging.info(
                                    f"Match found after Grobid enrichment (score: {best_score2}, "
                                    f"threshold: {adjusted_threshold2}{' (adjusted)' if adjusted_threshold2 < threshold else ''})"
                                )
                                return best_match2

                        else:
                            logging.info("⚠️ Grobid failed to extract reference")

                    except Exception as e:
                        logging.info(f"⚠️ Grobid processing error: {e}")
                else:
                    # Explain why Grobid was NOT used
                    reasons = []
                    if not self.use_grobid:
                        reasons.append("Grobid not enabled (use --use-grobid flag)")
                    if not self.grobid_processor:
                        reasons.append("Grobid processor failed to initialize")
                    if not processed_ref.unstructured:
                        reasons.append("No unstructured text available")

                    if reasons:
                        logging.info(f"\nℹ️ Grobid fallback skipped: {'; '.join(reasons)}")

                # Partial match without year (relaxed attempt) - OUTSIDE Grobid block
                if processed_ref.year:
                    logging.info("\n" + "="*70)
                    logging.info(f"🔄 TRYING WITHOUT YEAR")
                    logging.info("="*70)

                    processed_ref_no_year = Reference(**{k: v for k, v in processed_ref.__dict__.items()})
                    processed_ref_no_year.year = ""

                    best_match3, best_score3 = await run_sparql_matching_loop(processed_ref_no_year, threshold, use_doi=use_doi)

                    if best_match3 and best_score3 >= threshold * 0.9:
                        best_match3['score'] = best_score3
                        logging.info(f"Partial match without year (score={best_score3} < threshold)")
                        return best_match3

                logging.info("No match found after all attempts")
                all_scores = [s for s in [best_score, best_score2, best_score3] if s is not None and s > 0]
                highest_score = max(all_scores) if all_scores else None
                
                return {
                    'below_threshold': True,
                    'score': highest_score,
                    'score_original': best_score,
                    'score_grobid': best_score2 if best_score2 > 0 else None,
                    'score_no_year': best_score3 if best_score3 > 0 else None,
                    'grobid_attempted': best_score2 > 0,
                    'no_year_attempted': best_score3 > 0
                }

            except Exception as e:
                logging.error(f"Error processing reference: {e}")
                return None
            
    def _merge_reference_data(self, original_ref: Reference, grobid_ref: Reference, use_doi: bool = True):
        """Merge reference data from Grobid into original reference"""
        
        # Fields to merge ONLY if original is empty
        merge_if_empty = [
            ('year', 'year'),
            ('first_author_lastname', 'first_author_lastname'),
            ('volume', 'volume'),
        ]
        
        # Fields to PREFER from Grobid if available (more accurate)
        prefer_grobid = [
            ('article_title', 'article_title'),  # ← Grobid is very accurate 
        ]
        
        # Fields to KEEP original (Crossref più affidabile)
        keep_original = [
            ('first_page', 'first_page'),  # ← Crossref page è più affidabile
        ]
        
        merged_fields = []
        
        # Merge if empty
        for orig_field, grobid_field in merge_if_empty:
            orig_value = getattr(original_ref, orig_field, "")
            grobid_value = getattr(grobid_ref, grobid_field, "")
            
            if not orig_value and grobid_value:
                setattr(original_ref, orig_field, grobid_value)
                merged_fields.append(orig_field)
        
        # Prefer Grobid
        for orig_field, grobid_field in prefer_grobid:
            orig_value = getattr(original_ref, orig_field, "")
            grobid_value = getattr(grobid_ref, grobid_field, "")
            
            if not orig_value and grobid_value: 
                setattr(original_ref, orig_field, grobid_value)
                merged_fields.append(orig_field)
        
        if not original_ref.doi and grobid_ref.doi:
                original_ref.doi = grobid_ref.doi
                merged_fields.append('doi')
        
        if merged_fields:
            logging.debug(f"Merged fields: {', '.join(merged_fields)}")
            
    def _save_unmatched_references(self, unmatched_refs: List[Tuple[str, Reference, Optional[Dict]]], 
                                    output_file: str):
        """Save unmatched references with their metadata to CSV"""
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'reference_id',
                    'year',
                    'volume',
                    'first_page',
                    'first_author_lastname',
                    'article_title',
                    'volume_title',
                    'journal_title',
                    'doi',
                    'unstructured',
                    'best_score',
                    'score_original',           
                    'score_after_grobid',       
                    'score_without_year',       
                    'grobid_attempted',         
                    'threshold_failed'
                ])
                writer.writeheader()
                
                for ref_id, ref, score_info in unmatched_refs:
                    # Extract scores
                    if isinstance(score_info, dict):
                        best_score = score_info.get('score', 'N/A')
                        score_original = score_info.get('score_original', 'N/A')
                        score_grobid = score_info.get('score_grobid', 'N/A')
                        score_no_year = score_info.get('score_no_year', 'N/A')
                        grobid_attempted = 'Yes' if score_info.get('grobid_attempted') else 'No'
                    elif isinstance(score_info, (int, float)):
                        best_score = score_info
                        score_original = score_info
                        score_grobid = 'N/A'
                        score_no_year = 'N/A'
                        grobid_attempted = 'No'
                    else:
                        best_score = 'N/A'
                        score_original = 'N/A'
                        score_grobid = 'N/A'
                        score_no_year = 'N/A'
                        grobid_attempted = 'No'
                    
                    writer.writerow({
                        'reference_id': ref_id,
                        'year': ref.year or '',
                        'volume': ref.volume or '',
                        'first_page': ref.first_page or '',
                        'first_author_lastname': ref.first_author_lastname or '',
                        'article_title': ref.article_title or '',
                        'volume_title': ref.volume_title or '',
                        'journal_title': ref.journal_title or '',
                        'doi': ref.doi or '',
                        'unstructured': ref.unstructured or '',
                        'best_score': best_score,
                        'score_original': score_original,
                        'score_after_grobid': score_grobid,
                        'score_without_year': score_no_year,
                        'grobid_attempted': grobid_attempted,
                        'threshold_failed': 'Yes' if best_score != 'N/A' else 'No'
                    })
            
            logging.info(f"Unmatched references saved to: {output_file}")
        except Exception as e:
            logging.error(f"Error saving unmatched references: {e}")
            
    async def process_file(self, input_file: str, output_file: str, threshold: int = 26, use_doi: bool = True):
            """FIXED: Process file with proper error handling and use_doi parameter"""
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Input file does not exist: {input_file}")
                
            _, extension = os.path.splitext(input_file)

            try:
                if extension.lower() == '.json':
                    await self._process_crossref_file(input_file, output_file, threshold, use_doi)
                elif extension.lower() == '.xml':
                    await self._process_tei_file(input_file, output_file, threshold, use_doi)
                else:
                    raise ValueError(f"Unsupported file format: {extension}")
            except Exception as e:
                logging.error(f"Error processing file {input_file}: {e}")
                raise

    async def _process_crossref_file(self, input_file: str, output_file: str, threshold: int, use_doi: bool = True):        
        """Enhanced Crossref processing with comprehensive field tracking"""
        def clean_reference_text(text: str) -> str:
            """Fix common encoding issues in reference text"""
            replacements = {
                'âˆ†': 'Δ',  # Delta
                'Î"': 'Δ',   # Another common Delta corruption
                'â€"': '–',  # En-dash
                'â€"': '—',  # Em-dash
                "â€˜": "'",  # Left single quote
                "â€™": "'",  # Right single quote
                'â€œ': '"',  # Left double quote
                'â€': '"',   # Right double quote
                'Â': '',     # Non-breaking space artifact
            }
            
            for bad, good in replacements.items():
                text = text.replace(bad, good)
            
            return text
        
        # Initialize comprehensive stats with ALL fields
        stats = {
            'total_references': 0, 
            'matches_found': 0, 
            'errors': 0,
            'query_types': {},
            
            'doi_matches': 0,
            # Author stats
            'refs_with_author': 0,
            'author_exact_matches': 0,
            
            # Title stats
            'refs_with_title': 0,
            'title_exact_matches': 0,
            'title_fuzzy_matches': 0,
            
            # Year stats
            'refs_with_year': 0,
            'year_exact_matches': 0,
            'year_adjacent_matches': 0,
            
            # Volume stats
            'refs_with_volume': 0,
            'volume_matches': 0,
            
            # Page stats
            'refs_with_page': 0,
            'page_matches': 0,
            
            # DOI stats
            'refs_with_doi': 0,
            'doi_matches': 0,
            
            # Grobid stats
            'grobid_fallbacks': 0,
            'grobid_successes': 0
        }
        
        stats_lock = asyncio.Lock()
        unmatched_lock = asyncio.Lock()
        reference_semaphore = asyncio.Semaphore(10)
        async def process_single_reference(ref_data, index):
            """Process one reference concurrently"""
            async with reference_semaphore:
                try:
                    async with stats_lock:
                        stats['total_references'] += 1
                    
                    logging.info(f"\n{'='*70}")
                    logging.info(f"📄 Processing Reference #{index}/{total_refs}")
                    logging.info(f"{'='*70}")
                    
                    # Extract fields using helper methods
                    author_lastname = self._extract_author(ref_data)
                    article_title = self._extract_title(ref_data)
                    year_val = self._extract_year_from_data(ref_data)
                    volume_val = ref_data.get('volume', '')
                    page_val = self._extract_page(ref_data)
                    doi_val = ref_data.get('DOI', '')
                    unstructured = ref_data.get('unstructured', '')
                    
                    ref = Reference(
                        year=str(year_val) if year_val else "",
                        first_author_lastname=author_lastname,
                        article_title=article_title,
                        volume=volume_val,
                        first_page=page_val,
                        doi=doi_val,
                        unstructured=unstructured
                    )
                    
                    # Track fields (thread-safe)
                    async with stats_lock:
                        if ref.first_author_lastname:
                            stats['refs_with_author'] += 1
                        if ref.get_main_title():
                            stats['refs_with_title'] += 1
                        if ref.doi:
                            stats['refs_with_doi'] += 1
                        if ref.year:
                            stats['refs_with_year'] += 1
                        if ref.volume:
                            stats['refs_with_volume'] += 1
                        if ref.first_page:
                            stats['refs_with_page'] += 1
                    
                    # Process (concurrent with other refs)
                    match = await self.process_reference(ref, threshold, use_doi, stats, stats_lock)
                    if match and not match.get('below_threshold', False):
                        logging.info(f"\n{'✓'*70}")
                        logging.info(f"🎉 MATCH FOUND!")
                        logging.info(f"  Reference ID: ref_{index}")
                        logging.info(f"  Query Type: {match.get('query_type', 'unknown')}")
                        logging.info(f"  Score: {match.get('score', 0)}")
                        logging.info(f"{'✓'*70}\n")
                    else:
                        logging.info(f"\n{'✗'*70}")
                        logging.info(f"❌ NO MATCH FOUND")
                        logging.info(f"  Reference ID: ref_{index}")
                        logging.info(f"{'✗'*70}\n")
                    return {
                        'index': index,
                        'ref_id': f'ref_{index}',
                        'ref': ref,
                        'match': match,
                        'success': True
                    }
                    
                except Exception as e:
                    async with stats_lock:
                        stats['errors'] += 1
                    logging.error(f"Error processing {index}: {e}")
                    return {'index': index, 'success': False, 'error': str(e)}
        unmatched_refs = []
        
        # File reading with multiple encoding attempts
        data = None
        encodings = ['utf-8', 'utf-8-sig', 'cp1252', 'latin-1']
        
        logging.info(f"\n{'='*70}")
        logging.info(f"📂 PROCESSING CROSSREF FILE")
        logging.info(f"{'='*70}")
        logging.info(f"Input file: {os.path.basename(input_file)}")
        logging.info(f"Output file: {os.path.basename(output_file)}")
        logging.info(f"Threshold: {threshold}")
        logging.info(f"Use DOI: {use_doi}")
        logging.info(f"{'='*70}\n")
        
        for encoding in encodings:
            try:
                with open(input_file, 'r', encoding=encoding) as f:
                    data = json.load(f)
                logging.info(f"✅ Successfully read file with {encoding} encoding")
                break
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                if encoding == encodings[-1]:
                    logging.error(f"❌ Failed to read file with all encodings: {e}")
                    raise
                continue
        
        if not data:
            raise ValueError("Unable to load JSON data from file")

        # Validate JSON structure
        if 'message' not in data:
            logging.error(f"❌ Invalid JSON structure: missing 'message' key")
            logging.debug(f"Available keys: {list(data.keys())}")
            raise ValueError("Invalid Crossref JSON structure")

        # Check for 0 references
        ref_count = data['message'].get('reference-count', 0)

        if ref_count == 0 or 'reference' not in data['message']:
            logging.info(f"⚠️ File has 0 references (reference-count: {ref_count})")
            
            title = data['message'].get('title', ['Unknown'])
            if isinstance(title, list) and title:
                title = title[0]
            logging.info(f"📄 Paper: {str(title)[:80]}...")
            logging.info(f"🔗 DOI: {data['message'].get('DOI', 'Unknown')}")
            
            # Create empty output files
            os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
            
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                    'reference_id', 'article_title', 'matched_title',
                    'score', 'matched_doi', 'meta_id', 'query_type'
                ])
                writer.writeheader()
            
            stats_file = os.path.splitext(output_file)[0] + '_stats.txt'
            self._save_stats_file(stats, stats_file)
            
            unmatched_file = os.path.splitext(output_file)[0] + '_unmatched.csv'
            with open(unmatched_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'reference_id', 'year', 'volume', 'first_page',
                    'first_author_lastname', 'article_title', 'volume_title',
                    'journal_title', 'doi', 'unstructured', 'best_score',
                    'threshold_failed'
                ])
            
            logging.info(f"✅ Created empty output files (0 references)")
            return
        
        total_refs = len(data['message']['reference'])
        logging.info(f"📊 Found {total_refs} references in file\n")

        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

        try:
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

                # Create concurrent tasks
                total_refs = len(data['message']['reference'])
                logging.info(f"Creating {total_refs} concurrent tasks")

                tasks = []
                for i, ref_data in enumerate(data['message']['reference'], 1):
                    task = process_single_reference(ref_data, i)
                    tasks.append(task)

                # Process concurrently
                logging.info(f"Starting concurrent processing")
                results = []

                for coro in asyncio.as_completed(tasks):
                    result = await coro
                    results.append(result)
                    if len(results) % 10 == 0:
                        logging.info(f"Progress: {len(results)}/{total_refs}")

                # Sort by index
                results.sort(key=lambda x: x['index'])

                # Write results
                with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(
                        csvfile,
                        fieldnames=['reference_id', 'article_title', 'matched_title',
                                'score', 'matched_doi', 'meta_id', 'query_type']
                    )
                    writer.writeheader()
                    
                    for result in results:
                        if not result['success']:
                            continue
                        
                        match = result['match']
                        
                        if match and not match.get('below_threshold', False):
                            async with stats_lock:
                                stats['matches_found'] += 1
                                query_type = match.get('query_type', 'unknown')
                                stats['query_types'][query_type] = stats['query_types'].get(query_type, 0) + 1
                                self._track_final_match_stats(match, result['ref'], stats)
                            
                            writer.writerow({
                                'reference_id': result['ref_id'],
                                # 'article_title': result['ref'].article_title or 'N/A',
                                'article_title': result['ref'].get_main_title() or 'N/A',
                                'matched_title': match.get('title', {}).get('value', 'N/A'),
                                'score': match.get('score', 0),
                                'matched_doi': match.get('doi', {}).get('value', 'N/A'),
                                'meta_id': match.get('br', {}).get('value', 'N/A'),
                                'query_type': match.get('query_type', 'unknown')
                            })
                        else:
                            best_score = match.get('score') if match else None
                            async with unmatched_lock:
                                unmatched_refs.append((result['ref_id'], result['ref'], best_score))

            # Save unmatched references
            if unmatched_refs:
                unmatched_file = os.path.splitext(output_file)[0] + '_unmatched.csv'
                self._save_unmatched_references(unmatched_refs, unmatched_file)

            # Print and save stats
            logging.info(f"\n{'='*70}")
            logging.info(f"📊 FINAL STATISTICS")
            logging.info(f"{'='*70}")
            
            percentage = self._print_stats(stats, input_file)
            
            logging.info(f"\n📈 Detailed Breakdown:")
            logging.info(f"  References with author: {stats['refs_with_author']}/{stats['total_references']}")
            logging.info(f"  References with title: {stats['refs_with_title']}/{stats['total_references']}")
            logging.info(f"  References with DOI: {stats['refs_with_doi']}/{stats['total_references']}")
            logging.info(f"  References with year: {stats['refs_with_year']}/{stats['total_references']}")
            logging.info(f"  References with volume: {stats['refs_with_volume']}/{stats['total_references']}")
            logging.info(f"  References with page: {stats['refs_with_page']}/{stats['total_references']}")

            
            if stats['query_types']:
                logging.info(f"\n🔍 Query Type Distribution:")
                total_matches = sum(stats['query_types'].values())
                for qtype, count in sorted(stats['query_types'].items(), key=lambda x: x[1], reverse=True):
                    percentage_qtype = (count / total_matches * 100) if total_matches > 0 else 0
                    logging.info(f"  {qtype}: {count} ({percentage_qtype:.1f}%)")
            
            logging.info(f"{'='*70}\n")
            
            if stats['matches_found'] > 0:
                print(f"\n  ✓ {stats['matches_found']}/{stats['total_references']} matches " +
                    f"({percentage:.1f}%)" + 
                    (f" | {stats['errors']} errors" if stats['errors'] > 0 else ""))
            
            # Save stats file
            stats_file = os.path.splitext(output_file)[0] + '_stats.txt'
            self._save_stats_file(stats, stats_file)
            
        except Exception as e:
            logging.error(f"❌ Error writing output file {output_file}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            raise

    async def _process_tei_file(self, input_file: str, output_file: str, threshold: int, use_doi: bool = True):
        """Enhanced TEI processing with comprehensive field tracking"""
        
        # Initialize comprehensive stats with ALL fields (same as Crossref)
        stats = {
            'total_references': 0,
            'matches_found': 0,
            'errors': 0,
            'query_types': {},
            
            # Author stats
            'refs_with_author': 0,
            'author_exact_matches': 0,
            
            # Title stats
            'refs_with_title': 0,
            'title_exact_matches': 0,
            'title_fuzzy_matches': 0,
            
            # Year stats
            'refs_with_year': 0,
            'year_exact_matches': 0,
            'year_adjacent_matches': 0,
            
            # Volume stats
            'refs_with_volume': 0,
            'volume_matches': 0,
            
            # Page stats
            'refs_with_page': 0,
            'page_matches': 0,
            
            # DOI stats
            'refs_with_doi': 0,
            'doi_matches': 0
        }
        stats_lock = asyncio.Lock()
        unmatched_lock = asyncio.Lock()
        unmatched_refs = []
        reference_semaphore = asyncio.Semaphore(10)
        
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

        logging.info(f"\n{'='*70}")
        logging.info(f"📂 PROCESSING TEI FILE")
        logging.info(f"{'='*70}")
        logging.info(f"Input file: {os.path.basename(input_file)}")
        logging.info(f"Output file: {os.path.basename(output_file)}")
        logging.info(f"Threshold: {threshold}")
        logging.info(f"{'='*70}\n")

        try:
            tree = ET.parse(input_file)
            root = tree.getroot()
        except ET.ParseError as e:
            logging.error(f"XML parsing error for {input_file}: {e}")
            raise
        except Exception as e:
            logging.error(f"Error reading TEI file {input_file}: {e}")
            raise

        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        
        # Get bibl_structs and total_refs BEFORE defining the function
        bibl_structs = root.findall('.//tei:biblStruct', ns)
        total_refs = len(bibl_structs)
        logging.info(f"📊 Found {total_refs} biblStruct elements\n")
        
        async def process_single_bibl(bibl, index):
            """Process one TEI reference concurrently"""
            async with reference_semaphore:
                try:
                    async with stats_lock:
                        stats['total_references'] += 1
                    
                    logging.info(f"\n{'='*70}")
                    logging.info(f"📄 Processing Reference #{index}/{total_refs}")
                    logging.info(f"{'='*70}")
                    
                    ref = self._parse_bibl_struct(bibl, ns)
                    if not ref:
                        logging.info(f"⚠️ Could not parse biblStruct")
                        return {'index': index, 'success': False}
                    
                    # Track fields
                    async with stats_lock:
                        if ref.first_author_lastname:
                            stats['refs_with_author'] += 1
                        if ref.get_main_title():
                            stats['refs_with_title'] += 1
                        if ref.year:
                            stats['refs_with_year'] += 1
                        if ref.volume:
                            stats['refs_with_volume'] += 1
                        if ref.first_page:
                            stats['refs_with_page'] += 1
                        if ref.doi:
                            stats['refs_with_doi'] += 1
                    
                    # Process
                    match = await self.process_reference(ref, threshold, use_doi, stats, stats_lock)
                    
                    ref_id = bibl.get('{http://www.w3.org/XML/1998/namespace}id', f'ref_{index}')
                    
                    if match and not match.get('below_threshold', False):
                        logging.info(f"\n{'✓'*70}")
                        logging.info(f"🎉 MATCH FOUND!")
                        logging.info(f"  Reference ID: {ref_id}")
                        logging.info(f"  Query Type: {match.get('query_type', 'unknown')}")
                        logging.info(f"  Score: {match.get('score', 0)}")
                        logging.info(f"{'✓'*70}\n")
                    else:
                        logging.info(f"\n{'✗'*70}")
                        logging.info(f"❌ NO MATCH FOUND")
                        logging.info(f"  Reference ID: {ref_id}")
                        logging.info(f"{'✗'*70}\n")
                    
                    return {
                        'index': index,
                        'ref_id': ref_id,
                        'ref': ref,
                        'match': match,
                        'success': True
                    }
                    
                except Exception as e:
                    async with stats_lock:
                        stats['errors'] += 1
                    logging.error(f"Error processing {index}: {e}")
                    import traceback
                    logging.debug(traceback.format_exc())
                    return {'index': index, 'success': False}

        try:
            # Create concurrent tasks
            tasks = []
            for i, bibl in enumerate(bibl_structs, 1):
                task = process_single_bibl(bibl, i)
                tasks.append(task)

            # Process concurrently
            logging.info(f"🚀 Starting concurrent processing of {len(tasks)} references")
            results = []

            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                if len(results) % 10 == 0:
                    logging.info(f"Progress: {len(results)}/{total_refs}")

            # Sort by index
            results.sort(key=lambda x: x['index'])
            
            # WRITE RESULTS TO CSV
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
                
                for result in results:
                    if not result['success']:
                        continue
                    
                    match = result['match']
                    
                    if match and not match.get('below_threshold', False):
                        # Update stats
                        async with stats_lock:
                            stats['matches_found'] += 1
                            query_type = match.get('query_type', 'unknown')
                            stats['query_types'][query_type] = stats['query_types'].get(query_type, 0) + 1
                            self._track_final_match_stats(match, result['ref'], stats)
                        
                        # Write row
                        writer.writerow({
                            'reference_id': result['ref_id'],
                            'article_title': result['ref'].get_main_title() or 'N/A',
                            'matched_title': match.get('title', {}).get('value', 'N/A'),
                            'score': match.get('score', 0),
                            'matched_doi': match.get('doi', {}).get('value', 'N/A'),
                            'meta_id': match.get('br', {}).get('value', 'N/A'),
                            'query_type': match.get('query_type', 'unknown')
                        })
                    else:
                        if isinstance(match, dict):
                            best_score = match.get('score')
                            score_info = match 
                        else:
                            best_score = None
                            score_info = None
                        
                        async with unmatched_lock:
                            unmatched_refs.append((result['ref_id'], result['ref'], score_info))
            
            # Save unmatched references
            if unmatched_refs:
                unmatched_file = os.path.splitext(output_file)[0] + '_unmatched.csv'
                self._save_unmatched_references(unmatched_refs, unmatched_file)
            
            # Print and save stats
            logging.info(f"\n{'='*70}")
            logging.info(f"📊 FINAL STATISTICS")
            logging.info(f"{'='*70}")
            
            percentage = self._print_stats(stats, input_file)
            
            logging.info(f"\n📈 Detailed Breakdown:")
            logging.info(f"  References with author: {stats['refs_with_author']}/{stats['total_references']}")
            logging.info(f"  References with title: {stats['refs_with_title']}/{stats['total_references']}")
            logging.info(f"  References with DOI: {stats['refs_with_doi']}/{stats['total_references']}")
            
            if stats['query_types']:
                logging.info(f"\n🔍 Query Type Distribution:")
                total_matches = sum(stats['query_types'].values())
                for qtype, count in sorted(stats['query_types'].items(), key=lambda x: x[1], reverse=True):
                    percentage_qtype = (count / total_matches * 100) if total_matches > 0 else 0
                    logging.info(f"  {qtype}: {count} ({percentage_qtype:.1f}%)")
            
            logging.info(f"{'='*70}\n")
            
            if stats['matches_found'] > 0:
                print(f"\n  ✓ {stats['matches_found']}/{stats['total_references']} matches " +
                    f"({percentage:.1f}%)" + 
                    (f" | {stats['errors']} errors" if stats['errors'] > 0 else ""))
            
            # Save stats file
            stats_file = os.path.splitext(output_file)[0] + '_stats.txt'
            self._save_stats_file(stats, stats_file)
            
        except Exception as e:
            logging.error(f"❌ Error writing output file {output_file}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            raise
        
    @staticmethod
    def _parse_bibl_struct(bibl: ET.Element, ns: Dict) -> Optional[Reference]:
            """FIXED: More robust biblStruct parsing with error handling"""
            try:
                ref = Reference()

                # Extract year
                date = bibl.find('.//tei:date[@when]', ns)
                if date is not None:
                    when_attr = date.get('when', '')
                    if when_attr and len(when_attr) >= 4:
                        ref.year = when_attr[:4]

                # Extract first author surname
                authors = bibl.findall('.//tei:author/tei:persName/tei:surname', ns)
                if authors and authors[0].text:
                    ref.first_author_lastname = authors[0].text.strip()

                # Extract titles with priority: analytic > monograph
                analytic = bibl.find('tei:analytic', ns)
                if analytic is not None:
                    title_elem = analytic.find('tei:title', ns)
                    if title_elem is not None and title_elem.text:
                        ref.article_title = title_elem.text.strip()

                # If no analytic title, check monograph titles
                if not ref.article_title:
                    monogr = bibl.find('tei:monogr', ns)
                    if monogr is not None:
                        # Try book title first
                        m_title = monogr.find('tei:title[@level="m"]', ns)
                        if m_title is not None and m_title.text:
                            ref.volume_title = m_title.text.strip()
                        
                        # Then journal title
                        j_title = monogr.find('tei:title[@level="j"]', ns)
                        if j_title is not None and j_title.text:
                            ref.journal_title = j_title.text.strip()

                # Extract volume
                volume_elem = bibl.find('.//tei:biblScope[@unit="volume"]', ns)
                if volume_elem is not None and volume_elem.text:
                    ref.volume = volume_elem.text.strip()

                # Extract first page
                page_elem = bibl.find('.//tei:biblScope[@unit="page"]', ns)
                if page_elem is not None:
                    if page_elem.get('from'):
                        ref.first_page = page_elem.get('from').strip()
                    elif page_elem.text:
                        # Handle page ranges like "123-130"
                        page_text = page_elem.text.strip()
                        if '-' in page_text:
                            ref.first_page = page_text.split('-')[0].strip()
                        else:
                            ref.first_page = page_text

                return ref

            except Exception as e:
                logging.info(f"Error parsing biblStruct: {e}")
                return None

    @staticmethod
    def _print_stats(stats: Dict, source_file: str = "") -> float:
        """Enhanced stats printing with error reporting - returns percentage"""
        total = stats.get('total_references', 0)
        matches = stats.get('matches_found', 0)
        errors = stats.get('errors', 0)
        percentage = (matches / total * 100) if total > 0 else 0.0

        logging.info(f"=== Processing Statistics for {os.path.basename(source_file) if source_file else 'file'} ===")
        logging.info(f"Total references processed: {total}")
        logging.info(f"Total matches found: {matches}")
        logging.info(f"Processing errors: {errors}")
        logging.info(f"Match percentage: {percentage:.2f}%")
        logging.info("=" * 50)
        
        return percentage

    @staticmethod
    def _save_stats_file(stats: Dict, output_file: str):
        """
        Save comprehensive statistics including all field coverage and match data
        """
        try:
            total = stats.get('total_references', 0)
            matches = stats.get('matches_found', 0)
            errors = stats.get('errors', 0)
            percentage = (matches / total * 100) if total > 0 else 0.0
            
            # Get all coverage stats
            refs_with_doi = stats.get('refs_with_doi', 0)
            refs_with_title = stats.get('refs_with_title', 0)
            refs_with_author = stats.get('refs_with_author', 0)
            refs_with_year = stats.get('refs_with_year', 0)
            refs_with_volume = stats.get('refs_with_volume', 0)
            refs_with_page = stats.get('refs_with_page', 0)
            
            # Get all match stats
            doi_matches = stats.get('doi_matches', 0) 
            author_exact_matches = stats.get('author_exact_matches', 0)
            year_exact_matches = stats.get('year_exact_matches', 0)
            year_adjacent_matches = stats.get('year_adjacent_matches', 0)
            title_exact_matches = stats.get('title_exact_matches', 0)
            title_fuzzy_matches = stats.get('title_fuzzy_matches', 0)
            volume_matches = stats.get('volume_matches', 0)
            page_matches = stats.get('page_matches', 0)

            query_types = stats.get('query_types', {})

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Processing Statistics\n")
                f.write(f"{'='*50}\n")
                f.write(f"Total references processed: {total}\n")
                f.write(f"Total matches found: {matches}\n")
                f.write(f"Processing errors: {errors}\n")
                f.write(f"Match percentage: {percentage:.2f}%\n")
                
                # Write coverage breakdown
                f.write(f"\nCoverage Breakdown:\n")
                f.write(f"{'─'*50}\n")
                f.write(f"References with DOI: {refs_with_doi}\n")
                f.write(f"References with title: {refs_with_title}\n")
                f.write(f"References with author: {refs_with_author}\n")
                f.write(f"References with year: {refs_with_year}\n")
                f.write(f"References with volume: {refs_with_volume}\n")
                f.write(f"References with page: {refs_with_page}\n")

                # Write field match statistics
                f.write(f"\nField Match Statistics:\n")
                f.write(f"{'─'*50}\n")
                
                # DOI matches - CORRECTED
                f.write(f"DOI matches: {doi_matches}")  
                if refs_with_doi > 0:
                    doi_pct = (doi_matches / refs_with_doi * 100)
                    f.write(f" ({doi_pct:.1f}% of refs with DOI)\n")
                else:
                    f.write("\n")
                
                # Author matches
                if refs_with_author > 0:
                    author_percentage = (author_exact_matches / refs_with_author * 100)
                    f.write(f"Author exact matches: {author_exact_matches} ({author_percentage:.1f}%)\n")
                else:
                    f.write(f"Author exact matches: {author_exact_matches}\n")
                
                # Year matches
                if refs_with_year > 0:
                    year_percentage = ((year_exact_matches + year_adjacent_matches) / refs_with_year * 100)
                    f.write(f"Year exact matches: {year_exact_matches}\n")
                    f.write(f"Year adjacent matches (±1): {year_adjacent_matches}\n")
                    f.write(f"Year total match rate: {year_percentage:.1f}%\n")
                else:
                    f.write(f"Year exact matches: {year_exact_matches}\n")
                    f.write(f"Year adjacent matches (±1): {year_adjacent_matches}\n")
                
                # Title matches
                if refs_with_title > 0:
                    title_total = title_exact_matches + title_fuzzy_matches
                    title_percentage = (title_total / refs_with_title * 100)
                    f.write(f"Title exact matches: {title_exact_matches}\n")
                    f.write(f"Title fuzzy matches: {title_fuzzy_matches}\n")
                    f.write(f"Title total match rate: {title_percentage:.1f}%\n")
                else:
                    f.write(f"Title exact matches: {title_exact_matches}\n")
                    f.write(f"Title fuzzy matches: {title_fuzzy_matches}\n")
                
                # Volume matches
                if refs_with_volume > 0:
                    volume_percentage = (volume_matches / refs_with_volume * 100)
                    f.write(f"Volume matches: {volume_matches} ({volume_percentage:.1f}%)\n")
                else:
                    f.write(f"Volume matches: {volume_matches}\n")
                
                # Page matches
                if refs_with_page > 0:
                    page_percentage = (page_matches / refs_with_page * 100)
                    f.write(f"Page matches: {page_matches} ({page_percentage:.1f}%)\n")
                else:
                    f.write(f"Page matches: {page_matches}\n")
                
                # Write query type distribution
                if query_types:
                    f.write(f"\nQuery Type Distribution:\n")
                    f.write(f"{'─'*50}\n")
                    total_query = sum(query_types.values())
                    for qtype, count in sorted(query_types.items(), key=lambda x: x[1], reverse=True):
                        qtype_percentage = (count / total_query * 100) if total_query > 0 else 0
                        f.write(f"  {qtype}: {count} ({qtype_percentage:.1f}%)\n")

                # Write Grobid statistics
                grobid_fallbacks = stats.get('grobid_fallbacks', 0)
                grobid_successes = stats.get('grobid_successes', 0)
                if grobid_fallbacks > 0 or grobid_successes > 0:
                    f.write(f"\nGrobid Fallback Statistics:\n")
                    f.write(f"{'─'*50}\n")
                    f.write(f"Grobid fallback attempts: {grobid_fallbacks}\n")
                    f.write(f"Successful matches after Grobid: {grobid_successes}\n")
                    if grobid_fallbacks > 0:
                        grobid_success_rate = (grobid_successes / grobid_fallbacks * 100)
                        f.write(f"Grobid success rate: {grobid_success_rate:.1f}%\n")

                f.write(f"\nTimestamp: {datetime.now().isoformat()}\n")
            
            logging.info(f"Statistics saved to: {output_file}")
            
        except Exception as e:
            logging.error(f"Error writing stats file {output_file}: {e}")
            
    def _track_final_match_stats(self, match: Dict, ref: Reference, stats: Dict):
        """
        Track field contributions for the FINAL accepted match only.
        Called ONCE per successful match, not for every candidate.
        """
        if not match or not stats:
            return
        
        try:
            # DOI contribution
            if ref.doi:
                if 'doi' not in match:
                    logging.debug(f"⚠️ Reference has DOI but match result doesn't have 'doi' field")
                    logging.debug(f"   Ref DOI: {ref.doi}")
                    logging.debug(f"   Match keys: {list(match.keys())}")
                elif match['doi'] is None or (isinstance(match['doi'], dict) and not match['doi'].get('value')):
                    logging.debug(f"⚠️ Match has 'doi' field but it's None or empty")
                else:
                    ref_doi = ref.doi.lower().strip()

                    # Gestisci sia dict che string
                    if isinstance(match['doi'], dict):
                        result_doi = match['doi']['value'].lower().strip()
                    else:
                        result_doi = str(match['doi']).lower().strip()

                    if ref_doi == result_doi:
                        stats['doi_matches'] = stats.get('doi_matches', 0) + 1
                        logging.debug(f"✅ DOI MATCH: {ref_doi}")
                    else:
                        logging.debug(f"❌ DOI MISMATCH: ref={ref_doi} vs result={result_doi}")
            
            # Author contribution
            if ref.first_author_lastname and 'author_name' in match:
                try:
                    result_author = self._normalize_author_name(match['author_name']['value'])
                    ref_author = self._normalize_author_name(ref.first_author_lastname)
                    
                    if ref_author == result_author:
                        stats['author_exact_matches'] = stats.get('author_exact_matches', 0) + 1
                except Exception as e:
                    logging.debug(f"Error checking author match: {e}")
            
            # Year contribution
            if ref.year and 'pub_date' in match:
                try:
                    ref_year = self._extract_year(ref.year)
                    result_year = int(match['pub_date']['value'][:4])
                    
                    if ref_year == result_year:
                        stats['year_exact_matches'] = stats.get('year_exact_matches', 0) + 1
                    elif ref_year and abs(ref_year - result_year) == 1:
                        stats['year_adjacent_matches'] = stats.get('year_adjacent_matches', 0) + 1
                except (ValueError, KeyError, IndexError):
                    pass
            
            # Title contribution
            if 'title' in match and ref.get_main_title():
                try:
                    result_title = self._normalize_title(match['title']['value'])
                    titles_to_check = [ref.article_title, ref.volume_title, ref.journal_title]
                    
                    best_score = 0
                    for title in titles_to_check:
                        if title:
                            record_title = self._normalize_title(title)
                            
                            if record_title == result_title:
                                similarity = 100
                            else:
                                ratio = fuzz.ratio(record_title, result_title)
                                partial = fuzz.partial_ratio(record_title, result_title)
                                token_sort = fuzz.token_sort_ratio(record_title, result_title)
                                token_set = fuzz.token_set_ratio(record_title, result_title)
                                similarity = max(ratio, partial, token_sort, token_set)
                            
                            best_score = max(best_score, similarity)
                    
                    if best_score == 100:
                        stats['title_exact_matches'] = stats.get('title_exact_matches', 0) + 1
                    elif best_score >= 75:
                        stats['title_fuzzy_matches'] = stats.get('title_fuzzy_matches', 0) + 1
                except Exception as e:
                    logging.debug(f"Error checking title match: {e}")
            
            # Volume contribution
            if ref.volume and 'volume_num' in match:
                if ref.volume == match['volume_num']['value']:
                    stats['volume_matches'] = stats.get('volume_matches', 0) + 1
            
            # Page contribution
            if ref.first_page and 'start_page' in match:
                ref_page = ref.first_page.lstrip('0')
                result_page = match['start_page']['value'].lstrip('0')
                
                if ref_page == result_page:
                    stats['page_matches'] = stats.get('page_matches', 0) + 1
        
        except Exception as e:
            logging.info(f"Error tracking final match stats: {e}")
            
    def quick_validate_output(self, csv_file: str, stats_file: str) -> Tuple[bool, str]:
        """
        Quick validation - checks if files exist and aren't empty.
        Returns (is_valid, error_message)
        """
        # Check CSV exists and has data
        if not os.path.exists(csv_file):
            return False, "CSV missing"
        
        if os.path.getsize(csv_file) < 100:
            return False, "CSV empty"
        
        # Check stats exists and isn't tiny
        if not os.path.exists(stats_file):
            return False, "Stats missing"
        
        if os.path.getsize(stats_file) < 500:
            return False, "Stats too small"
        
        return True, "OK"

    async def process_file_with_retry(self, input_file: str, output_file: str,
                                      threshold: int = 26, use_doi: bool = True,
                                      max_retries: int = 2) -> bool:
        """
        Process with automatic retry if validation fails.
        Returns True if successful, False otherwise.
        """
        stats_file = os.path.splitext(output_file)[0] + '_stats.txt'
        
        for attempt in range(1, max_retries + 1):
            try:
                # Process the file using original method
                await self.process_file(input_file, output_file, threshold, use_doi)
                
                # Quick validation
                is_valid, reason = self.quick_validate_output(output_file, stats_file)
                
                if is_valid:
                    if attempt > 1:
                        logging.info(f"✅ {os.path.basename(input_file)}: Success on attempt {attempt}")
                    return True
                
                # If validation failed and we have retries left
                if attempt < max_retries:
                    logging.warning(f"⚠️ {os.path.basename(input_file)}: {reason}, retrying (attempt {attempt+1}/{max_retries})...")
                    # Clean up failed outputs
                    if os.path.exists(output_file):
                        os.remove(output_file)
                    if os.path.exists(stats_file):
                        os.remove(stats_file)
                    # Brief pause before retry
                    await asyncio.sleep(1)  
                else:
                    logging.error(f"❌ {os.path.basename(input_file)}: {reason} after {max_retries} attempts")
                    return False
                    
            except Exception as e:
                logging.error(f"❌ {os.path.basename(input_file)} attempt {attempt}: {e}")
                # Clean up on exception
                if os.path.exists(output_file):
                    try:
                        os.remove(output_file)
                    except:
                        pass
                if os.path.exists(stats_file):
                    try:
                        os.remove(stats_file)
                    except:
                        pass
                
                if attempt < max_retries:
                    logging.info(f"Retrying after exception...")
                    await asyncio.sleep(1)
                else:
                    logging.error(f"❌ All {max_retries} attempts failed")
                    return False
        
        return False
class BatchProcessor:
    """Memory-efficient batch processor with incremental checkpointing"""
    
    def __init__(self, reference_processor: 'ReferenceProcessor',
             batch_size: int = 3, pause_duration: int = 10, error_threshold: int = 10,  
             use_doi: bool = True, checkpoint_interval: int = 10):
        self.reference_processor = reference_processor
        self.batch_size = batch_size
        self.pause_duration = pause_duration
        self.error_threshold = error_threshold
        self.use_doi = use_doi
        self.checkpoint_interval = checkpoint_interval
        self._checkpoint_lock = threading.Lock()
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('batch_processing.log'),
                logging.StreamHandler()
            ]
        )




    async def process_directory(self, input_dir: str, output_dir: str, threshold: int = 26, 
                           checkpoint_file: str = 'processing_checkpoint.pkl') -> str:
        """Memory-efficient directory processing with progress tracking and aggregate stats"""
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
            
        os.makedirs(output_dir, exist_ok=True)
        
        with self._checkpoint_lock:
            processed_files = self.load_checkpoint(checkpoint_file)

        input_files = [
            f for f in os.listdir(input_dir) 
            if f.endswith(('.xml', '.json')) and f not in processed_files
        ]
        
        total_files = len(input_files)
        
        if not input_files:
            print("✓ No new files to process.")
            return

        print(f"\n📁 Processing {total_files} files from: {input_dir}")
        print(f"💾 Already processed: {len(processed_files)} files\n")

        # Aggregate stats
        aggregate_stats = {
            'total_references': 0,
            'matches_found': 0,
            'errors': 0,
            'query_types': {},
            'refs_with_author': 0,
            'author_exact_matches': 0,
            'refs_with_title': 0,
            'refs_with_doi': 0,
            'doi_matches': 0,              
            'refs_with_year': 0,           
            'refs_with_volume': 0,          
            'refs_with_page': 0,            
            'year_exact_matches': 0,        
            'year_adjacent_matches': 0,     
            'title_exact_matches': 0,      
            'title_fuzzy_matches': 0,       
            'volume_matches': 0,            
            'page_matches': 0,              
            'grobid_fallbacks': 0,         
            'grobid_successes': 0,          
            'files_processed': 0,
            'files_with_errors': 0
        }

        error_500_count = 0
        files_since_checkpoint = 0
        current_batch_processed = set()

        # Progress bar for overall file processing
        with tqdm(total=total_files, desc="Overall Progress", unit="file", 
                position=0, leave=True, ncols=100) as pbar:
            
            for i in range(0, total_files, self.batch_size):
                batch_files = input_files[i:i + self.batch_size]
                batch_num = i // self.batch_size + 1
                total_batches = (total_files - 1) // self.batch_size + 1
                
                pbar.set_description(f"Batch {batch_num}/{total_batches}")

                tasks = []
                file_info = []
                
                for filename in batch_files:
                    input_path = os.path.join(input_dir, filename)
                    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_matches.csv")
                    
                    # Create async task
                    task = self.reference_processor.process_file_with_retry(input_path, output_path, threshold, self.use_doi, max_retries=2)
                    tasks.append(task)
                    file_info.append((filename, output_path))
                
                # Run all tasks concurrently
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for result, (filename, output_path) in zip(results, file_info):
                    try:
                        # Check if result is an exception
                        if isinstance(result, Exception):
                            logging.error(f"Error processing {filename}: {result}")
                            error_500_count += 1
                            aggregate_stats['files_with_errors'] += 1
                            pbar.update(1)
                            
                            # Try to read stats even on error
                            stats_file = os.path.splitext(output_path)[0] + '_stats.txt'
                            if os.path.exists(stats_file):
                                try:
                                    file_stats = self._read_stats_file(stats_file)
                                    self._merge_stats(aggregate_stats, file_stats)
                                    aggregate_stats['files_processed'] += 1
                                    logging.info(f"Stats recovered after exception for {filename}")
                                except Exception as e:
                                    logging.info(f"Could not read stats after exception: {e}")
                            continue
                        
                        # Read stats file
                        stats_file = os.path.splitext(output_path)[0] + '_stats.txt'
                        if os.path.exists(stats_file):
                            try:
                                file_stats = self._read_stats_file(stats_file)
                                self._merge_stats(aggregate_stats, file_stats)
                                aggregate_stats['files_processed'] += 1
                                logging.debug(f"Read stats from {stats_file}: {file_stats}")
                            except Exception as e:
                                logging.info(f"Error reading stats from {stats_file}: {e}")
                        
                        # Update progress
                        pbar.set_postfix_str(f"✓ {filename[:30]}...")
                        current_batch_processed.add(filename)
                        files_since_checkpoint += 1
                        pbar.update(1)
                        
                        if files_since_checkpoint >= self.checkpoint_interval:
                            processed_files.update(current_batch_processed)
                            self.save_checkpoint(checkpoint_file, processed_files)
                            current_batch_processed.clear()
                            files_since_checkpoint = 0
                    
                    except Exception as e:
                        logging.error(f"Error handling result for {filename}: {e}")
                        aggregate_stats['files_with_errors'] += 1
                        pbar.update(1)
                if current_batch_processed:
                    with self._checkpoint_lock:
                        processed_files.update(current_batch_processed)
                        self.save_checkpoint(checkpoint_file, processed_files)
                    current_batch_processed.clear()
                
                if batch_num % 10 == 0:
                    self.compress_checkpoint(checkpoint_file)
                
                if error_500_count >= self.error_threshold:
                    pbar.write(f"\n⚠ Too many errors ({error_500_count}). Pausing 5 minutes...")
                    time.sleep(300)  
                    error_500_count = 0  

                if i + self.batch_size < total_files:
                    time.sleep(self.pause_duration)

            with self._checkpoint_lock:
                if current_batch_processed:
                    processed_files.update(current_batch_processed)
                self.save_checkpoint(checkpoint_file, processed_files)
        
        # Generate aggregate report
        print(f"\n✓ Complete! Processed {len(processed_files)} total files")
        print(f"\n📊 Generating aggregate report...")
        
        self._print_aggregate_stats(aggregate_stats, output_dir)
        self._save_aggregate_stats_file(aggregate_stats, output_dir)
        
        try:
            # Wait a moment for any in-flight writes to complete
            await asyncio.sleep(2)
            
            # Re-aggregate ALL stats files to catch any that were written late
            logging.info("Performing final aggregation of all stats files...")
            aggregate_stats = self._aggregate_all_stats(output_dir)  # 👈 DO THIS FIRST
            
            # VALIDATION: Check all outputs for failures (AFTER re-aggregation)
            logging.info("\n" + "="*80)
            logging.info("🔍 Validating all outputs...")
            logging.info("="*80)
            
            validation_results = self.validate_all_outputs(output_dir)
            
            if validation_results['failed_files']:
                failed_count = len(validation_results['failed_files'])
                logging.warning(f"\n⚠️  {failed_count} files failed validation:")
                for fname, reason in validation_results['failed_files']:
                    logging.warning(f"  • {fname}: {reason}")
                
                # Save failed files list
                failed_list_path = os.path.join(output_dir, 'FAILED_FILES.txt')
                with open(failed_list_path, 'w', encoding='utf-8') as f:
                    f.write(f"Files that failed validation: {failed_count}\n")
                    f.write("="*80 + "\n\n")
                    for fname, reason in validation_results['failed_files']:
                        f.write(f"{fname}: {reason}\n")
                
                logging.warning(f"Failed files list saved to: {failed_list_path}")
                logging.warning("\nTo fix: Run validate_and_cleanup.py")
            else:
                logging.info("✅ All files validated successfully!")
            
            # Generate report with complete data (LAST)
            generate_processing_report(output_dir, aggregate_stats)
            logging.info(f"✅ HTML report generated: {os.path.join(output_dir, 'processing_report.html')}")
            
            logging.info(f"\n{'='*50}")
            logging.info("Processing complete!")
            
        except Exception as e:
            logging.error(f"Error generating aggregate report: {e}")

    @staticmethod
    def load_checkpoint(checkpoint_file: str) -> Set[str]:
        """Load checkpoint with robust error handling"""
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, set):
                        logging.info(f"Loaded checkpoint: {len(data)} files already processed")
                        return data
                    elif isinstance(data, list):
                        logging.info(f"Loaded checkpoint (list): {len(data)} files already processed")
                        return set(data)
                    else:
                        logging.info(f"Invalid checkpoint format ({type(data)}), starting fresh")
                        return set()
            except (pickle.PickleError, EOFError, ValueError) as e:
                logging.error(f"Error loading checkpoint: {e}, starting fresh")
                # Backup corrupted checkpoint
                try:
                    backup_file = f"{checkpoint_file}.corrupted.{int(time.time())}"
                    os.rename(checkpoint_file, backup_file)
                    logging.info(f"Corrupted checkpoint backed up to: {backup_file}")
                except Exception:
                    pass
                return set()
        
        logging.info("No checkpoint found, starting fresh")
        return set()

    def save_checkpoint(self, checkpoint_file: str, processed_files: Set[str]):
        """Thread-safe checkpoint saving with atomic writes"""
        temp_file = None
        
        try:
            # Create temp file with unique name
            temp_file = f"{checkpoint_file}.tmp.{os.getpid()}.{threading.get_ident()}"
            
            # Write to temp file
            with open(temp_file, 'wb') as f:
                pickle.dump(processed_files, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Atomic operations for safety
            if os.path.exists(checkpoint_file):
                # On Unix: os.rename is atomic if on same filesystem
                # On Windows: need to remove first
                if os.name == 'nt':  # Windows
                    backup_file = f"{checkpoint_file}.backup"
                    try:
                        if os.path.exists(backup_file):
                            os.remove(backup_file)
                        # Copy current to backup
                        import shutil
                        shutil.copy2(checkpoint_file, backup_file)
                        # Remove original
                        os.remove(checkpoint_file)
                        # Rename temp to original
                        os.replace(temp_file, checkpoint_file)  # Atomic on all platforms
                        # Clean up backup
                        if os.path.exists(backup_file):
                            os.remove(backup_file)
                    except Exception as e:
                        logging.error(f"Windows atomic save failed: {e}")
                        raise
                else:  # Unix-like systems
                    os.rename(temp_file, checkpoint_file)
            else:
                os.rename(temp_file, checkpoint_file)
            
            temp_file = None  # Successfully saved
            
        except Exception as e:
            logging.error(f"Error saving checkpoint: {e}")
            # Clean up temp file if it exists
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass
            # Re-raise to alert caller
            raise  
    def compress_checkpoint(self, checkpoint_file: str, max_items: int = 10000):
            """Compress checkpoint if it grows too large"""
            try:
                processed_files = self.load_checkpoint(checkpoint_file)
                
                if len(processed_files) > max_items:
                    logging.info(
                        f"Checkpoint has {len(processed_files)} items, "
                        f"considering archiving older entries"
                    )
                    # Archive old checkpoint
                    archive_file = f"{checkpoint_file}.archive.{int(time.time())}"
                    with open(archive_file, 'wb') as f:
                        pickle.dump(processed_files, f)
                    logging.info(f"Archived large checkpoint to: {archive_file}")
                    
            except Exception as e:
                logging.error(f"Error compressing checkpoint: {e}")
    def get_processing_stats(self, checkpoint_file: str) -> Dict[str, any]:
        """Get statistics about processing progress"""
        if not os.path.exists(checkpoint_file):
            return {
                'processed_files': 0,
                'checkpoint_exists': False
            }
        
        try:
            processed_files = self.load_checkpoint(checkpoint_file)
            return {
                'processed_files': len(processed_files),
                'checkpoint_exists': True,
                'checkpoint_size_bytes': os.path.getsize(checkpoint_file)
            }
        except Exception as e:
            logging.error(f"Error getting stats: {e}")
            return {
                'processed_files': 0,
                'checkpoint_exists': True,
                'error': str(e)
            }

    def reset_checkpoint(self, checkpoint_file: str) -> bool:
        """Reset checkpoint - use with caution"""
        try:
            if os.path.exists(checkpoint_file):
                backup_file = f"{checkpoint_file}.reset_backup.{int(time.time())}"
                os.rename(checkpoint_file, backup_file)
                logging.info(f"Checkpoint reset. Backup saved to: {backup_file}")
                return True
            else:
                logging.info("No checkpoint to reset")
                return False
        except Exception as e:
            logging.error(f"Error resetting checkpoint: {e}")
            return False

    def process_directory_with_resume(self, input_dir: str, output_dir: str, 
                                     threshold: int = 26, 
                                     checkpoint_file: str = 'processing_checkpoint.pkl',
                                     force_restart: bool = False):
        """
        Process directory with automatic resume capability
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            threshold: Matching threshold
            checkpoint_file: Checkpoint file path
            force_restart: If True, ignore existing checkpoint and start fresh
        """
        if force_restart:
            logging.info("Force restart requested - resetting checkpoint")
            self.reset_checkpoint(checkpoint_file)
        
        # Show current progress
        stats = self.get_processing_stats(checkpoint_file)
        if stats['checkpoint_exists']:
            logging.info(f"Resuming from checkpoint: {stats['processed_files']} files already processed")
        
        # Process
        self.process_directory(input_dir, output_dir, threshold, checkpoint_file)
    
    def _read_stats_file(self, stats_file: str) -> Dict:
        """
        FIXED: Read stats from individual file stats.txt
        Now extracts ALL metadata including query types and author stats
        """
        stats = {
            'total_references': 0,
            'matches_found': 0,
            'errors': 0,
            'query_types': {},
            'refs_with_author': 0,
            'author_exact_matches': 0,
            'refs_with_title': 0,
            'refs_with_doi': 0,
            'doi_matches': 0,            
            'refs_with_year': 0,         
            'refs_with_volume': 0,      
            'refs_with_page': 0,         
            'year_exact_matches': 0,     
            'year_adjacent_matches': 0,  
            'title_exact_matches': 0,    
            'title_fuzzy_matches': 0,    
            'volume_matches': 0,        
            'page_matches': 0,          
            'grobid_fallbacks': 0,       
            'grobid_successes': 0        
        }
        
        try:
            with open(stats_file, 'r', encoding='utf-8') as f:
                content = f.read()
                total_match = re.search(r'Total references processed:\s*(\d+)', content)
                if total_match:
                    refs = int(total_match.group(1))
                    if refs > 0:
                        logging.debug(f"Reading {os.path.basename(stats_file)}: {refs} refs")
                else:
                    logging.warning(f"Could not find 'Total references' in {stats_file}")
                # Parse basic stats
                total_match = re.search(r'Total references processed:\s*(\d+)', content)
                matches_match = re.search(r'Total matches found:\s*(\d+)', content)
                errors_match = re.search(r'Processing errors:\s*(\d+)', content)
                
                if total_match:
                    stats['total_references'] = int(total_match.group(1))
                if matches_match:
                    stats['matches_found'] = int(matches_match.group(1))
                if errors_match:
                    stats['errors'] = int(errors_match.group(1))
                
                # Parse ALL coverage breakdown fields
                refs_author_match = re.search(r'References with author:\s*(\d+)', content)
                refs_title_match = re.search(r'References with title:\s*(\d+)', content)
                refs_doi_match = re.search(r'References with DOI:\s*(\d+)', content)
                refs_year_match = re.search(r'References with year:\s*(\d+)', content)
                refs_volume_match = re.search(r'References with volume:\s*(\d+)', content)
                refs_page_match = re.search(r'References with page:\s*(\d+)', content)
                
                if refs_author_match:
                    stats['refs_with_author'] = int(refs_author_match.group(1))
                if refs_title_match:
                    stats['refs_with_title'] = int(refs_title_match.group(1))
                if refs_doi_match:
                    stats['refs_with_doi'] = int(refs_doi_match.group(1))
                if refs_year_match:
                    stats['refs_with_year'] = int(refs_year_match.group(1))
                if refs_volume_match:
                    stats['refs_with_volume'] = int(refs_volume_match.group(1))
                if refs_page_match:
                    stats['refs_with_page'] = int(refs_page_match.group(1))
                
                # Parse field match statistics
                author_matches_match = re.search(r'Author exact matches:\s*(\d+)', content)
                year_exact_match = re.search(r'Year exact matches:\s*(\d+)', content)
                year_adjacent_match = re.search(r'Year adjacent matches.*?:\s*(\d+)', content)
                title_exact_match = re.search(r'Title exact matches:\s*(\d+)', content)
                title_fuzzy_match = re.search(r'Title fuzzy matches:\s*(\d+)', content)
                volume_matches_match = re.search(r'Volume matches:\s*(\d+)', content)
                page_matches_match = re.search(r'Page matches:\s*(\d+)', content)
                
                if author_matches_match:
                    stats['author_exact_matches'] = int(author_matches_match.group(1))
                if year_exact_match:
                    stats['year_exact_matches'] = int(year_exact_match.group(1))
                if year_adjacent_match:
                    stats['year_adjacent_matches'] = int(year_adjacent_match.group(1))
                if title_exact_match:
                    stats['title_exact_matches'] = int(title_exact_match.group(1))
                if title_fuzzy_match:
                    stats['title_fuzzy_matches'] = int(title_fuzzy_match.group(1))
                if volume_matches_match:
                    stats['volume_matches'] = int(volume_matches_match.group(1))
                if page_matches_match:
                    stats['page_matches'] = int(page_matches_match.group(1))
                
                # Parse query type distribution
                query_section_start = content.find('Query Type Distribution:')
                if query_section_start != -1:
                    remaining_text = content[query_section_start:]
                    
                    next_section = re.search(r'\n\n[A-Z]|\nTimestamp:', remaining_text)
                    if next_section:
                        query_section_text = remaining_text[:next_section.start()]
                    else:
                        query_section_text = remaining_text
                    
                    query_lines = query_section_text.split('\n')
                    
                    for line in query_lines:
                        match = re.search(r'^\s*([a-z_]+):\s*(\d+)', line)
                        if match:
                            query_type = match.group(1)
                            count = int(match.group(2))
                            stats['query_types'][query_type] = count

                # Parse Grobid statistics
                grobid_fallbacks_match = re.search(r'Grobid fallback attempts:\s*(\d+)', content)
                grobid_successes_match = re.search(r'Successful matches after Grobid:\s*(\d+)', content)
                doi_matches_match = re.search(r'DOI matches:\s*(\d+)', content)

                if grobid_fallbacks_match:
                    stats['grobid_fallbacks'] = int(grobid_fallbacks_match.group(1))
                if grobid_successes_match:
                    stats['grobid_successes'] = int(grobid_successes_match.group(1))
                if doi_matches_match:
                    stats['doi_matches'] = int(doi_matches_match.group(1))

        except Exception as e:
            logging.info(f"Error reading stats file {stats_file}: {e}")
            import traceback
            logging.debug(traceback.format_exc())
        
        return stats
    
    def _merge_stats(self, aggregate: Dict, file_stats: Dict):
        """
        FIXED: Merge file stats into aggregate stats
        Now includes query types and author statistics
        """
        aggregate['total_references'] += file_stats.get('total_references', 0)
        aggregate['matches_found'] += file_stats.get('matches_found', 0)
        aggregate['errors'] += file_stats.get('errors', 0)
        
        # Merge coverage stats
        aggregate['refs_with_author'] += file_stats.get('refs_with_author', 0)
        aggregate['refs_with_title'] += file_stats.get('refs_with_title', 0)
        aggregate['refs_with_doi'] += file_stats.get('refs_with_doi', 0)
        aggregate['refs_with_year'] += file_stats.get('refs_with_year', 0)     
        aggregate['refs_with_volume'] += file_stats.get('refs_with_volume', 0)   
        aggregate['refs_with_page'] += file_stats.get('refs_with_page', 0)       
        
        # Merge match stats
        aggregate['author_exact_matches'] += file_stats.get('author_exact_matches', 0)
        aggregate['year_exact_matches'] += file_stats.get('year_exact_matches', 0)      
        aggregate['year_adjacent_matches'] += file_stats.get('year_adjacent_matches', 0) 
        aggregate['title_exact_matches'] += file_stats.get('title_exact_matches', 0)   
        aggregate['title_fuzzy_matches'] += file_stats.get('title_fuzzy_matches', 0)    
        aggregate['volume_matches'] += file_stats.get('volume_matches', 0)              
        aggregate['page_matches'] += file_stats.get('page_matches', 0)                  
        aggregate['doi_matches'] += file_stats.get('doi_matches', 0)                    

        # Merge Grobid stats
        aggregate['grobid_fallbacks'] += file_stats.get('grobid_fallbacks', 0)
        aggregate['grobid_successes'] += file_stats.get('grobid_successes', 0)

        # Merge query types
        for qtype, count in file_stats.get('query_types', {}).items():
            aggregate['query_types'][qtype] = aggregate['query_types'].get(qtype, 0) + count
    def validate_all_outputs(self, output_dir: str) -> dict:
        """
        Validate ALL output files after processing.
        Returns dict with validation results.
        """
        
        results = {
            'total_files': 0,
            'valid_files': 0,
            'failed_files': []
        }
        
        # Find all CSV files
        csv_files = glob(os.path.join(output_dir, "*_matches.csv"))
        results['total_files'] = len(csv_files)
        
        for csv_file in csv_files:
            basename = os.path.basename(csv_file).replace("_matches.csv", "")
            stats_file = csv_file.replace("_matches.csv", "_matches_stats.txt")
            
            # Quick validation
            is_valid = True
            reason = None
            
            # Check stats file exists
            if not os.path.exists(stats_file):
                is_valid = False
                reason = "Missing stats"
            # Check CSV size
            elif os.path.getsize(csv_file) < 100:
                is_valid = False
                reason = "Empty CSV"
            # Check stats size
            elif os.path.getsize(stats_file) < 500:
                is_valid = False
                reason = "Stats too small"
            # Check for 0 references in stats
            else:
                try:
                    with open(stats_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if "Total references processed: 0" in content:
                        is_valid = False
                        reason = "0 references"
                except:
                    is_valid = False
                    reason = "Stats unreadable"
            
            if is_valid:
                results['valid_files'] += 1
            else:
                results['failed_files'].append((basename, reason))
                logging.warning(f"Failed validation: {basename} - {reason}")
        
        return results
    def _aggregate_all_stats(self, output_dir: str) -> Dict:
            """
            Aggregate ALL stats files in directory (not just checkpointed ones).
            This fixes the race condition by rescanning the directory.
            """
            
            # Initialize aggregate stats
            aggregate_stats = {
                'total_references': 0,
                'matches_found': 0,
                'errors': 0,
                'query_types': {},
                'refs_with_author': 0,
                'author_exact_matches': 0,
                'refs_with_title': 0,
                'refs_with_doi': 0,
                'doi_matches': 0,
                'refs_with_year': 0,
                'refs_with_volume': 0,
                'refs_with_page': 0,
                'year_exact_matches': 0,
                'year_adjacent_matches': 0,
                'title_exact_matches': 0,
                'title_fuzzy_matches': 0,
                'volume_matches': 0,
                'page_matches': 0,
                'grobid_fallbacks': 0,         
                'grobid_successes': 0,          
                'files_processed': 0,
                'files_with_errors': 0,
                'empty_files': 0,
                'total_files_attempted': 0
            }
            
            # Scan ALL stats files in directory
            stats_files = glob(os.path.join(output_dir, "*_matches_stats.txt"))
            
            logging.info(f"Aggregating {len(stats_files)} stats files from {output_dir}")
            
            for stats_file in stats_files:
                
                aggregate_stats['total_files_attempted'] += 1 
                                
                try:
                    file_stats = self._read_stats_file(stats_file)
                                    
                    # Only count files with actual references (skip empty files)
                    if file_stats.get('total_references', 0) > 0:
                        self._merge_stats(aggregate_stats, file_stats)
                        aggregate_stats['files_processed'] += 1
                        logging.debug(f"Aggregated {os.path.basename(stats_file)}: "
                                f"{file_stats.get('total_references', 0)} refs, "
                                f"{file_stats.get('matches_found', 0)} matches")
                    else:
                        aggregate_stats['empty_files'] += 1 
                        logging.debug(f"Skipping empty file: {os.path.basename(stats_file)}")
                                        
                except Exception as e:
                    logging.error(f"Error reading {os.path.basename(stats_file)}: {e}")
                    aggregate_stats['files_with_errors'] += 1
            
            logging.info(f"Aggregation complete: {aggregate_stats['files_processed']} files, "
                        f"{aggregate_stats['total_references']} refs, "
                        f"{aggregate_stats['matches_found']} matches")
            
            return aggregate_stats
    def _print_aggregate_stats(self, stats: Dict, output_dir: str):
        """Print aggregate statistics"""
        total = stats.get('total_references', 0)
        matches = stats.get('matches_found', 0)
        errors = stats.get('errors', 0)
        files_processed = stats.get('files_processed', 0)
        percentage = (matches / total * 100) if total > 0 else 0.0
        
        print(f"\n{'='*70}")
        print(f"📊 AGGREGATE STATISTICS FOR ALL FILES")
        print(f"{'='*70}")
        print(f"Files processed: {files_processed}")
        print(f"Total references: {total}")
        print(f"Total matches: {matches}")
        print(f"Processing errors: {errors}")
        print(f"Match rate: {percentage:.2f}%")
        
        if stats.get('refs_with_author', 0) > 0:
            author_percentage = (stats['author_exact_matches'] / stats['refs_with_author'] * 100)
            print(f"\nAuthor statistics:")
            print(f"  References with author: {stats['refs_with_author']}")
            print(f"  Author exact matches: {stats['author_exact_matches']} ({author_percentage:.1f}%)")
        
        if stats.get('query_types'):
            print(f"\nQuery type distribution:")
            total_query = sum(stats['query_types'].values())
            for qtype, count in sorted(stats['query_types'].items(), key=lambda x: x[1], reverse=True):
                qtype_percentage = (count / total_query * 100) if total_query > 0 else 0
                print(f"  {qtype}: {count} ({qtype_percentage:.1f}%)")
        
        print(f"{'='*70}\n")

    def _save_aggregate_stats_file(self, stats: Dict, output_dir: str):
        """Save comprehensive aggregate stats to file with all field statistics"""
        stats_file = os.path.join(output_dir, 'aggregate_stats.txt')
    
        try:
            total = stats.get('total_references', 0)
            matches = stats.get('matches_found', 0)
            errors = stats.get('errors', 0)
            files_processed = stats.get('files_processed', 0)
            percentage = (matches / total * 100) if total > 0 else 0.0
        
            with open(stats_file, 'w', encoding='utf-8') as f:
                f.write(f"Aggregate Processing Statistics\n")
                f.write(f"{'='*50}\n")
                f.write(f"Files processed: {files_processed}\n")
                f.write(f"Total references: {total}\n")
                f.write(f"Total matches: {matches}\n")
                f.write(f"Processing errors: {errors}\n")
                f.write(f"Match percentage: {percentage:.2f}%\n")
            
                # Coverage Breakdown (ALL fields)
                f.write(f"\nCoverage Breakdown:\n")
                f.write(f"{'─'*50}\n")
                
                refs_with_doi = stats.get('refs_with_doi', 0)
                refs_with_title = stats.get('refs_with_title', 0)
                refs_with_author = stats.get('refs_with_author', 0)
                refs_with_year = stats.get('refs_with_year', 0)
                refs_with_volume = stats.get('refs_with_volume', 0)
                refs_with_page = stats.get('refs_with_page', 0)
                
                f.write(f"References with DOI: {refs_with_doi}")
                if total > 0:
                    f.write(f" ({refs_with_doi/total*100:.1f}%)\n")
                else:
                    f.write("\n")
                    
                f.write(f"References with title: {refs_with_title}")
                if total > 0:
                    f.write(f" ({refs_with_title/total*100:.1f}%)\n")
                else:
                    f.write("\n")
                    
                f.write(f"References with author: {refs_with_author}")
                if total > 0:
                    f.write(f" ({refs_with_author/total*100:.1f}%)\n")
                else:
                    f.write("\n")
                    
                f.write(f"References with year: {refs_with_year}")
                if total > 0:
                    f.write(f" ({refs_with_year/total*100:.1f}%)\n")
                else:
                    f.write("\n")
                    
                f.write(f"References with volume: {refs_with_volume}")
                if total > 0:
                    f.write(f" ({refs_with_volume/total*100:.1f}%)\n")
                else:
                    f.write("\n")
                    
                f.write(f"References with page: {refs_with_page}")
                if total > 0:
                    f.write(f" ({refs_with_page/total*100:.1f}%)\n")
                else:
                    f.write("\n")
                
                # Field Match Statistics (ALL fields)
                f.write(f"\nField Match Statistics:\n")
                f.write(f"{'─'*50}\n")
                
                # DOI matches
                doi_matches = stats.get('doi_matches', 0)
                f.write(f"DOI matches: {doi_matches}")
                if refs_with_doi > 0:
                    doi_pct = (doi_matches / refs_with_doi * 100)
                    f.write(f" ({doi_pct:.1f}% of refs with DOI)\n")
                else:
                    f.write("\n")
                
                # Author matches
                author_exact_matches = stats.get('author_exact_matches', 0)
                f.write(f"Author exact matches: {author_exact_matches}")
                if refs_with_author > 0:
                    author_pct = (author_exact_matches / refs_with_author * 100)
                    f.write(f" ({author_pct:.1f}% of refs with author)\n")
                else:
                    f.write("\n")
                
                # Year matches
                year_exact_matches = stats.get('year_exact_matches', 0)
                year_adjacent_matches = stats.get('year_adjacent_matches', 0)
                year_total_matches = year_exact_matches + year_adjacent_matches
                f.write(f"Year exact matches: {year_exact_matches}\n")
                f.write(f"Year adjacent matches (±1): {year_adjacent_matches}\n")
                f.write(f"Year total matches: {year_total_matches}")
                if refs_with_year > 0:
                    year_pct = (year_total_matches / refs_with_year * 100)
                    f.write(f" ({year_pct:.1f}% of refs with year)\n")
                else:
                    f.write("\n")
                
                # Title matches
                title_exact_matches = stats.get('title_exact_matches', 0)
                title_fuzzy_matches = stats.get('title_fuzzy_matches', 0)
                title_total_matches = title_exact_matches + title_fuzzy_matches
                f.write(f"Title exact matches: {title_exact_matches}\n")
                f.write(f"Title fuzzy matches: {title_fuzzy_matches}\n")
                f.write(f"Title total matches: {title_total_matches}")
                if refs_with_title > 0:
                    title_pct = (title_total_matches / refs_with_title * 100)
                    f.write(f" ({title_pct:.1f}% of refs with title)\n")
                else:
                    f.write("\n")
                
                # Volume matches
                volume_matches = stats.get('volume_matches', 0)
                f.write(f"Volume matches: {volume_matches}")
                if refs_with_volume > 0:
                    volume_pct = (volume_matches / refs_with_volume * 100)
                    f.write(f" ({volume_pct:.1f}% of refs with volume)\n")
                else:
                    f.write("\n")
                
                # Page matches
                page_matches = stats.get('page_matches', 0)
                f.write(f"Page matches: {page_matches}")
                if refs_with_page > 0:
                    page_pct = (page_matches / refs_with_page * 100)
                    f.write(f" ({page_pct:.1f}% of refs with page)\n")
                else:
                    f.write("\n")
                
                # Query Type Distribution
                if stats.get('query_types'):
                    f.write(f"\nQuery Type Distribution:\n")
                    f.write(f"{'─'*50}\n")
                    total_query = sum(stats['query_types'].values())
                    for qtype, count in sorted(stats['query_types'].items(), key=lambda x: x[1], reverse=True):
                        qtype_percentage = (count / total_query * 100) if total_query > 0 else 0
                        f.write(f"  {qtype}: {count} ({qtype_percentage:.1f}%)\n")
                
                # Field Contribution to Matches
                if matches > 0:
                    f.write(f"\nField Contribution to Matches:\n")
                    f.write(f"{'─'*50}\n")
                    
                    if doi_matches > 0:
                        doi_contrib = (doi_matches / matches * 100)
                        f.write(f"DOI: {doi_contrib:.1f}% of all matches\n")
                        
                    if author_exact_matches > 0:
                        author_contrib = (author_exact_matches / matches * 100)
                        f.write(f"Author: {author_contrib:.1f}% of all matches\n")
                        
                    if year_total_matches > 0:
                        year_contrib = (year_total_matches / matches * 100)
                        f.write(f"Year: {year_contrib:.1f}% of all matches\n")
                        
                    if title_total_matches > 0:
                        title_contrib = (title_total_matches / matches * 100)
                        f.write(f"Title: {title_contrib:.1f}% of all matches\n")
                        
                    if volume_matches > 0:
                        volume_contrib = (volume_matches / matches * 100)
                        f.write(f"Volume: {volume_contrib:.1f}% of all matches\n")
                        
                    if page_matches > 0:
                        page_contrib = (page_matches / matches * 100)
                        f.write(f"Page: {page_contrib:.1f}% of all matches\n")
            
                f.write(f"\nTimestamp: {datetime.now().isoformat()}\n")
        
            logging.info(f"Aggregate statistics saved to: {stats_file}")
        
        except Exception as e:
            logging.error(f"Error writing aggregate stats file: {e}")

def batch_process_with_recovery(input_dir: str, output_dir: str, 
                                threshold: int = 26, 
                                use_grobid: bool = False,
                                batch_size: int = 3,
                                checkpoint_interval: int = 10,
                                error_threshold: int = 10):  
    """
    Convenient wrapper for batch processing with automatic recovery
    """
    try:
        # Initialize processor
        processor = ReferenceProcessor(
            use_grobid=use_grobid
        )
        
        # Initialize batch processor
        batch_processor = BatchProcessor(
        reference_processor=processor,
        batch_size=batch_size,
        pause_duration=10,
        error_threshold=error_threshold, 
        use_doi=True,
        checkpoint_interval=checkpoint_interval
        )
        
        # Process with resume capability
        batch_processor.process_directory_with_resume(
            input_dir=input_dir,
            output_dir=output_dir,
            threshold=threshold,
            force_restart=False
        )
        
        # Print final stats
        stats = batch_processor.get_processing_stats('processing_checkpoint.pkl')
        logging.info(f"Processing complete. Total files processed: {stats['processed_files']}")
        
        if use_grobid:
            logging.info(f"Grobid fallbacks used: {processor.grobid_fallback_count}")
            
    except KeyboardInterrupt:
        logging.info("Processing interrupted by user. Progress has been saved.")
    except Exception as e:
        logging.error(f"Batch processing failed: {e}")
        raise


async def process_single(processor: ReferenceProcessor, input_file: str, output_file: str = None, 
                  threshold: int = 26, use_doi: bool = True):
    """FIXED: Added file validation and use_doi parameter"""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' does not exist.")

    if output_file is None:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = f"{base_name}_matches_GS.csv"

    print(f"Processing file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Using DOI: {'Yes' if use_doi else 'No'}")
    
    try:
        await processor.process_file(input_file, output_file, threshold, use_doi)

        print("Processing completed successfully!")
    except Exception as e:
        logging.error(f"Error processing file: {e}")
        raise


async def main():
    """FIXED: Enhanced argument parsing and validation"""
    parser = argparse.ArgumentParser(
        description='Process references from Crossref JSON or TEI XML files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('input', help='Path to input file or directory')
    parser.add_argument('--batch', '-b', action='store_true', 
                       help='Process all files in the input directory')
    parser.add_argument('--output', '-o', 
                       help='Output file (single) or directory (batch)')
    parser.add_argument('--threshold', '-t', type=int, default=26, 
                       help='Matching score threshold')
    parser.add_argument('--use-grobid', action='store_true', 
                       help='Enable Grobid for unstructured references')
    parser.add_argument('--grobid-config', type=str,
                       help='Path to Grobid config file')
    parser.add_argument('--use-doi', action='store_true', default=True,
                       help='Use DOI in queries (default: True)')
    parser.add_argument('--no-doi', dest='use_doi', action='store_false',
                       help='Disable DOI usage in queries')
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT,
                       help='SPARQL query timeout in seconds')
    parser.add_argument('--max-retries', type=int, default=DEFAULT_MAX_RETRIES,
                       help='Maximum number of retries for failed queries')
    parser.add_argument('--batch-size', type=int, default=3,
                       help='Number of files to process in each batch')
    parser.add_argument('--pause-duration', type=int, default=10,
                       help='Pause between batches in seconds')
    parser.add_argument('--error-threshold', type=int, default=10,
                    help='Maximum server errors before stopping (default: 10)')
    parser.add_argument('--log-level', type=str, 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Logging level')
    parser.add_argument('--rate-limit', type=float, default=2.5,
                       help='Requests per second (default: 2.5)')
    parser.add_argument('--burst-size', type=int, default=10,
                       help='Maximum concurrent requests (default: 10)')
    args = parser.parse_args()
    log_level = getattr(logging, args.log_level)
    setup_logging(log_level=log_level)
    # Validation
    if not os.path.exists(args.input):
        parser.error(f"Input path does not exist: {args.input}")

    if args.batch and not os.path.isdir(args.input):
        parser.error("Input must be a directory when using --batch")

    if not args.batch and os.path.isdir(args.input):
        parser.error("Input appears to be a directory. Use --batch for directory processing")

    # Threshold validation
    if args.threshold < 0 or args.threshold > 100:
        parser.error("Threshold must be between 0 and 100")

    # Initialize processor
    processor = None  

    # Initialize processor
    try:
        # 1. Create a config object first
        config = MatcherConfig()
        
        # 2. Update the config object with your arguments
        config.max_retries = args.max_retries
        config.default_timeout = args.timeout
        config.requests_per_second = args.rate_limit  
        config.burst_size = args.burst_size         
        # 3. Pass the configured object to the processor
        processor = ReferenceProcessor(
            use_grobid=args.use_grobid,
            grobid_config=args.grobid_config,
            config=config  # <-- Pass the config here
        )
        
    except Exception as e:
        parser.error(f"Failed to initialize processor: {e}")
        return 1  # <-- Added this to force exit

    # Process files
    try:
        if args.batch:
            batch_processor = BatchProcessor(
            processor,
            batch_size=args.batch_size,
            pause_duration=args.pause_duration,
            error_threshold=args.error_threshold,  
            use_doi=args.use_doi,
                )
            output_dir = args.output or f"{args.input}_processed"
            await batch_processor.process_directory(args.input, output_dir, args.threshold)
            
            if args.use_grobid:
                print(f"Total Grobid fallbacks performed: {processor.grobid_fallback_count}")
        else:
            await process_single(
                processor, 
                args.input, 
                args.output, 
                args.threshold, 
                args.use_doi
            )
            if args.use_grobid:
                print(f"Total Grobid fallbacks performed: {processor.grobid_fallback_count}")

    except KeyboardInterrupt:
        logging.info("Processing interrupted by user")
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        return 1

    return 0

def generate_file_breakdown_box(stats: Dict, is_aggregate: bool) -> str:
    """Generate a visual breakdown box for file processing"""
    if not is_aggregate:
        return ""
    
    total_attempted = stats.get('total_files_attempted', 0)
    files_processed = stats.get('files_processed', 0)
    empty_files = stats.get('empty_files', 0)
    
    if total_attempted == 0:
        return ""
    
    success_percent = (files_processed / total_attempted * 100) if total_attempted > 0 else 0
    empty_percent = (empty_files / total_attempted * 100) if total_attempted > 0 else 0
    
    return f'''
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 25px; border-radius: 12px; margin: 20px 0; color: white;">
            <h3 style="margin: 0 0 15px 0; color: white;">📊 File Processing Breakdown</h3>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
                <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 32px; font-weight: bold;">{total_attempted}</div>
                    <div style="font-size: 12px; opacity: 0.9;">TOTAL ATTEMPTED</div>
                </div>
                <div style="background: rgba(39,174,96,0.3); padding: 15px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 32px; font-weight: bold;">{files_processed}</div>
                    <div style="font-size: 12px; opacity: 0.9;">WITH DATA ({success_percent:.1f}%)</div>
                </div>
                <div style="background: rgba(243,156,18,0.3); padding: 15px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 32px; font-weight: bold;">{empty_files}</div>
                    <div style="font-size: 12px; opacity: 0.9;">EMPTY SOURCE ({empty_percent:.1f}%)</div>
                </div>
            </div>
            <div style="margin-top: 15px; padding: 12px; background: rgba(255,255,255,0.1); 
                        border-radius: 6px; font-size: 13px;">
                <strong>ℹ️ Note:</strong> Empty files have 0 references in the source data (not processing failures)
            </div>
        </div>'''
def generate_processing_report(output_dir: str, stats: Dict):
    """Generate enhanced HTML report with all field statistics"""
    
    report_file = os.path.join(output_dir, "processing_report.html")
    
    # Determine report type
    is_aggregate = stats.get('files_processed', 0) > 1
    files_count = stats.get('files_processed', 1)
    total_attempted = stats.get('total_files_attempted', files_count)
    title_suffix = f" - {files_count}/{total_attempted} Files" if is_aggregate else ""
    
    # Calculate basic percentages
    total_refs = stats.get('total_references', 0)
    matches = stats.get('matches_found', 0)
    errors = stats.get('errors', 0)
    match_rate = (matches / total_refs * 100) if total_refs > 0 else 0
    
    # Calculate field statistics
    def calc_field_stats(refs_with_field, field_matches):
        if refs_with_field > 0:
            coverage = (refs_with_field / total_refs * 100) if total_refs > 0 else 0
            match_rate = (field_matches / refs_with_field * 100)
            contribution = (field_matches / matches * 100) if matches > 0 else 0
            return coverage, match_rate, contribution
        return 0, 0, 0
    
    # DOI stats
    refs_with_doi = stats.get('refs_with_doi', 0)
    doi_matches = stats.get('doi_matches', 0)
    doi_coverage, doi_match_rate, doi_contribution = calc_field_stats(refs_with_doi, doi_matches)
    
    # Author stats
    refs_with_author = stats.get('refs_with_author', 0)
    author_matches = stats.get('author_exact_matches', 0)
    author_coverage, author_match_rate, author_contribution = calc_field_stats(refs_with_author, author_matches)
    
    # Title stats
    refs_with_title = stats.get('refs_with_title', 0)
    title_exact = stats.get('title_exact_matches', 0)
    title_fuzzy = stats.get('title_fuzzy_matches', 0)
    title_total = title_exact + title_fuzzy
    title_coverage, title_match_rate, title_contribution = calc_field_stats(refs_with_title, title_total)
    
    # Year stats
    refs_with_year = stats.get('refs_with_year', 0)
    year_exact = stats.get('year_exact_matches', 0)
    year_adjacent = stats.get('year_adjacent_matches', 0)
    year_total = year_exact + year_adjacent
    year_coverage, year_match_rate, year_contribution = calc_field_stats(refs_with_year, year_total)
    
    # Volume stats
    refs_with_volume = stats.get('refs_with_volume', 0)
    volume_matches = stats.get('volume_matches', 0)
    volume_coverage, volume_match_rate, volume_contribution = calc_field_stats(refs_with_volume, volume_matches)
    
    # Page stats
    refs_with_page = stats.get('refs_with_page', 0)
    page_matches = stats.get('page_matches', 0)
    page_coverage, page_match_rate, page_contribution = calc_field_stats(refs_with_page, page_matches)

    # Grobid stats
    grobid_fallbacks = stats.get('grobid_fallbacks', 0)
    grobid_successes = stats.get('grobid_successes', 0)
    grobid_success_rate = (grobid_successes / grobid_fallbacks * 100) if grobid_fallbacks > 0 else 0

    # Generate query type rows HTML
    query_type_rows_html = generate_query_type_rows(stats)
    
    # Generate aggregate stats link if applicable
    aggregate_stats_link = ''
    if is_aggregate:
        aggregate_stats_link = '<li><a href="aggregate_stats.txt">📊 <strong>Aggregate Stats</strong> - Text summary of all files</a></li>'
    
    # Generate files processed metadata
    files_metadata = ''
    if is_aggregate:
        total_attempted = stats.get('total_files_attempted', files_count)
        empty_files = stats.get('empty_files', 0)
        
        files_metadata = f'''
            <p><strong>📁 Total Files Attempted:</strong> {total_attempted}</p>
            <p><strong>✅ Files with Data:</strong> {files_count}</p>
            <p><strong>📭 Empty Files (0 refs in source):</strong> {empty_files}</p>'''
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Reference Matching Report{title_suffix}</title>
        <meta charset="utf-8">
        <style>
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            .container {{ 
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            }}
            h1 {{ 
                color: #2c3e50;
                border-bottom: 4px solid #3498db;
                padding-bottom: 15px;
                margin-top: 0;
                font-size: 2.5em;
            }}
            h2 {{ 
                color: #34495e;
                margin-top: 40px;
                margin-bottom: 20px;
                font-size: 1.8em;
                border-left: 5px solid #3498db;
                padding-left: 15px;
            }}
            .metadata {{
                background: #ecf0f1;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 30px;
            }}
            .metadata p {{
                margin: 5px 0;
                color: #555;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .stat-box {{ 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 25px;
                border-radius: 12px;
                text-align: center;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                transition: transform 0.3s ease;
            }}
            .stat-box:hover {{
                transform: translateY(-5px);
                box-shadow: 0 6px 20px rgba(0,0,0,0.15);
            }}
            .stat-value {{ 
                font-size: 42px;
                font-weight: bold;
                color: white;
                margin: 10px 0;
            }}
            .stat-label {{ 
                color: rgba(255,255,255,0.9);
                font-size: 14px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            .success {{ color: #27ae60; }}
            .warning {{ color: #f39c12; }}
            .error {{ color: #e74c3c; }}
            
            table {{ 
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            th {{ 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px;
                text-align: left;
                font-weight: 600;
                text-transform: uppercase;
                font-size: 14px;
                letter-spacing: 0.5px;
            }}
            td {{ 
                padding: 12px 15px;
                border-bottom: 1px solid #ecf0f1;
            }}
            tr:hover {{ 
                background: #f8f9fa;
            }}
            tr:last-child td {{
                border-bottom: none;
            }}
            
            .progress-bar {{ 
                width: 100%;
                height: 40px;
                background: #ecf0f1;
                border-radius: 20px;
                overflow: hidden;
                box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
                position: relative;
            }}
            .progress-fill {{ 
                height: 100%;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                transition: width 0.5s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                font-size: 16px;
            }}
            
            .info-card {{
                background: #f8f9fa;
                border-left: 4px solid #3498db;
                padding: 15px;
                margin: 20px 0;
                border-radius: 5px;
            }}
            
            .log-links {{
                background: #ecf0f1;
                padding: 20px;
                border-radius: 10px;
                margin-top: 30px;
            }}
            .log-links ul {{
                list-style: none;
                padding: 0;
                margin: 0;
            }}
            .log-links li {{
                margin: 10px 0;
                padding: 10px;
                background: white;
                border-radius: 5px;
                transition: background 0.3s ease;
            }}
            .log-links li:hover {{
                background: #e3e7eb;
            }}
            .log-links a {{
                color: #3498db;
                text-decoration: none;
                font-weight: 500;
                display: flex;
                align-items: center;
            }}
            .log-links a:hover {{
                color: #2980b9;
            }}
            
            .badge {{
                display: inline-block;
                padding: 5px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: bold;
                margin-left: 10px;
            }}
            .badge-success {{
                background: #27ae60;
                color: white;
            }}
            .badge-warning {{
                background: #f39c12;
                color: white;
            }}
            .badge-info {{
                background: #3498db;
                color: white;
            }}
            
            @media print {{
                body {{
                    background: white;
                }}
                .container {{
                    box-shadow: none;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Reference Matching Report{title_suffix}</h1>
            
            <div class="metadata">
                <p><strong>📅 Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                {files_metadata}
                <p><strong>🎯 Report Type:</strong> {"Aggregate (Multiple Files)" if is_aggregate else "Single File"}</p>
            </div>

            {generate_file_breakdown_box(stats, is_aggregate)}
            <h2>📈 Summary Statistics</h2>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-label">Total References</div>
                    <div class="stat-value">{total_refs:,}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Matches Found</div>
                    <div class="stat-value">{matches:,}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Match Rate</div>
                    <div class="stat-value">{match_rate:.1f}%</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Errors</div>
                    <div class="stat-value">{errors:,}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">🔧 Grobid Fallbacks</div>
                    <div class="stat-value">{grobid_fallbacks:,}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">✅ Grobid Successes</div>
                    <div class="stat-value">{grobid_successes:,}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Grobid Success Rate</div>
                    <div class="stat-value">{grobid_success_rate:.1f}%</div>
                </div>
            </div>

            <h2>📊 Match Rate Progress</h2>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {match_rate:.1f}%">
                    {match_rate:.1f}% ({matches:,}/{total_refs:,})
                </div>
            </div>
            
            <div class="info-card">
                <strong>ℹ️ Coverage Breakdown</strong>
                <ul style="margin: 10px 0 0 20px;">
                    <li>References with DOI: {refs_with_doi:,} ({doi_coverage:.1f}%)</li>
                    <li>References with Title: {refs_with_title:,} ({title_coverage:.1f}%)</li>
                    <li>References with Author: {refs_with_author:,} ({author_coverage:.1f}%)</li>
                    <li>References with Year: {refs_with_year:,} ({year_coverage:.1f}%)</li>
                    <li>References with Volume: {refs_with_volume:,} ({volume_coverage:.1f}%)</li>
                    <li>References with Page: {refs_with_page:,} ({page_coverage:.1f}%)</li>
                </ul>
            </div>
            
            <h2>🔍 Query Type Breakdown</h2>
            <table>
                <thead>
                    <tr>
                        <th>Query Type</th>
                        <th style="text-align: center;">Count</th>
                        <th style="text-align: center;">Percentage</th>
                        <th style="text-align: center;">Status</th>
                    </tr>
                </thead>
                <tbody>
                    {query_type_rows_html}
                </tbody>
            </table>
            
            <h2>📊 Field-by-Field Match Analysis</h2>
            
            <h3>🔗 DOI Statistics</h3>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th style="text-align: right;">Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>References with DOI</strong></td>
                        <td style="text-align: right;">{refs_with_doi:,}</td>
                    </tr>
                    <tr>
                        <td><strong>DOI Matches</strong></td>
                        <td style="text-align: right;" class="success"><strong>{doi_matches:,}</strong></td>
                    </tr>
                    <tr>
                        <td><strong>DOI Match Rate</strong></td>
                        <td style="text-align: right;">
                            <strong>{doi_match_rate:.1f}%</strong>
                            <span class="badge {"badge-success" if doi_match_rate >= 50 else "badge-warning"}">
                                {"Good" if doi_match_rate >= 50 else "Needs Improvement"}
                            </span>
                        </td>
                    </tr>
                    <tr>
                        <td><strong>DOI Contribution to Matches</strong></td>
                        <td style="text-align: right;">
                            {doi_contribution:.1f}% of all matches
                        </td>
                    </tr>
                </tbody>
            </table>
            
            <h3>👤 Author Statistics</h3>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th style="text-align: right;">Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>References with Author</strong></td>
                        <td style="text-align: right;">{refs_with_author:,}</td>
                    </tr>
                    <tr>
                        <td><strong>Author Exact Matches</strong></td>
                        <td style="text-align: right;" class="success"><strong>{author_matches:,}</strong></td>
                    </tr>
                    <tr>
                        <td><strong>Author Match Rate</strong></td>
                        <td style="text-align: right;">
                            <strong>{author_match_rate:.1f}%</strong>
                            <span class="badge {"badge-success" if author_match_rate >= 50 else "badge-warning"}">
                                {"Good" if author_match_rate >= 50 else "Needs Improvement"}
                            </span>
                        </td>
                    </tr>
                    <tr>
                        <td><strong>Author Contribution to Matches</strong></td>
                        <td style="text-align: right;">
                            {author_contribution:.1f}% of all matches
                        </td>
                    </tr>
                </tbody>
            </table>
            
            <h3>📰 Title Statistics</h3>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th style="text-align: right;">Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>References with Title</strong></td>
                        <td style="text-align: right;">{refs_with_title:,}</td>
                    </tr>
                    <tr>
                        <td><strong>Title Exact Matches</strong></td>
                        <td style="text-align: right;" class="success"><strong>{title_exact:,}</strong></td>
                    </tr>
                    <tr>
                        <td><strong>Title Fuzzy Matches</strong></td>
                        <td style="text-align: right;"><strong>{title_fuzzy:,}</strong></td>
                    </tr>
                    <tr>
                        <td><strong>Title Total Matches</strong></td>
                        <td style="text-align: right;" class="success"><strong>{title_total:,}</strong></td>
                    </tr>
                    <tr>
                        <td><strong>Title Match Rate</strong></td>
                        <td style="text-align: right;">
                            <strong>{title_match_rate:.1f}%</strong>
                            <span class="badge {"badge-success" if title_match_rate >= 50 else "badge-warning"}">
                                {"Good" if title_match_rate >= 50 else "Needs Improvement"}
                            </span>
                        </td>
                    </tr>
                    <tr>
                        <td><strong>Title Contribution to Matches</strong></td>
                        <td style="text-align: right;">
                            {title_contribution:.1f}% of all matches
                        </td>
                    </tr>
                </tbody>
            </table>
            
            <h3>📅 Year Statistics</h3>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th style="text-align: right;">Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>References with Year</strong></td>
                        <td style="text-align: right;">{refs_with_year:,}</td>
                    </tr>
                    <tr>
                        <td><strong>Year Exact Matches</strong></td>
                        <td style="text-align: right;" class="success"><strong>{year_exact:,}</strong></td>
                    </tr>
                    <tr>
                        <td><strong>Year Adjacent Matches (±1)</strong></td>
                        <td style="text-align: right;"><strong>{year_adjacent:,}</strong></td>
                    </tr>
                    <tr>
                        <td><strong>Year Total Matches</strong></td>
                        <td style="text-align: right;" class="success"><strong>{year_total:,}</strong></td>
                    </tr>
                    <tr>
                        <td><strong>Year Match Rate</strong></td>
                        <td style="text-align: right;">
                            <strong>{year_match_rate:.1f}%</strong>
                            <span class="badge {"badge-success" if year_match_rate >= 50 else "badge-warning"}">
                                {"Good" if year_match_rate >= 50 else "Needs Improvement"}
                            </span>
                        </td>
                    </tr>
                    <tr>
                        <td><strong>Year Contribution to Matches</strong></td>
                        <td style="text-align: right;">
                            {year_contribution:.1f}% of all matches
                        </td>
                    </tr>
                </tbody>
            </table>
            
            <h3>📚 Volume Statistics</h3>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th style="text-align: right;">Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>References with Volume</strong></td>
                        <td style="text-align: right;">{refs_with_volume:,}</td>
                    </tr>
                    <tr>
                        <td><strong>Volume Matches</strong></td>
                        <td style="text-align: right;" class="success"><strong>{volume_matches:,}</strong></td>
                    </tr>
                    <tr>
                        <td><strong>Volume Match Rate</strong></td>
                        <td style="text-align: right;">
                            <strong>{volume_match_rate:.1f}%</strong>
                            <span class="badge {"badge-success" if volume_match_rate >= 50 else "badge-warning"}">
                                {"Good" if volume_match_rate >= 50 else "Needs Improvement"}
                            </span>
                        </td>
                    </tr>
                    <tr>
                        <td><strong>Volume Contribution to Matches</strong></td>
                        <td style="text-align: right;">
                            {volume_contribution:.1f}% of all matches
                        </td>
                    </tr>
                </tbody>
            </table>
            
            <h3>📄 Page Statistics</h3>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th style="text-align: right;">Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>References with Page</strong></td>
                        <td style="text-align: right;">{refs_with_page:,}</td>
                    </tr>
                    <tr>
                        <td><strong>Page Matches</strong></td>
                        <td style="text-align: right;" class="success"><strong>{page_matches:,}</strong></td>
                    </tr>
                    <tr>
                        <td><strong>Page Match Rate</strong></td>
                        <td style="text-align: right;">
                            <strong>{page_match_rate:.1f}%</strong>
                            <span class="badge {"badge-success" if page_match_rate >= 50 else "badge-warning"}">
                                {"Good" if page_match_rate >= 50 else "Needs Improvement"}
                            </span>
                        </td>
                    </tr>
                    <tr>
                        <td><strong>Page Contribution to Matches</strong></td>
                        <td style="text-align: right;">
                            {page_contribution:.1f}% of all matches
                        </td>
                    </tr>
                </tbody>
            </table>

            <h3>🔧 Grobid Fallback Statistics</h3>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th style="text-align: right;">Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Grobid Fallback Attempts</strong></td>
                        <td style="text-align: right;">{grobid_fallbacks:,}</td>
                    </tr>
                    <tr>
                        <td><strong>Successful Matches After Grobid</strong></td>
                        <td style="text-align: right;" class="success"><strong>{grobid_successes:,}</strong></td>
                    </tr>
                    <tr>
                        <td><strong>Grobid Success Rate</strong></td>
                        <td style="text-align: right;">
                            <strong>{grobid_success_rate:.1f}%</strong>
                            <span class="badge {"badge-success" if grobid_success_rate >= 50 else "badge-warning"}">
                                {"Good" if grobid_success_rate >= 50 else "Needs Improvement"}
                            </span>
                        </td>
                    </tr>
                    <tr>
                        <td><strong>Grobid Contribution to Total Matches</strong></td>
                        <td style="text-align: right;">
                            {(grobid_successes / matches * 100) if matches > 0 else 0:.1f}% of all matches
                        </td>
                    </tr>
                </tbody>
            </table>

            <h2>📁 Log Files</h2>
            <div class="log-links">
                <ul>
                    <li>
                        <a href="reference_matching_main.log">
                            📄 <strong>Main Log</strong> - All processing logs
                        </a>
                    </li>
                    <li>
                        <a href="reference_matching_authors.log">
                            👤 <strong>Authors Log</strong> - Author extraction and matching details
                        </a>
                    </li>
                    <li>
                        <a href="reference_matching_queries.log">
                            🔍 <strong>Queries Log</strong> - SPARQL query execution logs
                        </a>
                    </li>
                    <li>
                        <a href="reference_matching_scores.log">
                            🎯 <strong>Scores Log</strong> - Match scoring details
                        </a>
                    </li>
                    <li>
                        <a href="reference_matching_errors.log">
                            ⚠️ <strong>Errors Log</strong> - Errors and warnings
                        </a>
                    </li>
                    {aggregate_stats_link}
                </ul>
            </div>
            
            <div class="info-card" style="margin-top: 30px; border-left-color: #9b59b6;">
                <strong>💡 Interpretation Guide</strong>
                <ul style="margin: 10px 0 0 20px;">
                    <li><strong>Match Rate 60-100%:</strong> Excellent coverage - Most references found in OpenCitations</li>
                    <li><strong>Match Rate 40-60%:</strong> Good coverage - Typical for mixed datasets</li>
                    <li><strong>Match Rate 20-40%:</strong> Fair coverage - Many conference papers or older publications</li>
                    <li><strong>Match Rate 0-20%:</strong> Low coverage - Check data quality or database availability</li>
                </ul>
                <strong style="margin-top: 15px; display: block;">📊 Field Contribution Guide</strong>
                <ul style="margin: 10px 0 0 20px;">
                    <li><strong>High contribution (>60%):</strong> This field is critical for matching</li>
                    <li><strong>Medium contribution (30-60%):</strong> This field is important</li>
                    <li><strong>Low contribution (<30%):</strong> This field has limited impact</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        # logging.info(f"📄 HTML report generated: {report_file}")
    except Exception as e:
        logging.error(f"Error generating report: {e}")
def generate_query_type_rows(stats: Dict) -> str:
    """Generate HTML table rows for query type breakdown with enhanced styling"""
    query_types = stats.get('query_types', {})
    if not query_types:
        return '<tr><td colspan="4" style="text-align: center; color: #95a5a6;">No query type data available</td></tr>'
    
    total = sum(query_types.values())
    rows = []
    
    # Define icons and descriptions for query types
    query_info = {
        'doi_title': ('🔗📰', 'DOI + Title'),
        'year_and_doi': ('📅🔗', 'Year + DOI'),
        'author_title': ('👤📰', 'Author + Title'),
        'year_author_page': ('📅👤📄', 'Year + Author + Page'),
        'year_volume_page': ('📅📚📄', 'Year + Volume + Page'),
        'year_author_volume': ('📅👤📚', 'Year + Author + Volume')
    }
    
    for qtype, count in sorted(query_types.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total * 100) if total > 0 else 0
        icon, description = query_info.get(qtype, ('🔍', qtype))
        
        # Badge color based on count
        if percentage >= 40:
            badge_class = 'badge-success'
            status = 'Primary'
        elif percentage >= 20:
            badge_class = 'badge-info'
            status = 'Secondary'
        else:
            badge_class = 'badge-warning'
            status = 'Minor'
        
        rows.append(f"""
            <tr>
                <td>
                    <span style="font-size: 18px;">{icon}</span>
                    <strong style="margin-left: 10px;">{qtype}</strong>
                    <br>
                    <small style="color: #7f8c8d; margin-left: 30px;">{description}</small>
                </td>
                <td style="text-align: center;"><strong>{count:,}</strong></td>
                <td style="text-align: center;">{percentage:.1f}%</td>
                <td style="text-align: center;">
                    <span class="badge {badge_class}">{status}</span>
                </td>
            </tr>
        """)
    
    return "\n".join(rows)
if __name__ == "__main__":
    asyncio.run(main())    