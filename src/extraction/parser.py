"""
HTML parser utilities for SEC filings.
"""
from bs4 import BeautifulSoup
from typing import Optional, List
import re
import html
import logging

logger = logging.getLogger(__name__)


class SECFilingParser:
    """Parse HTML SEC filings and extract relevant sections."""

    def __init__(self, html_content: str, filename: str = "unknown"):
        """
        Initialize parser with HTML content.

        Args:
            html_content: Raw HTML string of SEC filing
            filename: Filename for logging purposes
        """
        self.filename = filename
        self.soup = BeautifulSoup(html_content, 'html.parser')
        # Get text and decode HTML entities (&#8217; -> ')
        raw_text = self.soup.get_text()
        # Decode HTML entities and normalize whitespace
        text = html.unescape(raw_text)
        # Replace non-breaking spaces and other unicode whitespace with regular space
        text = text.replace('\xa0', ' ')
        text = text.replace('\u200b', '')  # zero-width space
        self.text = text
    
    def _is_toc_content(self, text: str) -> bool:
        """
        Check if text looks like Table of Contents (many page numbers).

        TOC typically has patterns like "Overview38Results42" (word followed by page number).
        """
        # Count patterns like "Word123" or "Word 123" (word followed by page number)
        toc_pattern = r'[A-Za-z]{3,}\s*\d{1,3}(?=[A-Z]|\s|$)'
        matches = re.findall(toc_pattern, text[:2000])

        # If more than 10 such patterns in first 2000 chars, likely TOC
        return len(matches) > 10

    def extract_section(self, section_name: str) -> Optional[str]:
        """
        Extract a specific section from the filing.

        Args:
            section_name: Section to extract (e.g., "Risk Factors", "MD&A")

        Returns:
            Extracted section text or None if not found
        """
        # Flexible patterns for section headers (handles various formats)
        # Note: 10-K uses Item 7 for MD&A, 10-Q uses Item 2
        patterns = {
            "risk_factors": [
                # GE-style "STRATEGIC RISKS" section header (2018+ iXBRL format)
                # Must be before Item 1A to avoid matching TOC entry
                # Handles: "STRATEGIC RISKS. Strategic", "STRATEGIC RISKS  Strategic", "RISKSStrategic"
                r"STRATEGIC\s+RISKS\.?\s*Strategic\s+risk",
                # 10-K and 10-Q Part II Item 1A (standard format)
                # Note: [.\s:\-–—]+ handles period, space, colon, hyphen, en-dash, em-dash
                # \s+ allows newlines between words (e.g., "Risk\n    Factors")
                r"ITEM\s+1A[.\s:\-–—]+RISK\s+FACTORS",
                r"Item\s+1A[.\s:\-–—]+Risk\s+Factors",
                # Item 1A with parenthesis (e.g., "Item 1A. (Risk Factors and...")
                r"ITEM\s+1A[.\s:\-–—]+\(?RISK\s+FACTORS",
                r"Item\s+1A[.\s:\-–—]+\(?Risk\s+Factors",
                # Item 1A with "Business Risk" (e.g., CAT 2007)
                r"Item\s+1A[.\s:\-–—]+Business\s+Risk",
                # Standalone headers (older filings like Citigroup 2006-2008)
                # Match RISK FACTORS as section title (not in middle of sentence)
                r"(?:^|\s{2,})RISK\s+FACTORS(?:\s*$|\s{2,})",
                # Some older filings
                r"Risk\s+Factors\s+and\s+Cautionary\s+Statements",
            ],
            "mda": [
                # 10-K: Item 7
                r"ITEM\s*7[.\s:\-]+MANAGEMENT.{0,5}S?\s+DISCUSSION\s+AND\s+ANALYSIS",
                r"ITEM\s*7[.\s:\-]+Management.{0,5}s?\s+Discussion\s+and\s+Analysis",
                # 10-Q: Item 2 (Part I)
                r"ITEM\s*2[.\s:\-]+MANAGEMENT.{0,5}S?\s+DISCUSSION\s+AND\s+ANALYSIS",
                r"ITEM\s*2[.\s:\-]+Management.{0,5}s?\s+Discussion\s+and\s+Analysis",
                # Generic/fallback patterns (older filings)
                r"Management.{0,5}s?\s+Discussion\s+and\s+Analysis\s+of\s+Financial",
                r"FINANCIAL\s+REVIEW\s+Results\s+of\s+Operations",
                r"Financial\s+Review\s+Results\s+of\s+Operations",
            ],
            "liquidity": [
                r"Liquidity\s+and\s+Capital\s+Resources",
                r"LIQUIDITY\s+AND\s+CAPITAL\s+RESOURCES",
                r"Financial\s+Condition,?\s+Liquidity",
            ],
            "business": [
                r"ITEM\s*1[.\s:\-]+BUSINESS",
                r"ITEM\s*1[.\s:\-]+Business",
            ]
        }

        section_patterns = patterns.get(section_name.lower(), [section_name])

        # Minimum section length to filter out Table of Contents entries
        # Note: TOC entries are typically 500-1000 chars, real sections vary
        # We also have _is_toc_content() check, so 1500 is safe
        MIN_SECTION_LENGTH = 1500

        # Try to find section with each pattern
        for pattern in section_patterns:
            # Find ALL matches, not just the first one
            matches = list(re.finditer(pattern, self.text, re.IGNORECASE))

            if not matches:
                continue

            # For each match, calculate section length and pick the best one
            # Prefer: longest non-TOC section, then longest overall
            best_section = None
            best_length = 0
            best_is_toc = True

            for match in matches:
                start_pos = match.start()

                # Find next major section header (ITEM X. followed by section title)
                # Must be a section HEADER, not a reference like "See Item 7. Management's..."
                # Headers are typically preceded by newline(s) or significant whitespace
                # Pattern: newline/whitespace + ITEM + number + period + title
                next_section = re.search(
                    r'(?:^|\n)\s*ITEM\s+\d+[A-Z]?\.\s*[A-Z]',  # e.g., "\nItem 1B. U" or "\n  Item 2. P"
                    self.text[start_pos + 500:],
                    re.IGNORECASE
                )

                if next_section:
                    end_pos = start_pos + 500 + next_section.start()
                    section_text = self.text[start_pos:end_pos].strip()
                else:
                    # Take substantial chunk if no next section found
                    section_text = self.text[start_pos:start_pos + 80000].strip()

                is_toc = self._is_toc_content(section_text)
                section_len = len(section_text)

                # Prefer non-TOC sections over TOC sections
                # If both are TOC or both are non-TOC, prefer longer one
                if (not is_toc and best_is_toc) or \
                   (is_toc == best_is_toc and section_len > best_length):
                    best_length = section_len
                    best_section = section_text
                    best_is_toc = is_toc

            # Only return if section is substantial (not just a TOC entry)
            if best_section and best_length >= MIN_SECTION_LENGTH:
                if best_is_toc:
                    logger.debug(f"Warning: section '{section_name}' may be TOC ({best_length} chars)")
                else:
                    logger.debug(f"Found section '{section_name}' ({best_length} chars)")
                return best_section

        # Only warn for critical sections (risk_factors, mda), debug for optional ones
        if section_name in ('risk_factors', 'mda'):
            logger.warning(f"Section '{section_name}' not found in {self.filename}")
        else:
            logger.debug(f"Section '{section_name}' not found in {self.filename}")
        return None
    
    def extract_all_sections(self) -> dict:
        """
        Extract all major sections from the filing.
        
        Returns:
            Dictionary with section names as keys and content as values
        """
        sections = {}
        
        for section_key in ['risk_factors', 'mda', 'liquidity', 'business']:
            content = self.extract_section(section_key)
            if content:
                sections[section_key] = content
        
        return sections
    
    def clean_text(self, text: str, max_length: Optional[int] = None) -> str:
        """
        Clean extracted text (remove excessive whitespace, etc.).
        
        Args:
            text: Raw text to clean
            max_length: Optional maximum length to truncate to
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common SEC filing artifacts
        text = re.sub(r'\d+\s*Table of Contents', '', text, flags=re.IGNORECASE)
        
        # Truncate if needed
        if max_length and len(text) > max_length:
            text = text[:max_length] + "..."
        
        return text.strip()
    
    def extract_tables(self) -> List[str]:
        """
        Extract financial tables from the filing.
        
        Returns:
            List of table contents as strings
        """
        tables = []
        for table in self.soup.find_all('table'):
            table_text = table.get_text(separator=' | ', strip=True)
            if table_text and len(table_text) > 50:  # Ignore tiny tables
                tables.append(table_text)
        
        return tables
    
    def chunk_text(self, text: str, chunk_size: int = 4000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks for LLM processing.
        
        Args:
            text: Text to chunk
            chunk_size: Maximum chunk size in characters
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end near the chunk boundary
                sentence_end = text.rfind('. ', start + chunk_size - 200, end)
                if sentence_end > start:
                    end = sentence_end + 1
            
            chunks.append(text[start:end])
            start = end - overlap
        
        return chunks
