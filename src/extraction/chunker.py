"""
Token-based text chunking for LLM processing.

Uses tiktoken for accurate token counting (OpenAI tokenizer).
"""
import tiktoken
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class TokenChunker:
    """
    Split text into chunks based on token count.

    Uses tiktoken for accurate token counting compatible with OpenAI models.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        chunk_size: int = 4000,
        overlap: int = 200
    ):
        """
        Initialize chunker.

        Args:
            model: Model name for tokenizer selection
            chunk_size: Maximum tokens per chunk
            overlap: Token overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

        # Get tokenizer for the model
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base (used by GPT-4, GPT-3.5-turbo)
            self.encoding = tiktoken.get_encoding("cl100k_base")

        logger.debug(f"Initialized chunker: {chunk_size} tokens, {overlap} overlap")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        tokens = self.encoding.encode(text)
        total_tokens = len(tokens)

        if total_tokens <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < total_tokens:
            end = min(start + self.chunk_size, total_tokens)

            # Get chunk tokens and decode back to text
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)

            # Try to break at sentence boundary if not at the end
            if end < total_tokens:
                # Look for sentence end in last 20% of chunk
                search_start = int(len(chunk_text) * 0.8)
                last_period = chunk_text.rfind('. ', search_start)

                if last_period > 0:
                    chunk_text = chunk_text[:last_period + 1]
                    # Recalculate end based on actual text used
                    end = start + len(self.encoding.encode(chunk_text))

            chunks.append(chunk_text.strip())

            # Move start with overlap
            start = end - self.overlap

            # Prevent infinite loop
            if start >= total_tokens - self.overlap:
                break

        logger.debug(f"Split {total_tokens} tokens into {len(chunks)} chunks")
        return chunks

    def chunk_sections(
        self,
        sections: dict,
        max_total_tokens: Optional[int] = None
    ) -> List[str]:
        """
        Chunk multiple sections, prioritizing important ones.

        Args:
            sections: Dict of section_name -> text
            max_total_tokens: Optional limit on total tokens

        Returns:
            List of chunks from all sections
        """
        # Priority order for sections
        priority = ['mda', 'risk_factors', 'liquidity', 'business']

        all_chunks = []
        total_tokens = 0

        for section_name in priority:
            if section_name not in sections:
                continue

            section_text = sections[section_name]

            # Check if we'd exceed limit
            if max_total_tokens:
                section_tokens = self.count_tokens(section_text)
                if total_tokens + section_tokens > max_total_tokens:
                    # Truncate section to fit
                    remaining = max_total_tokens - total_tokens
                    if remaining < 500:  # Not worth adding
                        break
                    tokens = self.encoding.encode(section_text)[:remaining]
                    section_text = self.encoding.decode(tokens)

            # Chunk this section
            chunks = self.chunk_text(section_text)

            # Add section marker to first chunk
            if chunks:
                chunks[0] = f"[SECTION: {section_name.upper()}]\n\n{chunks[0]}"

            all_chunks.extend(chunks)
            total_tokens += sum(self.count_tokens(c) for c in chunks)

            if max_total_tokens and total_tokens >= max_total_tokens:
                break

        return all_chunks


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-4.1-mini"
) -> float:
    """
    Estimate API cost for given token counts.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model name

    Returns:
        Estimated cost in USD
    """
    # Pricing as of 2026 (per 1M tokens)
    pricing = {
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
        "gpt-4.1": {"input": 2.00, "output": 8.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-5-mini": {"input": 1.00, "output": 4.00},
    }

    prices = pricing.get(model, pricing["gpt-4.1-mini"])

    input_cost = (input_tokens / 1_000_000) * prices["input"]
    output_cost = (output_tokens / 1_000_000) * prices["output"]

    return input_cost + output_cost
