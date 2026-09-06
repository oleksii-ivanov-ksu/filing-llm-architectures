# LLM-Extractor Module

## Overview

The LLM-Extractor extracts **factual information only** from SEC filings (10-K and 10-Q) without making predictions or recommendations.

## Components

### 1. `schemas.py`
Pydantic models for structured fact extraction:
- `ExtractedFacts`: Main schema with financial metrics, risks, guidance
- `RiskFactor`: Risk category, severity, description
- `Guidance`: Management forward-looking statements

### 2. `parser.py`
HTML parsing utilities:
- `SECFilingParser`: Extract sections (MD&A, Risk Factors, etc.)
- Text cleaning and chunking
- Table extraction

### 3. `prompts.py`
LLM prompts for extraction:
- System prompt emphasizing FACTS ONLY
- Extraction prompt with JSON schema
- Specialized prompts for risks and MD&A

### 4. `extractor.py`
Main extraction logic:
- `FilingExtractor`: Orchestrates extraction process
- Uses OpenAI API with structured output
- Batch processing support

## Usage

```python
from src.extraction import FilingExtractor
from pathlib import Path

# Initialize
extractor = FilingExtractor(
    api_key="your-openai-key",
    model="gpt-4o-mini",
    temperature=0.1,
    max_tokens=4000
)

# Extract from single filing
facts = extractor.extract_from_file(
    filing_path=Path("data/raw/AAPL/filings/10-K_2019-10-30.html"),
    ticker="AAPL",
    filing_type="10-K",
    filing_date="2019-10-30"
)

# Batch extract all filings for a ticker
results = extractor.extract_batch(
    filings_dir=Path("data/raw/AAPL/filings"),
    ticker="AAPL",
    output_dir=Path("data/extracted/AAPL")
)
```

## Running Extraction

Use the provided script:

```bash
python scripts/extract_facts.py
```

This will:
1. Process all 10 companies
2. Extract facts from all 320 filings
3. Save results to `data/extracted/{TICKER}/`

## Output Format

Each extracted filing is saved as JSON:

```json
{
  "ticker": "AAPL",
  "filing_date": "2019-10-30",
  "filing_type": "10-K",
  "fiscal_period": "FY 2019",
  "revenue": "$260.2 billion",
  "revenue_change": "+9% YoY",
  "gross_margin": "37.8%",
  "risk_factors": [...],
  "guidance": {...},
  "key_events": [...]
}
```

## Key Design Principles

1. **Facts Only**: No predictions, opinions, or recommendations
2. **Structured Output**: JSON format for downstream processing
3. **Source Attribution**: All facts extracted from original filing
4. **Low Temperature**: 0.1 for consistent, factual extraction
5. **Validation**: Pydantic schemas ensure data quality

## Next Steps

After extraction, the facts feed into:
- **Analyst Signals Module**: Combines with analyst consensus data
- **LLM-Reasoner**: Generates investment views with confidence levels
