"""
Unified data verification script for the selected universe.
Generates detailed terminal report with all statistics.

Shows separately:
1. Downloaded files (actual HTML files in directories)
2. Filing links in filings.json (what should be downloadable)

This helps identify where problems occurred:
- Links exist but files missing = download problem
- Links missing = SEC search/API problem

Usage:
    python scripts/verify_universe.py
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import yaml
import pandas as pd


def load_config():
    """Load universe configuration."""
    with open('config/universe.yaml', 'r') as f:
        return yaml.safe_load(f)


def load_sector_data():
    """Load sector information from constituents.csv."""
    constituents_path = Path('data/constituents.csv')
    if not constituents_path.exists():
        return {}

    df = pd.read_csv(constituents_path)
    return dict(zip(df['Symbol'], df['GICS Sector']))


def parse_filing_year(filename, filing_type):
    """Extract year from filing filename."""
    try:
        if filing_type == '10-K':
            # Format: 10-K_YYYY-MM-DD.html
            return int(filename.split('_')[1][:4])
        elif filing_type == '10-Q':
            # Format: 10-Q_YYYY-MM-DD.html
            date_str = filename.split('_')[1].replace('.html', '')
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            month = dt.month
            year = dt.year
            # Determine fiscal quarter from filing month
            if 1 <= month <= 3:
                return (year - 1, 'Q4')  # Q4 of previous year
            elif 4 <= month <= 6:
                return (year, 'Q1')
            elif 7 <= month <= 9:
                return (year, 'Q2')
            else:
                return (year, 'Q3')
    except:
        return None


def get_filings_from_json(ticker_dir, start_year, end_year):
    """Get filing info from filings.json (links that were found)."""
    result = {
        '10k_links': [],
        '10q_links': [],
    }

    filings_json = ticker_dir / 'filings.json'
    if not filings_json.exists():
        return None

    try:
        with open(filings_json, 'r') as f:
            filings = json.load(f)
    except:
        return None

    for filing in filings:
        form_type = filing.get('formType', '')
        date_str = filing.get('filingDate', '') or filing.get('acceptedDate', '')

        if not date_str:
            continue

        try:
            dt = datetime.strptime(date_str[:10], '%Y-%m-%d')
            year = dt.year
            month = dt.month
        except:
            continue

        if form_type == '10-K':
            if start_year <= year <= end_year:
                result['10k_links'].append(year)
        elif form_type == '10-Q':
            # Determine fiscal quarter
            if 1 <= month <= 3:
                q_year, q = year - 1, 'Q4'
            elif 4 <= month <= 6:
                q_year, q = year, 'Q1'
            elif 7 <= month <= 9:
                q_year, q = year, 'Q2'
            else:
                q_year, q = year, 'Q3'

            if start_year <= q_year <= end_year:
                result['10q_links'].append(f"{q_year}-{q}")

    result['10k_links'] = sorted(set(result['10k_links']))
    result['10q_links'] = sorted(set(result['10q_links']))

    return result


def get_downloaded_filings(ticker_dir, start_year, end_year):
    """Get actually downloaded filing files."""
    result = {
        '10k_files': [],
        '10q_files': [],
    }

    filings_dir = ticker_dir / 'filings'
    if not filings_dir.exists():
        return result

    for f in filings_dir.iterdir():
        if f.name.startswith('10-K_') and f.suffix == '.html':
            year = parse_filing_year(f.name, '10-K')
            if year and start_year <= year <= end_year:
                result['10k_files'].append(year)
        elif f.name.startswith('10-Q_') and f.suffix == '.html':
            parsed = parse_filing_year(f.name, '10-Q')
            if parsed:
                q_year, q = parsed
                if start_year <= q_year <= end_year:
                    result['10q_files'].append(f"{q_year}-{q}")

    result['10k_files'] = sorted(set(result['10k_files']))
    # Filter Q4 (covered by 10-K)
    result['10q_files'] = sorted(set([q for q in result['10q_files'] if not q.endswith('-Q4')]))

    return result


def get_data_coverage(ticker, raw_data_dir, start_year, end_year):
    """Get detailed data coverage for a ticker."""
    ticker_dir = raw_data_dir / ticker
    if not ticker_dir.exists():
        return None

    years = list(range(start_year, end_year + 1))

    # Get filings from JSON (links)
    json_filings = get_filings_from_json(ticker_dir, start_year, end_year)

    # Get downloaded files
    downloaded = get_downloaded_filings(ticker_dir, start_year, end_year)

    result = {
        'ticker': ticker,
        'has_filings_json': json_filings is not None,
        # From filings.json (links found)
        '10k_links': json_filings['10k_links'] if json_filings else [],
        '10q_links': json_filings['10q_links'] if json_filings else [],
        # Actually downloaded files
        '10k_files': downloaded['10k_files'],
        '10q_files': downloaded['10q_files'],
        # Analyst estimates
        'analyst': {'found': [], 'missing': [], 'by_year': {}},
        # Financial statements
        'income_statement': {'quarterly': 0, 'annual': 0},
        'balance_sheet': {'quarterly': 0, 'annual': 0},
        'cash_flow': {'quarterly': 0, 'annual': 0},
        # Prices
        'prices': {'count': 0, 'start': None, 'end': None, 'gaps': []}
    }

    # Calculate missing
    result['10k_links_missing'] = [y for y in years if y not in result['10k_links']]
    result['10k_files_missing'] = [y for y in years if y not in result['10k_files']]

    # Expected 10-Q: Q1, Q2, Q3 for each year
    expected_10q = [f"{y}-{q}" for y in years for q in ['Q1', 'Q2', 'Q3']]
    # Filter Q4 from links too
    result['10q_links'] = [q for q in result['10q_links'] if not q.endswith('-Q4')]
    result['10q_links_missing'] = [q for q in expected_10q if q not in result['10q_links']]
    result['10q_files_missing'] = [q for q in expected_10q if q not in result['10q_files']]

    # Check analyst estimates
    est_file = ticker_dir / 'analyst_estimates.json'
    if est_file.exists():
        try:
            with open(est_file, 'r') as f:
                estimates = json.load(f)

            year_quarters = defaultdict(set)
            for est in estimates:
                date_str = est.get('date', '')
                if date_str:
                    try:
                        dt = datetime.strptime(date_str[:10], '%Y-%m-%d')
                        if start_year <= dt.year <= end_year:
                            q = (dt.month - 1) // 3 + 1
                            year_quarters[dt.year].add(q)
                            result['analyst']['found'].append(f"{dt.year}-Q{q}")
                    except:
                        pass

            result['analyst']['by_year'] = {y: len(qs) for y, qs in year_quarters.items()}
            result['analyst']['missing'] = [y for y in years if result['analyst']['by_year'].get(y, 0) < 3]
        except:
            pass

    # Check financial statements
    for stmt_name in ['income_statement', 'balance_sheet', 'cash_flow']:
        stmt_file = ticker_dir / f'{stmt_name}.json'
        if stmt_file.exists():
            try:
                with open(stmt_file, 'r') as f:
                    stmt_data = json.load(f)
                result[stmt_name]['quarterly'] = len(stmt_data.get('quarterly', []))
                result[stmt_name]['annual'] = len(stmt_data.get('annual', []))
            except:
                pass

    # Check prices
    prices_file = ticker_dir / 'prices.csv'
    if prices_file.exists():
        try:
            df = pd.read_csv(prices_file)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')

            start_date = datetime(start_year, 1, 1)
            end_date = datetime(end_year, 12, 31)
            df_period = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

            if len(df_period) > 0:
                result['prices']['count'] = len(df_period)
                result['prices']['start'] = df_period['date'].min().strftime('%Y-%m-%d')
                result['prices']['end'] = df_period['date'].max().strftime('%Y-%m-%d')

                # Find gaps > 7 days
                dates = df_period['date'].tolist()
                for i in range(1, len(dates)):
                    diff = (dates[i] - dates[i-1]).days
                    if diff > 7:
                        result['prices']['gaps'].append({
                            'from': dates[i-1].strftime('%Y-%m-%d'),
                            'to': dates[i].strftime('%Y-%m-%d'),
                            'days': diff
                        })
        except:
            pass

    return result


def print_ticker_summary(cov, num_years):
    """Print one-line summary for a ticker."""
    ticker = cov['ticker']
    expected_10q = num_years * 3

    # Filings - links vs downloaded
    _10k_links = len(cov['10k_links'])
    _10k_files = len(cov['10k_files'])
    _10q_links = len(cov['10q_links'])
    _10q_files = len(cov['10q_files'])

    # Status
    status_10k = "OK" if _10k_files == num_years else ("DL" if _10k_links > _10k_files else "!")
    status_10q = "OK" if _10q_files == expected_10q else ("DL" if _10q_links > _10q_files else "!")

    _prices = cov['prices']['count']

    # Financial statements
    inc_q = cov['income_statement']['quarterly']
    inc_a = cov['income_statement']['annual']
    bal_q = cov['balance_sheet']['quarterly']
    bal_a = cov['balance_sheet']['annual']
    cf_q = cov['cash_flow']['quarterly']
    cf_a = cov['cash_flow']['annual']

    print(f"  {ticker:6} 10-K: {_10k_files:2}/{num_years} [{status_10k:2}]  "
          f"10-Q: {_10q_files:2}/{expected_10q} [{status_10q:2}]  "
          f"Inc: {inc_q:2}q/{inc_a:2}a  Bal: {bal_q:2}q/{bal_a:2}a  CF: {cf_q:2}q/{cf_a:2}a  "
          f"Prices: {_prices:5}")


def _compress_year_ranges(years):
    """Convert list of years to compressed range string like '2014-2020, 2022-2024'."""
    if not years:
        return ""
    years = sorted(years)
    ranges = []
    start = years[0]
    end = years[0]

    for year in years[1:]:
        if year == end + 1:
            end = year
        else:
            ranges.append(f"{start}-{end}" if start != end else str(start))
            start = end = year

    ranges.append(f"{start}-{end}" if start != end else str(start))
    return ", ".join(ranges)


def print_detailed_issues(cov, num_years):
    """Print detailed issues for a ticker."""
    issues = []

    expected_10q = num_years * 3

    # 10-K issues
    if cov['10k_links_missing']:
        issues.append(f"10-K links missing for years: {cov['10k_links_missing']}")

    links_not_downloaded_10k = [y for y in cov['10k_links'] if y not in cov['10k_files']]
    if links_not_downloaded_10k:
        issues.append(f"10-K links found but NOT downloaded: {links_not_downloaded_10k}")

    # 10-Q issues
    if cov['10q_links_missing']:
        missing_by_year = defaultdict(list)
        for q in cov['10q_links_missing']:
            year, quarter = q.split('-')
            missing_by_year[year].append(quarter)
        missing_str = ', '.join([f"{y}: {','.join(qs)}" for y, qs in sorted(missing_by_year.items())])
        issues.append(f"10-Q links missing: {missing_str}")

    links_not_downloaded_10q = [q for q in cov['10q_links'] if q not in cov['10q_files']]
    if links_not_downloaded_10q:
        missing_by_year = defaultdict(list)
        for q in links_not_downloaded_10q:
            year, quarter = q.split('-')
            missing_by_year[year].append(quarter)
        missing_str = ', '.join([f"{y}: {','.join(qs)}" for y, qs in sorted(missing_by_year.items())])
        issues.append(f"10-Q links found but NOT downloaded: {missing_str}")

    if cov['analyst']['missing']:
        issues.append(f"Analyst estimates incomplete: {cov['analyst']['missing']}")

    # Financial statement issues
    if cov['income_statement']['quarterly'] == 0 and cov['income_statement']['annual'] == 0:
        issues.append("Income statement: NO DATA")
    if cov['balance_sheet']['quarterly'] == 0 and cov['balance_sheet']['annual'] == 0:
        issues.append("Balance sheet: NO DATA")
    if cov['cash_flow']['quarterly'] == 0 and cov['cash_flow']['annual'] == 0:
        issues.append("Cash flow: NO DATA")

    if cov['prices']['gaps']:
        gaps = cov['prices']['gaps'][:3]
        gap_str = ', '.join([f"{g['from']}→{g['to']} ({g['days']}d)" for g in gaps])
        if len(cov['prices']['gaps']) > 3:
            gap_str += f" (+{len(cov['prices']['gaps']) - 3} more)"
        issues.append(f"Price gaps: {gap_str}")

    if issues:
        print(f"\n  {cov['ticker']}:")
        for issue in issues:
            print(f"    - {issue}")


def main():
    config = load_config()

    # Support both old format (core/holdout) and new format (tickers)
    universe = config['universe']
    if 'tickers' in universe:
        all_tickers = universe['tickers']
        core = all_tickers
        holdout = []
    else:
        core = universe.get('core', [])
        holdout = universe.get('holdout', [])
        all_tickers = core + holdout

    start_year = int(config['period']['start'][:4])
    end_year = int(config['period']['end'][:4])
    num_years = end_year - start_year + 1

    print("=" * 120)
    print("DATA VERIFICATION REPORT")
    print("=" * 120)
    print(f"\nPeriod: {start_year}-{end_year} ({num_years} years)")
    print(f"Universe: {len(all_tickers)} companies")

    print(f"\nExpected per company:")
    print(f"  - 10-K annual reports: {num_years} (1 per year)")
    print(f"  - 10-Q quarterly reports: {num_years * 3} (Q1, Q2, Q3 per year; Q4 covered by 10-K)")
    print(f"  - Analyst estimates: ~{num_years * 4} quarters")
    print(f"  - Trading days: ~{num_years * 252}")

    raw_data_dir = Path("data/raw")

    # Check which tickers have data directories
    tickers_with_dir = []
    tickers_without_dir = []
    for ticker in all_tickers:
        if (raw_data_dir / ticker).exists():
            tickers_with_dir.append(ticker)
        else:
            tickers_without_dir.append(ticker)

    print(f"\n  Tickers with data directory: {len(tickers_with_dir)}/{len(all_tickers)}")
    if tickers_without_dir:
        print(f"  Tickers WITHOUT data directory: {len(tickers_without_dir)}")
        if len(tickers_without_dir) <= 20:
            print(f"    {', '.join(tickers_without_dir)}")
        else:
            print(f"    {', '.join(tickers_without_dir[:20])}... (+{len(tickers_without_dir) - 20} more)")

    # Collect all coverage data
    coverages = []
    for ticker in tickers_with_dir:
        cov = get_data_coverage(ticker, raw_data_dir, start_year, end_year)
        if cov:
            coverages.append(cov)

    if not coverages:
        print("\nNo data found for any ticker!")
        return

    # Summary statistics
    print("\n" + "=" * 120)
    print("FILINGS COVERAGE SUMMARY")
    print("=" * 120)
    print("\nLegend: [OK] = complete, [DL] = links found but download incomplete, [!] = links missing")
    print()

    # Show first 50 tickers with summary
    for i, cov in enumerate(coverages[:50]):
        print_ticker_summary(cov, num_years)

    if len(coverages) > 50:
        print(f"\n  ... and {len(coverages) - 50} more tickers")

    # Overall statistics
    print("\n" + "=" * 120)
    print("OVERALL STATISTICS")
    print("=" * 120)

    # Filings stats
    total_10k_links = sum(len(c['10k_links']) for c in coverages)
    total_10k_files = sum(len(c['10k_files']) for c in coverages)
    total_10q_links = sum(len(c['10q_links']) for c in coverages)
    total_10q_files = sum(len(c['10q_files']) for c in coverages)

    expected_10k = len(coverages) * num_years
    expected_10q = len(coverages) * num_years * 3

    print(f"\n  SEC Filings:")
    print(f"    10-K links found:      {total_10k_links:,} / {expected_10k:,} expected ({100*total_10k_links/expected_10k:.1f}%)")
    print(f"    10-K files downloaded: {total_10k_files:,} / {expected_10k:,} expected ({100*total_10k_files/expected_10k:.1f}%)")
    print(f"    10-Q links found:      {total_10q_links:,} / {expected_10q:,} expected ({100*total_10q_links/expected_10q:.1f}%)")
    print(f"    10-Q files downloaded: {total_10q_files:,} / {expected_10q:,} expected ({100*total_10q_files/expected_10q:.1f}%)")

    # Financial Statements
    total_inc_q = sum(c['income_statement']['quarterly'] for c in coverages)
    total_inc_a = sum(c['income_statement']['annual'] for c in coverages)
    total_bal_q = sum(c['balance_sheet']['quarterly'] for c in coverages)
    total_bal_a = sum(c['balance_sheet']['annual'] for c in coverages)
    total_cf_q = sum(c['cash_flow']['quarterly'] for c in coverages)
    total_cf_a = sum(c['cash_flow']['annual'] for c in coverages)

    tickers_with_financials = sum(1 for c in coverages if c['income_statement']['quarterly'] > 0)

    print(f"\n  Financial Statements:")
    print(f"    Tickers with financial data: {tickers_with_financials}/{len(coverages)}")
    print(f"    Income Statement:  {total_inc_q:,} quarterly, {total_inc_a:,} annual")
    print(f"    Balance Sheet:     {total_bal_q:,} quarterly, {total_bal_a:,} annual")
    print(f"    Cash Flow:         {total_cf_q:,} quarterly, {total_cf_a:,} annual")

    # Prices
    total_prices = sum(c['prices']['count'] for c in coverages)
    tickers_with_prices = sum(1 for c in coverages if c['prices']['count'] > 0)
    print(f"\n  Prices:")
    print(f"    Tickers with price data: {tickers_with_prices}/{len(coverages)}")
    print(f"    Total price observations: {total_prices:,}")

    # Completeness breakdown
    print(f"\n  Completeness Breakdown:")

    complete_10k_links = sum(1 for c in coverages if not c['10k_links_missing'])
    complete_10k_files = sum(1 for c in coverages if not c['10k_files_missing'])
    complete_10q_links = sum(1 for c in coverages if not c['10q_links_missing'])
    complete_10q_files = sum(1 for c in coverages if not c['10q_files_missing'])

    print(f"    Companies with all 10-K links:      {complete_10k_links}/{len(coverages)}")
    print(f"    Companies with all 10-K downloaded: {complete_10k_files}/{len(coverages)}")
    print(f"    Companies with all 10-Q links:      {complete_10q_links}/{len(coverages)}")
    print(f"    Companies with all 10-Q downloaded: {complete_10q_files}/{len(coverages)}")

    # Download issues (links exist but files missing)
    download_issues_10k = sum(1 for c in coverages
                              if len(c['10k_links']) > len(c['10k_files']))
    download_issues_10q = sum(1 for c in coverages
                              if len(c['10q_links']) > len(c['10q_files']))

    print(f"\n  Download Issues (links found but files missing):")
    print(f"    10-K download issues: {download_issues_10k} companies")
    print(f"    10-Q download issues: {download_issues_10q} companies")

    # Detailed issues
    has_issues = [c for c in coverages if
                  c['10k_links_missing'] or c['10k_files_missing'] or
                  c['10q_links_missing'] or c['10q_files_missing'] or
                  c['analyst']['missing'] or c['prices']['gaps'] or
                  (c['income_statement']['quarterly'] == 0 and c['income_statement']['annual'] == 0) or
                  (c['balance_sheet']['quarterly'] == 0 and c['balance_sheet']['annual'] == 0) or
                  (c['cash_flow']['quarterly'] == 0 and c['cash_flow']['annual'] == 0)]

    if has_issues:
        print("\n" + "=" * 120)
        print(f"DETAILED ISSUES ({len(has_issues)} companies with issues)")
        print("=" * 120)

        # Show first 30 with issues
        for cov in has_issues[:30]:
            print_detailed_issues(cov, num_years)

        if len(has_issues) > 30:
            print(f"\n  ... and {len(has_issues) - 30} more companies with issues")
    else:
        print("\n" + "=" * 120)
        print("NO ISSUES FOUND - ALL DATA COMPLETE")
        print("=" * 120)

    # ============================================================
    # VALID DATA SUMMARY (based on links, not downloaded files)
    # ============================================================
    print("\n" + "=" * 120)
    print("VALID DATA SUMMARY (companies and years with complete filing links)")
    print("=" * 120)

    years = list(range(start_year, end_year + 1))

    # For each company, find years with complete data (10-K + all 3 10-Qs)
    valid_company_years = {}
    for cov in coverages:
        ticker = cov['ticker']
        valid_years = []

        for year in years:
            # Check 10-K link exists for this year
            has_10k = year in cov['10k_links']

            # Check all 3 10-Q links exist (Q1, Q2, Q3)
            expected_10q = [f"{year}-Q1", f"{year}-Q2", f"{year}-Q3"]
            has_all_10q = all(q in cov['10q_links'] for q in expected_10q)

            if has_10k and has_all_10q:
                valid_years.append(year)

        if valid_years:
            valid_company_years[ticker] = valid_years

    # Companies with complete data for ALL years
    fully_complete = [t for t, yrs in valid_company_years.items() if len(yrs) == num_years]

    print(f"\n  Companies with COMPLETE data for all {num_years} years: {len(fully_complete)}")
    if fully_complete:
        # Print in rows of 10
        for i in range(0, len(fully_complete), 10):
            print(f"    {', '.join(fully_complete[i:i+10])}")

    # Summary for easy copy-paste
    print("\n" + "-" * 120)
    print("USABLE DATASET SUMMARY")
    print("-" * 120)

    # Analyze ALL possible (start, end) year combinations
    print("\n  Analyzing all possible date ranges...")

    all_periods = {}  # (start, end) -> list of tickers
    for s in years:
        for e in years:
            if e >= s:
                required_years = set(range(s, e + 1))
                companies_with_range = []
                for ticker, yrs in valid_company_years.items():
                    if required_years.issubset(set(yrs)):
                        companies_with_range.append(ticker)
                if companies_with_range:
                    all_periods[(s, e)] = sorted(companies_with_range)

    # Create scored list: (start, end, years_span, n_companies, score)
    scored = []
    for (s, e), tickers in all_periods.items():
        years_span = e - s + 1
        n_companies = len(tickers)
        score = years_span * n_companies
        scored.append((s, e, years_span, n_companies, score, tickers))

    # LIST 1: Top by NUMBER OF COMPANIES (minimum 5 years)
    print("\n  " + "=" * 80)
    print("  TOP PERIODS BY NUMBER OF COMPANIES (min 5 years)")
    print("  " + "=" * 80)
    by_companies = [x for x in scored if x[2] >= 5]  # min 5 years
    by_companies.sort(key=lambda x: (-x[3], -x[2]))  # sort by companies desc, then years desc

    seen_company_counts = set()
    for s, e, years_span, n_companies, score, tickers in by_companies[:15]:
        # Skip if we already showed this company count with longer period
        if n_companies in seen_company_counts:
            continue
        seen_company_counts.add(n_companies)
        print(f"    {s}-{e} ({years_span:2} years): {n_companies:3} companies")

    # LIST 2: Top by PERIOD LENGTH (minimum 10 companies)
    print("\n  " + "=" * 80)
    print("  TOP PERIODS BY LENGTH (min 10 companies)")
    print("  " + "=" * 80)
    by_length = [x for x in scored if x[3] >= 10]  # min 10 companies
    by_length.sort(key=lambda x: (-x[2], -x[3]))  # sort by years desc, then companies desc

    seen_lengths = set()
    for s, e, years_span, n_companies, score, tickers in by_length[:15]:
        # Skip if we already showed this length with more companies
        if years_span in seen_lengths:
            continue
        seen_lengths.add(years_span)
        print(f"    {s}-{e} ({years_span:2} years): {n_companies:3} companies")

    # LIST 3: Top by SCORE (years × companies) - best trade-off
    print("\n  " + "=" * 80)
    print("  TOP PERIODS BY SCORE (years × companies)")
    print("  " + "=" * 80)
    by_score = [x for x in scored if x[2] >= 5 and x[3] >= 10]  # min 5 years, 10 companies
    by_score.sort(key=lambda x: -x[4])  # sort by score desc

    for s, e, years_span, n_companies, score, tickers in by_score[:10]:
        print(f"    {s}-{e} ({years_span:2} years): {n_companies:3} companies  [score: {score}]")

    # Output Python-usable data for top 3 best periods
    print("\n  " + "-" * 80)
    print("  RECOMMENDED DATASETS (Python-ready)")
    print("  " + "-" * 80)

    if by_score:
        for i, (s, e, years_span, n_companies, score, tickers) in enumerate(by_score[:3]):
            print(f"\n  # Option {i+1}: {s}-{e} ({years_span} years, {n_companies} companies)")
            print(f"  PERIOD_{i+1} = ({s}, {e})")
            print(f"  TICKERS_{i+1} = {tickers}")

    # ============================================================
    # BALANCED PORTFOLIO SELECTION (2006-2025)
    # ============================================================
    print("\n" + "=" * 120)
    print("BALANCED PORTFOLIO SELECTION (2006-2025, 100 companies)")
    print("=" * 120)

    # Load sector data
    sector_data = load_sector_data()
    if not sector_data:
        print("\n  ERROR: Could not load sector data from data/constituents.csv")
    else:
        # Get companies with complete 2006-2025 data
        target_start, target_end = 2006, 2025
        required_years = set(range(target_start, target_end + 1))

        complete_tickers = []
        for ticker, yrs in valid_company_years.items():
            if required_years.issubset(set(yrs)):
                complete_tickers.append(ticker)

        print(f"\n  Companies with complete data for {target_start}-{target_end}: {len(complete_tickers)}")

        # Get sector for each ticker
        ticker_sectors = {}
        unknown_sector = []
        for ticker in complete_tickers:
            sector = sector_data.get(ticker)
            if sector:
                ticker_sectors[ticker] = sector
            else:
                unknown_sector.append(ticker)

        if unknown_sector:
            print(f"  Tickers without sector info: {len(unknown_sector)} ({', '.join(unknown_sector[:10])}...)")

        # Group by sector
        by_sector = defaultdict(list)
        for ticker, sector in ticker_sectors.items():
            by_sector[sector].append(ticker)

        print(f"\n  Sector distribution of complete tickers:")
        for sector in sorted(by_sector.keys()):
            print(f"    {sector:35}: {len(by_sector[sector]):3} companies")

        # Select 100 companies balanced by sector
        # Strategy: proportional allocation, at least 2 per sector, round-robin for remainder
        total_to_select = 100
        n_sectors = len(by_sector)

        if len(ticker_sectors) < total_to_select:
            print(f"\n  WARNING: Only {len(ticker_sectors)} companies with sector info, need {total_to_select}")
            total_to_select = len(ticker_sectors)

        # Calculate proportional allocation
        sector_allocation = {}
        total_available = sum(len(tickers) for tickers in by_sector.values())

        for sector, tickers in by_sector.items():
            # Proportional share, minimum 2 per sector
            proportion = len(tickers) / total_available
            allocated = max(2, int(proportion * total_to_select))
            # Don't allocate more than available
            sector_allocation[sector] = min(allocated, len(tickers))

        # Adjust to exactly 100
        current_total = sum(sector_allocation.values())
        sectors_sorted = sorted(by_sector.keys(), key=lambda s: len(by_sector[s]), reverse=True)

        while current_total < total_to_select:
            for sector in sectors_sorted:
                if sector_allocation[sector] < len(by_sector[sector]):
                    sector_allocation[sector] += 1
                    current_total += 1
                    if current_total >= total_to_select:
                        break

        while current_total > total_to_select:
            for sector in reversed(sectors_sorted):
                if sector_allocation[sector] > 2:
                    sector_allocation[sector] -= 1
                    current_total -= 1
                    if current_total <= total_to_select:
                        break

        # Select tickers from each sector (sorted alphabetically for reproducibility)
        selected_100 = []
        for sector in sorted(by_sector.keys()):
            n_select = sector_allocation[sector]
            tickers_sorted = sorted(by_sector[sector])
            selected_100.extend(tickers_sorted[:n_select])

        print(f"\n  Selected {len(selected_100)} companies balanced by sector:")
        selected_by_sector = defaultdict(list)
        for ticker in selected_100:
            selected_by_sector[ticker_sectors[ticker]].append(ticker)

        for sector in sorted(selected_by_sector.keys()):
            tickers = selected_by_sector[sector]
            print(f"    {sector:35}: {len(tickers):2} - {', '.join(sorted(tickers))}")

        # Split into two portfolios of 50 (alternating to balance sectors)
        portfolio_1 = []
        portfolio_2 = []

        for sector in sorted(selected_by_sector.keys()):
            tickers = sorted(selected_by_sector[sector])
            for i, ticker in enumerate(tickers):
                if i % 2 == 0:
                    portfolio_1.append(ticker)
                else:
                    portfolio_2.append(ticker)

        # Balance if unequal
        while len(portfolio_1) > 50 and len(portfolio_2) < 50:
            portfolio_2.append(portfolio_1.pop())
        while len(portfolio_2) > 50 and len(portfolio_1) < 50:
            portfolio_1.append(portfolio_2.pop())

        portfolio_1 = sorted(portfolio_1)
        portfolio_2 = sorted(portfolio_2)

        print(f"\n  " + "-" * 80)
        print(f"  PORTFOLIO 1 ({len(portfolio_1)} companies) - Core/Training")
        print(f"  " + "-" * 80)
        p1_by_sector = defaultdict(list)
        for t in portfolio_1:
            p1_by_sector[ticker_sectors[t]].append(t)
        for sector in sorted(p1_by_sector.keys()):
            print(f"    {sector:35}: {len(p1_by_sector[sector]):2} - {', '.join(sorted(p1_by_sector[sector]))}")

        print(f"\n  " + "-" * 80)
        print(f"  PORTFOLIO 2 ({len(portfolio_2)} companies) - Holdout/Validation")
        print(f"  " + "-" * 80)
        p2_by_sector = defaultdict(list)
        for t in portfolio_2:
            p2_by_sector[ticker_sectors[t]].append(t)
        for sector in sorted(p2_by_sector.keys()):
            print(f"    {sector:35}: {len(p2_by_sector[sector]):2} - {', '.join(sorted(p2_by_sector[sector]))}")

        # Python-ready output
        print(f"\n  " + "=" * 80)
        print(f"  PYTHON-READY OUTPUT")
        print(f"  " + "=" * 80)
        print(f"\n  PERIOD = ({target_start}, {target_end})")
        print(f"\n  PORTFOLIO_1 = {portfolio_1}")
        print(f"\n  PORTFOLIO_2 = {portfolio_2}")
        print(f"\n  ALL_100 = {sorted(selected_100)}")

    print("\n" + "=" * 120)
    print("VERIFICATION COMPLETE")
    print("=" * 120)


if __name__ == "__main__":
    main()