"""
Prompts for LLM Reasoner.

System prompt defines the role and guidelines.
User prompt template is filled with company-specific data.
"""

REASONER_SYSTEM_PROMPT = """You are a quantitative investment analyst. Your task is to generate investment views based on SEC filings and analyst data.

A "view" is your expected EXCESS return for the stock over the next quarter, expressed in BASIS POINTS (bps). 100 bps = 1%.

Guidelines:
- Range: -300 to +300 bps (extreme cases)
- Typical range: -100 to +100 bps
- 0 bps = no view (neutral)

Consider these factors:
1. Sentiment & Guidance: Is management confident or cautious?
2. Risk Factors: Any new or elevated risks?
3. Key Events: M&A, restructuring, litigation?
4. Earnings Surprise: Beat or miss vs consensus?
5. Analyst Dispersion: High dispersion = more uncertainty

Confidence levels:
- HIGH: Clear signals, low uncertainty
- MEDIUM: Mixed signals or moderate uncertainty
- LOW: Conflicting signals or high uncertainty

IMPORTANT:
- Base your view ONLY on the information provided
- Do NOT use external knowledge about the company
- Be conservative - avoid extreme views without strong evidence
- Explain your reasoning briefly (2-3 sentences)

Respond in JSON format with exactly these fields:
{
  "view_bps": <integer from -300 to 300>,
  "confidence": "<high|medium|low>",
  "reasoning": "<2-3 sentence explanation>"
}"""


REASONER_USER_PROMPT = """Generate an investment view for {ticker} as of {view_date} (after {fiscal_period} filing).

## Company: {ticker} ({sector})

## Latest 10-K Summary (filed {latest_10k_date})
- Risk Summary: {risk_summary}
- Key Risks: {top_risks}
- Key Events: {annual_events}

## Latest 10-Q Summary (filed {latest_10q_date})
- Sentiment: {sentiment} - {sentiment_rationale}
- Guidance: {guidance_direction}
  - Revenue outlook: {revenue_outlook}
  - Earnings outlook: {earnings_outlook}
- Recent Events: {quarterly_events}

## Most Recent Earnings ({earnings_period}, filed {earnings_filing_date})
- EPS: Actual ${eps_actual} vs Estimate ${eps_estimate} ({eps_surprise}% surprise)
- Revenue: Actual ${revenue_actual}B vs Estimate ${revenue_estimate}B ({revenue_surprise}% surprise)
- Analyst Coverage: {num_analysts} analysts
- Estimate Dispersion: {dispersion}% (spread in analyst estimates)

## Recent Performance
- 1-month return: {return_1m}%
- 3-month return: {return_3m}%
- 60-day volatility: {volatility}%

---

Based on the above, provide your investment view in JSON format:
- view_bps: Expected excess return in basis points (-300 to +300)
- confidence: high, medium, or low
- reasoning: Brief explanation (2-3 sentences)"""


def format_user_prompt(input_data: "ReasonerInput") -> str:
    """
    Format user prompt with data from ReasonerInput.

    Args:
        input_data: ReasonerInput object with all necessary data

    Returns:
        Formatted prompt string
    """
    # Extract risk summary (top 3 risks)
    risk_summary = "No significant risks identified"
    top_risks = "N/A"
    if input_data.annual_risk_factors:
        high_risks = [r for r in input_data.annual_risk_factors if r.severity == "high"]
        top_3 = high_risks[:3] if high_risks else input_data.annual_risk_factors[:3]
        top_risks = "; ".join([f"{r.category}: {r.description[:100]}" for r in top_3])
        risk_summary = input_data.annual_risk_summary or f"{len(input_data.annual_risk_factors)} risk factors identified"

    # Annual events
    annual_events = ", ".join(input_data.annual_key_events[:5]) if input_data.annual_key_events else "No significant events"

    # Quarterly events
    quarterly_events = ", ".join(input_data.quarterly_key_events[:5]) if input_data.quarterly_key_events else "No significant events"

    # Guidance
    guidance = input_data.quarterly_guidance
    guidance_direction = guidance.direction if guidance and guidance.direction else "Not provided"
    revenue_outlook = guidance.revenue_outlook if guidance and guidance.revenue_outlook else "Not specified"
    earnings_outlook = guidance.earnings_outlook if guidance and guidance.earnings_outlook else "Not specified"

    # Earnings surprise
    surprise = input_data.earnings_surprise
    earnings_period = surprise.period if surprise else "N/A"
    earnings_filing_date = surprise.filing_date if surprise else "N/A"
    eps_actual = f"{surprise.eps_actual:.2f}" if surprise and surprise.eps_actual else "N/A"
    eps_estimate = f"{surprise.eps_estimate:.2f}" if surprise and surprise.eps_estimate else "N/A"
    eps_surprise = f"{surprise.eps_surprise_pct:+.1f}" if surprise and surprise.eps_surprise_pct else "N/A"
    revenue_actual = f"{surprise.revenue_actual:.1f}" if surprise and surprise.revenue_actual else "N/A"
    revenue_estimate = f"{surprise.revenue_estimate:.1f}" if surprise and surprise.revenue_estimate else "N/A"
    revenue_surprise = f"{surprise.revenue_surprise_pct:+.1f}" if surprise and surprise.revenue_surprise_pct else "N/A"
    num_analysts = surprise.num_analysts if surprise and surprise.num_analysts else "N/A"
    dispersion = f"{surprise.estimate_dispersion * 100:.1f}" if surprise and surprise.estimate_dispersion else "N/A"

    # Price metrics
    return_1m = f"{input_data.price_return_1m:+.1f}" if input_data.price_return_1m is not None else "N/A"
    return_3m = f"{input_data.price_return_3m:+.1f}" if input_data.price_return_3m is not None else "N/A"
    volatility = f"{input_data.price_volatility_60d:.1f}" if input_data.price_volatility_60d is not None else "N/A"

    return REASONER_USER_PROMPT.format(
        ticker=input_data.ticker,
        view_date=input_data.view_date.strftime("%Y-%m-%d"),
        fiscal_period=input_data.fiscal_period,
        sector=input_data.sector,
        latest_10k_date=input_data.latest_10k_date or "N/A",
        risk_summary=risk_summary,
        top_risks=top_risks,
        annual_events=annual_events,
        latest_10q_date=input_data.latest_10q_date or "N/A",
        sentiment=input_data.quarterly_sentiment or "neutral",
        sentiment_rationale=input_data.quarterly_sentiment_rationale or "No rationale provided",
        guidance_direction=guidance_direction,
        revenue_outlook=revenue_outlook,
        earnings_outlook=earnings_outlook,
        quarterly_events=quarterly_events,
        earnings_period=earnings_period,
        earnings_filing_date=earnings_filing_date,
        eps_actual=eps_actual,
        eps_estimate=eps_estimate,
        eps_surprise=eps_surprise,
        revenue_actual=revenue_actual,
        revenue_estimate=revenue_estimate,
        revenue_surprise=revenue_surprise,
        num_analysts=num_analysts,
        dispersion=dispersion,
        return_1m=return_1m,
        return_3m=return_3m,
        volatility=volatility,
    )
