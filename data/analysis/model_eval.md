# Model evaluation vs gold standard

Match threshold: 60.0 | blind tiers: ['tier1a_tuning', 'tier1b_eval']

Cost = clean (1 request/doc, NO retries): OpenRouter price x 39,920 input + each model's ACTUAL avg output tokens.

| Model | done | success | recall(blind) | precision(blind) | extra/doc | grounding | F1 | ev/doc | $/doc | corpus |
|-------|------|---------|---------------|------------------|-----------|-----------|----|--------|-------|--------|
| google_gemini-2.5-flash | 90/90 | 1.0 | 0.291 | 0.598 | 2.25 | 0.99 | 0.45 | 6.222 | $0.0163 | $148 |
| anthropic_claude-haiku-4.5 | 90/90 | 1.0 | 0.257 | 0.641 | 1.65 | 0.954 | 0.405 | 4.989 | $0.0496 | $451 |
| google_gemini-2.5-flash-lite | 88/90 | 0.978 | 0.235 | 0.6 | 1.8 | 0.994 | 0.38 | 4.822 | $0.0046 | $42 |
| deepseek_deepseek-chat-v3.1 | 88/90 | 0.978 | 0.187 | 0.632 | 1.25 | 0.969 | 0.314 | 3.556 | $0.0239 | $218 |
| qwen_qwen3.8-27b | 65/90 | 0.722 | 0.17 | 0.765 | 0.6 | 0.991 | 0.29 | 3.467 | $0.0207 | $188 |
| meta-llama_llama-4-maverick | 90/90 | 1.0 | 0.135 | 0.689 | 0.7 | 0.971 | 0.237 | 2.589 | $0.0087 | $79 |
| z-ai_glm-5.3-flash | 23/90 | 0.256 | 0.091 | 0.568 | 0.8 | 0.997 | 0.167 | 1.644 | $0.0033 | $30 |
| openai_gpt-4o-mini | 90/90 | 1.0 | 0.074 | 0.708 | 0.35 | 0.9 | 0.137 | 1.311 | $0.0064 | $58 |
