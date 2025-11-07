"""THINK task executor instruction seed prompt for DSPy.

This module provides an optimized instruction prompt and demos specifically
for THINK tasks - deep reasoning, strategic analysis, and multi-source synthesis.
"""

import dspy

EXECUTOR_THINK_PROMPT = r"""
# Executor (THINK) — Instruction Prompt

Role
Execute THINK tasks: perform deep reasoning, strategic analysis, pattern recognition, and multi-source synthesis to generate insights.

Task Characteristics (THINK)
- Primary goal: Generate strategic insights through deep analytical reasoning
- Reasoning-heavy: Requires multi-step logical thinking and pattern recognition
- Multi-source synthesis: Combine information from various sources and perspectives
- Complex output: Structured analysis with supporting evidence and reasoning chains

Execution Guidelines (THINK-Specific)
1. Multi-perspective analysis: Consider problem from multiple angles and viewpoints
2. Evidence-based reasoning: Build arguments on concrete data and verifiable facts
3. Pattern recognition: Identify trends, correlations, and causal relationships
4. Strategic thinking: Consider short-term and long-term implications
5. Structured approach: Use frameworks (SWOT, decision trees, comparative analysis)
6. Critical evaluation: Question assumptions and identify limitations
7. Synthesis: Combine insights from multiple sources into coherent conclusions

Output Contract (strict)
- `output` (string): Structured analysis with clear reasoning chain and evidence
- `sources` (list[str]): All information sources, frameworks, and data references used

Quality Standards (THINK)
- Depth: Go beyond surface-level observations to root causes and mechanisms
- Rigor: Use logical reasoning with clear premises and conclusions
- Comprehensiveness: Address all relevant aspects of the problem
- Clarity: Present complex ideas in accessible, well-organized format
- Actionability: Include practical implications and recommendations
- Objectivity: Acknowledge biases, uncertainties, and alternative viewpoints

Common THINK Patterns
- Strategic analysis: Gather context → identify factors → analyze relationships → synthesize insights → recommend actions
- Problem diagnosis: Define problem → investigate causes → analyze patterns → identify root causes → propose solutions
- Trend analysis: Collect data points → identify patterns → analyze drivers → project implications → assess impacts
- Decision analysis: Define options → evaluate criteria → analyze tradeoffs → assess risks → recommend choice
- Comparative analysis: Identify dimensions → gather data → compare systematically → highlight differences → draw conclusions

Output Format Guidelines
- Executive summary: Key findings and recommendations (2-3 sentences)
- Context: Relevant background and scope
- Analysis: Structured exploration of the question (use sections/frameworks)
- Evidence: Data points, examples, and supporting information
- Implications: What this means practically
- Limitations: Uncertainties, caveats, and alternative interpretations
"""

EXECUTOR_THINK_DEMOS = [
    # Demo 1: Strategic analysis (technology)
    dspy.Example(
        goal="Why do some social media platforms succeed while others fail?",
        output=(
            "Strategic Analysis: Social Media Platform Success Factors\n\n"
            "Key Success Factors:\n\n"
            "1. Network Effects & Critical Mass\n"
            "- Winners achieved rapid growth in specific niches (Facebook: colleges, TikTok: Gen Z)\n"
            "- Metcalfe's Law: Platform value grows exponentially with user base\n"
            "- Failed platforms (Google+, Friendster) couldn't overcome existing network lock-in\n\n"
            "2. Content Differentiation\n"
            "- Successful platforms offered unique formats: Instagram (photos), TikTok (short video), Twitter (microblogging)\n"
            "- Copycat platforms without innovation failed (Vine after TikTok)\n\n"
            "3. Creator Economy Lock-In\n"
            "- Platforms that monetize creators build stickiness (YouTube Partner Program)\n"
            "- Creator investment in audience becomes switching cost\n\n"
            "4. Timing & Market Readiness\n"
            "- TikTok succeeded with mobile-first design during smartphone maturity\n"
            "- Technology readiness: Instagram needed iPhone camera improvements\n\n"
            "Failure Patterns:\n"
            "- Too early: Friendster (2002) faced scaling issues\n"
            "- Too late: Google+ (2011) couldn't overcome Facebook network effects\n"
            "- No differentiation: Ello, Vero (Facebook clones)\n\n"
            "Implications:\n"
            "- New entrants must create new content format OR target underserved niche\n"
            "- Defensibility comes from creator lock-in and unique algorithm IP\n"
            "- Network effects alone insufficient without differentiation"
        ),
        sources=[
            "Platform growth data (SimilarWeb)",
            "Network effects theory (Metcalfe's Law)",
            "Creator economy reports",
            "Business strategy frameworks"
        ]
    ).with_inputs("goal"),

    # Demo 2: Problem diagnosis
    dspy.Example(
        goal="Why do most New Year's resolutions fail by February?",
        output=(
            "Problem Diagnosis: New Year's Resolution Failure\n\n"
            "Root Causes:\n\n"
            "1. Implementation Gap (Primary)\n"
            "- Resolutions are outcome goals (\"lose weight\") not process goals (\"gym Mon/Wed/Fri 7am\")\n"
            "- Missing implementation intentions: when, where, how specifics\n"
            "- Research: Implementation intentions increase success 2-3x (Gollwitzer)\n\n"
            "2. Motivation Depletion\n"
            "- January motivation is emotional peak (fresh start effect)\n"
            "- Motivation decreases with stress and novelty wearoff\n"
            "- Successful change relies on systems/habits, not willpower\n\n"
            "3. Unrealistic Expectations\n"
            "- People expect linear progress; reality has plateaus and setbacks\n"
            "- Single failure interpreted as total failure (all-or-nothing thinking)\n\n"
            "4. Environment Not Designed for Success\n"
            "- Context unchanged: same cues trigger old behaviors\n"
            "- Friction too high for new behavior, too low for old behavior\n\n"
            "Why Some Succeed:\n"
            "- Specific: \"Read 20 pages before bed\" vs \"Read more\"\n"
            "- Tied to routine: \"After brushing teeth\" (habit stacking)\n"
            "- Resilience plan: \"If I miss a day, I'll do double pages next day\"\n"
            "- Accountability: Workout partner, public commitment\n\n"
            "Recommendations:\n"
            "- Replace \"I will X\" with \"Every [time], I will [action] at [location]\"\n"
            "- Start ridiculously small (2min/day) to build consistency\n"
            "- Design environment: increase friction for bad habits\n"
            "- Pre-commit to recovery plan for setbacks"
        ),
        sources=[
            "Gollwitzer & Sheeran - Implementation intentions meta-analysis",
            "BJ Fogg - Behavior Model",
            "Prochaska - Stages of Change",
            "Atomic Habits - Environment design"
        ]
    ).with_inputs("goal"),

    # Demo 3: Trend analysis
    dspy.Example(
        goal="Analyze remote work adoption trend and predict long-term trajectory",
        output=(
            "Trend Analysis: Remote Work Trajectory\n\n"
            "Historical Context:\n"
            "- Pre-2020: ~5-7% regular remote work\n"
            "- COVID peak: 65-70% remote (Mar 2020)\n"
            "- Current (2024): ~30-35% hybrid/remote\n"
            "- Prediction: Stabilizing at 25-35% long-term\n\n"
            "Key Drivers (Pushing Adoption):\n"
            "1. Cost savings: $10-15K/employee/year (office space reduction)\n"
            "2. Talent access: Geographic constraints removed, global hiring\n"
            "3. Quality of life: 54min/day commute eliminated (US Census)\n"
            "4. Technology maturation: Zoom, Slack, async workflows established\n\n"
            "Key Constraints (Limiting Adoption):\n"
            "1. Collaboration tax: Cross-team collaboration dropped 25% when remote (Microsoft research)\n"
            "2. Management overhead: Trust issues, performance measurement harder\n"
            "3. Career development: Remote workers report 20% lower promotion rates\n"
            "4. Social connection: Fully remote workers report higher loneliness\n\n"
            "Long-Term Equilibrium (5-10 years):\n"
            "- Hybrid dominant: 2-3 days office, 2-3 days remote (25-35% workforce)\n"
            "- Full remote niche: 10-15% in async-friendly roles\n"
            "- Full office: 50-60% (presence-required industries)\n\n"
            "Industry Variation:\n"
            "- Tech/Software: 40-50% hybrid/remote\n"
            "- Finance/Legal: 20-30% (compliance barriers)\n"
            "- Healthcare/Education: 5-15% (physical presence required)\n\n"
            "Implications:\n"
            "- Office demand permanently 20-30% below pre-COVID\n"
            "- Suburban/secondary cities see residential demand increase\n"
            "- Employers must design intentional hybrid policies"
        ),
        sources=[
            "US Census - Commute statistics",
            "FlexJobs - Remote work surveys",
            "Microsoft Research - Collaboration study",
            "Stanford - Remote work outcomes",
            "Real estate reports (CBRE)"
        ]
    ).with_inputs("goal"),

    # Demo 4: Decision analysis
    dspy.Example(
        goal="Should I rent or buy a house? Analyze the decision factors.",
        output=(
            "Decision Analysis: Rent vs Buy Housing\n\n"
            "Key Decision Variables:\n"
            "1. Time horizon: How long will you stay? (most important)\n"
            "2. Financial capacity: Down payment, emergency fund, income stability\n"
            "3. Local market: Price-to-rent ratio, appreciation trend\n"
            "4. Personal preferences: Stability vs flexibility\n\n"
            "Financial Analysis:\n\n"
            "Break-even: ~5-7 years to cover transaction costs\n\n"
            "Example ($400K house, 20% down, 6.5% mortgage):\n"
            "- Monthly cost: $3,010 (mortgage $2,027 + tax $500 + insurance $150 + maintenance $333)\n"
            "- Equivalent rent: $2,500/mo\n"
            "- Building equity: ~$600/mo (early years)\n"
            "- Transaction cost to sell: ~$30K (7.5%)\n\n"
            "Price-to-Rent Ratio (Key Metric):\n"
            "- Formula: Home Price ÷ Annual Rent\n"
            "- Ratio < 15: Strong buy market\n"
            "- Ratio 15-20: Neutral\n"
            "- Ratio > 20: Strong rent market\n"
            "- Example: $400K house / $30K rent = 13.3 → Favors buying\n\n"
            "Decision Matrix:\n\n"
            "STRONG BUY signals:\n"
            "- Staying 7+ years\n"
            "- 20%+ down + 6mo emergency fund\n"
            "- Price-to-rent < 15\n"
            "- Value stability and control\n\n"
            "RENT signals:\n"
            "- Staying < 5 years\n"
            "- Limited savings (< 10% down)\n"
            "- Price-to-rent > 20\n"
            "- Value flexibility and mobility\n\n"
            "Common Misconceptions:\n"
            "- Myth: \"Renting is throwing money away\"\n"
            "  Reality: Mortgage interest, tax, maintenance also not equity (70-80% of early payments)\n"
            "- Myth: \"Home always appreciates\"\n"
            "  Reality: Real appreciation ~0-2% annually long-term\n\n"
            "Recommendations:\n"
            "1. Calculate YOUR price-to-rent ratio in target area\n"
            "2. Assess time horizon honestly\n"
            "3. Ensure cushion: 20% down + 6mo emergency fund + $10K maintenance reserve\n"
            "4. If borderline, rent 1-2 more years while saving"
        ),
        sources=[
            "Price-to-rent calculations (Zillow, Redfin)",
            "Mortgage amortization schedules",
            "Case-Shiller Index - Historical appreciation",
            "National Association of Realtors - Transaction costs",
            "Personal finance frameworks"
        ]
    ).with_inputs("goal"),

    # Demo 5: Comparative analysis (crypto)
    dspy.Example(
        goal="Compare Bitcoin and Ethereum as investment assets",
        output=(
            "Comparative Analysis: Bitcoin vs Ethereum\n\n"
            "Core Philosophy:\n"
            "- BTC: Digital gold, store of value, sound money\n"
            "- ETH: Programmable platform, DeFi infrastructure, Web3 foundation\n\n"
            "Technical Comparison:\n\n"
            "Supply:\n"
            "- BTC: Fixed 21M cap, ~19.6M mined (93%), disinflationary\n"
            "- ETH: No hard cap, ~120M supply, deflationary post-EIP-1559 (burn mechanism)\n\n"
            "Consensus:\n"
            "- BTC: Proof of Work (PoW), energy-intensive, maximally secure\n"
            "- ETH: Proof of Stake (PoS) post-Merge, 99.95% less energy\n\n"
            "Programmability:\n"
            "- BTC: Limited scripting, intentionally constrained\n"
            "- ETH: Turing-complete smart contracts, full programmability\n\n"
            "Use Cases:\n\n"
            "Bitcoin:\n"
            "1. Store of value (\"digital gold\")\n"
            "2. Cross-border payments\n"
            "3. Corporate/nation-state reserve asset\n"
            "4. Speculation on scarcity premium\n\n"
            "Ethereum:\n"
            "1. DeFi infrastructure (Aave, Uniswap, stablecoins)\n"
            "2. NFT platform and tokenization\n"
            "3. Smart contract settlement layer\n"
            "4. Staking yield (3-5% annual)\n"
            "5. Speculation on utility value\n\n"
            "Investment Characteristics:\n\n"
            "Bitcoin:\n"
            "- Market cap: ~$850B (#1)\n"
            "- Volatility: 60-80% annual (lower than ETH)\n"
            "- Regulatory: Commodity (CFTC), spot ETFs approved\n"
            "- Upside: Digital gold thesis, institutional adoption\n"
            "- Downside: Energy concerns, slower innovation\n\n"
            "Ethereum:\n"
            "- Market cap: ~$275B (#2)\n"
            "- Volatility: 70-100% annual, 1.2-1.5x beta to BTC\n"
            "- Regulatory: Security vs commodity debate, ETFs approved 2024\n"
            "- Upside: DeFi growth, value accrual from usage (burn + staking)\n"
            "- Downside: Competition (Solana, L2s), regulatory risk\n\n"
            "Portfolio Allocation:\n\n"
            "Conservative: 80% BTC, 20% ETH (lower volatility)\n"
            "Balanced: 60% BTC, 40% ETH (moderate risk)\n"
            "Aggressive: 40% BTC, 60% ETH (bet on utility growth)\n\n"
            "Key Decision Factors:\n\n"
            "Choose BTC if:\n"
            "- Believe in digital gold thesis\n"
            "- Want lower volatility (relatively)\n"
            "- Prefer simplicity and certainty\n\n"
            "Choose ETH if:\n"
            "- Believe in Web3/DeFi thesis\n"
            "- Want utility value exposure\n"
            "- Comfortable with higher volatility for higher upside\n\n"
            "Recommendation:\n"
            "Hold both in ratio matching risk tolerance (60/40 BTC/ETH baseline)\n"
            "- Rebalance periodically\n"
            "- Only capital you can afford to lose\n"
            "- Dollar-cost average to reduce timing risk"
        ),
        sources=[
            "CoinGecko/CoinMarketCap - Market data",
            "Bitcoin whitepaper (Satoshi, 2008)",
            "Ethereum whitepaper (Vitalik, 2013)",
            "On-chain metrics (Glassnode)",
            "DeFi TVL data (DefiLlama)"
        ]
    ).with_inputs("goal"),

    # Demo 6: Pattern recognition
    dspy.Example(
        goal="What patterns from past tech disruptions might apply to AI?",
        output=(
            "Pattern Recognition: Historical Tech Disruptions → AI\n\n"
            "Historical Cases:\n"
            "- Electricity (1880s-1930s): 50yr adoption, 2-3x productivity gain\n"
            "- Automobiles (1900s-1950s): 40yr adoption, enabled suburbanization\n"
            "- Internet (1990s-2020s): 25yr adoption, winner-take-most dynamics\n\n"
            "Common Patterns:\n\n"
            "1. Adoption S-Curve\n"
            "- Slow start → Rapid growth → Saturation\n"
            "- Timeline compressing: Electricity (50yr), Internet (25yr), Smartphones (10yr)\n"
            "- AI prediction: 10-20yr adoption (already at ~5% for GPT-level tools)\n\n"
            "2. Job Displacement Then Net Creation\n"
            "- Initial: Visible job losses (blacksmiths, travel agents)\n"
            "- 10-20 years later: Net job creation in new categories (auto mechanics, web developers)\n"
            "- New jobs unimaginable pre-disruption (\"social media manager\" in 1995)\n"
            "- AI: Short-term displacement (data entry, basic writing), long-term new roles (AI trainers, prompt engineers)\n\n"
            "3. Productivity Paradox\n"
            "- Initial years: Productivity gains NOT visible in data\n"
            "- Example: Computers everywhere 1980s, productivity gains only clear by late 1990s\n"
            "- AI: Current gains may be understated; wait 5-10 years for full impact\n\n"
            "4. Winner-Take-Most\n"
            "- Disruptions favor consolidation (network effects, economies of scale)\n"
            "- Examples: Standard Oil, AT&T, Microsoft, Google\n"
            "- AI: OpenAI/Google/Anthropic likely dominate, regulatory scrutiny by 2030s\n\n"
            "5. Regulatory Lag\n"
            "- Regulation lags technology by 10-20 years\n"
            "- Examples: Auto safety (1960s), Internet privacy (GDPR 2018)\n"
            "- AI: Current wild-west; expect regulation by 2030-2035\n\n"
            "6. Complementary Innovation Required\n"
            "- Electricity needed: motors, appliances, wiring standards\n"
            "- Internet needed: browsers, payment systems, cloud infrastructure\n"
            "- AI needs: Better interfaces, fine-tuning tools, evaluation frameworks\n\n"
            "AI-Specific Differences:\n"
            "- Faster adoption: Digital distribution, zero marginal cost (ChatGPT: 100M users in 2 months)\n"
            "- Broader scope: Impacts knowledge work, not just physical labor\n"
            "- Cognitive automation: Affects high-skill workers MORE than previous disruptions\n\n"
            "Predictions:\n\n"
            "Next 5 Years (2025-2030):\n"
            "- Visible job displacement in routine cognitive work\n"
            "- Productivity paradox: Unclear aggregate gains\n"
            "- Winner-take-most: 2-3 dominant providers\n\n"
            "Next 10 Years (2030-2035):\n"
            "- 40-60% knowledge workers use AI daily\n"
            "- Net job creation starts (new categories emerge)\n"
            "- Comprehensive AI regulation passed\n"
            "- Clear productivity gains in GDP\n\n"
            "Implications:\n"
            "- Workers: Invest in skills that complement AI (creativity, judgment)\n"
            "- Companies: Adopt now or risk competitive disadvantage\n"
            "- Policymakers: Start regulatory frameworks now (don't wait 20 years)"
        ),
        sources=[
            "Economic history - Electricity adoption (David, Paul)",
            "Automobile industry history",
            "Internet adoption data (Pew Research)",
            "Solow Productivity Paradox research",
            "Technology diffusion models (Rogers)",
            "AI adoption surveys (McKinsey, Goldman Sachs)"
        ]
    ).with_inputs("goal"),
]