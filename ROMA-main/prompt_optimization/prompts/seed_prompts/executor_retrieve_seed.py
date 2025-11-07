"""RETRIEVE task executor instruction seed prompt for DSPy.

This module provides an optimized instruction prompt and demos specifically
for RETRIEVE tasks - fast data fetching with minimal reasoning.
"""

import dspy

EXECUTOR_RETRIEVE_PROMPT = r"""
# Executor (RETRIEVE) — Instruction Prompt

Role
Execute RETRIEVE tasks efficiently: fetch, extract, and present data from external sources with minimal processing.

Task Characteristics (RETRIEVE)
- Primary goal: Get specific data points quickly (metrics, values, statistics, records)
- Speed priority: Fast retrieval over deep analysis
- Tool-heavy: Requires API calls, database queries, web searches
- Simple output: Direct data presentation without extensive interpretation

Execution Guidelines (RETRIEVE-Specific)
1. Direct tool usage: Immediately use the most appropriate tool for the data request
2. Minimal reasoning: Fetch data without extensive analysis or interpretation
3. Fresh data priority: Always fetch current/real-time data when available
4. Multiple sources: Cross-reference critical data from multiple sources when possible
5. Error recovery: If primary source fails, try alternative sources immediately
6. Format consistency: Present numeric data with proper units and precision
7. Single-pass execution: Complete retrieval in 1-3 tool calls maximum

Output Contract (strict)
- `output` (string): The requested data formatted clearly and concisely
- `sources` (list[str]): Specific tools, APIs, or data sources used for retrieval

Quality Standards (RETRIEVE)
- Speed: Minimize tool calls and iterations - get data directly
- Accuracy: Use official/authoritative sources when available
- Precision: Include exact values with appropriate decimal places and units
- Freshness: Include timestamps or "as of" dates when relevant
- Clarity: Present data in clean, structured format (tables, lists, or prose)
- Source transparency: Always cite where data came from

Common RETRIEVE Patterns
- Single value query: Tool call → extract value → format with units → cite source
- Multi-value query: Tool call → extract multiple fields → tabular format → cite
- Comparison query: Multiple tool calls → side-by-side presentation → cite each
- Historical query: Time-series tool call → specify date range → cite
- Search query: Web/database search → extract relevant results → cite

Error Handling (RETRIEVE-Specific)
- API failures: Try alternative sources before reporting failure
- Rate limits: State limitation clearly, provide partial data if available
- Missing data: Explicitly state "Data not available from [source]"
- Stale data: Include timestamp and warn if data may be outdated
- Ambiguous queries: Ask for clarification or make reasonable assumptions
- Invalid inputs: Suggest corrections or alternatives

Output Format Guidelines
- Single value: "Metric: Value Units (Source, Timestamp)"
- Multiple values: Use bullet lists, tables, or JSON as appropriate
- Comparisons: Side-by-side format with clear labels
- Time series: Include date ranges and data frequency
- Large datasets: Summarize key statistics, offer to save full data
"""

EXECUTOR_RETRIEVE_DEMOS = [
    # Demo 1: Simple API data retrieval (weather)
    dspy.Example(
        goal="What is the current weather in San Francisco?",
        output=(
            "San Francisco, CA Weather:\n"
            "Temperature: 62°F (17°C)\n"
            "Conditions: Partly cloudy\n"
            "Humidity: 68%\n"
            "Wind: 12 mph NW\n"
            "UV Index: 5 (Moderate)\n"
            "Data as of: 2025-10-20 14:30 PDT"
        ),
        sources=["Weather API - /current/San Francisco"]
    ).with_inputs("goal"),

    # Demo 2: Simple data point retrieval (crypto)
    dspy.Example(
        goal="Get the current price of Bitcoin in USD",
        output=(
            "Bitcoin (BTC) price: $43,521.37 USD\n"
            "24h change: +2.34%\n"
            "Market cap: $852.4B\n"
            "Data as of: 2025-10-20 14:32 UTC"
        ),
        sources=["CoinGecko API - /coins/bitcoin"]
    ).with_inputs("goal"),

    # Demo 3: Database query result
    dspy.Example(
        goal="How many active users registered in the last 30 days?",
        output=(
            "User Registration Statistics (Last 30 days):\n\n"
            "Total new registrations: 1,247 users\n"
            "Active users (1+ login): 892 users (71.5%)\n"
            "Daily average: 41.6 registrations/day\n"
            "Peak day: Oct 15 (87 registrations)\n\n"
            "Breakdown by source:\n"
            "- Organic: 623 (49.9%)\n"
            "- Referral: 387 (31.0%)\n"
            "- Paid ads: 237 (19.1%)\n\n"
            "Query executed: 2025-10-20 14:35 UTC"
        ),
        sources=["PostgreSQL - users table, analytics_events table"]
    ).with_inputs("goal"),

    # Demo 4: Multi-item comparison (crypto)
    dspy.Example(
        goal="Compare the prices of Bitcoin, Ethereum, and Solana",
        output=(
            "Price Comparison (2025-10-20 14:35 UTC):\n\n"
            "1. Bitcoin (BTC)\n"
            "   Price: $43,521.37\n"
            "   24h Change: +2.34%\n"
            "   Market Cap: $852.4B (#1)\n\n"
            "2. Ethereum (ETH)\n"
            "   Price: $2,287.56\n"
            "   24h Change: +1.87%\n"
            "   Market Cap: $275.1B (#2)\n\n"
            "3. Solana (SOL)\n"
            "   Price: $98.42\n"
            "   24h Change: +5.12%\n"
            "   Market Cap: $45.3B (#5)"
        ),
        sources=["CoinGecko API - /coins/markets"]
    ).with_inputs("goal"),

    # Demo 3: DeFi protocol TVL
    dspy.Example(
        goal="Get the total value locked (TVL) in Uniswap protocol",
        output=(
            "Uniswap TVL: $3.847 billion (as of 2025-10-20)\n\n"
            "Chain breakdown:\n"
            "- Ethereum: $2.145B (55.8%)\n"
            "- Polygon: $0.892B (23.2%)\n"
            "- Arbitrum: $0.534B (13.9%)\n"
            "- Optimism: $0.276B (7.1%)\n\n"
            "24h change: +1.2%\n"
            "Rank: #4 by TVL (all protocols)"
        ),
        sources=["DefiLlama API - /protocol/uniswap"]
    ).with_inputs("goal"),

    # Demo 4: Trading volume query
    dspy.Example(
        goal="What is the 24-hour trading volume for BTC/USDT on Binance?",
        output=(
            "BTC/USDT Trading Volume (Binance Spot):\n\n"
            "24h Volume (BTC): 45,823.52 BTC\n"
            "24h Volume (USDT): $1,994,827,431\n"
            "Number of trades: 2,847,392\n\n"
            "Price range (24h):\n"
            "- High: $44,125.00\n"
            "- Low: $42,387.50\n"
            "- Current: $43,521.37\n\n"
            "Data as of: 2025-10-20 14:40 UTC"
        ),
        sources=["Binance API - /api/v3/ticker/24hr"]
    ).with_inputs("goal"),

    # Demo 5: Token holder information
    dspy.Example(
        goal="How many holders does USDC have on Ethereum?",
        output=(
            "USDC (USD Coin) Holder Statistics on Ethereum:\n\n"
            "Total holders: 2,847,392 addresses\n"
            "Total supply: 24.8B USDC\n"
            "Contract: 0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48\n\n"
            "Top holder concentration:\n"
            "- Top 10 holders: 42.3% of supply\n"
            "- Top 100 holders: 67.8% of supply\n"
            "- Remaining holders: 32.2% of supply\n\n"
            "Data as of: 2025-10-20 (Ethereum mainnet)"
        ),
        sources=["Arkham Intelligence API - /tokens/ethereum/usdc"]
    ).with_inputs("goal"),

    # Demo 6: Search-based retrieval
    dspy.Example(
        goal="Find the latest news about Ethereum ETF approval",
        output=(
            "Latest Ethereum ETF News (as of 2025-10-20):\n\n"
            "1. SEC Approves 8 Spot Ethereum ETFs (May 23, 2024)\n"
            "   - Approved: BlackRock, Fidelity, Grayscale, Bitwise, VanEck, ARK, Invesco, Franklin\n"
            "   - Trading began: July 23, 2024\n"
            "   - Source: SEC.gov, Bloomberg\n\n"
            "2. Ethereum ETFs See $2.1B Inflows (October 2025)\n"
            "   - Total AUM: $8.4 billion across all ETFs\n"
            "   - BlackRock leads with $3.2B AUM\n"
            "   - Source: Bloomberg ETF Research\n\n"
            "3. Institutional Adoption Growing\n"
            "   - 47% of surveyed institutions plan ETH ETF allocation\n"
            "   - Average allocation: 3-5% of crypto portfolio\n"
            "   - Source: Coinbase Institutional Survey"
        ),
        sources=["Serper News Search - 'Ethereum ETF approval latest'"]
    ).with_inputs("goal"),

    # Demo 7: Error handling - missing data
    dspy.Example(
        goal="Get the price of SHIB token on Solana blockchain",
        output=(
            "Data retrieval issue:\n\n"
            "SHIB (Shiba Inu) is primarily an ERC-20 token on Ethereum, not natively on Solana.\n\n"
            "Available SHIB data:\n"
            "- Ethereum (native): $0.00000842\n"
            "- Wrapped versions may exist on Solana but are not tracked by major APIs\n\n"
            "Recommendation: Please specify if you're looking for:\n"
            "1. SHIB price on Ethereum (native chain)\n"
            "2. A wrapped/bridged version on Solana\n"
            "3. A different token with similar name on Solana"
        ),
        sources=["CoinGecko API - /coins/shiba-inu", "Manual chain verification"]
    ).with_inputs("goal"),

    # Demo 8: Search-based retrieval (research papers)
    dspy.Example(
        goal="Find recent research papers on transformer architecture improvements",
        output=(
            "Recent Transformer Architecture Research (2025):\n\n"
            "1. \"Efficient Transformers with Linear Attention\" (Mar 2025)\n"
            "   Authors: Chen et al., Stanford University\n"
            "   Key finding: 40% faster inference with minimal accuracy loss\n"
            "   Citation: arXiv:2503.12847\n\n"
            "2. \"Sparse Mixture-of-Experts at Scale\" (Feb 2025)\n"
            "   Authors: Zhang et al., Google Research\n"
            "   Key finding: 10x parameter efficiency using dynamic routing\n"
            "   Citation: arXiv:2502.08934\n\n"
            "3. \"Context Length Extension via Sliding Window\" (Jan 2025)\n"
            "   Authors: Kumar et al., Meta AI\n"
            "   Key finding: 1M token context with constant memory\n"
            "   Citation: arXiv:2501.05621"
        ),
        sources=["arXiv Search API - 'transformer architecture 2025'", "Google Scholar"]
    ).with_inputs("goal"),

    # Demo 9: Error handling - ambiguous request
    dspy.Example(
        goal="Get the status of order 12345",
        output=(
            "Data retrieval issue:\n\n"
            "Multiple orders found with ID containing '12345':\n\n"
            "1. Order #ORD-12345-A (Customer: John Smith)\n"
            "   Status: Shipped (tracking: 1Z9XX8420398765432)\n"
            "   Date: 2025-10-15\n\n"
            "2. Order #ORD-12345-B (Customer: Jane Doe)\n"
            "   Status: Processing\n"
            "   Date: 2025-10-19\n\n"
            "Please specify:\n"
            "- Full order ID (e.g., ORD-12345-A), or\n"
            "- Customer name or email, or\n"
            "- Order date range"
        ),
        sources=["Order Management System - orders table"]
    ).with_inputs("goal"),

    # Demo 10: Cross-source verification (crypto)
    dspy.Example(
        goal="Verify the current Bitcoin price from multiple sources",
        output=(
            "Bitcoin Price Verification (2025-10-20 14:45 UTC):\n\n"
            "Source comparison:\n"
            "1. CoinGecko: $43,521.37\n"
            "2. Binance: $43,519.80 (BTC/USDT spot)\n"
            "3. Coinbase: $43,524.15 (BTC/USD)\n\n"
            "Average: $43,521.77\n"
            "Spread: $4.35 (0.01%)\n"
            "Consensus: $43,521 ± $2\n\n"
            "Note: Minor differences are normal due to:\n"
            "- Exchange-specific liquidity\n"
            "- Data refresh timing (±10 seconds)\n"
            "- USD vs USDT pricing\n\n"
            "All sources agree within 0.01% - price is reliable."
        ),
        sources=[
            "CoinGecko API - /coins/bitcoin",
            "Binance API - /ticker/price",
            "Coinbase API - /products/BTC-USD/ticker"
        ]
    ).with_inputs("goal"),
]