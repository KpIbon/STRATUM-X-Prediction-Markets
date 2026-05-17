"use server"

import { generateText } from "ai"
import { openai } from "@ai-sdk/openai"

// Check if OpenAI API key is available
const isOpenAIConfigured = !!process.env.OPENAI_API_KEY

export async function getAiStrategyAdvice(marketCondition: string, currentStrategy?: string): Promise<string> {
  try {
    // Check if OpenAI API key is configured
    if (!isOpenAIConfigured) {
      return "OpenAI API key is not configured. Please add the OPENAI_API_KEY environment variable to enable AI-powered strategy advice."
    }

    const { text } = await generateText({
      model: openai("gpt-4o"),
      prompt: `
        As an algorithmic trading expert, provide advice for a trading strategy based on the following market conditions:
        ${marketCondition}
        
        ${currentStrategy ? `The current strategy is: ${currentStrategy}` : ""}
        
        Provide specific recommendations for indicators, entry/exit conditions, and risk management parameters.
        Format your response in a clear, concise manner that can be easily understood by traders.
      `,
      system:
        "You are an expert algorithmic trading assistant for AlgoSensei, a no-code trading strategy builder platform. Provide professional, accurate, and actionable trading strategy advice.",
    })

    return text
  } catch (error) {
    console.error("Error getting AI strategy advice:", error)
    if (error.message?.includes("API key")) {
      return "OpenAI API key is missing or invalid. Please add a valid OPENAI_API_KEY to your environment variables."
    }
    return "Unable to generate strategy advice at this time. Please try again later."
  }
}

export async function analyzeBacktestResults(results: string): Promise<string> {
  try {
    // Check if OpenAI API key is configured
    if (!isOpenAIConfigured) {
      return "OpenAI API key is not configured. Please add the OPENAI_API_KEY environment variable to enable AI-powered analysis."
    }

    const { text } = await generateText({
      model: openai("gpt-4o"),
      prompt: `
        Analyze the following backtest results and provide insights and optimization suggestions:
        ${results}
        
        Include analysis of:
        1. Overall performance metrics
        2. Risk-adjusted returns
        3. Potential weaknesses in the strategy
        4. Specific optimization suggestions
      `,
      system:
        "You are an expert algorithmic trading assistant for AlgoSensei. Provide professional, data-driven analysis of backtest results with actionable optimization suggestions.",
    })

    return text
  } catch (error) {
    console.error("Error analyzing backtest results:", error)
    if (error.message?.includes("API key")) {
      return "OpenAI API key is missing or invalid. Please add a valid OPENAI_API_KEY to your environment variables."
    }
    return "Unable to analyze backtest results at this time. Please try again later."
  }
}

// Function to provide mock AI strategy advice when OpenAI API is not available
// Making this async to comply with Server Action requirements
export async function getMockStrategyAdvice(marketCondition: string): Promise<string> {
  // Simple keyword-based response system
  const isBullish =
    marketCondition.toLowerCase().includes("bullish") || marketCondition.toLowerCase().includes("uptrend")
  const isBearish =
    marketCondition.toLowerCase().includes("bearish") || marketCondition.toLowerCase().includes("downtrend")
  const isVolatile =
    marketCondition.toLowerCase().includes("volatile") || marketCondition.toLowerCase().includes("volatility")

  if (isBullish) {
    return `
Based on the bullish market conditions you've described, here's a strategy recommendation:

STRATEGY: Trend-Following with Pullback Entry

INDICATORS:
- Moving Averages: Use 20 EMA and 50 EMA for trend direction
- RSI (14): For identifying oversold conditions in an uptrend
- Volume: Confirm trend strength with increasing volume

ENTRY CONDITIONS:
- Price is above both 20 EMA and 50 EMA (confirming uptrend)
- Wait for a pullback where price approaches but remains above the 20 EMA
- Enter when RSI recovers from an oversold condition (crosses above 40)

EXIT CONDITIONS:
- Take profit at predetermined levels (e.g., previous resistance or 1:2 risk-reward ratio)
- Set trailing stop loss to lock in profits as the trend continues
- Exit if price closes below the 50 EMA

RISK MANAGEMENT:
- Limit position size to 2-3% of portfolio per trade
- Set initial stop loss below the 50 EMA or recent swing low
- Consider scaling out of positions at multiple profit targets

This strategy takes advantage of the bullish trend while waiting for optimal entry points during temporary pullbacks.
`
  } else if (isBearish) {
    return `
Based on the bearish market conditions you've described, here's a strategy recommendation:

STRATEGY: Controlled Short Selling with Confirmation

INDICATORS:
- Moving Averages: 20 EMA and 50 EMA for trend confirmation
- MACD: For bearish momentum confirmation
- Volume: To confirm selling pressure

ENTRY CONDITIONS:
- Price is below both 20 EMA and 50 EMA (confirming downtrend)
- MACD histogram is negative and expanding
- Enter short positions on bounces to the 20 EMA that fail (resistance tests)

EXIT CONDITIONS:
- Take profit at predetermined support levels
- Cover position when MACD shows bullish divergence
- Exit if price closes above the 20 EMA

RISK MANAGEMENT:
- Limit position size to 1-2% of portfolio (more conservative in bearish markets)
- Set stop loss above recent swing high or the 50 EMA
- Consider using options or defined-risk instruments instead of direct short selling
- Be prepared to exit quickly if market sentiment shifts

This strategy aims to capitalize on the bearish trend while maintaining strict risk controls, which is essential in downtrending markets.
`
  } else if (isVolatile) {
    return `
Based on the volatile market conditions you've described, here's a strategy recommendation:

STRATEGY: Volatility Breakout with Confirmation

INDICATORS:
- Bollinger Bands (20, 2): To measure and visualize volatility
- ATR (14): To quantify volatility and set appropriate stop losses
- RSI (14): For identifying potential reversal points

ENTRY CONDITIONS:
- Wait for Bollinger Band squeeze (bands narrowing) indicating potential breakout
- Enter in the direction of the breakout when price closes outside the bands
- Confirm with increased volume and RSI momentum in breakout direction

EXIT CONDITIONS:
- Take profit when price reaches the opposite Bollinger Band
- Exit if price reverses and crosses back inside the bands
- Use time-based exits (e.g., close position if no significant movement within 3 periods)

RISK MANAGEMENT:
- Smaller position sizes (1-2% of portfolio) due to higher volatility
- Set stop losses at 1.5-2x ATR from entry point
- Consider using options strategies that benefit from volatility (e.g., straddles)
- Split entries into multiple smaller positions to average into the trade

This strategy is designed to capitalize on explosive price movements that often follow periods of compression, while maintaining strict risk controls appropriate for volatile conditions.
`
  } else {
    return `
Based on the market conditions you've described, here's a general strategy recommendation:

STRATEGY: Adaptive Moving Average Strategy

INDICATORS:
- Moving Averages: 10 EMA, 20 SMA, and 50 SMA
- RSI (14): For identifying potential overbought/oversold conditions
- Volume: For confirmation of price movements

ENTRY CONDITIONS:
- For long positions: Price crosses above 20 SMA with 10 EMA > 20 SMA
- For short positions: Price crosses below 20 SMA with 10 EMA < 20 SMA
- Confirm with increasing volume in the direction of the trade

EXIT CONDITIONS:
- Take profit at predetermined levels based on support/resistance
- Exit long positions when price crosses below 10 EMA
- Exit short positions when price crosses above 10 EMA

RISK MANAGEMENT:
- Position size limited to 2% of portfolio per trade
- Set stop loss at recent swing high/low or based on ATR
- Consider scaling in/out of positions to manage risk

This adaptive strategy can work in various market conditions by following the intermediate trend while using shorter-term signals for entries and exits.

Note: For more specific advice, please provide additional details about the particular market conditions you're observing.
`
  }
}
