"use server"

import { createHmac } from "crypto"

const API_URL = "https://api.binance.com"

interface MarketData {
  symbol: string
  price: string
  volume: string
  change24h: string
}

interface KlineData {
  openTime: number
  open: string
  high: string
  low: string
  close: string
  volume: string
  closeTime: number
}

export async function getMarketData(symbols: string[] = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]): Promise<MarketData[]> {
  try {
    const responses = await Promise.all(
      symbols.map((symbol) => fetch(`${API_URL}/api/v3/ticker/24hr?symbol=${symbol}`).then((res) => res.json())),
    )

    return responses.map((data) => ({
      symbol: data.symbol,
      price: data.lastPrice,
      volume: data.volume,
      change24h: data.priceChangePercent,
    }))
  } catch (error) {
    console.error("Error fetching market data:", error)
    throw new Error("Failed to fetch market data")
  }
}

export async function getHistoricalData(symbol: string, interval = "1d", limit = 100): Promise<KlineData[]> {
  try {
    const response = await fetch(`${API_URL}/api/v3/klines?symbol=${symbol}&interval=${interval}&limit=${limit}`)

    const data = await response.json()

    return data.map((kline) => ({
      openTime: kline[0],
      open: kline[1],
      high: kline[2],
      low: kline[3],
      close: kline[4],
      volume: kline[5],
      closeTime: kline[6],
    }))
  } catch (error) {
    console.error("Error fetching historical data:", error)
    throw new Error("Failed to fetch historical data")
  }
}

// This function requires authentication and should only be used server-side
export async function executeTrade(symbol: string, side: "BUY" | "SELL", quantity: string, price?: string) {
  try {
    const timestamp = Date.now()
    const apiKey = process.env.BINANCE_API_KEY
    const apiSecret = process.env.BINANCE_API_SECRET

    if (!apiKey || !apiSecret) {
      throw new Error("Binance API credentials not configured")
    }

    const params = new URLSearchParams({
      symbol,
      side,
      type: price ? "LIMIT" : "MARKET",
      quantity,
      timestamp: timestamp.toString(),
      ...(price && { price }),
    })

    const signature = createHmac("sha256", apiSecret).update(params.toString()).digest("hex")

    params.append("signature", signature)

    const response = await fetch(`${API_URL}/api/v3/order`, {
      method: "POST",
      headers: {
        "X-MBX-APIKEY": apiKey,
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: params,
    })

    return await response.json()
  } catch (error) {
    console.error("Error executing trade:", error)
    throw new Error("Failed to execute trade")
  }
}
