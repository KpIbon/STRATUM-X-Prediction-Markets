"use client"

import { useEffect, useState } from "react"
import { LineChart, ArrowUpRight, ArrowDownRight, RefreshCw } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { getMarketData } from "@/lib/binance"

interface MarketData {
  symbol: string
  price: string
  volume: string
  change24h: string
}

export function MarketDataWidget() {
  const [marketData, setMarketData] = useState<MarketData[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [selectedSymbol, setSelectedSymbol] = useState("BTCUSDT")
  const [refreshing, setRefreshing] = useState(false)

  const fetchData = async () => {
    try {
      setRefreshing(true)
      const data = await getMarketData(["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"])
      setMarketData(data)
    } catch (error) {
      console.error("Error fetching market data:", error)
    } finally {
      setIsLoading(false)
      setRefreshing(false)
    }
  }

  useEffect(() => {
    fetchData()

    // Set up polling for real-time updates
    const interval = setInterval(fetchData, 30000) // Update every 30 seconds

    return () => clearInterval(interval)
  }, [])

  return (
    <Card className="bg-black/80 border-teal-500/30 backdrop-blur-md">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <div>
          <CardTitle className="text-xl text-white flex items-center">
            <LineChart className="mr-2 h-5 w-5 text-teal-500" />
            Market Data
          </CardTitle>
          <CardDescription className="text-gray-400">Real-time cryptocurrency prices via Binance</CardDescription>
        </div>
        <Button
          variant="outline"
          size="sm"
          className="h-8 border-teal-500/50 text-teal-400 hover:bg-teal-900/20"
          onClick={fetchData}
          disabled={refreshing}
        >
          <RefreshCw className={`h-4 w-4 ${refreshing ? "animate-spin" : ""}`} />
        </Button>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue={selectedSymbol} onValueChange={setSelectedSymbol} className="w-full">
          <TabsList className="grid grid-cols-5 bg-gray-900/50">
            {marketData.map((data) => (
              <TabsTrigger
                key={data.symbol}
                value={data.symbol}
                className="data-[state=active]:bg-teal-500 data-[state=active]:text-black"
              >
                {data.symbol.replace("USDT", "")}
              </TabsTrigger>
            ))}
          </TabsList>

          {marketData.map((data) => (
            <TabsContent key={data.symbol} value={data.symbol} className="mt-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-900/50 rounded-lg p-3">
                  <div className="text-gray-400 text-sm mb-1">Price</div>
                  <div className="text-xl font-bold text-white">${Number.parseFloat(data.price).toLocaleString()}</div>
                </div>
                <div className="bg-gray-900/50 rounded-lg p-3">
                  <div className="text-gray-400 text-sm mb-1">24h Change</div>
                  <div
                    className={`text-xl font-bold flex items-center ${
                      Number.parseFloat(data.change24h) >= 0 ? "text-green-500" : "text-red-500"
                    }`}
                  >
                    {Number.parseFloat(data.change24h) >= 0 ? (
                      <ArrowUpRight className="mr-1 h-5 w-5" />
                    ) : (
                      <ArrowDownRight className="mr-1 h-5 w-5" />
                    )}
                    {Number.parseFloat(data.change24h).toFixed(2)}%
                  </div>
                </div>
                <div className="bg-gray-900/50 rounded-lg p-3 col-span-2">
                  <div className="text-gray-400 text-sm mb-1">24h Volume</div>
                  <div className="text-lg font-bold text-white">
                    {Number.parseFloat(data.volume).toLocaleString()} {data.symbol.replace("USDT", "")}
                  </div>
                </div>
              </div>
              <Button
                className="mt-4 w-full bg-teal-500 hover:bg-teal-600 text-black font-bold"
                onClick={() =>
                  window.open(`https://www.binance.com/en/trade/${data.symbol}?theme=dark&type=spot`, "_blank")
                }
              >
                View on Binance
              </Button>
            </TabsContent>
          ))}
        </Tabs>
      </CardContent>
    </Card>
  )
}
