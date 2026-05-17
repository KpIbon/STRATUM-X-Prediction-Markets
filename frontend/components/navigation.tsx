"use client"

import { useRouter, usePathname } from "next/navigation"
import { Button } from "@/components/ui/button"
import { ArrowLeft, BarChart2, Home, Layers, Network, TrendingUp, Users } from "lucide-react"

export function Navigation() {
  const router = useRouter()
  const pathname = usePathname()

  return (
    <>
      <Button
        variant="ghost"
        className="absolute top-4 left-4 z-20 text-white hover:text-teal-400 hover:bg-transparent"
        onClick={() => router.push("/")}
      >
        <ArrowLeft className="mr-2 h-4 w-4" />
        Back to Home
      </Button>

      <div className="absolute bottom-8 left-0 right-0 z-20 flex justify-center">
        <div className="bg-black/50 backdrop-blur-md border border-gray-800 rounded-full px-2 py-1">
          <div className="flex space-x-1">
            <NavButton
              icon={<Home className="h-5 w-5" />}
              label="Home"
              onClick={() => router.push("/")}
              active={pathname === "/"}
            />
            <NavButton
              icon={<Layers className="h-5 w-5" />}
              label="Strategy"
              onClick={() => router.push("/strategy")}
              active={pathname === "/strategy"}
            />
            <NavButton
              icon={<BarChart2 className="h-5 w-5" />}
              label="Backtest"
              onClick={() => router.push("/backtest")}
              active={pathname === "/backtest"}
            />
            <NavButton
              icon={<TrendingUp className="h-5 w-5" />}
              label="Trading"
              onClick={() => router.push("/trading")}
              active={pathname === "/trading"}
            />
            <NavButton
              icon={<Users className="h-5 w-5" />}
              label="Community"
              onClick={() => router.push("/community")}
              active={pathname === "/community"}
            />
            <NavButton
              icon={<Network className="h-5 w-5" />}
              label="Studio"
              onClick={() => router.push("/studio")}
              active={pathname === "/studio"}
            />
          </div>
        </div>
      </div>
    </>
  )
}

function NavButton({ icon, label, onClick, active }) {
  return (
    <Button
      variant="ghost"
      className={`flex flex-col items-center justify-center px-4 py-2 space-y-1 rounded-full ${
        active ? "bg-teal-500/20 text-teal-400" : "text-gray-400 hover:text-teal-400 hover:bg-transparent"
      }`}
      onClick={onClick}
    >
      {icon}
      <span className="text-xs">{label}</span>
    </Button>
  )
}
