import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "STRATUM-X | Adaptive Forecasting Intelligence",
  description: "Prediction Markets · Regime Detection · Ensemble Forecasting",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="bg-black text-white antialiased">
        {children}
      </body>
    </html>
  );
}
