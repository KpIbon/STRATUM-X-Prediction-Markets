export const metadata = {
  title: "STRATUM-X | Adaptive Forecasting Intelligence",
};

export default function Home() {
  return (
    <main className="min-h-screen bg-black text-white flex flex-col items-center justify-center p-8">
      <div className="text-center max-w-2xl">
        <div className="inline-block px-4 py-1.5 rounded-full border border-cyan-500/30 text-cyan-400 text-sm mb-6">
          Hackathon Ready
        </div>
        <h1 className="text-5xl font-bold mb-4">
          <span className="text-cyan-400">STRATUM</span>
          <span className="text-white">-X</span>
        </h1>
        <p className="text-xl text-gray-400 mb-2">Adaptive Forecasting Intelligence</p>
        <p className="text-gray-600 mb-8">Prediction Markets · Regime Detection · Ensemble Forecasting</p>
        <div className="grid grid-cols-3 gap-4 mt-8">
          {[
            ["Agents", "6 autonomous modules"],
            ["Regimes", "4-state classifier"],
            ["Explain", "7-layer reasoning"],
          ].map(([title, desc]) => (
            <div key={title} className="border border-gray-800 rounded-lg p-4 text-left">
              <div className="text-cyan-400 text-sm font-mono mb-1">{title}</div>
              <div className="text-gray-500 text-xs">{desc}</div>
            </div>
          ))}
        </div>
      </div>
    </main>
  );
}
