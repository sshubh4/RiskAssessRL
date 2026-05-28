import { useState, useEffect, useRef, useCallback } from 'react'
import Topbar from './components/Topbar'
import LeftPanel from './components/LeftPanel'
import TradingChart from './components/TradingChart'
import PortfolioChart from './components/PortfolioChart'
import StatStrip from './components/StatStrip'
import ComparisonTable from './components/ComparisonTable'

const WS_URL = `ws://${window.location.hostname}:8000/ws/simulate`

const RUN_COLORS = ['#2962ff', '#f6c90e', '#9c27b0']

export function computeStats(steps, initial, bahHistory) {
  if (!steps?.length) return {}
  const hist = [initial, ...steps.map(s => s.portfolio_value)]
  const last = hist[hist.length - 1]
  const totalReturn = (last - initial) / initial * 100

  const returns = hist.slice(1).map((v, i) => (v - hist[i]) / hist[i])
  const rf = 0.04 / 252
  const excess = returns.map(r => r - rf)
  const mean = excess.reduce((a, b) => a + b, 0) / excess.length
  const variance = excess.reduce((a, b) => a + (b - mean) ** 2, 0) / excess.length
  const std = Math.sqrt(variance)
  const sharpe = std > 1e-8 ? Math.sqrt(252) * mean / std : 0

  let peak = -Infinity, maxDD = 0
  for (const v of hist) {
    if (v > peak) peak = v
    const dd = (v - peak) / peak
    if (dd < maxDD) maxDD = dd
  }

  const buys = steps.filter(s => s.action === 0)
  const sells = steps.filter(s => s.action === 1)
  const trades = Math.min(buys.length, sells.length)
  let winning = 0
  for (let i = 0; i < trades; i++) {
    if (sells[i]?.portfolio_value > buys[i]?.portfolio_value) winning++
  }
  const winRate = trades > 0 ? (winning / trades * 100) : 0

  let vsBah = null
  if (bahHistory?.length) {
    const bahReturn = (bahHistory[bahHistory.length - 1] - bahHistory[0]) / bahHistory[0] * 100
    vsBah = totalReturn - bahReturn
  }

  return { portfolioValue: last, totalReturn, sharpe, maxDD, trades, winRate, vsBah, capital: initial }
}

// Default simulation params per algo type
export function defaultSimParams(algo) {
  const isValueBased = ['DQN', 'DDQN'].includes(algo?.toUpperCase())
  return isValueBased
    ? { epsilon: 0.05, risk_aversion: 0.5, position_size: 1.0, temperature: 1.0, action_threshold: 0.0 }
    : { epsilon: 0.0,  risk_aversion: 0.5, position_size: 1.0, temperature: 1.0, action_threshold: 0.0 }
}

export default function App() {
  const [ticker, setTicker]       = useState('AAPL')
  const [algo, setAlgo]           = useState('DQN')
  const [capital, setCapital]     = useState(100_000)
  const [commission, setCommission] = useState(0.001)
  const [speed, setSpeed]         = useState(5)
  const [mode, setMode]           = useState('backtest')   // 'backtest' | 'simulate'
  const [running, setRunning]     = useState(false)
  const [simParams, setSimParams] = useState(defaultSimParams('DQN'))

  const [ohlcv, setOhlcv]           = useState([])
  const [bahHistory, setBahHistory] = useState([])
  const [comparison, setComparison] = useState([])
  const [bahData, setBahData]       = useState(null)
  const [dataFreshness, setDataFreshness] = useState(null)
  const [modelsLoaded, setModelsLoaded]   = useState(0)

  const [steps, setSteps]               = useState([])
  const [portfolioHistory, setPortfolioHistory] = useState([])
  const [showTable, setShowTable]       = useState(true)

  // Saved runs for compare-params feature (up to 3)
  const [savedRuns, setSavedRuns] = useState([])

  const wsRef    = useRef(null)
  const stepsRef = useRef([])

  // When algo changes, reset sim params to defaults for that algo type
  useEffect(() => {
    setSimParams(defaultSimParams(algo))
  }, [algo])

  // Load data on ticker change
  useEffect(() => {
    fetch(`/api/data?ticker=${ticker}`)
      .then(r => r.json())
      .then(d => {
        const data = d.data ?? []
        setOhlcv(data)
        const prices = data.map(x => x.close)
        if (prices.length) {
          const shares = Math.floor(capital / prices[0])
          const cash = capital - shares * prices[0]
          setBahHistory(prices.map(p => cash + shares * p))
        }
      }).catch(() => {})

    fetch('/api/algorithms')
      .then(r => r.json())
      .then(d => { setComparison(d.results ?? []); setBahData(d.buy_and_hold ?? null) })
      .catch(() => {})

    fetch('/health')
      .then(r => r.json())
      .then(d => { setDataFreshness(d.data_freshness); setModelsLoaded(d.models_loaded ?? 0) })
      .catch(() => {})
  }, [ticker])

  // Recompute B&H when capital changes
  useEffect(() => {
    if (!ohlcv.length) return
    const prices = ohlcv.map(x => x.close)
    const shares = Math.floor(capital / prices[0])
    const cash = capital - shares * prices[0]
    setBahHistory(prices.map(p => cash + shares * p))
  }, [capital, ohlcv])

  const runSimulation = useCallback(() => {
    if (running) return
    if (wsRef.current) { wsRef.current.close(); wsRef.current = null }

    setRunning(true)
    setSteps([])
    setPortfolioHistory([capital])
    stepsRef.current = []

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    const payload = { algo, capital, speed, ticker, commission, ...simParams }
    ws.onopen = () => ws.send(JSON.stringify(payload))

    ws.onmessage = e => {
      const msg = JSON.parse(e.data)
      if (msg.type === 'step') {
        stepsRef.current.push(msg.data)
        setSteps(prev => [...prev, msg.data])
        setPortfolioHistory(prev => [...prev, msg.data.portfolio_value])
      } else if (msg.type === 'done' || msg.type === 'complete') {
        setRunning(false)
        fetch('/api/algorithms')
          .then(r => r.json())
          .then(d => { setComparison(d.results ?? []); setBahData(d.buy_and_hold ?? null) })
          .catch(() => {})
      } else if (msg.type === 'error') {
        console.error('WS error:', msg.message)
        setRunning(false)
      }
    }

    ws.onerror = () => setRunning(false)
    ws.onclose = () => setRunning(false)
  }, [algo, capital, speed, ticker, commission, simParams, running])

  const saveRun = useCallback(() => {
    if (!stepsRef.current.length) return
    const hist = [capital, ...stepsRef.current.map(s => s.portfolio_value)]
    const idx = savedRuns.length % RUN_COLORS.length
    const color = RUN_COLORS[idx]
    const label = `Run ${savedRuns.length + 1}`
    const newRun = {
      id:    Date.now(),
      label,
      algo,
      color,
      params: { ...simParams, commission },
      portfolioHistory: hist,
      steps: [...stepsRef.current],
    }
    setSavedRuns(prev => {
      const next = [...prev, newRun]
      return next.length > 3 ? next.slice(next.length - 3) : next
    })
  }, [savedRuns.length, capital, algo, simParams, commission])

  const clearRuns = useCallback(() => setSavedRuns([]), [])

  const bahSlice = bahHistory.slice(0, portfolioHistory.length)
  const stats = computeStats(stepsRef.current, capital, bahSlice)

  const lastPrice  = ohlcv.length ? ohlcv[ohlcv.length - 1].close : null
  const prevPrice  = ohlcv.length > 1 ? ohlcv[ohlcv.length - 2].close : null
  const priceChange    = lastPrice && prevPrice ? lastPrice - prevPrice : null
  const priceChangePct = lastPrice && prevPrice ? (priceChange / prevPrice) * 100 : null

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: '#131722', overflow: 'hidden' }}>
      <Topbar
        ticker={ticker} onTicker={setTicker}
        lastPrice={lastPrice} priceChange={priceChange} priceChangePct={priceChangePct}
        dataFreshness={dataFreshness} modelsLoaded={modelsLoaded} running={running}
      />

      <div style={{ display: 'flex', flex: 1, overflow: 'hidden', minHeight: 0 }}>
        <LeftPanel
          algo={algo} onAlgo={setAlgo}
          capital={capital} onCapital={setCapital}
          commission={commission} onCommission={setCommission}
          speed={speed} onSpeed={setSpeed}
          mode={mode} onMode={setMode}
          simParams={simParams} onSimParams={setSimParams}
          onRun={runSimulation} running={running}
          stats={stats} modelsLoaded={modelsLoaded}
          savedRuns={savedRuns}
          onSaveRun={saveRun}
          onClearRuns={clearRuns}
          hasCurrentRun={stepsRef.current.length > 0}
        />

        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', minWidth: 0 }}>
          <div style={{ flex: '3', minHeight: 0, borderBottom: '1px solid #2a2e39' }}>
            <TradingChart ohlcv={ohlcv} steps={steps} ticker={ticker} />
          </div>
          <StatStrip stats={stats} algo={algo} />
          <div style={{ flex: '2', minHeight: 0 }}>
            <PortfolioChart
              portfolioHistory={portfolioHistory}
              bahHistory={bahSlice}
              ohlcv={ohlcv}
              initialCapital={capital}
              savedRuns={savedRuns}
            />
          </div>
        </div>

      </div>

      <div style={{
        height: showTable ? '160px' : '32px', flexShrink: 0,
        borderTop: '1px solid #2a2e39', background: '#1e2329',
        transition: 'height 0.2s ease', overflow: 'hidden',
      }}>
        <div
          style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            padding: '6px 16px', cursor: 'pointer',
            borderBottom: showTable ? '1px solid #2a2e39' : 'none' }}
          onClick={() => setShowTable(t => !t)}
        >
          <span style={{ fontSize: '10px', color: '#787b86', textTransform: 'uppercase',
            letterSpacing: '0.08em', fontWeight: 600 }}>Algorithm Comparison</span>
          <span style={{ color: '#787b86', fontSize: '11px' }}>{showTable ? '▼' : '▲'}</span>
        </div>
        {showTable && (
          <div style={{ height: '128px', overflowY: 'auto' }}>
            <ComparisonTable results={comparison} bahData={bahData} />
          </div>
        )}
      </div>
    </div>
  )
}
