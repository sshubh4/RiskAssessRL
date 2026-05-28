import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import Topbar from './components/Topbar'
import LeftPanel from './components/LeftPanel'
import TradingChart from './components/TradingChart'
import PortfolioChart from './components/PortfolioChart'
import StatStrip from './components/StatStrip'
import ComparisonTable from './components/ComparisonTable'
import AlgorithmsTab from './components/AlgorithmsTab'

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

  const [ohlcv, setOhlcv]               = useState([])
  const [testStartIdx, setTestStartIdx] = useState(0)
  const [bahHistory, setBahHistory]     = useState([])   // live-computed AAPL B&H fallback
  const [comparison, setComparison]     = useState([])
  const [bahData, setBahData]           = useState(null)
  const [spyBenchmark, setSpyBenchmark]   = useState(null)  // from comparison.json
  const [aaplBenchmark, setAaplBenchmark] = useState(null)  // from comparison.json
  const [regime, setRegime]               = useState(null)
  const [dataFreshness, setDataFreshness] = useState(null)
  const [modelsLoaded, setModelsLoaded]   = useState(0)
  const [models, setModels]               = useState({})  // per-algo loaded status

  const [steps, setSteps]               = useState([])
  const [portfolioHistory, setPortfolioHistory] = useState([])
  const [view, setView]                 = useState('charts')  // 'charts' | 'compare'

  // Backtest-specific risk management params
  const [btStopLoss,   setBtStopLoss]   = useState(0)    // 0 = OFF, else integer % e.g. 15 → -15%
  const [btTakeProfit, setBtTakeProfit] = useState(0)    // 0 = OFF, else integer % e.g. 25 → +25%
  const [btMaxDD,      setBtMaxDD]      = useState(0)    // 0 = OFF, else integer % e.g. 20 → -20%
  const [btPosSize,    setBtPosSize]    = useState(1.0)  // fraction 0.1–1.0

  // Custom test date range (indices into ohlcv); synced with ticker load
  const [btRangeStart, setBtRangeStart] = useState(0)   // updated after ohlcv loads
  const [btRangeEnd,   setBtRangeEnd]   = useState(0)   // updated after ohlcv loads

  // Saved runs for compare-params feature (up to 3)
  const [savedRuns, setSavedRuns] = useState([])

  const wsRef    = useRef(null)
  const stepsRef = useRef([])

  // When algo changes, reset sim params to defaults for that algo type
  useEffect(() => {
    setSimParams(defaultSimParams(algo))
  }, [algo])

  // Sync test range to the canonical split whenever ohlcv/ticker changes
  useEffect(() => {
    if (testStartIdx > 0 && ohlcv.length > 0) {
      setBtRangeStart(testStartIdx)
      setBtRangeEnd(ohlcv.length - 1)
    }
  }, [testStartIdx, ohlcv.length])

  // Load data on ticker change
  useEffect(() => {
    fetch(`/api/data?ticker=${ticker}`)
      .then(r => r.json())
      .then(d => {
        const data = d.data ?? []
        const tsi  = d.test_start_idx ?? Math.floor(data.length * 0.8)
        setOhlcv(data)
        setTestStartIdx(tsi)
        if (d.regime) setRegime(d.regime)
        // Live-computed AAPL B&H — used as fallback when comparison.json has no portfolio_history
        const testPrices = data.slice(tsi).map(x => x.close)
        if (testPrices.length) {
          const shares = Math.floor(capital / testPrices[0])
          const cash   = capital - shares * testPrices[0]
          setBahHistory(testPrices.map(p => cash + shares * p))
        }
      }).catch(() => {})

    fetch('/api/algorithms')
      .then(r => r.json())
      .then(d => {
        setComparison(d.results ?? [])
        setBahData(d.buy_and_hold ?? null)
        const bm = d.benchmarks ?? {}
        setSpyBenchmark(bm.sp500  && bm.sp500.portfolio_history?.length  ? bm.sp500  : null)
        setAaplBenchmark(bm.aapl_hold && bm.aapl_hold.portfolio_history?.length ? bm.aapl_hold : null)
      })
      .catch(() => {})

    fetch('/health')
      .then(r => r.json())
      .then(d => {
        setDataFreshness(d.data_freshness)
        setModelsLoaded(d.models_loaded ?? 0)
        setModels(d.models ?? {})
      })
      .catch(() => {})
  }, [ticker])

  // Recompute B&H when capital changes (always anchored to test period start)
  useEffect(() => {
    if (!ohlcv.length) return
    const testPrices = ohlcv.slice(testStartIdx).map(x => x.close)
    if (!testPrices.length) return
    const shares = Math.floor(capital / testPrices[0])
    const cash   = capital - shares * testPrices[0]
    setBahHistory(testPrices.map(p => cash + shares * p))
  }, [capital, ohlcv, testStartIdx])

  const runSimulation = useCallback(() => {
    if (running) return
    if (wsRef.current) { wsRef.current.close(); wsRef.current = null }

    setRunning(true)
    setSteps([])
    setPortfolioHistory([capital])
    stepsRef.current = []

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    // In backtest mode use bt-specific params; in simulate mode use simParams
    const payload = mode === 'backtest'
      ? {
          algo, capital, speed, ticker, commission,
          position_size: btPosSize, risk_aversion: 0.5,
          epsilon: 0.0, temperature: 1.0, action_threshold: 0.0,
          ...(btStopLoss   > 0 ? { stop_loss:         btStopLoss   / 100 } : {}),
          ...(btTakeProfit > 0 ? { take_profit:        btTakeProfit / 100 } : {}),
          ...(btMaxDD      > 0 ? { max_drawdown_kill:  btMaxDD      / 100 } : {}),
          test_start_idx: btRangeStart,
          test_end_idx:   btRangeEnd + 1,   // API is exclusive end
        }
      : { algo, capital, speed, ticker, commission, ...simParams }
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
          .then(d => {
            setComparison(d.results ?? [])
            setBahData(d.buy_and_hold ?? null)
            const bm = d.benchmarks ?? {}
            setSpyBenchmark(bm.sp500  && bm.sp500.portfolio_history?.length  ? bm.sp500  : null)
            setAaplBenchmark(bm.aapl_hold && bm.aapl_hold.portfolio_history?.length ? bm.aapl_hold : null)
          })
          .catch(() => {})
      } else if (msg.type === 'error') {
        console.error('WS error:', msg.message)
        setRunning(false)
      }
    }

    ws.onerror = () => setRunning(false)
    ws.onclose = () => setRunning(false)
  }, [algo, capital, speed, ticker, commission, simParams, running,
      mode, btStopLoss, btTakeProfit, btMaxDD, btPosSize, btRangeStart, btRangeEnd])

  const runAllBacktest = useCallback(async () => {
    if (running) return
    setRunning(true)
    setSteps([])
    setPortfolioHistory([capital])
    stepsRef.current = []
    try {
      const res = await fetch('/api/run_all_backtest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ticker, capital, commission,
          position_size:     btPosSize,
          stop_loss:         btStopLoss   > 0 ? btStopLoss   / 100 : null,
          take_profit:       btTakeProfit > 0 ? btTakeProfit / 100 : null,
          max_drawdown_kill: btMaxDD      > 0 ? btMaxDD      / 100 : null,
          test_start_idx:    btRangeStart,
          test_end_idx:      btRangeEnd + 1,
        }),
      })
      const d = await res.json()
      setComparison(d.results ?? [])
      setBahData(d.buy_and_hold ?? null)
      const bm = d.benchmarks ?? {}
      setSpyBenchmark(bm.sp500  && bm.sp500.portfolio_history?.length  ? bm.sp500  : null)
      setAaplBenchmark(bm.aapl_hold && bm.aapl_hold.portfolio_history?.length ? bm.aapl_hold : null)
      setView('compare')    // auto-switch to compare tab
    } catch (err) {
      console.error('Run All failed:', err)
    } finally {
      setRunning(false)
    }
  }, [running, ticker, capital, commission, btStopLoss, btTakeProfit, btMaxDD, btPosSize,
      btRangeStart, btRangeEnd])

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

  // Called from AlgorithmsTab — applies recommended settings and switches to backtest
  const applyAlgoConfig = useCallback((algoId, cfg) => {
    setAlgo(algoId)
    setMode('backtest')
    if (cfg) {
      if (cfg.stop_loss    != null) setBtStopLoss(cfg.stop_loss)
      if (cfg.take_profit  != null) setBtTakeProfit(cfg.take_profit)
      if (cfg.position_size != null) setBtPosSize(cfg.position_size)
      // Also update simulate params so the simulate tab reflects the recommendation
      setSimParams(prev => ({
        ...prev,
        ...(cfg.epsilon      != null ? { epsilon:          cfg.epsilon      } : {}),
        ...(cfg.temperature  != null ? { temperature:      cfg.temperature  } : {}),
        ...(cfg.risk_aversion != null ? { risk_aversion:   cfg.risk_aversion } : {}),
        ...(cfg.position_size != null ? { position_size:   cfg.position_size } : {}),
      }))
    }
    setView('charts')  // switch to charts so user can see the run
  }, [])

  const bahSlice = bahHistory.slice(0, portfolioHistory.length)

  // Use SPY benchmark (scaled to current capital) for "vs benchmark" stat.
  // Fall back to live-computed AAPL B&H if comparison.json has no SPY history yet.
  const spySlice = useMemo(() => {
    if (!spyBenchmark?.portfolio_history?.length) return bahSlice
    const scale = capital / 100_000
    return spyBenchmark.portfolio_history
      .slice(0, portfolioHistory.length)
      .map(v => v * scale)
  }, [spyBenchmark, portfolioHistory.length, capital, bahSlice])

  const aaplBahSlice = useMemo(() => {
    if (!aaplBenchmark?.portfolio_history?.length) return bahSlice
    const scale = capital / 100_000
    return aaplBenchmark.portfolio_history.map(v => v * scale)
  }, [aaplBenchmark, capital, bahSlice])

  const stats = computeStats(stepsRef.current, capital, spySlice)

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
          btStopLoss={btStopLoss}     onBtStopLoss={setBtStopLoss}
          btTakeProfit={btTakeProfit} onBtTakeProfit={setBtTakeProfit}
          btMaxDD={btMaxDD}           onBtMaxDD={setBtMaxDD}
          btPosSize={btPosSize}       onBtPosSize={setBtPosSize}
          btRangeStart={btRangeStart} onBtRangeStart={setBtRangeStart}
          btRangeEnd={btRangeEnd}     onBtRangeEnd={setBtRangeEnd}
          onRunAll={runAllBacktest}
          ohlcv={ohlcv} testStartIdx={testStartIdx}
        />

        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', minWidth: 0 }}>

          {/* View toggle tab bar */}
          <div style={{
            display: 'flex', flexShrink: 0,
            borderBottom: '1px solid #2a2e39', background: '#1e2329',
          }}>
            {[['charts', 'Charts'], ['compare', 'Compare Algos'], ['models', 'Models']].map(([id, label]) => (
              <button key={id} onClick={() => setView(id)}
                style={{
                  padding: '7px 18px', border: 'none', background: 'transparent',
                  cursor: 'pointer', fontSize: 10, fontWeight: 700,
                  textTransform: 'uppercase', letterSpacing: '0.07em',
                  color: view === id ? '#d1d4dc' : '#4c525e',
                  borderBottom: view === id ? '2px solid #2962ff' : '2px solid transparent',
                  transition: 'color 0.15s',
                }}>
                {label}
              </button>
            ))}
          </div>

          {/* Charts view */}
          {view === 'charts' && (
            <>
              <div style={{ flex: '3', minHeight: 0, borderBottom: '1px solid #2a2e39' }}>
                <TradingChart ohlcv={ohlcv} steps={steps} ticker={ticker}
                  testStartIdx={testStartIdx} regime={regime} />
              </div>
              <StatStrip stats={stats} algo={algo} />
              <div style={{ flex: '2', minHeight: 0 }}>
                <PortfolioChart
                  portfolioHistory={portfolioHistory}
                  spyHistory={spySlice}
                  aaplBahHistory={aaplBahSlice}
                  ohlcv={ohlcv.slice(testStartIdx)}
                  initialCapital={capital}
                  savedRuns={savedRuns}
                  algo={algo}
                />
              </div>
            </>
          )}

          {/* Compare view — full height */}
          {view === 'compare' && (
            <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: '16px 20px' }}>
              <ComparisonTable
                results={comparison}
                bahData={bahData}
                spyBenchmark={spyBenchmark}
                aaplBenchmark={aaplBenchmark}
              />
            </div>
          )}

          {/* Models showcase tab */}
          {view === 'models' && (
            <div style={{ flex: 1, minHeight: 0, overflowY: 'auto' }}>
              <AlgorithmsTab
                results={comparison}
                models={models}
                spyBenchmark={spyBenchmark}
                onApply={applyAlgoConfig}
              />
            </div>
          )}

        </div>

      </div>
    </div>
  )
}
