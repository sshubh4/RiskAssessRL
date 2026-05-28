import { useMemo } from 'react'

const ALGOS = [
  { id: 'Random', short: 'Rnd' },
  { id: 'DQN',    short: 'DQN' },
  { id: 'DDQN',   short: 'DDQN' },
  { id: 'A2C',    short: 'A2C' },
  { id: 'PPO',    short: 'PPO' },
]

const RUN_COLORS = ['#2962ff', '#f6c90e', '#9c27b0']

// ── atoms ────────────────────────────────────────────────────────────────────

function Divider() {
  return <div style={{ height: 1, background: '#2a2e39', margin: '6px 0' }} />
}

function Label({ children }) {
  return (
    <div style={{ fontSize: 10, fontWeight: 600, color: '#4c525e',
      textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 5 }}>
      {children}
    </div>
  )
}

function Slider({ label, value, min, max, step, onChange, disabled, fmt, hint }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
        <span style={{ fontSize: 11, color: '#787b86' }}>{label}</span>
        <span style={{ fontSize: 11, fontWeight: 700, color: '#d1d4dc', fontVariantNumeric: 'tabular-nums' }}>
          {fmt ? fmt(value) : value}
        </span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))} disabled={disabled}
        style={{ width: '100%', accentColor: '#2962ff', cursor: disabled ? 'not-allowed' : 'pointer' }} />
      {hint && <div style={{ fontSize: 9, color: '#4c525e', marginTop: 1 }}>{hint}</div>}
    </div>
  )
}

function SpeedPills({ speed, onSpeed, disabled }) {
  return (
    <div style={{ display: 'flex', gap: 4 }}>
      {[1, 5, 10, 20].map(s => (
        <button key={s} onClick={() => onSpeed(s)} disabled={disabled}
          style={{
            flex: 1, padding: '4px 0', borderRadius: 3,
            border: speed === s ? '1px solid #2962ff' : '1px solid #2a2e39',
            background: speed === s ? 'rgba(41,98,255,0.15)' : '#2a2e39',
            color: speed === s ? '#5c8df6' : '#787b86',
            fontSize: 11, fontWeight: speed === s ? 700 : 400,
            cursor: disabled ? 'not-allowed' : 'pointer',
          }}>
          {s}×
        </button>
      ))}
    </div>
  )
}

function RunButton({ onClick, disabled, running, label }) {
  return (
    <button onClick={onClick} disabled={disabled} style={{
      width: '100%', padding: '10px 0', borderRadius: 5, border: 'none',
      background: disabled
        ? 'rgba(41,98,255,0.25)'
        : 'linear-gradient(135deg, #2962ff 0%, #1648cc 100%)',
      color: '#fff', fontSize: 13, fontWeight: 700,
      cursor: disabled ? 'not-allowed' : 'pointer',
      display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
      letterSpacing: '0.02em',
    }}>
      {running ? <><span style={{ animation: 'spin 1s linear infinite', display: 'inline-block' }}>⟳</span> Running…</> : <><span>▶</span>{label}</>}
    </button>
  )
}

// ── explainer ────────────────────────────────────────────────────────────────

function buildExplainer(algo, p) {
  const vb = ['DQN', 'DDQN'].includes(algo?.toUpperCase())
  const lines = []
  if (vb) {
    const ep = Math.round(p.epsilon * 100)
    lines.push(ep === 0 ? 'ε=0: Pure exploitation — always picks best known action'
      : ep < 30 ? `ε=${p.epsilon.toFixed(2)}: Explores randomly ${ep}% of the time`
      : `ε=${p.epsilon.toFixed(2)}: High randomness — testing worst-case behavior`)
    if (p.action_threshold > 0.05)
      lines.push(`Threshold=${p.action_threshold.toFixed(2)}: Only trades when Q-gap > ${p.action_threshold.toFixed(2)}`)
  } else {
    lines.push(p.temperature < 0.6 ? `T=${p.temperature.toFixed(1)}: Deterministic — strong preference for top action`
      : p.temperature > 1.4 ? `T=${p.temperature.toFixed(1)}: High entropy — nearly uniform action selection`
      : `T=${p.temperature.toFixed(1)}: Balanced exploration vs exploitation`)
    if (p.action_threshold > 0.15)
      lines.push(`Confidence=${p.action_threshold.toFixed(2)}: Trades only when ${Math.round(p.action_threshold * 100)}%+ confident`)
  }
  if (p.risk_aversion < 0.2) lines.push('Risk=low: Ignores volatility — pure return chasing')
  else if (p.risk_aversion > 0.7) lines.push(`Risk=${p.risk_aversion.toFixed(1)}: Heavy volatility penalty — fewer trades`)
  if (p.position_size < 1) lines.push(`Size=${Math.round(p.position_size * 100)}%: Deploys ${Math.round(p.position_size * 100)}% of capital per trade`)
  return lines
}

// ── tabs ─────────────────────────────────────────────────────────────────────

function BacktestTab({
  capital, onCapital, commission, onCommission, speed, onSpeed,
  stopLoss, onStopLoss, takeProfit, onTakeProfit, maxDD, onMaxDD,
  posSize, onPosSize,
  rangeStart, onRangeStart, rangeEnd, onRangeEnd,
  ohlcv, testStartIdx,
  onRun, onRunAll, running,
}) {
  const totalBars = ohlcv.length
  const selDays   = rangeEnd - rangeStart + 1
  const selYears  = selDays > 0 ? (selDays / 252).toFixed(1) : '—'
  const startDate = ohlcv[rangeStart]?.date ?? '—'
  const endDate   = ohlcv[rangeEnd]?.date   ?? '—'
  // Warn if custom range differs from canonical test period
  const isCustom  = rangeStart !== testStartIdx || rangeEnd !== totalBars - 1

  const anyRiskActive = stopLoss > 0 || takeProfit > 0 || maxDD > 0

  return (
    <div style={{ padding: '10px 14px' }}>

      {/* Test date range pickers */}
      {totalBars > 0 && (
        <div style={{ marginBottom: 12 }}>
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6,
          }}>
            <div style={{ fontSize: 10, fontWeight: 600, color: '#4c525e',
              textTransform: 'uppercase', letterSpacing: '0.08em' }}>
              Test Range
            </div>
            {isCustom && (
              <span style={{ fontSize: 9, fontWeight: 600, color: '#f5c842',
                background: 'rgba(245,200,66,0.12)', border: '1px solid rgba(245,200,66,0.3)',
                padding: '1px 5px', borderRadius: 3 }}>
                CUSTOM
              </span>
            )}
          </div>

          {/* Start slider */}
          <div style={{ marginBottom: 8 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
              <span style={{ fontSize: 11, color: '#787b86' }}>Start</span>
              <span style={{ fontSize: 11, fontWeight: 700, color: '#d1d4dc',
                fontVariantNumeric: 'tabular-nums' }}>
                {startDate}
              </span>
            </div>
            <input type="range"
              min={0} max={Math.max(0, rangeEnd - 20)} step={1}
              value={rangeStart}
              disabled={running}
              onChange={e => onRangeStart(Number(e.target.value))}
              style={{ width: '100%', accentColor: '#2962ff',
                cursor: running ? 'not-allowed' : 'pointer' }} />
          </div>

          {/* End slider */}
          <div style={{ marginBottom: 6 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
              <span style={{ fontSize: 11, color: '#787b86' }}>End</span>
              <span style={{ fontSize: 11, fontWeight: 700, color: '#d1d4dc',
                fontVariantNumeric: 'tabular-nums' }}>
                {endDate}
              </span>
            </div>
            <input type="range"
              min={Math.min(totalBars - 1, rangeStart + 20)}
              max={totalBars - 1} step={1}
              value={rangeEnd}
              disabled={running}
              onChange={e => onRangeEnd(Number(e.target.value))}
              style={{ width: '100%', accentColor: '#2962ff',
                cursor: running ? 'not-allowed' : 'pointer' }} />
          </div>

          {/* Summary row */}
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            background: '#2a2e39', borderRadius: 4, padding: '5px 8px',
          }}>
            <span style={{ fontSize: 10, color: '#787b86' }}>
              {selDays} days · {selYears} yrs
            </span>
            {isCustom && (
              <button onClick={() => { onRangeStart(testStartIdx); onRangeEnd(totalBars - 1) }}
                disabled={running}
                style={{
                  fontSize: 9, color: '#5c8df6', background: 'transparent', border: 'none',
                  cursor: 'pointer', padding: 0, fontWeight: 600,
                }}>
                Reset ↺
              </button>
            )}
          </div>
        </div>
      )}

      <Slider label="Capital" value={capital} min={10000} max={500000} step={5000}
        disabled={running} onChange={onCapital} fmt={v => `$${v.toLocaleString()}`} />
      <Slider label="Commission" value={commission} min={0} max={0.005} step={0.0001}
        disabled={running} onChange={onCommission} fmt={v => `${(v * 100).toFixed(2)}%`}
        hint="Per-side transaction cost" />
      <Slider label="Position Size" value={posSize} min={0.1} max={1.0} step={0.1}
        disabled={running} onChange={onPosSize}
        fmt={v => `${Math.round(v * 100)}%`}
        hint="% of capital deployed per trade" />

      <Divider />

      {/* Risk management section */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
        <div style={{ fontSize: 10, fontWeight: 600, color: '#4c525e',
          textTransform: 'uppercase', letterSpacing: '0.08em' }}>
          Risk Management
        </div>
        {anyRiskActive && (
          <span style={{ fontSize: 9, color: '#26a69a', fontWeight: 600 }}>ACTIVE</span>
        )}
      </div>

      <Slider label="Stop Loss" value={stopLoss} min={0} max={30} step={1}
        disabled={running} onChange={onStopLoss}
        fmt={v => v === 0 ? 'OFF' : `-${v}%`}
        hint={stopLoss > 0 ? `Force-sell if position falls ${stopLoss}%` : 'No automatic exit on loss'} />
      <Slider label="Take Profit" value={takeProfit} min={0} max={50} step={5}
        disabled={running} onChange={onTakeProfit}
        fmt={v => v === 0 ? 'OFF' : `+${v}%`}
        hint={takeProfit > 0 ? `Lock in gains when position up ${takeProfit}%` : 'No automatic profit lock'} />
      <Slider label="Max DD Kill" value={maxDD} min={0} max={50} step={5}
        disabled={running} onChange={onMaxDD}
        fmt={v => v === 0 ? 'OFF' : `-${v}%`}
        hint={maxDD > 0 ? `Halt if portfolio draws down ${maxDD}% from peak` : 'No portfolio kill switch'} />

      <Divider />
      <Label>Playback Speed</Label>
      <SpeedPills speed={speed} onSpeed={onSpeed} disabled={running} />
      <div style={{ height: 10 }} />

      <RunButton onClick={onRun} disabled={running} running={running} label="Run Backtest" />
      <div style={{ height: 6 }} />

      {/* Run All Algos button */}
      <button onClick={onRunAll} disabled={running}
        style={{
          width: '100%', padding: '9px 0', borderRadius: 5,
          border: '1px solid #2a2e39',
          background: running ? 'transparent' : '#2a2e39',
          color: running ? '#4c525e' : '#787b86',
          fontSize: 11, fontWeight: 600,
          cursor: running ? 'not-allowed' : 'pointer',
          display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
          transition: 'background 0.1s, color 0.1s',
        }}
        onMouseEnter={e => { if (!running) { e.currentTarget.style.background = '#363c4e'; e.currentTarget.style.color = '#d1d4dc' }}}
        onMouseLeave={e => { e.currentTarget.style.background = running ? 'transparent' : '#2a2e39'; e.currentTarget.style.color = running ? '#4c525e' : '#787b86' }}
      >
        <span>⚖</span> Run All 5 Algos &amp; Compare
      </button>
      <div style={{ fontSize: 9, color: '#4c525e', marginTop: 4, textAlign: 'center', lineHeight: 1.5 }}>
        Applies the same risk rules to all strategies · opens Compare tab
      </div>
    </div>
  )
}

function SimulateTab({ algo, p, setP, onRun, running, capital, onCapital,
  commission, onCommission, speed, onSpeed,
  savedRuns, onSaveRun, onClearRuns, hasCurrentRun }) {

  const vb = ['DQN', 'DDQN'].includes(algo?.toUpperCase())
  const set = (k, v) => setP(prev => ({ ...prev, [k]: v }))
  const explainer = useMemo(() => buildExplainer(algo, p), [algo, p])

  return (
    <div style={{ padding: '10px 14px', overflowY: 'auto' }}>
      <Slider label="Capital" value={capital} min={10000} max={500000} step={5000}
        disabled={running} onChange={onCapital} fmt={v => `$${v.toLocaleString()}`} />
      <Slider label="Commission" value={commission} min={0} max={0.005} step={0.0001}
        disabled={running} onChange={onCommission} fmt={v => `${(v * 100).toFixed(2)}%`} />
      <Divider />

      <Label>{vb ? 'DQN Controls' : `${algo} Controls`}</Label>
      {vb ? (
        <>
          <Slider label="Exploration ε" value={p.epsilon} min={0} max={1} step={0.01}
            disabled={running} onChange={v => set('epsilon', v)} fmt={v => v.toFixed(2)}
            hint="Higher = more random actions" />
          <Slider label="Action Threshold" value={p.action_threshold} min={0} max={1} step={0.01}
            disabled={running} onChange={v => set('action_threshold', v)} fmt={v => v.toFixed(2)}
            hint="Min Q-gap to trade vs hold" />
        </>
      ) : (
        <>
          <Slider label="Temperature" value={p.temperature} min={0.1} max={2} step={0.05}
            disabled={running} onChange={v => set('temperature', v)} fmt={v => v.toFixed(2)}
            hint="Higher = more random (softmax)" />
          <Slider label="Min Confidence" value={p.action_threshold} min={0} max={0.9} step={0.01}
            disabled={running} onChange={v => set('action_threshold', v)} fmt={v => v.toFixed(2)}
            hint="Min probability to trade" />
        </>
      )}
      <Slider label="Risk Aversion" value={p.risk_aversion} min={0} max={1} step={0.05}
        disabled={running} onChange={v => set('risk_aversion', v)} fmt={v => v.toFixed(2)}
        hint="Higher = heavier volatility penalty" />
      <Slider label="Position Size" value={p.position_size} min={0.1} max={1} step={0.05}
        disabled={running} onChange={v => set('position_size', v)}
        fmt={v => `${Math.round(v * 100)}%`} hint="% of capital per trade" />

      {/* Live explainer */}
      <div style={{ background: '#2a2e39', borderRadius: 5, borderLeft: '3px solid #2962ff',
        padding: '8px 10px', marginBottom: 10 }}>
        <div style={{ fontSize: 9, fontWeight: 700, color: '#5c8df6',
          textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 5 }}>
          What This Means
        </div>
        {explainer.map((l, i) => (
          <div key={i} style={{ fontSize: 10, color: '#787b86', lineHeight: 1.6 }}>{l}</div>
        ))}
      </div>

      <Label>Playback Speed</Label>
      <SpeedPills speed={speed} onSpeed={onSpeed} disabled={running} />
      <div style={{ height: 10 }} />
      <RunButton onClick={onRun} disabled={running} running={running} label="Run Simulation" />

      <Divider />
      {/* Compare Runs */}
      <div style={{ marginTop: 4 }}>
        <Label>Compare Runs</Label>
        {savedRuns.length === 0 && (
          <div style={{ fontSize: 10, color: '#4c525e', marginBottom: 8 }}>
            Run then click "+ Save" to overlay up to 3 parameter sets on the portfolio chart.
          </div>
        )}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginBottom: 8 }}>
          {savedRuns.map(run => {
            const hist = run.portfolioHistory
            const ret = ((hist[hist.length - 1] - hist[0]) / hist[0] * 100).toFixed(1)
            const retColor = parseFloat(ret) >= 0 ? '#26a69a' : '#ef5350'
            return (
              <div key={run.id} style={{
                display: 'flex', alignItems: 'center', gap: 8, padding: '6px 8px',
                background: '#2a2e39', borderRadius: 4, borderLeft: `3px solid ${run.color}`,
              }}>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: '#d1d4dc' }}>
                    {run.label} · {run.algo}
                  </div>
                  <div style={{ fontSize: 9, color: '#787b86', marginTop: 1 }}>
                    {['DQN','DDQN'].includes(run.algo?.toUpperCase())
                      ? `ε=${run.params.epsilon?.toFixed(2)}`
                      : `T=${run.params.temperature?.toFixed(2)}`
                    } · RA={run.params.risk_aversion?.toFixed(1)} · {Math.round((run.params.position_size ?? 1) * 100)}%
                  </div>
                </div>
                <span style={{ fontSize: 12, fontWeight: 700, color: retColor,
                  fontVariantNumeric: 'tabular-nums' }}>
                  {parseFloat(ret) >= 0 ? '+' : ''}{ret}%
                </span>
              </div>
            )
          })}
        </div>
        <div style={{ display: 'flex', gap: 6 }}>
          <button onClick={onSaveRun} disabled={!hasCurrentRun || running}
            style={{
              flex: 1, padding: '6px 0', borderRadius: 4,
              border: '1px solid #2962ff',
              background: (!hasCurrentRun || running) ? 'transparent' : 'rgba(41,98,255,0.1)',
              color: (!hasCurrentRun || running) ? '#4c525e' : '#5c8df6',
              fontSize: 11, fontWeight: 600,
              cursor: (!hasCurrentRun || running) ? 'not-allowed' : 'pointer',
            }}>
            + Save Run
          </button>
          {savedRuns.length > 0 && (
            <button onClick={onClearRuns}
              style={{ padding: '6px 10px', borderRadius: 4, border: '1px solid #2a2e39',
                background: 'transparent', color: '#787b86', fontSize: 11, cursor: 'pointer' }}>
              Clear
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

// ── main export ───────────────────────────────────────────────────────────────

export default function LeftPanel({
  algo, onAlgo, capital, onCapital, commission, onCommission,
  speed, onSpeed, mode, onMode, simParams, onSimParams,
  onRun, running, savedRuns, onSaveRun, onClearRuns, hasCurrentRun,
  btStopLoss, onBtStopLoss, btTakeProfit, onBtTakeProfit,
  btMaxDD, onBtMaxDD, btPosSize, onBtPosSize,
  btRangeStart, onBtRangeStart, btRangeEnd, onBtRangeEnd,
  onRunAll, ohlcv = [], testStartIdx = 0,
}) {
  return (
    <div style={{
      width: 220, flexShrink: 0,
      background: '#1e2329', borderRight: '1px solid #2a2e39',
      display: 'flex', flexDirection: 'column', overflow: 'hidden',
    }}>
      {/* Algo pills */}
      <div style={{ padding: '10px 14px 8px', flexShrink: 0 }}>
        <Label>Strategy</Label>
        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
          {ALGOS.map(a => (
            <button key={a.id} onClick={() => !running && onAlgo(a.id)}
              style={{
                padding: '4px 8px', borderRadius: 4, border: 'none',
                background: algo === a.id ? '#2962ff' : '#2a2e39',
                color: algo === a.id ? '#fff' : '#787b86',
                fontSize: 11, fontWeight: algo === a.id ? 700 : 400,
                cursor: running ? 'not-allowed' : 'pointer',
              }}>
              {a.short}
            </button>
          ))}
        </div>
      </div>

      {/* Mode tabs */}
      <div style={{ display: 'flex', borderBottom: '1px solid #2a2e39', flexShrink: 0 }}>
        {['backtest', 'simulate'].map(m => (
          <button key={m} onClick={() => onMode(m)}
            style={{
              flex: 1, padding: '8px 4px', border: 'none', background: 'transparent',
              cursor: 'pointer', fontSize: 10, fontWeight: 700, textTransform: 'uppercase',
              letterSpacing: '0.07em',
              color: mode === m ? '#d1d4dc' : '#4c525e',
              borderBottom: mode === m ? '2px solid #2962ff' : '2px solid transparent',
            }}>
            {m}
          </button>
        ))}
      </div>

      {/* Tab content — scrollable */}
      <div style={{ flex: 1, overflowY: 'auto', minHeight: 0 }}>
        {mode === 'backtest'
          ? <BacktestTab
              capital={capital}         onCapital={onCapital}
              commission={commission}   onCommission={onCommission}
              speed={speed}             onSpeed={onSpeed}
              stopLoss={btStopLoss}     onStopLoss={onBtStopLoss}
              takeProfit={btTakeProfit} onTakeProfit={onBtTakeProfit}
              maxDD={btMaxDD}           onMaxDD={onBtMaxDD}
              posSize={btPosSize}       onPosSize={onBtPosSize}
              rangeStart={btRangeStart} onRangeStart={onBtRangeStart}
              rangeEnd={btRangeEnd}     onRangeEnd={onBtRangeEnd}
              ohlcv={ohlcv}             testStartIdx={testStartIdx}
              onRun={onRun}             onRunAll={onRunAll}
              running={running}
            />
          : <SimulateTab algo={algo} p={simParams} setP={onSimParams}
              onRun={onRun} running={running}
              capital={capital} onCapital={onCapital}
              commission={commission} onCommission={onCommission}
              speed={speed} onSpeed={onSpeed}
              savedRuns={savedRuns} onSaveRun={onSaveRun}
              onClearRuns={onClearRuns} hasCurrentRun={hasCurrentRun} />
        }
      </div>

      <style>{`@keyframes spin { from { transform: rotate(0deg) } to { transform: rotate(360deg) } }`}</style>
    </div>
  )
}
