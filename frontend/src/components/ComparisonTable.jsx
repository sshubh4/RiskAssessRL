const COLS = [
  { key: 'algo',                  label: 'Algorithm',   width: '110px' },
  { key: 'final_portfolio_value', label: 'Final Value', width: '110px',
    fmt: v => v != null ? `$${v.toLocaleString('en-US', { maximumFractionDigits: 0 })}` : '—' },
  { key: 'total_return_pct',      label: 'Return',      width: '85px',
    fmt: v => v != null ? `${v >= 0 ? '+' : ''}${v.toFixed(2)}%` : '—' },
  { key: 'sharpe_ratio',          label: 'Sharpe',      width: '76px',
    fmt: v => v != null ? v.toFixed(3) : '—' },
  { key: 'max_drawdown',          label: 'Max DD',      width: '76px',
    fmt: v => v != null ? `${(v * 100).toFixed(2)}%` : '—' },
  { key: 'win_rate',              label: 'Win Rate',    width: '76px',
    fmt: v => v != null ? `${v.toFixed(1)}%` : '—' },
  { key: 'n_trades',              label: 'Trades',      width: '64px' },
  { key: 'vs_spy',                label: 'vs S&P 500',  width: '90px',   computed: true,
    fmt: v => v != null ? `${v >= 0 ? '+' : ''}${v.toFixed(2)}%` : '—' },
  { key: 'vs_aapl',               label: 'vs AAPL B&H', width: '90px',   computed: true,
    fmt: v => v != null ? `${v >= 0 ? '+' : ''}${v.toFixed(2)}%` : '—' },
]

function getBestValues(rows) {
  const best = {}
  for (const col of COLS) {
    if (col.key === 'algo' || col.key === 'n_trades' || col.computed) continue
    const vals = rows.map(r => r[col.key]).filter(v => v != null && !isNaN(v))
    if (!vals.length) continue
    best[col.key] = col.key === 'max_drawdown'
      ? Math.max(...vals)   // closest to 0 = best
      : Math.max(...vals)
  }
  return best
}

function cellColor(col, value, isBest) {
  if (col.key === 'algo') return '#d1d4dc'
  if (col.key === 'total_return_pct') return isBest ? '#26a69a' : (value >= 0 ? '#26a69a' : '#ef5350')
  if (col.key === 'sharpe_ratio')     return isBest ? '#26a69a' : (value >= 1 ? '#26a69a' : value >= 0 ? '#d1d4dc' : '#ef5350')
  if (col.key === 'max_drawdown')     return isBest ? '#26a69a' : (value < -0.15 ? '#ef5350' : '#d1d4dc')
  if (col.key === 'final_portfolio_value') return isBest ? '#26a69a' : '#d1d4dc'
  if (col.key === 'win_rate')         return value == null ? '#4c525e' : value >= 50 ? '#26a69a' : '#ef5350'
  if (col.key === 'vs_spy' || col.key === 'vs_aapl') return value == null ? '#4c525e' : value >= 0 ? '#26a69a' : '#ef5350'
  return '#787b86'
}

// ── main export ────────────────────────────────────────────────────────────

export default function ComparisonTable({ results, bahData, spyBenchmark, aaplBenchmark }) {
  if (!results?.length) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center',
        justifyContent: 'center', height: '100%', gap: 8 }}>
        <span style={{ fontSize: 28, opacity: 0.2 }}>⚖</span>
        <span style={{ color: '#4c525e', fontSize: '12px' }}>
          Run a backtest to populate algorithm comparison
        </span>
      </div>
    )
  }

  const spyReturn  = spyBenchmark?.total_return_pct  ?? null
  const aaplReturn = aaplBenchmark?.total_return_pct ?? bahData?.total_return_pct ?? null

  // Enrich agent rows with computed delta columns
  const agentRows = results.map(r => ({
    ...r,
    vs_spy:  spyReturn  != null ? round2(r.total_return_pct - spyReturn)  : null,
    vs_aapl: aaplReturn != null ? round2(r.total_return_pct - aaplReturn) : null,
  }))

  const best = getBestValues(agentRows)

  // Benchmark rows — shown below agents, styled differently
  const benchmarkRows = [
    spyBenchmark  ? { ...spyBenchmark,  _benchmark: true, vs_spy: null, vs_aapl: null } : null,
    aaplBenchmark ? { ...aaplBenchmark, _benchmark: true, vs_spy: null, vs_aapl: null }
      : bahData   ? { ...bahData, algo: 'Buy&Hold', name: 'Buy & Hold', _benchmark: true, vs_spy: null, vs_aapl: null }
      : null,
  ].filter(Boolean)

  return (
    <div style={{ width: '100%' }}>

      {/* Section header */}
      <div style={{ marginBottom: 14 }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: '#d1d4dc', letterSpacing: '0.04em', marginBottom: 3 }}>
          Algorithm Performance Comparison
        </div>
        <div style={{ fontSize: 10, color: '#4c525e' }}>
          All agents evaluated on the same test period · Blue bar = best value in column
        </div>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px' }}>
          <thead>
            <tr style={{ background: '#1a1e27' }}>
              {COLS.map(col => (
                <th key={col.key} style={{
                  padding: '10px 14px',
                  textAlign: 'left',
                  fontSize: '9px',
                  fontWeight: 700,
                  color: (col.key === 'vs_spy' || col.key === 'vs_aapl') ? '#5c8df6' : '#4c525e',
                  textTransform: 'uppercase',
                  letterSpacing: '0.09em',
                  borderBottom: '1px solid #2a2e39',
                  width: col.width,
                  whiteSpace: 'nowrap',
                }}>
                  {col.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {/* ── Agent rows ─────────────────────────────────────────────────── */}
            {agentRows.map((row, i) => (
              <tr key={i}
                style={{ borderBottom: '1px solid #1e2329', background: 'transparent', transition: 'background 0.12s' }}
                onMouseEnter={e => e.currentTarget.style.background = '#2a2e39'}
                onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
              >
                {COLS.map(col => {
                  const raw       = row[col.key]
                  const formatted = col.fmt ? col.fmt(raw) : (raw ?? '—')
                  const isBest    = !col.computed && best[col.key] != null && raw === best[col.key]
                  const color     = cellColor(col, raw, isBest)
                  return (
                    <td key={col.key} style={{
                      padding: '11px 14px',
                      color,
                      fontWeight: col.key === 'algo' ? 600 : (isBest ? 700 : 400),
                      fontVariantNumeric: 'tabular-nums',
                      whiteSpace: 'nowrap',
                      position: 'relative',
                    }}>
                      {isBest && col.key !== 'algo' && col.key !== 'n_trades' && (
                        <span style={{
                          position: 'absolute', left: 4, top: '50%', transform: 'translateY(-50%)',
                          width: 3, height: 16, background: '#2962ff', borderRadius: 2,
                        }} />
                      )}
                      {formatted}
                    </td>
                  )
                })}
              </tr>
            ))}

            {/* ── Benchmark section divider ───────────────────────────────────── */}
            {benchmarkRows.length > 0 && (
              <tr>
                <td colSpan={COLS.length} style={{
                  padding: '8px 14px 4px',
                  fontSize: 9, fontWeight: 700, color: '#4c525e',
                  textTransform: 'uppercase', letterSpacing: '0.1em',
                  borderTop: '1px solid #2a2e39',
                  borderBottom: 'none',
                }}>
                  Benchmarks
                </td>
              </tr>
            )}

            {/* ── Benchmark rows ─────────────────────────────────────────────── */}
            {benchmarkRows.map((row, i) => (
              <tr key={`bm-${i}`}
                style={{ background: 'rgba(120,123,134,0.06)', borderBottom: '1px solid #1e2329' }}>
                {COLS.map(col => {
                  const raw       = row[col.key]
                  const formatted = col.fmt ? col.fmt(raw) : (raw ?? '—')
                  const color     = col.key === 'algo'
                    ? (row.color ?? '#787b86')
                    : cellColor(col, raw, false)
                  return (
                    <td key={col.key} style={{
                      padding: '9px 14px',
                      color,
                      fontStyle: 'italic',
                      fontWeight: col.key === 'algo' ? 600 : 400,
                      fontVariantNumeric: 'tabular-nums',
                      whiteSpace: 'nowrap',
                    }}>
                      {col.key === 'algo' ? (row.name ?? row.algo) : formatted}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* ── Info note ───────────────────────────────────────────────────────── */}
      <div style={{
        marginTop: 16, padding: '10px 14px',
        background: 'rgba(41,98,255,0.06)',
        border: '1px solid rgba(41,98,255,0.18)',
        borderRadius: 6, fontSize: 10, color: '#787b86', lineHeight: 1.7,
      }}>
        <span style={{ color: '#5c8df6', fontWeight: 600 }}>ℹ</span>
        {'  '}All strategies start with the same capital on the same date as the RL agents.
        {' '}<span style={{ color: '#d1d4dc' }}>S&P 500 (SPY) is the primary benchmark</span>
        {' '}— beating it on a risk-adjusted basis (Sharpe ratio) is the real goal.
        {' '}Beating AAPL buy-and-hold during a sustained bull market is intentionally difficult —
        {' '}RL agents optimise for Sharpe, not raw return.
      </div>
    </div>
  )
}

function round2(v) { return Math.round(v * 100) / 100 }
