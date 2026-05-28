const COLS = [
  { key: 'algo',                 label: 'Algorithm',   width: '110px' },
  { key: 'final_portfolio_value',label: 'Final Value', width: '100px',
    fmt: v => v != null ? `$${v.toLocaleString('en-US', { maximumFractionDigits: 0 })}` : '—' },
  { key: 'total_return_pct',     label: 'Return',      width: '80px',
    fmt: v => v != null ? `${v >= 0 ? '+' : ''}${v.toFixed(2)}%` : '—' },
  { key: 'sharpe_ratio',         label: 'Sharpe',      width: '72px',
    fmt: v => v != null ? v.toFixed(3) : '—' },
  { key: 'max_drawdown',         label: 'Max DD',      width: '72px',
    fmt: v => v != null ? `${(v * 100).toFixed(2)}%` : '—' },
  { key: 'n_steps',              label: 'Steps',       width: '60px' },
]

function getBestValues(rows) {
  const best = {}
  for (const col of COLS) {
    if (col.key === 'algo' || col.key === 'n_steps') continue
    const vals = rows.map(r => r[col.key]).filter(v => v != null && !isNaN(v))
    if (!vals.length) continue
    if (col.key === 'max_drawdown') {
      best[col.key] = Math.max(...vals)   // closest to 0 is best
    } else {
      best[col.key] = Math.max(...vals)
    }
  }
  return best
}

function cellColor(col, value, isBest) {
  if (col.key === 'algo') return '#d1d4dc'
  if (col.key === 'total_return_pct') {
    if (isBest) return '#26a69a'
    return value >= 0 ? '#26a69a' : '#ef5350'
  }
  if (col.key === 'sharpe_ratio') {
    if (isBest) return '#26a69a'
    return value >= 1 ? '#26a69a' : value >= 0 ? '#d1d4dc' : '#ef5350'
  }
  if (col.key === 'max_drawdown') {
    if (isBest) return '#26a69a'
    return value < -0.15 ? '#ef5350' : '#d1d4dc'
  }
  if (col.key === 'final_portfolio_value') {
    return isBest ? '#26a69a' : '#d1d4dc'
  }
  return '#787b86'
}

export default function ComparisonTable({ results, bahData }) {
  if (!results?.length) {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        height: '100%', color: '#4c525e', fontSize: '12px',
      }}>
        Run a backtest or evaluate agents to see comparison data
      </div>
    )
  }

  const allRows = bahData ? [...results, bahData] : results
  const best = getBestValues(allRows)

  return (
    <div style={{ overflowX: 'auto', width: '100%' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11px' }}>
        <thead>
          <tr style={{ background: '#1e2329' }}>
            {COLS.map(col => (
              <th key={col.key} style={{
                padding: '6px 12px',
                textAlign: 'left',
                fontSize: '9px',
                fontWeight: 600,
                color: '#4c525e',
                textTransform: 'uppercase',
                letterSpacing: '0.08em',
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
          {allRows.map((row, i) => {
            const isBah = row.algo === 'Buy&Hold'
            return (
              <tr
                key={i}
                style={{
                  borderBottom: '1px solid #1e2329',
                  background: isBah ? 'rgba(120,123,134,0.05)' : 'transparent',
                  transition: 'background 0.1s',
                  cursor: 'pointer',
                }}
                onMouseEnter={e => e.currentTarget.style.background = '#2a2e39'}
                onMouseLeave={e => e.currentTarget.style.background = isBah ? 'rgba(120,123,134,0.05)' : 'transparent'}
              >
                {COLS.map(col => {
                  const raw = row[col.key]
                  const formatted = col.fmt ? col.fmt(raw) : (raw ?? '—')
                  const isBest = best[col.key] != null && raw === best[col.key]
                  const color = cellColor(col, raw, isBest)

                  return (
                    <td key={col.key} style={{
                      padding: '7px 12px',
                      color,
                      fontWeight: col.key === 'algo' ? 600 : (isBest ? 700 : 400),
                      fontVariantNumeric: 'tabular-nums',
                      whiteSpace: 'nowrap',
                      position: 'relative',
                    }}>
                      {isBest && col.key !== 'algo' && col.key !== 'n_steps' && (
                        <span style={{
                          position: 'absolute',
                          left: '4px', top: '50%', transform: 'translateY(-50%)',
                          width: '3px', height: '14px',
                          background: '#2962ff',
                          borderRadius: '2px',
                        }} />
                      )}
                      {formatted}
                    </td>
                  )
                })}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
