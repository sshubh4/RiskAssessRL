function Stat({ label, value, color, border }) {
  return (
    <div style={{
      flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center',
      justifyContent: 'center', padding: '0 8px',
      borderRight: border ? '1px solid #2a2e39' : 'none',
    }}>
      <div style={{
        fontSize: 15, fontWeight: 700, color: color || '#d1d4dc',
        fontVariantNumeric: 'tabular-nums', lineHeight: 1.2,
      }}>
        {value ?? '—'}
      </div>
      <div style={{ fontSize: 9, color: '#4c525e', textTransform: 'uppercase',
        letterSpacing: '0.07em', marginTop: 2 }}>
        {label}
      </div>
    </div>
  )
}

export default function StatStrip({ stats, algo }) {
  if (!stats?.portfolioValue) return null

  const { totalReturn, sharpe, maxDD, winRate, trades, vsBah } = stats
  const retColor  = totalReturn >= 0 ? '#26a69a' : '#ef5350'
  const sharpeColor = sharpe >= 1 ? '#26a69a' : sharpe >= 0 ? '#d1d4dc' : '#ef5350'
  const bahColor  = vsBah == null ? '#787b86' : vsBah >= 0 ? '#26a69a' : '#ef5350'

  return (
    <div style={{
      height: 44, flexShrink: 0,
      display: 'flex', alignItems: 'stretch',
      background: '#1e2329',
      borderTop: '1px solid #2a2e39',
      borderBottom: '1px solid #2a2e39',
    }}>
      <Stat label="Total Return" border
        value={`${totalReturn >= 0 ? '+' : ''}${totalReturn.toFixed(2)}%`}
        color={retColor} />
      <Stat label="Sharpe Ratio" border
        value={sharpe.toFixed(3)}
        color={sharpeColor} />
      <Stat label="Max Drawdown" border
        value={`${(maxDD * 100).toFixed(2)}%`}
        color="#ef5350" />
      <Stat label="Win Rate" border
        value={trades > 0 ? `${winRate.toFixed(0)}%` : '—'}
        color={winRate >= 50 ? '#26a69a' : '#ef5350'} />
      <Stat label="Trades"  border
        value={trades}
        color="#d1d4dc" />
      <Stat label="vs S&P 500"
        value={vsBah != null ? `${vsBah >= 0 ? '+' : ''}${vsBah.toFixed(2)}%` : '—'}
        color={bahColor} />
    </div>
  )
}
