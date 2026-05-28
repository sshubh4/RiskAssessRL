function Card({ label, value, sub, valueColor, large }) {
  return (
    <div style={{
      background: '#2a2e39',
      borderRadius: '6px',
      padding: '12px',
      marginBottom: '8px',
    }}>
      <div style={{ fontSize: '10px', color: '#787b86', textTransform: 'uppercase',
        letterSpacing: '0.08em', marginBottom: '4px' }}>
        {label}
      </div>
      <div style={{
        fontSize: large ? '20px' : '16px',
        fontWeight: 700,
        color: valueColor || '#d1d4dc',
        fontVariantNumeric: 'tabular-nums',
        lineHeight: 1.2,
      }}>
        {value ?? '—'}
      </div>
      {sub && (
        <div style={{ fontSize: '10px', color: '#787b86', marginTop: '3px' }}>{sub}</div>
      )}
    </div>
  )
}

function StatGrid({ items }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px', marginBottom: '8px' }}>
      {items.map(({ label, value, color }) => (
        <div key={label} style={{
          background: '#2a2e39', borderRadius: '5px', padding: '9px 10px',
        }}>
          <div style={{ fontSize: '9px', color: '#787b86', textTransform: 'uppercase',
            letterSpacing: '0.07em', marginBottom: '4px' }}>{label}</div>
          <div style={{ fontSize: '13px', fontWeight: 700, color: color || '#d1d4dc',
            fontVariantNumeric: 'tabular-nums' }}>
            {value ?? '—'}
          </div>
        </div>
      ))}
    </div>
  )
}

function ProgressBar({ value, max, color }) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0
  return (
    <div style={{
      height: '4px', background: '#2a2e39', borderRadius: '2px', overflow: 'hidden',
    }}>
      <div style={{
        height: '100%', width: `${pct}%`,
        background: color || '#2962ff',
        borderRadius: '2px',
        transition: 'width 0.3s ease',
      }} />
    </div>
  )
}

function VsBahBar({ stratReturn, bahReturn }) {
  const maxAbs = Math.max(Math.abs(stratReturn ?? 0), Math.abs(bahReturn ?? 0), 1)
  const stratPct = ((stratReturn ?? 0) / maxAbs) * 50
  const bahPct   = ((bahReturn  ?? 0) / maxAbs) * 50
  return (
    <div style={{ marginBottom: '10px' }}>
      {[
        { label: 'Strategy',    val: stratReturn, pct: stratPct,
          color: stratReturn >= 0 ? '#26a69a' : '#ef5350' },
        { label: 'Buy & Hold',  val: bahReturn,   pct: bahPct,
          color: bahReturn >= 0 ? '#787b86' : '#ef5350' },
      ].map(({ label, val, pct, color }) => (
        <div key={label} style={{ marginBottom: '6px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '3px' }}>
            <span style={{ fontSize: '10px', color: '#787b86' }}>{label}</span>
            <span style={{ fontSize: '10px', fontWeight: 600, color,
              fontVariantNumeric: 'tabular-nums' }}>
              {val != null ? `${val >= 0 ? '+' : ''}${val.toFixed(2)}%` : '—'}
            </span>
          </div>
          <ProgressBar value={Math.abs(pct)} max={50} color={color} />
        </div>
      ))}
    </div>
  )
}

export default function RightPanel({
  stats, lastPrice, priceChange, priceChangePct,
  portfolioHistory, bahHistory, capital, running, steps,
}) {
  const hasRun    = stats?.portfolioValue != null
  const portValue = hasRun ? stats.portfolioValue : capital
  const pnl       = hasRun ? portValue - capital : 0
  const pnlPct    = hasRun ? stats.totalReturn : 0

  const bahFinalReturn = bahHistory?.length
    ? (bahHistory[bahHistory.length - 1] - (bahHistory[0] ?? capital)) / (bahHistory[0] ?? capital) * 100
    : null

  // Progress of simulation
  const totalExpected = steps?.length || 0
  const progress = running ? 0 : (totalExpected > 0 ? 100 : 0)

  return (
    <div style={{
      width: '200px',
      flexShrink: 0,
      background: '#1e2329',
      borderLeft: '1px solid #2a2e39',
      overflowY: 'auto',
      padding: '12px',
      display: 'flex',
      flexDirection: 'column',
      gap: '2px',
    }}>
      {/* Price card */}
      {lastPrice && (
        <Card
          label="Last Price"
          large
          value={`$${lastPrice.toFixed(2)}`}
          valueColor={priceChange >= 0 ? '#26a69a' : '#ef5350'}
          sub={priceChange != null
            ? `${priceChange >= 0 ? '+' : ''}${priceChange.toFixed(2)} (${priceChangePct >= 0 ? '+' : ''}${priceChangePct.toFixed(2)}%)`
            : null}
        />
      )}

      {/* Portfolio card */}
      <Card
        label="Portfolio Value"
        large
        value={`$${portValue.toLocaleString('en-US', { maximumFractionDigits: 0 })}`}
        valueColor={pnl >= 0 ? '#26a69a' : '#ef5350'}
        sub={hasRun
          ? `${pnl >= 0 ? '+' : ''}$${Math.abs(pnl).toLocaleString('en-US', { maximumFractionDigits: 0 })} (${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(2)}%)`
          : `Initial: $${capital.toLocaleString()}`}
      />

      {/* Stats 2×2 grid */}
      <StatGrid items={[
        {
          label: 'Sharpe',
          value: hasRun ? stats.sharpe.toFixed(2) : '—',
          color: hasRun
            ? (stats.sharpe >= 1 ? '#26a69a' : stats.sharpe >= 0 ? '#d1d4dc' : '#ef5350')
            : '#787b86',
        },
        {
          label: 'Max DD',
          value: hasRun ? `${(stats.maxDD * 100).toFixed(1)}%` : '—',
          color: '#ef5350',
        },
        {
          label: 'Win Rate',
          value: hasRun && stats.trades > 0 ? `${stats.winRate.toFixed(0)}%` : '—',
          color: hasRun && stats.winRate >= 50 ? '#26a69a' : '#ef5350',
        },
        {
          label: 'Trades',
          value: hasRun ? stats.trades : '—',
          color: '#d1d4dc',
        },
      ]} />

      {/* vs B&H comparison bars */}
      <div style={{ background: '#2a2e39', borderRadius: '6px', padding: '10px', marginBottom: '8px' }}>
        <div style={{ fontSize: '10px', color: '#787b86', textTransform: 'uppercase',
          letterSpacing: '0.08em', marginBottom: '8px' }}>
          vs Buy & Hold
        </div>
        <VsBahBar
          stratReturn={hasRun ? stats.totalReturn : null}
          bahReturn={bahFinalReturn}
        />
        {hasRun && stats.vsBah != null && (
          <div style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            borderTop: '1px solid #363c4e', paddingTop: '8px', marginTop: '4px',
          }}>
            <span style={{ fontSize: '10px', color: '#787b86' }}>Alpha</span>
            <span style={{
              fontSize: '12px', fontWeight: 700,
              color: stats.vsBah >= 0 ? '#26a69a' : '#ef5350',
              fontVariantNumeric: 'tabular-nums',
            }}>
              {stats.vsBah >= 0 ? '+' : ''}{stats.vsBah.toFixed(2)}%
            </span>
          </div>
        )}
      </div>

      {/* Simulation progress */}
      {(running || totalExpected > 0) && (
        <div style={{ background: '#2a2e39', borderRadius: '6px', padding: '10px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
            <span style={{ fontSize: '10px', color: '#787b86', textTransform: 'uppercase',
              letterSpacing: '0.08em' }}>
              {running ? 'Simulating' : 'Complete'}
            </span>
            <span style={{ fontSize: '10px', color: '#787b86' }}>
              {totalExpected} steps
            </span>
          </div>
          <ProgressBar
            value={running ? 50 : 100}
            max={100}
            color={running ? '#2962ff' : '#26a69a'}
          />
        </div>
      )}

      <div style={{ flex: 1 }} />
    </div>
  )
}
