import { useState, useEffect } from 'react'

const TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'SPY']

function Clock() {
  const [time, setTime] = useState('')
  useEffect(() => {
    const fmt = () => {
      const now = new Date()
      return now.toLocaleTimeString('en-US', {
        hour: '2-digit', minute: '2-digit', second: '2-digit',
        timeZone: 'America/New_York', hour12: false,
      }) + ' ET'
    }
    setTime(fmt())
    const id = setInterval(() => setTime(fmt()), 1000)
    return () => clearInterval(id)
  }, [])
  return <span style={{ color: '#787b86', fontSize: '12px', fontVariantNumeric: 'tabular-nums' }}>{time}</span>
}

export default function Topbar({
  ticker, onTicker,
  lastPrice, priceChange, priceChangePct,
  dataFreshness, modelsLoaded, running,
}) {
  const up = priceChange >= 0

  return (
    <div style={{
      height: '48px',
      background: '#1e2329',
      borderBottom: '1px solid #2a2e39',
      display: 'flex',
      alignItems: 'center',
      padding: '0 16px',
      gap: '8px',
      flexShrink: 0,
      userSelect: 'none',
    }}>
      {/* Logo / Brand */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginRight: '12px' }}>
        <div style={{
          width: '28px', height: '28px',
          background: 'linear-gradient(135deg, #2962ff 0%, #1a3fa3 100%)',
          borderRadius: '6px',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: '13px', fontWeight: 700, color: '#fff',
        }}>R</div>
        <span style={{ fontWeight: 700, fontSize: '14px', color: '#d1d4dc', letterSpacing: '-0.02em' }}>
          RiskAssessRL
        </span>
      </div>

      {/* Divider */}
      <div style={{ width: '1px', height: '24px', background: '#2a2e39', margin: '0 4px' }} />

      {/* Ticker pills */}
      <div style={{ display: 'flex', gap: '4px' }}>
        {TICKERS.map(t => (
          <button
            key={t}
            onClick={() => onTicker(t)}
            style={{
              padding: '4px 10px',
              borderRadius: '4px',
              border: ticker === t ? '1px solid #2962ff' : '1px solid transparent',
              background: ticker === t ? 'rgba(41,98,255,0.15)' : 'transparent',
              color: ticker === t ? '#5c8df6' : '#787b86',
              fontWeight: ticker === t ? 600 : 400,
              fontSize: '12px',
              cursor: 'pointer',
              transition: 'all 0.12s',
            }}
          >
            {t}
          </button>
        ))}
      </div>

      {/* Divider */}
      <div style={{ width: '1px', height: '24px', background: '#2a2e39', margin: '0 4px' }} />

      {/* Price display */}
      {lastPrice && (
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '6px' }}>
          <span style={{ fontSize: '16px', fontWeight: 600, color: '#d1d4dc', fontVariantNumeric: 'tabular-nums' }}>
            ${lastPrice.toFixed(2)}
          </span>
          {priceChange !== null && (
            <span style={{
              fontSize: '12px',
              color: up ? '#26a69a' : '#ef5350',
              fontVariantNumeric: 'tabular-nums',
            }}>
              {up ? '+' : ''}{priceChange.toFixed(2)} ({up ? '+' : ''}{priceChangePct.toFixed(2)}%)
            </span>
          )}
        </div>
      )}

      {/* Spacer */}
      <div style={{ flex: 1 }} />

      {/* Status badges */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        {running && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <div style={{
              width: '6px', height: '6px', borderRadius: '50%',
              background: '#2962ff',
              animation: 'pulse 1s infinite',
            }} />
            <span style={{ fontSize: '11px', color: '#5c8df6', fontWeight: 500 }}>SIMULATING</span>
          </div>
        )}

        <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
          <div style={{
            width: '6px', height: '6px', borderRadius: '50%',
            background: modelsLoaded >= 4 ? '#26a69a' : modelsLoaded > 0 ? '#f6c90e' : '#787b86',
          }} />
          <span style={{ fontSize: '11px', color: '#787b86' }}>
            {modelsLoaded}/4 models
          </span>
        </div>

        {dataFreshness && (
          <span style={{ fontSize: '11px', color: '#787b86' }}>
            Data: {dataFreshness}
          </span>
        )}

        <Clock />
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
        }
      `}</style>
    </div>
  )
}
