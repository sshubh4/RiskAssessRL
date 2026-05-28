import { useRef, useEffect, useState, useMemo } from 'react'

const PAD = { top: 12, right: 14, bottom: 28, left: 62 }
const VOL_FRAC = 0.18

function niceTicks(min, max, count = 5) {
  const range = max - min
  if (range === 0) return [min]
  const raw = range / (count - 1)
  const mag = Math.pow(10, Math.floor(Math.log10(raw)))
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(m => m >= raw) ?? raw
  const lo = Math.ceil(min / step) * step
  const ticks = []
  for (let t = lo; t <= max + step * 0.01; t += step)
    ticks.push(parseFloat(t.toFixed(10)))
  return ticks
}

export default function TradingChart({ ohlcv, steps, ticker }) {
  const wrapRef = useRef(null)
  const [dim, setDim] = useState({ w: 600, h: 300 })
  const [hov, setHov] = useState(null)

  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => {
      const { width, height } = e.contentRect
      if (width > 10 && height > 10) setDim({ w: width, h: height })
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const c = useMemo(() => {
    const { w, h } = dim
    if (!ohlcv?.length) return null
    const n = ohlcv.length
    const cW = w - PAD.left - PAD.right
    const cH = h - PAD.top - PAD.bottom
    const pH = cH * (1 - VOL_FRAC)  // price area height
    const vH = cH * VOL_FRAC         // volume area height

    const priceMin = Math.min(...ohlcv.map(d => d.low))
    const priceMax = Math.max(...ohlcv.map(d => d.high))
    const pad = (priceMax - priceMin) * 0.06
    const yLo = priceMin - pad, yHi = priceMax + pad

    const xStep = cW / n
    const cw = Math.max(1, xStep * 0.62)
    const xOf = i => PAD.left + (i + 0.5) * xStep
    const yOf = p => PAD.top + pH * (1 - (p - yLo) / (yHi - yLo))
    const volMax = Math.max(...ohlcv.map(d => d.volume))
    const vBase = PAD.top + cH
    const yVol = v => PAD.top + pH + vH * (1 - (v / volMax) * 0.92)

    const pTicks = niceTicks(yLo, yHi, 5)
    const tCount = Math.min(6, n)
    const tIdxs = Array.from({ length: tCount }, (_, i) => Math.floor(i * (n - 1) / (tCount - 1)))

    const markers = {}
    for (const s of (steps ?? [])) {
      if (s.action === 0 || s.action === 1) markers[s.step] = s.action
    }

    return { w, h, n, cW, cH, pH, vH, xStep, cw, xOf, yOf, yVol, vBase, pTicks, tIdxs, markers }
  }, [ohlcv, steps, dim])

  const { w, h } = dim
  if (!ohlcv?.length || !c) {
    return (
      <div ref={wrapRef} style={{ width: '100%', height: '100%', display: 'flex',
        alignItems: 'center', justifyContent: 'center', background: '#131722' }}>
        <span style={{ color: '#4c525e', fontSize: '12px' }}>Loading chart…</span>
      </div>
    )
  }

  const { xOf, yOf, yVol, vBase, pTicks, tIdxs, markers, cw, xStep, pH } = c
  const hovBar = hov != null ? ohlcv[hov] : null

  return (
    <div ref={wrapRef} style={{ width: '100%', height: '100%', position: 'relative',
      background: '#131722', userSelect: 'none' }}>

      {/* OHLC hover tooltip */}
      {hovBar && (
        <div style={{
          position: 'absolute', top: 8, left: PAD.left + 4, zIndex: 20,
          display: 'flex', gap: 10, alignItems: 'center',
          fontSize: 11, pointerEvents: 'none', fontVariantNumeric: 'tabular-nums',
        }}>
          <span style={{ color: '#4c525e' }}>{hovBar.date}</span>
          {[['O', hovBar.open], ['H', hovBar.high], ['L', hovBar.low], ['C', hovBar.close]].map(([l, v]) => (
            <span key={l} style={{ color: l === 'C' ? (hovBar.close >= hovBar.open ? '#26a69a' : '#ef5350') : '#d1d4dc' }}>
              {l} <strong>{v.toFixed(2)}</strong>
            </span>
          ))}
        </div>
      )}

      <svg width={w} height={h} style={{ display: 'block' }}
        onMouseLeave={() => setHov(null)}>

        {/* Y grid + price labels */}
        {pTicks.map(p => {
          const y = yOf(p)
          if (y < PAD.top - 2 || y > PAD.top + pH + 2) return null
          return (
            <g key={p}>
              <line x1={PAD.left} y1={y} x2={w - PAD.right} y2={y}
                stroke="#1e2329" strokeWidth={1} />
              <text x={PAD.left - 6} y={y + 3.5} textAnchor="end"
                fill="#4c525e" fontSize={10}>
                {p >= 1000 ? `${(p / 1000).toFixed(1)}k` : p.toFixed(0)}
              </text>
            </g>
          )
        })}

        {/* Volume separator */}
        <line x1={PAD.left} y1={PAD.top + c.pH} x2={w - PAD.right} y2={PAD.top + c.pH}
          stroke="#1e2329" strokeWidth={1} />

        {/* Candles + volume + markers */}
        {ohlcv.map((bar, i) => {
          const x = xOf(i), up = bar.close >= bar.open
          const col = up ? '#26a69a' : '#ef5350'
          const yO = yOf(bar.open), yC = yOf(bar.close)
          const yH = yOf(bar.high), yL = yOf(bar.low)
          const bodyY = Math.min(yO, yC), bodyH = Math.max(1, Math.abs(yC - yO))
          const vy = yVol(bar.volume)
          const act = markers[i]

          return (
            <g key={i} onMouseEnter={() => setHov(i)} style={{ cursor: 'crosshair' }}>
              {/* hover band */}
              {hov === i && (
                <rect x={x - xStep / 2} y={PAD.top} width={xStep} height={c.cH}
                  fill="rgba(255,255,255,0.025)" />
              )}
              {/* wick */}
              <line x1={x} y1={yH} x2={x} y2={yL} stroke={col} strokeWidth={1} />
              {/* body */}
              <rect x={x - cw / 2} y={bodyY} width={cw} height={bodyH} fill={col} />
              {/* volume */}
              <rect x={x - cw / 2} y={vy} width={cw} height={vBase - vy}
                fill={col} opacity={0.35} />
              {/* buy marker ▲ */}
              {act === 0 && (
                <polygon points={`${x},${yL + 12} ${x - 5},${yL + 20} ${x + 5},${yL + 20}`}
                  fill="#2962ff" />
              )}
              {/* sell marker ▼ */}
              {act === 1 && (
                <polygon points={`${x},${yH - 12} ${x - 5},${yH - 20} ${x + 5},${yH - 20}`}
                  fill="#ef5350" />
              )}
            </g>
          )
        })}

        {/* X axis date labels */}
        {tIdxs.map(i => {
          if (i >= ohlcv.length) return null
          const d = new Date(ohlcv[i].date + 'T00:00:00')
          const label = d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
          return (
            <text key={i} x={xOf(i)} y={h - 8} textAnchor="middle" fill="#4c525e" fontSize={10}>
              {label}
            </text>
          )
        })}

        {/* Chart title */}
        {!hovBar && (
          <>
            <text x={PAD.left} y={PAD.top + 10} fill="#d1d4dc" fontSize={12} fontWeight="600">{ticker}</text>
            <text x={PAD.left + 38} y={PAD.top + 10} fill="#4c525e" fontSize={10}>Daily · Test period</text>
          </>
        )}

        {/* Legend */}
        <polygon points={`${w - 82},${PAD.top + 6} ${w - 87},${PAD.top + 13} ${w - 77},${PAD.top + 13}`}
          fill="#2962ff" />
        <text x={w - 74} y={PAD.top + 13} fill="#4c525e" fontSize={10}>Buy</text>
        <polygon points={`${w - 40},${PAD.top + 13} ${w - 45},${PAD.top + 6} ${w - 35},${PAD.top + 6}`}
          fill="#ef5350" />
        <text x={w - 32} y={PAD.top + 13} fill="#4c525e" fontSize={10}>Sell</text>
      </svg>
    </div>
  )
}
