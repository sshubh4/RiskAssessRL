import { useRef, useEffect, useState, useMemo, useCallback } from 'react'

const PAD = { top: 28, right: 14, bottom: 28, left: 62 }
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

const MIN_BARS     = 10
const DEFAULT_BARS = 252   // ~1 trading year shown by default

// Timeframe definitions: label → approximate trading days
const TIMEFRAMES = [
  { label: '1M',  bars: 21  },
  { label: '3M',  bars: 63  },
  { label: '6M',  bars: 126 },
  { label: '1Y',  bars: 252 },
  { label: 'ALL', bars: null },  // null = full dataset
]

export default function TradingChart({ ohlcv, steps, ticker, testStartIdx = 0, regime = null }) {
  const wrapRef = useRef(null)
  const [dim, setDim]   = useState({ w: 600, h: 300 })
  const [hov, setHov]   = useState(null)

  // ── Viewport state ───────────────────────────────────────────────
  // zoom      = number of bars visible in the viewport
  // panOff    = bars from the right edge that are hidden (0 = latest bar at right)
  // activeTf  = currently selected timeframe label (for button highlight)
  const [zoom, setZoom]         = useState(DEFAULT_BARS)
  const [panOff, setPanOff]     = useState(0)
  const [activeTf, setActiveTf] = useState('1Y')

  const dragRef = useRef(null)   // { x: clientX, panStart: number }

  // Reset viewport whenever the dataset changes
  useEffect(() => {
    const n = ohlcv?.length ?? 0
    setZoom(Math.min(DEFAULT_BARS, n || DEFAULT_BARS))
    setPanOff(0)
    setHov(null)
    setActiveTf('1Y')
  }, [ohlcv?.length])

  // ── Timeframe click handler ──────────────────────────────────────
  const applyTimeframe = useCallback((tf) => {
    const n    = ohlcv?.length ?? 0
    const bars = tf.bars ?? n            // ALL → show everything
    setZoom(Math.min(Math.max(bars, MIN_BARS), n || DEFAULT_BARS))
    setPanOff(0)                         // always snap to latest
    setActiveTf(tf.label)
    setHov(null)
  }, [ohlcv?.length])

  // ── Derived slice ────────────────────────────────────────────────
  const totalBars  = ohlcv?.length ?? 0
  const barsToShow = Math.min(Math.max(zoom, MIN_BARS), totalBars)
  const clampedPan = Math.min(panOff, Math.max(0, totalBars - barsToShow))
  const sliceEnd   = totalBars - clampedPan
  const sliceStart = Math.max(0, sliceEnd - barsToShow)
  const visible    = useMemo(
    () => (ohlcv ?? []).slice(sliceStart, sliceEnd),
    [ohlcv, sliceStart, sliceEnd]
  )

  // ── Resize observer ──────────────────────────────────────────────
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

  // ── Wheel zoom — non-passive to allow preventDefault ─────────────
  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const handler = (e) => {
      e.preventDefault()
      const factor = e.deltaY > 0 ? 1.15 : 1 / 1.15
      setZoom(prev =>
        Math.round(Math.min(Math.max(prev * factor, MIN_BARS), totalBars || DEFAULT_BARS))
      )
      setActiveTf(null)   // manual zoom clears the timeframe highlight
    }
    el.addEventListener('wheel', handler, { passive: false })
    return () => el.removeEventListener('wheel', handler)
  }, [totalBars])

  // ── Drag to pan ──────────────────────────────────────────────────
  const onMouseDown = useCallback((e) => {
    if (e.button !== 0) return
    dragRef.current = { x: e.clientX, panStart: clampedPan }
  }, [clampedPan])

  const onMouseMove = useCallback((e) => {
    if (!dragRef.current) return
    const cW = dim.w - PAD.left - PAD.right
    const barsPerPx = barsToShow / cW
    const dx = e.clientX - dragRef.current.x
    const dBars  = Math.round(-dx * barsPerPx)
    const newPan = Math.max(0, Math.min(totalBars - barsToShow, dragRef.current.panStart + dBars))
    if (dBars !== 0) setActiveTf(null)   // manual pan clears highlight
    setPanOff(newPan)
  }, [barsToShow, dim.w, totalBars])

  const onMouseUp = useCallback(() => { dragRef.current = null }, [])

  // ── Layout ───────────────────────────────────────────────────────
  const c = useMemo(() => {
    const { w, h } = dim
    if (!visible.length) return null
    const n  = visible.length
    const cW = w - PAD.left - PAD.right
    const cH = h - PAD.top - PAD.bottom
    const pH = cH * (1 - VOL_FRAC)
    const vH = cH * VOL_FRAC

    const priceMin = Math.min(...visible.map(d => d.low))
    const priceMax = Math.max(...visible.map(d => d.high))
    const pad  = (priceMax - priceMin) * 0.06
    const yLo  = priceMin - pad
    const yHi  = priceMax + pad

    const xStep = cW / n
    const cw    = Math.max(1, xStep * 0.62)
    const xOf   = i => PAD.left + (i + 0.5) * xStep
    const yOf   = p => PAD.top + pH * (1 - (p - yLo) / (yHi - yLo))
    const volMax = Math.max(...visible.map(d => d.volume))
    const vBase  = PAD.top + cH
    const yVol   = v => PAD.top + pH + vH * (1 - (v / volMax) * 0.92)

    const pTicks = niceTicks(yLo, yHi, 5)
    const tCount = Math.min(6, n)
    const tIdxs  = Array.from({ length: tCount }, (_, i) =>
      Math.floor(i * (n - 1) / (tCount - 1 || 1))
    )

    return { w, h, n, cW, cH, pH, vH, xStep, cw, xOf, yOf, yVol, vBase, pTicks, tIdxs }
  }, [visible, dim])

  // ── Buy/sell markers ─────────────────────────────────────────────
  // step.step is 0-indexed within the test slice.
  // Full-ohlcv index = testStartIdx + step.step.
  // Visible-slice index = full index − sliceStart.
  const markers = useMemo(() => {
    const m = {}
    for (const s of (steps ?? [])) {
      if (s.action === 0 || s.action === 1) {
        const globalIdx = testStartIdx + s.step
        const visIdx    = globalIdx - sliceStart
        if (visIdx >= 0 && visIdx < (visible?.length ?? 0)) m[visIdx] = s.action
      }
    }
    return m
  }, [steps, testStartIdx, sliceStart, visible?.length])

  // ── Test-period highlight range within visible ───────────────────
  const testVisStart = Math.max(0, testStartIdx - sliceStart)
  const testVisEnd   = visible?.length ?? 0   // test extends to last bar

  const { w, h } = dim
  const hovBar    = hov != null ? visible[hov] : null
  const isDragging = !!dragRef.current

  if (!visible.length || !c) {
    return (
      <div ref={wrapRef} style={{ width: '100%', height: '100%', display: 'flex',
        alignItems: 'center', justifyContent: 'center', background: '#131722' }}>
        <span style={{ color: '#4c525e', fontSize: '12px' }}>Loading chart…</span>
      </div>
    )
  }

  const { xOf, yOf, yVol, vBase, pTicks, tIdxs, cw, xStep, pH, cH } = c

  return (
    <div
      ref={wrapRef}
      style={{
        width: '100%', height: '100%', position: 'relative',
        background: '#131722', userSelect: 'none',
        cursor: isDragging ? 'grabbing' : 'crosshair',
      }}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onMouseLeave={() => { onMouseUp(); setHov(null) }}
    >

      {/* ── Unified header row ─────────────────────────────────────────────── */}
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: PAD.top,
        display: 'flex', alignItems: 'center',
        paddingLeft: PAD.left, paddingRight: PAD.right,
        zIndex: 20, gap: 8,
      }}>

        {/* Left: ticker + date range  OR  OHLC hover values */}
        <div style={{
          display: 'flex', gap: 8, alignItems: 'center', flex: 1, minWidth: 0,
          fontSize: 11, pointerEvents: 'none', fontVariantNumeric: 'tabular-nums',
          overflow: 'hidden',
        }}>
          {hovBar ? (
            <>
              <span style={{ color: '#4c525e', flexShrink: 0 }}>{hovBar.date}</span>
              {[['O', hovBar.open], ['H', hovBar.high], ['L', hovBar.low], ['C', hovBar.close]].map(([l, v]) => (
                <span key={l} style={{
                  flexShrink: 0,
                  color: l === 'C' ? (hovBar.close >= hovBar.open ? '#26a69a' : '#ef5350') : '#d1d4dc',
                }}>
                  {l} <strong>{v.toFixed(2)}</strong>
                </span>
              ))}
            </>
          ) : (
            <>
              <span style={{ fontSize: 12, fontWeight: 700, color: '#d1d4dc', flexShrink: 0 }}>
                {ticker}
              </span>
              <span style={{ fontSize: 10, color: '#4c525e', flexShrink: 0, fontVariantNumeric: 'tabular-nums' }}>
                {`${visible[0]?.date ?? ''} → ${visible[visible.length - 1]?.date ?? ''}`}
              </span>
              {regime?.label && (
                <span style={{
                  background: regime.color + '22',
                  border: `1px solid ${regime.color}`,
                  color: regime.color,
                  padding: '1px 6px', borderRadius: 3,
                  fontSize: 9, fontWeight: 600, letterSpacing: '0.03em',
                  flexShrink: 0,
                }}>
                  {regime.label} · SPY {regime.spy_return >= 0 ? '+' : ''}{regime.spy_return}%
                </span>
              )}
            </>
          )}
        </div>

        {/* Right: timeframe pills */}
        <div style={{ display: 'flex', gap: 2, flexShrink: 0, pointerEvents: 'auto' }}>
          {TIMEFRAMES.map(tf => {
            const active   = activeTf === tf.label
            const disabled = tf.bars != null && totalBars < tf.bars
            return (
              <button key={tf.label}
                onClick={() => !disabled && applyTimeframe(tf)}
                disabled={disabled}
                style={{
                  padding: '2px 7px', borderRadius: 3,
                  border: active ? '1px solid #2962ff' : '1px solid transparent',
                  background: active ? 'rgba(41,98,255,0.18)' : 'transparent',
                  color: disabled ? '#2a2e39' : active ? '#5c8df6' : '#4c525e',
                  fontSize: 10, fontWeight: active ? 700 : 500,
                  cursor: disabled ? 'not-allowed' : 'pointer',
                  transition: 'all 0.1s', lineHeight: 1,
                }}>
                {tf.label}
              </button>
            )
          })}
        </div>

      </div>

      {/* Mini scrollbar */}
      {totalBars > barsToShow && (() => {
        const trackW = w - PAD.left - PAD.right - 2
        const thumbW = Math.max(16, trackW * barsToShow / totalBars)
        const maxPan = totalBars - barsToShow
        const thumbX = PAD.left + (trackW - thumbW) * (1 - clampedPan / Math.max(1, maxPan))
        return (
          <div style={{
            position: 'absolute', bottom: 2, left: PAD.left, width: trackW,
            height: 3, background: 'rgba(255,255,255,0.05)', borderRadius: 2, pointerEvents: 'none',
          }}>
            <div style={{
              position: 'absolute', left: thumbX - PAD.left, width: thumbW,
              height: '100%', background: 'rgba(41,98,255,0.45)', borderRadius: 2,
            }} />
          </div>
        )
      })()}

      <svg width={w} height={h} style={{ display: 'block' }}
        onMouseLeave={() => setHov(null)}>

        {/* Test-period shading */}
        {testVisStart < testVisEnd && (() => {
          const x1 = xOf(testVisStart) - xStep / 2
          const x2 = xOf(testVisEnd - 1) + xStep / 2
          return <rect x={x1} y={PAD.top} width={Math.max(0, x2 - x1)} height={cH}
            fill="rgba(41,98,255,0.04)" />
        })()}

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
        <line x1={PAD.left} y1={PAD.top + pH} x2={w - PAD.right} y2={PAD.top + pH}
          stroke="#1e2329" strokeWidth={1} />

        {/* Candles + volume bars + markers */}
        {visible.map((bar, i) => {
          const x  = xOf(i)
          const up = bar.close >= bar.open
          const col = up ? '#26a69a' : '#ef5350'
          const yO = yOf(bar.open), yC = yOf(bar.close)
          const yH = yOf(bar.high), yL = yOf(bar.low)
          const bodyY = Math.min(yO, yC)
          const bodyH = Math.max(1, Math.abs(yC - yO))
          const vy  = yVol(bar.volume)
          const act = markers[i]

          return (
            <g key={i} onMouseEnter={() => setHov(i)} style={{ cursor: 'crosshair' }}>
              {hov === i && (
                <rect x={x - xStep / 2} y={PAD.top} width={xStep} height={cH}
                  fill="rgba(255,255,255,0.025)" />
              )}
              <line x1={x} y1={yH} x2={x} y2={yL} stroke={col} strokeWidth={1} />
              <rect x={x - cw / 2} y={bodyY} width={cw} height={bodyH} fill={col} />
              <rect x={x - cw / 2} y={vy} width={cw} height={vBase - vy}
                fill={col} opacity={0.35} />
              {act === 0 && (
                <polygon points={`${x},${yL + 12} ${x - 5},${yL + 20} ${x + 5},${yL + 20}`}
                  fill="#2962ff" />
              )}
              {act === 1 && (
                <polygon points={`${x},${yH - 12} ${x - 5},${yH - 20} ${x + 5},${yH - 20}`}
                  fill="#ef5350" />
              )}
            </g>
          )
        })}

        {/* Hover crosshair */}
        {hov != null && (
          <line x1={xOf(hov)} y1={PAD.top} x2={xOf(hov)} y2={PAD.top + cH}
            stroke="rgba(255,255,255,0.1)" strokeWidth={1} strokeDasharray="3 3" />
        )}

        {/* X axis date labels */}
        {tIdxs.map(i => {
          if (i >= visible.length) return null
          const d = new Date(visible[i].date + 'T00:00:00')
          const label = d.toLocaleDateString('en-US', { month: 'short', year: '2-digit' })
          return (
            <text key={i} x={xOf(i)} y={h - 8} textAnchor="middle" fill="#4c525e" fontSize={9}>
              {label}
            </text>
          )
        })}

        {/* TEST period label */}
        {testVisStart < testVisEnd && testVisStart < visible.length && (
          <text
            x={Math.min(xOf(testVisStart) + 4, w - PAD.right - 30)}
            y={PAD.top + 22}
            fill="rgba(41,98,255,0.45)" fontSize={9} fontWeight="700">
            TEST
          </text>
        )}

        {/* Legend */}
        <polygon
          points={`${w - 82},${PAD.top + 6} ${w - 87},${PAD.top + 13} ${w - 77},${PAD.top + 13}`}
          fill="#2962ff" />
        <text x={w - 74} y={PAD.top + 13} fill="#4c525e" fontSize={10}>Buy</text>
        <polygon
          points={`${w - 40},${PAD.top + 13} ${w - 45},${PAD.top + 6} ${w - 35},${PAD.top + 6}`}
          fill="#ef5350" />
        <text x={w - 32} y={PAD.top + 13} fill="#4c525e" fontSize={10}>Sell</text>
      </svg>
    </div>
  )
}
