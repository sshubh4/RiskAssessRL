import { useRef, useEffect, useState, useMemo } from 'react'

const PAD = { top: 12, right: 14, bottom: 28, left: 62 }

function niceTicks(min, max, count = 5) {
  const range = max - min
  if (range === 0) return [min, max]
  const raw = range / (count - 1)
  const mag = Math.pow(10, Math.floor(Math.log10(raw)))
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(m => m >= raw) ?? raw
  const lo = Math.ceil(min / step) * step
  const ticks = []
  for (let t = lo; t <= max + step * 0.01; t += step)
    ticks.push(parseFloat(t.toFixed(10)))
  return ticks
}

function fmtDollar(v) {
  if (v >= 1_000_000) return `$${(v / 1_000_000).toFixed(1)}M`
  if (v >= 1_000) return `$${(v / 1_000).toFixed(0)}k`
  return `$${v.toFixed(0)}`
}

function makePath(data, xOf, yOf) {
  if (!data?.length) return ''
  return data.map((d, i) => `${i === 0 ? 'M' : 'L'}${xOf(i)},${yOf(d)}`).join(' ')
}

export default function PortfolioChart({ portfolioHistory, bahHistory, ohlcv, initialCapital, savedRuns }) {
  const wrapRef = useRef(null)
  const [dim, setDim] = useState({ w: 600, h: 200 })

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

  // Build dated series from histories + ohlcv date mapping
  const series = useMemo(() => {
    const result = []

    const mapToPoints = hist => hist?.slice(1).map((v, i) => v).filter(v => v != null) ?? []

    const port = mapToPoints(portfolioHistory)
    const bah  = mapToPoints(bahHistory)

    if (port.length) {
      const portColor = port[port.length - 1] >= (initialCapital ?? 100000) ? '#26a69a' : '#ef5350'
      result.push({ id: 'current', label: 'Strategy', data: port, color: portColor, width: 2, dash: '' })
    }
    if (bah.length) {
      result.push({ id: 'bah', label: 'Buy & Hold', data: bah, color: '#787b86', width: 1, dash: '4 3' })
    }
    for (const run of (savedRuns ?? [])) {
      const d = mapToPoints(run.portfolioHistory)
      if (d.length) result.push({ id: String(run.id), label: run.label, data: d, color: run.color, width: 1.5, dash: '' })
    }
    return result
  }, [portfolioHistory, bahHistory, savedRuns, initialCapital])

  const c = useMemo(() => {
    const { w, h } = dim
    const allVals = series.flatMap(s => s.data)
    if (!allVals.length) return null

    const n = Math.max(...series.map(s => s.data.length))
    const cW = w - PAD.left - PAD.right
    const cH = h - PAD.top - PAD.bottom

    const rawMin = Math.min(...allVals)
    const rawMax = Math.max(...allVals)
    const pad = (rawMax - rawMin) * 0.06 || rawMax * 0.02
    const yLo = rawMin - pad, yHi = rawMax + pad

    const xOf = i => PAD.left + (i / (n - 1 || 1)) * cW
    const yOf = v => PAD.top + cH * (1 - (v - yLo) / (yHi - yLo))

    const ticks = niceTicks(yLo, yHi, 4)

    // X axis: use ohlcv dates if available
    const tCount = Math.min(5, n)
    const tIdxs = Array.from({ length: tCount }, (_, i) => Math.floor(i * (n - 1) / (tCount - 1 || 1)))

    return { w, h, cW, cH, n, xOf, yOf, ticks, tIdxs }
  }, [series, dim])

  const hasData = series.length > 0

  return (
    <div ref={wrapRef} style={{ width: '100%', height: '100%', position: 'relative',
      background: '#131722', userSelect: 'none' }}>

      {!hasData && (
        <div style={{ position: 'absolute', inset: 0, display: 'flex',
          alignItems: 'center', justifyContent: 'center' }}>
          <span style={{ color: '#4c525e', fontSize: '12px' }}>
            Run a backtest to see portfolio performance
          </span>
        </div>
      )}

      {hasData && c && (() => {
        const { w, h, cW, cH, xOf, yOf, ticks, tIdxs } = c

        return (
          <svg width={w} height={h} style={{ display: 'block' }}>
            {/* Grid lines + Y labels */}
            {ticks.map(t => {
              const y = yOf(t)
              if (y < PAD.top - 2 || y > PAD.top + cH + 2) return null
              return (
                <g key={t}>
                  <line x1={PAD.left} y1={y} x2={w - PAD.right} y2={y}
                    stroke="#1e2329" strokeWidth={1} />
                  <text x={PAD.left - 6} y={y + 3.5} textAnchor="end"
                    fill="#4c525e" fontSize={10}>
                    {fmtDollar(t)}
                  </text>
                </g>
              )
            })}

            {/* Initial capital reference line */}
            {initialCapital && (() => {
              const y = yOf(initialCapital)
              if (y >= PAD.top && y <= PAD.top + cH) {
                return (
                  <line x1={PAD.left} y1={y} x2={w - PAD.right} y2={y}
                    stroke="#363c4e" strokeWidth={1} strokeDasharray="2 4" />
                )
              }
            })()}

            {/* Gradient fill under current strategy line */}
            {series.find(s => s.id === 'current') && (() => {
              const cur = series.find(s => s.id === 'current')
              const gradId = 'portGrad'
              const n = cur.data.length
              const pathD = makePath(cur.data, xOf, yOf)
              const areaD = pathD + ` L${xOf(n - 1)},${PAD.top + cH} L${xOf(0)},${PAD.top + cH} Z`
              return (
                <>
                  <defs>
                    <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={cur.color} stopOpacity="0.15" />
                      <stop offset="100%" stopColor={cur.color} stopOpacity="0" />
                    </linearGradient>
                  </defs>
                  <path d={areaD} fill={`url(#${gradId})`} />
                </>
              )
            })()}

            {/* Lines */}
            {series.map(s => {
              const n = s.data.length
              if (n < 2) return null
              return (
                <path key={s.id}
                  d={makePath(s.data, xOf, yOf)}
                  fill="none"
                  stroke={s.color}
                  strokeWidth={s.width}
                  strokeDasharray={s.dash || undefined}
                />
              )
            })}

            {/* X axis: dates from ohlcv if available */}
            {tIdxs.map(i => {
              const bar = ohlcv?.[i]
              if (!bar) return null
              const d = new Date(bar.date + 'T00:00:00')
              const label = d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
              return (
                <text key={i} x={xOf(i)} y={h - 8} textAnchor="middle" fill="#4c525e" fontSize={10}>
                  {label}
                </text>
              )
            })}

            {/* Legend top-left */}
            <text x={PAD.left} y={PAD.top + 10} fill="#4c525e" fontSize={10}
              fontWeight="600" textTransform="uppercase" letterSpacing="0.06em">
              Portfolio
            </text>
            {series.map((s, i) => {
              const x = PAD.left + 60 + i * 80
              return (
                <g key={s.id}>
                  <line x1={x} y1={PAD.top + 7} x2={x + 14} y2={PAD.top + 7}
                    stroke={s.color} strokeWidth={s.id === 'bah' ? 1 : 2}
                    strokeDasharray={s.dash || undefined} />
                  <text x={x + 18} y={PAD.top + 11} fill={s.color} fontSize={10}>{s.label}</text>
                </g>
              )
            })}
          </svg>
        )
      })()}
    </div>
  )
}
