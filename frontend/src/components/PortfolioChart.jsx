import { useRef, useEffect, useState, useMemo } from 'react'

const PAD = { top: 44, right: 14, bottom: 28, left: 62 }

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
  if (v >= 1_000)     return `$${(v / 1_000).toFixed(0)}k`
  return `$${v.toFixed(0)}`
}

function fmtPct(v, initial) {
  const pct = (v - initial) / initial * 100
  return `${pct >= 0 ? '+' : ''}${pct.toFixed(1)}%`
}

function makePath(data, xOf, yOf) {
  if (!data?.length) return ''
  return data.map((d, i) => `${i === 0 ? 'M' : 'L'}${xOf(i)},${yOf(d)}`).join(' ')
}

export default function PortfolioChart({
  portfolioHistory, spyHistory, aaplBahHistory,
  ohlcv, initialCapital, savedRuns, algo,
}) {
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

  const ic = initialCapital ?? 100_000

  const series = useMemo(() => {
    const result = []
    const take = hist => hist?.slice(1).filter(v => v != null) ?? []

    // ── Current strategy ─────────────────────────────────────────────
    const port = take(portfolioHistory)
    if (port.length) {
      const last = port[port.length - 1]
      result.push({
        id: 'current', label: algo ? `${algo} Strategy` : 'Strategy',
        data: port, color: last >= ic ? '#2962ff' : '#ef5350',
        width: 2, dash: '',
      })
    }

    // ── S&P 500 benchmark ─────────────────────────────────────────────
    const spy = take(spyHistory)
    if (spy.length) {
      result.push({
        id: 'spy', label: 'S&P 500',
        data: spy, color: '#f5c842',
        width: 1.5, dash: '',
      })
    }

    // ── AAPL Buy & Hold ───────────────────────────────────────────────
    const bah = take(aaplBahHistory)
    if (bah.length) {
      result.push({
        id: 'bah', label: 'AAPL B&H',
        data: bah, color: '#787b86',
        width: 1, dash: '4 3',
      })
    }

    // ── Saved runs ────────────────────────────────────────────────────
    for (const run of (savedRuns ?? [])) {
      const d = take(run.portfolioHistory)
      if (d.length) result.push({
        id: String(run.id), label: run.label,
        data: d, color: run.color,
        width: 1.5, dash: '',
      })
    }

    return result
  }, [portfolioHistory, spyHistory, aaplBahHistory, savedRuns, algo, ic])

  const c = useMemo(() => {
    const { w, h } = dim
    const allVals = series.flatMap(s => s.data)
    if (!allVals.length) return null

    const n  = Math.max(...series.map(s => s.data.length))
    const cW = w - PAD.left - PAD.right
    const cH = h - PAD.top - PAD.bottom

    const rawMin = Math.min(...allVals, ic)
    const rawMax = Math.max(...allVals, ic)
    const pad    = (rawMax - rawMin) * 0.06 || rawMax * 0.02
    const yLo    = rawMin - pad
    const yHi    = rawMax + pad

    const xOf  = i => PAD.left + (i / (n - 1 || 1)) * cW
    const yOf  = v => PAD.top + cH * (1 - (v - yLo) / (yHi - yLo))
    const ticks = niceTicks(yLo, yHi, 4)

    const tCount = Math.min(5, n)
    const tIdxs  = Array.from({ length: tCount }, (_, i) =>
      Math.floor(i * (n - 1) / (tCount - 1 || 1))
    )
    return { w, h, cW, cH, n, xOf, yOf, ticks, tIdxs }
  }, [series, dim, ic])

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

        // Legend layout — positioned in the 44px top padding row
        // each entry: line swatch + label + final value + pct
        const legendItems = series.map(s => {
          const last = s.data[s.data.length - 1] ?? ic
          return { ...s, last, pct: fmtPct(last, ic) }
        })

        return (
          <svg width={w} height={h} style={{ display: 'block' }}>

            {/* ── Legend strip ──────────────────────────────────────────────── */}
            {legendItems.map((s, i) => {
              const retColor = s.last >= ic ? '#26a69a' : '#ef5350'
              const lx = PAD.left + i * 150
              if (lx + 140 > w - PAD.right) return null  // overflow guard
              return (
                <g key={s.id}>
                  {/* line swatch */}
                  <line x1={lx} y1={14} x2={lx + 16} y2={14}
                    stroke={s.color} strokeWidth={s.id === 'bah' ? 1.5 : 2}
                    strokeDasharray={s.dash || undefined} />
                  {/* label */}
                  <text x={lx + 20} y={11} fill={s.color} fontSize={9} fontWeight="700">
                    {s.label}
                  </text>
                  {/* dollar value */}
                  <text x={lx + 20} y={22} fill="#d1d4dc" fontSize={9}
                    fontVariantNumeric="tabular-nums">
                    {fmtDollar(s.last)}
                  </text>
                  {/* pct */}
                  <text x={lx + 20 + 36} y={22} fill={retColor} fontSize={9} fontWeight="600"
                    fontVariantNumeric="tabular-nums">
                    {s.pct}
                  </text>
                </g>
              )
            })}

            {/* ── Y grid + labels ───────────────────────────────────────────── */}
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

            {/* ── Initial capital reference line ────────────────────────────── */}
            {(() => {
              const y = yOf(ic)
              if (y < PAD.top || y > PAD.top + cH) return null
              return <line x1={PAD.left} y1={y} x2={w - PAD.right} y2={y}
                stroke="#363c4e" strokeWidth={1} strokeDasharray="2 4" />
            })()}

            {/* ── Gradient fill under current strategy ─────────────────────── */}
            {series.find(s => s.id === 'current') && (() => {
              const cur    = series.find(s => s.id === 'current')
              const n      = cur.data.length
              const pathD  = makePath(cur.data, xOf, yOf)
              const areaD  = pathD + ` L${xOf(n - 1)},${PAD.top + cH} L${xOf(0)},${PAD.top + cH} Z`
              return (
                <>
                  <defs>
                    <linearGradient id="portGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={cur.color} stopOpacity="0.12" />
                      <stop offset="100%" stopColor={cur.color} stopOpacity="0" />
                    </linearGradient>
                  </defs>
                  <path d={areaD} fill="url(#portGrad)" />
                </>
              )
            })()}

            {/* ── Lines ─────────────────────────────────────────────────────── */}
            {[...series].reverse().map(s => {
              if (s.data.length < 2) return null
              return (
                <path key={s.id}
                  d={makePath(s.data, xOf, yOf)}
                  fill="none"
                  stroke={s.color}
                  strokeWidth={s.width}
                  strokeDasharray={s.dash || undefined}
                  opacity={s.id === 'current' ? 1 : 0.75}
                />
              )
            })}

            {/* ── X axis dates ──────────────────────────────────────────────── */}
            {tIdxs.map(i => {
              const bar = ohlcv?.[i]
              if (!bar) return null
              const d = new Date(bar.date + 'T00:00:00')
              const label = d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
              return (
                <text key={i} x={xOf(i)} y={h - 8} textAnchor="middle"
                  fill="#4c525e" fontSize={10}>
                  {label}
                </text>
              )
            })}
          </svg>
        )
      })()}
    </div>
  )
}
