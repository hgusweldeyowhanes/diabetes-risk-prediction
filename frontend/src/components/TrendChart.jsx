import { useMemo, useState } from 'react'
import { useI18n } from '../i18n/I18nContext'

function pct(value) {
  if (value == null || Number.isNaN(Number(value))) return null
  const n = Number(value)
  return n <= 1 ? Math.round(n * 100) : Math.round(n)
}

export default function TrendChart({ points = [] }) {
  const { t } = useI18n()
  const [series, setSeries] = useState('probability')

  const chart = useMemo(() => {
    const ordered = [...points].sort(
      (a, b) => new Date(a.created_at) - new Date(b.created_at),
    )
    if (ordered.length < 2) return null

    const values = ordered.map((p) => {
      if (series === 'glucose') return Number(p.glucose)
      if (series === 'bmi') return Number(p.bmi)
      return pct(p.probability) ?? 0
    })
    const min = Math.min(...values)
    const max = Math.max(...values)
    const pad = max === min ? 1 : (max - min) * 0.12
    const lo = min - pad
    const hi = max + pad
    const w = 640
    const h = 220
    const left = 44
    const right = 16
    const top = 16
    const bottom = 36
    const innerW = w - left - right
    const innerH = h - top - bottom

    const coords = values.map((v, i) => {
      const x = left + (ordered.length === 1 ? innerW / 2 : (i / (ordered.length - 1)) * innerW)
      const y = top + innerH - ((v - lo) / (hi - lo || 1)) * innerH
      return { x, y, v, point: ordered[i] }
    })
    const path = coords.map((c, i) => `${i === 0 ? 'M' : 'L'} ${c.x.toFixed(1)} ${c.y.toFixed(1)}`).join(' ')
    const labels = [
      { y: top, text: series === 'probability' ? `${Math.round(hi)}%` : hi.toFixed(1) },
      { y: top + innerH, text: series === 'probability' ? `${Math.round(lo)}%` : lo.toFixed(1) },
    ]
    return { w, h, path, coords, labels, ordered }
  }, [points, series])

  if (!chart) {
    return <p className="muted">{t('trendEmpty')}</p>
  }

  return (
    <div className="trend-panel panel">
      <div className="trend-toolbar no-print">
        <h3>{t('trendTitle')}</h3>
        <div className="trend-toggles">
          {[
            ['probability', t('mlRisk')],
            ['glucose', t('glucose')],
            ['bmi', t('bmi')],
          ].map(([id, label]) => (
            <button
              key={id}
              type="button"
              className={`chip ${series === id ? 'is-on' : ''}`}
              onClick={() => setSeries(id)}
            >
              {label}
            </button>
          ))}
        </div>
      </div>
      <svg viewBox={`0 0 ${chart.w} ${chart.h}`} className="trend-svg" role="img" aria-label={t('trendTitle')}>
        <line x1="44" y1="16" x2="44" y2="184" className="trend-axis" />
        <line x1="44" y1="184" x2="624" y2="184" className="trend-axis" />
        {chart.labels.map((l) => (
          <text key={l.y} x="40" y={l.y + 4} className="trend-label" textAnchor="end">
            {l.text}
          </text>
        ))}
        <path d={chart.path} className="trend-line" fill="none" />
        {chart.coords.map((c) => (
          <g key={c.point.prediction_id || c.point.created_at}>
            <circle cx={c.x} cy={c.y} r="4.5" className="trend-dot" />
            <title>
              {new Date(c.point.created_at).toLocaleString()} · {c.v}
              {series === 'probability' ? '%' : ''}
            </title>
          </g>
        ))}
        {chart.ordered.map((p, i) => {
          const c = chart.coords[i]
          if (i !== 0 && i !== chart.ordered.length - 1 && chart.ordered.length > 3) return null
          return (
            <text key={`x-${p.prediction_id || i}`} x={c.x} y="208" className="trend-label" textAnchor="middle">
              {new Date(p.created_at).toLocaleDateString()}
            </text>
          )
        })}
      </svg>
    </div>
  )
}
