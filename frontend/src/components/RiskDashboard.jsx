export default function RiskDashboard({ result }) {
  if (!result) return null
  const pct = Math.round((result.probability || 0) * 100)
  const risk = result.risk_level || '—'
  const votes = result.model_votes || {}

  return (
    <section className="risk-dashboard">
      <div className="risk-hero">
        <p className="eyebrow">Your GlycoPath score</p>
        <h2 className={result.prediction === 'Diabetic' ? 'danger' : 'ok'}>
          {result.prediction === 'Diabetic' ? 'Elevated diabetes risk' : 'Lower diabetes risk'}
        </h2>
        <p className="risk-tier">{risk}</p>
        <div className="meter">
          <div className="meter-fill" style={{ width: `${pct}%` }} />
        </div>
        <p className="meter-label">
          {pct}% estimated risk · call confidence{' '}
          {Math.round((result.confidence || 0) * 100)}%
        </p>
        {result.probability_ci && (
          <p className="muted small">
            Likely range {Math.round(result.probability_ci[0] * 100)}–
            {Math.round(result.probability_ci[1] * 100)}%
          </p>
        )}
      </div>

      <div className="vote-grid">
        {Object.entries(votes).map(([name, v]) => (
          <div key={name} className="vote">
            <span>{name}</span>
            <strong className={v ? 'danger' : 'ok'}>{v ? 'Risk+' : 'Risk−'}</strong>
          </div>
        ))}
      </div>
    </section>
  )
}
