import { useI18n } from '../i18n/I18nContext'

export default function RiskDashboard({ result }) {
  const { t } = useI18n()
  if (!result) return null
  const pct = Math.round((result.probability || 0) * 100)
  const risk = result.risk_level || '—'
  const votes = result.model_votes || {}
  const screen = result.screening

  return (
    <section className="risk-dashboard">
      <div className="risk-hero">
        <p className="eyebrow">{t('scoreEyebrow')}</p>
        <h2 className={result.prediction === 'Diabetic' ? 'danger' : 'ok'}>
          {result.prediction === 'Diabetic' ? t('elevated') : t('lower')}
        </h2>
        <p className="risk-tier">{risk}</p>
        <div className="meter">
          <div className="meter-fill" style={{ width: `${pct}%` }} />
        </div>
        <p className="meter-label">
          {pct}% {t('estimated')} · {t('confidence')}{' '}
          {Math.round((result.confidence || 0) * 100)}%
        </p>
        {result.probability_ci && (
          <p className="muted small">
            {t('likelyRange')} {Math.round(result.probability_ci[0] * 100)}–
            {Math.round(result.probability_ci[1] * 100)}%
          </p>
        )}
      </div>

      {screen && (
        <div className="screen-card">
          <h3>{t('checklist')}</h3>
          <p className="risk-tier">
            {screen.label} · {screen.score}/{screen.max}
          </p>
          <p className="muted small">{t('checklistNote')}</p>
        </div>
      )}

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
