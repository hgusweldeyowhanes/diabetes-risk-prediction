import { useEffect, useState } from 'react'
import Disclaimer from '../components/Disclaimer'
import RiskDashboard from '../components/RiskDashboard'
import ResultsDisplay from '../components/ResultsDisplay'
import { useI18n } from '../i18n/I18nContext'

export default function Results() {
  const { t } = useI18n()
  const [result, setResult] = useState(null)

  useEffect(() => {
    try {
      const raw = sessionStorage.getItem('last_prediction')
      if (raw) setResult(JSON.parse(raw))
    } catch {
      setResult(null)
    }
  }, [])

  return (
    <section className="page">
      <div className="page-head">
        <div>
          <h1>{t('resultsTitle')}</h1>
          <p className="lede short">{t('resultsLede')}</p>
        </div>
        {result && (
          <button type="button" className="btn ghost no-print" onClick={() => window.print()}>
            {t('print')}
          </button>
        )}
      </div>
      <Disclaimer compact />
      <RiskDashboard result={result} />
      <ResultsDisplay result={result} />
      {result?.dataset_note && <p className="muted small">{result.dataset_note}</p>}
    </section>
  )
}
