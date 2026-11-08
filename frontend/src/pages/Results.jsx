import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import ClinicalReport from '../components/ClinicalReport'
import Disclaimer from '../components/Disclaimer'
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

  const reportPath = result?.prediction_id ? `/report/${result.prediction_id}` : '/report'

  return (
    <section className="page">
      <div className="page-head">
        <div>
          <h1>{t('resultsTitle')}</h1>
          <p className="lede short">{t('resultsLede')}</p>
        </div>
        {result && (
          <div className="btn-row no-print">
            <Link className="btn primary" to={reportPath}>
              {t('openReport')}
            </Link>
            <button type="button" className="btn ghost" onClick={() => window.print()}>
              {t('print')}
            </button>
          </div>
        )}
      </div>
      <Disclaimer compact />
      <ClinicalReport result={result} />
    </section>
  )
}
