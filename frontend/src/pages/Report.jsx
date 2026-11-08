import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import ClinicalReport from '../components/ClinicalReport'
import { useAuth } from '../auth/AuthContext'
import { useI18n } from '../i18n/I18nContext'
import { api } from '../services/api'

function normalizeHistoryRow(row) {
  return {
    ...row,
    prediction_id: row.id,
    clinical_guidance: row.clinical_guidance || [],
    local_explanation: row.local_explanation || [],
    counterfactuals: row.counterfactuals || [],
    dataset_note: row.dataset_note,
  }
}

export default function Report() {
  const { predictionId } = useParams()
  const { t } = useI18n()
  const { user } = useAuth()
  const [result, setResult] = useState(null)
  const [meta, setMeta] = useState({})
  const [error, setError] = useState('')

  useEffect(() => {
    let cancelled = false

    async function load() {
      setError('')
      if (predictionId) {
        if (!user) {
          setError(t('reportNeedsLogin'))
          return
        }
        try {
          const row = await api.historyDetail(predictionId)
          if (cancelled) return
          setResult(normalizeHistoryRow(row))
          setMeta({ created_at: row.created_at })
        } catch (err) {
          if (!cancelled) setError(err.message)
        }
        return
      }

      try {
        const raw = sessionStorage.getItem('last_prediction')
        if (!raw) {
          setResult(null)
          return
        }
        const parsed = JSON.parse(raw)
        if (!cancelled) {
          setResult(parsed)
          setMeta({ created_at: parsed.created_at })
        }
      } catch {
        if (!cancelled) setResult(null)
      }
    }

    load()
    return () => {
      cancelled = true
    }
  }, [predictionId, user, t])

  return (
    <section className="page report-page">
      <div className="page-head no-print">
        <div>
          <h1>{t('reportTitle')}</h1>
          <p className="lede short">{t('reportLede')}</p>
        </div>
        <div className="btn-row">
          <button type="button" className="btn primary" onClick={() => window.print()} disabled={!result}>
            {t('print')}
          </button>
          <Link className="btn ghost" to="/results">
            {t('navResults')}
          </Link>
        </div>
      </div>
      {error && <p className="error">{error}</p>}
      <ClinicalReport result={result} meta={meta} />
    </section>
  )
}
