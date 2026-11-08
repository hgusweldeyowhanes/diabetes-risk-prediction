import { useEffect, useState } from 'react'
import { Link, Navigate, useParams } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'
import TrendChart from '../components/TrendChart'
import { useI18n } from '../i18n/I18nContext'
import { api } from '../services/api'

function displayName(p) {
  if (!p) return ''
  const name = `${p.first_name || ''} ${p.last_name || ''}`.trim()
  return name || p.external_id || `#${p.id}`
}

function riskPct(probability) {
  if (probability == null) return ''
  const n = Number(probability)
  return `${Math.round(n <= 1 ? n * 100 : n)}%`
}

export default function PatientDetail() {
  const { id } = useParams()
  const { t } = useI18n()
  const { user, ready } = useAuth()
  const [patient, setPatient] = useState(null)
  const [predictions, setPredictions] = useState([])
  const [trend, setTrend] = useState([])
  const [error, setError] = useState('')
  const [notes, setNotes] = useState('')
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    if (!user || !id) return
    Promise.all([api.getPatient(id), api.patientPredictions(id), api.patientTrend(id)])
      .then(([p, preds, points]) => {
        setPatient(p)
        setNotes(p.notes || '')
        setPredictions(Array.isArray(preds) ? preds : [])
        setTrend(Array.isArray(points) ? points : [])
      })
      .catch((e) => setError(e.message))
  }, [user, id])

  if (!ready) return null
  if (!user) return <Navigate to="/login" replace />

  async function saveNotes(e) {
    e.preventDefault()
    setSaving(true)
    setError('')
    try {
      const updated = await api.updatePatient(id, { notes })
      setPatient(updated)
    } catch (err) {
      setError(err.message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <section className="page">
      <div className="page-head">
        <div>
          <p className="eyebrow">
            <Link to="/patients">{t('navPatients')}</Link>
          </p>
          <h1>{displayName(patient) || t('patient')}</h1>
          <p className="lede short">{t('dossierLede')}</p>
        </div>
        <Link className="btn primary no-print" to={`/predict?patient=${id}`}>
          {t('assessPatient')}
        </Link>
      </div>
      {error && <p className="error">{error}</p>}

      {patient && (
        <div className="dossier-grid">
          <section className="panel">
            <h3>{t('demographics')}</h3>
            <dl className="kv">
              <div>
                <dt>{t('externalId')}</dt>
                <dd>{patient.external_id || '—'}</dd>
              </div>
              <div>
                <dt>{t('sex')}</dt>
                <dd>
                  {patient.sex === 'F' ? t('sexF') : patient.sex === 'M' ? t('sexM') : t('sexU')}
                </dd>
              </div>
              <div>
                <dt>{t('dateOfBirth')}</dt>
                <dd>{patient.date_of_birth || '—'}</dd>
              </div>
              <div>
                <dt>{t('assessments')}</dt>
                <dd>{patient.prediction_count ?? predictions.length}</dd>
              </div>
            </dl>
            <form onSubmit={saveNotes} className="notes-form no-print">
              <label className="field">
                <span>{t('notes')}</span>
                <textarea rows={4} value={notes} onChange={(e) => setNotes(e.target.value)} />
              </label>
              <button type="submit" className="btn ghost" disabled={saving}>
                {saving ? t('saving') : t('saveNotes')}
              </button>
            </form>
          </section>

          <TrendChart points={trend} />
        </div>
      )}

      <section className="panel">
        <h3>{t('assessmentHistory')}</h3>
        {predictions.length === 0 ? (
          <p className="muted">{t('historyEmpty')}</p>
        ) : (
          <div className="data-table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>{t('when')}</th>
                  <th>{t('mlRisk')}</th>
                  <th>{t('screenScore')}</th>
                  <th>{t('glucose')}</th>
                  <th>{t('bmi')}</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {[...predictions].reverse().map((row) => (
                  <tr key={row.id}>
                    <td>{new Date(row.created_at).toLocaleString()}</td>
                    <td>
                      {row.risk_level || (row.prediction_result ? t('elevated') : t('lower'))}
                      {row.probability != null ? ` · ${riskPct(row.probability)}` : ''}
                    </td>
                    <td>
                      {row.screening
                        ? `${row.screening.label} (${row.screening.score}/${row.screening.max})`
                        : '—'}
                    </td>
                    <td>{row.glucose}</td>
                    <td>{row.bmi}</td>
                    <td>
                      <Link to={`/report/${row.id}`}>{t('openReport')}</Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </section>
  )
}
