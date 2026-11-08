import { useEffect, useState } from 'react'
import { Link, Navigate } from 'react-router-dom'
import TrendChart from '../components/TrendChart'
import { useAuth } from '../auth/AuthContext'
import { useI18n } from '../i18n/I18nContext'
import { api } from '../services/api'

function riskPct(probability) {
  if (probability == null) return ''
  const n = Number(probability)
  return `${Math.round(n <= 1 ? n * 100 : n)}%`
}

export default function History() {
  const { t } = useI18n()
  const { user, ready } = useAuth()
  const [rows, setRows] = useState([])
  const [trend, setTrend] = useState([])
  const [error, setError] = useState('')

  useEffect(() => {
    if (!user) return
    Promise.all([api.history(), api.historyTrend()])
      .then(([history, points]) => {
        setRows(Array.isArray(history) ? history : [])
        setTrend(Array.isArray(points) ? points : [])
      })
      .catch((e) => setError(e.message))
  }, [user])

  if (!ready) return null
  if (!user) return <Navigate to="/login" replace />

  return (
    <section className="page">
      <h1>{t('historyTitle')}</h1>
      <p className="lede short">{t('historyLede')}</p>
      {error && <p className="error">{error}</p>}
      <TrendChart points={trend} />
      {rows.length === 0 ? (
        <p className="muted">
          {t('historyEmpty')} <Link to="/predict">{t('ctaAssess')}</Link>
        </p>
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
                <th>{t('age')}</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
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
                  <td>{row.age}</td>
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
  )
}
