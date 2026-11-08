import { useEffect, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import PatientForm from '../components/PatientForm'
import Disclaimer from '../components/Disclaimer'
import { useAuth } from '../auth/AuthContext'
import { useI18n } from '../i18n/I18nContext'
import { api } from '../services/api'

export default function Predict() {
  const { t } = useI18n()
  const { user } = useAuth()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [patients, setPatients] = useState([])
  const [patientId, setPatientId] = useState('')
  const navigate = useNavigate()
  const [params] = useSearchParams()

  useEffect(() => {
    const fromQuery = params.get('patient') || ''
    if (fromQuery) setPatientId(fromQuery)
  }, [params])

  useEffect(() => {
    if (!user) {
      setPatients([])
      return
    }
    api
      .listPatients()
      .then((rows) => setPatients(Array.isArray(rows) ? rows : []))
      .catch(() => setPatients([]))
  }, [user])

  async function handleSubmit(form) {
    setLoading(true)
    setError('')
    try {
      const payload = {
        ...form,
        patient_id: patientId ? Number(patientId) : null,
      }
      const result = await api.predict(payload)
      sessionStorage.setItem('last_prediction', JSON.stringify(result))
      navigate(result.prediction_id ? `/report/${result.prediction_id}` : '/results')
    } catch (err) {
      setError(err.message || 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  function patientLabel(p) {
    const name = `${p.first_name || ''} ${p.last_name || ''}`.trim()
    return name || p.external_id || `#${p.id}`
  }

  return (
    <section className="page">
      <h1>{t('predictTitle')}</h1>
      <p className="lede short">{t('predictLede')}</p>
      <Disclaimer compact />
      {error && <p className="error">{error}</p>}
      {user && (
        <label className="field patient-picker no-print">
          <span>{t('linkPatient')}</span>
          <select value={patientId} onChange={(e) => setPatientId(e.target.value)}>
            <option value="">{t('noPatient')}</option>
            {patients.map((p) => (
              <option key={p.id} value={p.id}>
                {patientLabel(p)}
              </option>
            ))}
          </select>
          <small className="field-help">{t('linkPatientHelp')}</small>
        </label>
      )}
      <PatientForm onSubmit={handleSubmit} loading={loading} />
    </section>
  )
}
