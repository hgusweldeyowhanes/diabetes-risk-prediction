import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import PatientForm from '../components/PatientForm'
import Disclaimer from '../components/Disclaimer'
import { useI18n } from '../i18n/I18nContext'
import { api } from '../services/api'

export default function Predict() {
  const { t } = useI18n()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const navigate = useNavigate()

  async function handleSubmit(form) {
    setLoading(true)
    setError('')
    try {
      const result = await api.predict(form)
      sessionStorage.setItem('last_prediction', JSON.stringify(result))
      navigate('/results')
    } catch (err) {
      setError(err.message || 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="page">
      <h1>{t('predictTitle')}</h1>
      <p className="lede short">{t('predictLede')}</p>
      <Disclaimer compact />
      {error && <p className="error">{error}</p>}
      <PatientForm onSubmit={handleSubmit} loading={loading} />
    </section>
  )
}
