import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import PatientForm from '../components/PatientForm'
import { api } from '../services/api'

export default function Predict() {
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
      <h1>Clinical markers</h1>
      <p className="lede short">
        Enter values from a lab visit. Missing or zero readings for glucose,
        blood pressure, BMI, or insulin are filled carefully by the model.
      </p>
      {error && <p className="error">{error}</p>}
      <PatientForm onSubmit={handleSubmit} loading={loading} />
    </section>
  )
}
