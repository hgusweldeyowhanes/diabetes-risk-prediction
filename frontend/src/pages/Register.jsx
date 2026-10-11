import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'
import { useI18n } from '../i18n/I18nContext'

export default function Register() {
  const { t } = useI18n()
  const { register } = useAuth()
  const navigate = useNavigate()
  const [form, setForm] = useState({ username: '', email: '', password: '', first_name: '', last_name: '' })
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  function update(key, value) {
    setForm((prev) => ({ ...prev, [key]: value }))
  }

  async function handleSubmit(e) {
    e.preventDefault()
    setLoading(true)
    setError('')
    try {
      await register(form)
      navigate('/predict')
    } catch (err) {
      setError(err.message || 'Register failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="page narrow">
      <h1>{t('registerTitle')}</h1>
      {error && <p className="error">{error}</p>}
      <form className="patient-form" onSubmit={handleSubmit}>
        <label className="field">
          <span>{t('username')}</span>
          <input value={form.username} onChange={(e) => update('username', e.target.value)} required />
        </label>
        <label className="field">
          <span>{t('email')}</span>
          <input type="email" value={form.email} onChange={(e) => update('email', e.target.value)} />
        </label>
        <label className="field">
          <span>{t('firstName')}</span>
          <input value={form.first_name} onChange={(e) => update('first_name', e.target.value)} />
        </label>
        <label className="field">
          <span>{t('lastName')}</span>
          <input value={form.last_name} onChange={(e) => update('last_name', e.target.value)} />
        </label>
        <label className="field">
          <span>{t('password')}</span>
          <input type="password" value={form.password} onChange={(e) => update('password', e.target.value)} required />
        </label>
        <button className="btn primary" type="submit" disabled={loading}>
          {t('submitRegister')}
        </button>
      </form>
      <p className="muted">
        {t('haveAccount')} <Link to="/login">{t('navLogin')}</Link>
      </p>
    </section>
  )
}
