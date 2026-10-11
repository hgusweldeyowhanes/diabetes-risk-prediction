import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'
import { useI18n } from '../i18n/I18nContext'

export default function Login() {
  const { t } = useI18n()
  const { login } = useAuth()
  const navigate = useNavigate()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e) {
    e.preventDefault()
    setLoading(true)
    setError('')
    try {
      await login(username, password)
      navigate('/history')
    } catch (err) {
      setError(err.message || 'Login failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="page narrow">
      <h1>{t('loginTitle')}</h1>
      {error && <p className="error">{error}</p>}
      <form className="patient-form" onSubmit={handleSubmit}>
        <label className="field">
          <span>{t('username')}</span>
          <input value={username} onChange={(e) => setUsername(e.target.value)} required />
        </label>
        <label className="field">
          <span>{t('password')}</span>
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required />
        </label>
        <button className="btn primary" type="submit" disabled={loading}>
          {t('submitLogin')}
        </button>
      </form>
      <p className="muted">
        {t('needAccount')} <Link to="/register">{t('navRegister')}</Link>
      </p>
    </section>
  )
}
