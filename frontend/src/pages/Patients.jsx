import { useEffect, useState } from 'react'
import { Link, Navigate } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'
import { useI18n } from '../i18n/I18nContext'
import { api } from '../services/api'

const EMPTY = {
  first_name: '',
  last_name: '',
  external_id: '',
  sex: 'U',
  date_of_birth: '',
  notes: '',
}

export default function Patients() {
  const { t } = useI18n()
  const { user, ready } = useAuth()
  const [rows, setRows] = useState([])
  const [form, setForm] = useState(EMPTY)
  const [open, setOpen] = useState(false)
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)

  function load() {
    api
      .listPatients()
      .then(setRows)
      .catch((e) => setError(e.message))
  }

  useEffect(() => {
    if (user) load()
  }, [user])

  if (!ready) return null
  if (!user) return <Navigate to="/login" replace />

  async function handleCreate(e) {
    e.preventDefault()
    setSaving(true)
    setError('')
    try {
      const payload = {
        ...form,
        date_of_birth: form.date_of_birth || null,
      }
      await api.createPatient(payload)
      setForm(EMPTY)
      setOpen(false)
      load()
    } catch (err) {
      setError(err.message)
    } finally {
      setSaving(false)
    }
  }

  function displayName(p) {
    const name = `${p.first_name || ''} ${p.last_name || ''}`.trim()
    return name || p.external_id || `#${p.id}`
  }

  return (
    <section className="page">
      <div className="page-head">
        <div>
          <h1>{t('patientsTitle')}</h1>
          <p className="lede short">{t('patientsLede')}</p>
        </div>
        <button type="button" className="btn primary no-print" onClick={() => setOpen((v) => !v)}>
          {open ? t('cancel') : t('addPatient')}
        </button>
      </div>
      {error && <p className="error">{error}</p>}

      {open && (
        <form className="panel patient-create" onSubmit={handleCreate}>
          <div className="form-grid">
            <label className="field">
              <span>{t('firstName')}</span>
              <input
                value={form.first_name}
                onChange={(e) => setForm({ ...form, first_name: e.target.value })}
              />
            </label>
            <label className="field">
              <span>{t('lastName')}</span>
              <input
                value={form.last_name}
                onChange={(e) => setForm({ ...form, last_name: e.target.value })}
              />
            </label>
            <label className="field">
              <span>{t('externalId')}</span>
              <input
                value={form.external_id}
                onChange={(e) => setForm({ ...form, external_id: e.target.value })}
                placeholder={t('externalIdHelp')}
              />
            </label>
            <label className="field">
              <span>{t('sex')}</span>
              <select value={form.sex} onChange={(e) => setForm({ ...form, sex: e.target.value })}>
                <option value="F">{t('sexF')}</option>
                <option value="M">{t('sexM')}</option>
                <option value="U">{t('sexU')}</option>
              </select>
            </label>
            <label className="field">
              <span>{t('dateOfBirth')}</span>
              <input
                type="date"
                value={form.date_of_birth}
                onChange={(e) => setForm({ ...form, date_of_birth: e.target.value })}
              />
            </label>
          </div>
          <label className="field">
            <span>{t('notes')}</span>
            <textarea
              rows={3}
              value={form.notes}
              onChange={(e) => setForm({ ...form, notes: e.target.value })}
            />
          </label>
          <button type="submit" className="btn primary" disabled={saving}>
            {saving ? t('saving') : t('savePatient')}
          </button>
        </form>
      )}

      {rows.length === 0 ? (
        <p className="muted">{t('patientsEmpty')}</p>
      ) : (
        <div className="data-table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>{t('patient')}</th>
                <th>{t('externalId')}</th>
                <th>{t('assessments')}</th>
                <th>{t('mlRisk')}</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {rows.map((p) => (
                <tr key={p.id}>
                  <td>{displayName(p)}</td>
                  <td>{p.external_id || '—'}</td>
                  <td>{p.prediction_count ?? 0}</td>
                  <td>
                    {p.latest_risk_level || '—'}
                    {p.latest_probability != null
                      ? ` · ${Math.round((p.latest_probability <= 1 ? p.latest_probability * 100 : p.latest_probability))}%`
                      : ''}
                  </td>
                  <td>
                    <Link to={`/patients/${p.id}`}>{t('openDossier')}</Link>
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
