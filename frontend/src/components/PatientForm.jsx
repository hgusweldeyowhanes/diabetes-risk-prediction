import { useState } from 'react'
import { useI18n } from '../i18n/I18nContext'

const DEFAULTS = {
  sex: 'F',
  pregnancies: 1,
  glucose: 120,
  blood_pressure: 70,
  skin_thickness: 20,
  insulin: 80,
  bmi: 28,
  diabetes_pedigree_function: 0.5,
  age: 40,
  waist_cm: '',
  physically_active: true,
  family_history: false,
}

export default function PatientForm({ onSubmit, loading }) {
  const { t } = useI18n()
  const [form, setForm] = useState(DEFAULTS)
  const male = form.sex === 'M'

  function update(key, value) {
    setForm((prev) => ({ ...prev, [key]: value }))
  }

  function handleSubmit(e) {
    e.preventDefault()
    onSubmit({
      ...form,
      pregnancies: male ? 0 : Number(form.pregnancies),
      glucose: Number(form.glucose),
      blood_pressure: Number(form.blood_pressure),
      skin_thickness: Number(form.skin_thickness),
      insulin: Number(form.insulin),
      bmi: Number(form.bmi),
      diabetes_pedigree_function: Number(form.diabetes_pedigree_function),
      age: Number(form.age),
      waist_cm: form.waist_cm === '' ? null : Number(form.waist_cm),
    })
  }

  const fields = [
    { key: 'glucose', label: t('glucose'), help: t('glucoseHelp'), min: 0, max: 400, step: 1 },
    { key: 'blood_pressure', label: t('bloodPressure'), help: t('bloodPressureHelp'), min: 0, max: 202, step: 1 },
    { key: 'skin_thickness', label: t('skinThickness'), help: t('skinThicknessHelp'), min: 0, max: 200, step: 1 },
    { key: 'insulin', label: t('insulin'), help: t('insulinHelp'), min: 0, max: 900, step: 1 },
    { key: 'bmi', label: t('bmi'), help: t('bmiHelp'), min: 0, max: 70, step: 0.1 },
    { key: 'diabetes_pedigree_function', label: t('dpf'), help: t('dpfHelp'), min: 0, max: 3, step: 0.001 },
    { key: 'age', label: t('age'), help: t('ageHelp'), min: 1, max: 120, step: 1 },
    { key: 'waist_cm', label: t('waist'), help: t('waistHelp'), min: 40, max: 200, step: 1, optional: true },
  ]

  return (
    <form className="patient-form" onSubmit={handleSubmit}>
      <div className="form-grid">
        <label className="field">
          <span>{t('sex')}</span>
          <select value={form.sex} onChange={(e) => update('sex', e.target.value)}>
            <option value="F">{t('sexF')}</option>
            <option value="M">{t('sexM')}</option>
            <option value="U">{t('sexU')}</option>
          </select>
        </label>
        {!male && (
          <label className="field">
            <span>{t('pregnancies')}</span>
            <input
              type="number"
              min={0}
              max={20}
              step={1}
              value={form.pregnancies}
              onChange={(e) => update('pregnancies', e.target.value)}
              required
            />
            <small className="field-help">{t('pregnanciesHelp')}</small>
          </label>
        )}
        {fields.map((f) => (
          <label key={f.key} className="field">
            <span>{f.label}</span>
            <input
              type="number"
              min={f.min}
              max={f.max}
              step={f.step}
              value={form[f.key]}
              onChange={(e) => update(f.key, e.target.value)}
              required={!f.optional}
            />
            <small className="field-help">{f.help}</small>
          </label>
        ))}
      </div>
      <div className="check-row">
        <label className="check">
          <input
            type="checkbox"
            checked={form.physically_active}
            onChange={(e) => update('physically_active', e.target.checked)}
          />
          {t('active')}
        </label>
        <label className="check">
          <input
            type="checkbox"
            checked={form.family_history}
            onChange={(e) => update('family_history', e.target.checked)}
          />
          {t('familyHx')}
        </label>
      </div>
      {male && <p className="notice">{t('maleWarning')}</p>}
      <button type="submit" className="btn primary" disabled={loading}>
        {loading ? t('scoring') : t('getScore')}
      </button>
    </form>
  )
}
