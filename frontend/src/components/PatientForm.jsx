import { useState } from 'react'

const FIELDS = [
  { key: 'pregnancies', label: 'Pregnancies', min: 0, max: 20, step: 1 },
  { key: 'glucose', label: 'Glucose', min: 0, max: 400, step: 1 },
  { key: 'blood_pressure', label: 'Blood pressure', min: 0, max: 202, step: 1 },
  { key: 'skin_thickness', label: 'Skin thickness', min: 0, max: 200, step: 1 },
  { key: 'insulin', label: 'Insulin', min: 0, max: 900, step: 1 },
  { key: 'bmi', label: 'BMI', min: 0, max: 70, step: 0.1 },
  {
    key: 'diabetes_pedigree_function',
    label: 'Diabetes pedigree function',
    min: 0,
    max: 3,
    step: 0.001,
  },
  { key: 'age', label: 'Age', min: 1, max: 120, step: 1 },
]

const DEFAULTS = {
  pregnancies: 1,
  glucose: 120,
  blood_pressure: 70,
  skin_thickness: 20,
  insulin: 80,
  bmi: 28,
  diabetes_pedigree_function: 0.5,
  age: 40,
}

export default function PatientForm({ onSubmit, loading }) {
  const [form, setForm] = useState(DEFAULTS)

  function update(key, value) {
    setForm((prev) => ({ ...prev, [key]: value === '' ? '' : Number(value) }))
  }

  function handleSubmit(e) {
    e.preventDefault()
    onSubmit(form)
  }

  return (
    <form className="patient-form" onSubmit={handleSubmit}>
      <div className="form-grid">
        {FIELDS.map((f) => (
          <label key={f.key} className="field">
            <span>{f.label}</span>
            <input
              type="number"
              min={f.min}
              max={f.max}
              step={f.step}
              value={form[f.key]}
              onChange={(e) => update(f.key, e.target.value)}
              required
            />
          </label>
        ))}
      </div>
      <button type="submit" className="btn primary" disabled={loading}>
        {loading ? 'Scoring risk…' : 'Get my risk score'}
      </button>
    </form>
  )
}
