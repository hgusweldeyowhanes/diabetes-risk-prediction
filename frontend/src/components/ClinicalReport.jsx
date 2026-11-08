import Disclaimer from './Disclaimer'
import ResultsDisplay from './ResultsDisplay'
import RiskDashboard from './RiskDashboard'
import { useI18n } from '../i18n/I18nContext'

function inputRows(result) {
  const raw = result?.input_data || {}
  const pairs = [
    ['Pregnancies', raw.Pregnancies ?? result?.pregnancies],
    ['Glucose', raw.Glucose ?? result?.glucose],
    ['BloodPressure', raw.BloodPressure ?? result?.blood_pressure],
    ['SkinThickness', raw.SkinThickness ?? result?.skin_thickness],
    ['Insulin', raw.Insulin ?? result?.insulin],
    ['BMI', raw.BMI ?? result?.bmi],
    ['DiabetesPedigreeFunction', raw.DiabetesPedigreeFunction ?? result?.diabetes_pedigree_function],
    ['Age', raw.Age ?? result?.age],
  ]
  return pairs.filter(([, v]) => v !== undefined && v !== null)
}

export default function ClinicalReport({ result, meta = {} }) {
  const { t } = useI18n()
  if (!result) {
    return <p className="muted">{t('noResult')}</p>
  }

  const inputs = inputRows(result)
  const when = meta.created_at || result.created_at

  return (
    <article className="clinical-report">
      <header className="report-header">
        <div>
          <p className="eyebrow">{t('brand')}</p>
          <h1>{t('reportTitle')}</h1>
          <p className="lede short">{t('reportLede')}</p>
        </div>
        <div className="report-meta">
          {when && (
            <p>
              <strong>{t('when')}:</strong> {new Date(when).toLocaleString()}
            </p>
          )}
          {meta.patientName && (
            <p>
              <strong>{t('patient')}:</strong> {meta.patientName}
            </p>
          )}
          {result.prediction_id || result.id ? (
            <p>
              <strong>ID:</strong> {result.prediction_id || result.id}
            </p>
          ) : null}
        </div>
      </header>

      <Disclaimer compact />
      <RiskDashboard result={result} />

      {inputs.length > 0 && (
        <section className="panel">
          <h3>{t('inputMarkers')}</h3>
          <table>
            <thead>
              <tr>
                <th>{t('feature')}</th>
                <th>{t('value')}</th>
              </tr>
            </thead>
            <tbody>
              {inputs.map(([feature, value]) => (
                <tr key={feature}>
                  <td>{feature}</td>
                  <td>{value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      <ResultsDisplay result={result} />
      {(result.dataset_note || meta.dataset_note) && (
        <p className="muted small">{result.dataset_note || meta.dataset_note}</p>
      )}
      <p className="muted small report-footer">{t('reportFooter')}</p>
    </article>
  )
}
