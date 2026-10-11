import { useI18n } from '../i18n/I18nContext'

export default function ResultsDisplay({ result }) {
  const { t } = useI18n()
  if (!result) {
    return <p className="muted">{t('noResult')}</p>
  }

  const guidance = result.clinical_guidance || []
  const cfs = result.counterfactuals || []
  const local = result.local_explanation || []

  return (
    <div className="results-stack">
      {guidance.length > 0 && (
        <section className="panel">
          <h3>{t('guidance')}</h3>
          <ul>
            {guidance.map((g, i) => (
              <li key={i}>{g}</li>
            ))}
          </ul>
        </section>
      )}

      {local.length > 0 && (
        <section className="panel">
          <h3>{t('drivers')}</h3>
          <table>
            <thead>
              <tr>
                <th>{t('feature')}</th>
                <th>{t('value')}</th>
                <th>{t('z')}</th>
                <th>{t('role')}</th>
              </tr>
            </thead>
            <tbody>
              {local.map((row) => (
                <tr key={row.feature}>
                  <td>{row.feature}</td>
                  <td>{Number(row.value).toFixed(2)}</td>
                  <td>{Number(row.z_score).toFixed(2)}</td>
                  <td>{row.direction}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {cfs.length > 0 && (
        <section className="panel">
          <h3>{t('whatIf')}</h3>
          <table>
            <thead>
              <tr>
                <th>{t('scenario')}</th>
                <th>{t('probability')}</th>
                <th>{t('delta')}</th>
                <th>{t('tier')}</th>
              </tr>
            </thead>
            <tbody>
              {cfs.map((cf) => (
                <tr key={cf.scenario}>
                  <td>{cf.scenario}</td>
                  <td>{cf.probability}%</td>
                  <td className={cf.delta < 0 ? 'ok' : 'danger'}>{cf.delta}%</td>
                  <td>{cf.risk_level}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </div>
  )
}
