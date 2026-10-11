import { useEffect, useState } from 'react'
import { api } from '../services/api'
import { useI18n } from '../i18n/I18nContext'

export default function Analytics() {
  const { t } = useI18n()
  const [model, setModel] = useState(null)
  const [health, setHealth] = useState(null)
  const [error, setError] = useState('')
  const [techniques, setTechniques] = useState(null)

  useEffect(() => {
    try {
      const raw = sessionStorage.getItem('last_prediction')
      if (raw) setTechniques(JSON.parse(raw).techniques)
    } catch {
      /* ignore */
    }

    Promise.all([api.health(), api.activeModel().catch(() => null)])
      .then(([h, m]) => {
        setHealth(h)
        setModel(m)
      })
      .catch((e) => setError(e.message))
  }, [])

  const plots = (() => {
    try {
      const r = JSON.parse(sessionStorage.getItem('last_prediction') || '{}')
      return r.analytics?.plots || {}
    } catch {
      return {}
    }
  })()

  return (
    <section className="page">
      <h1>{t('insightsTitle')}</h1>
      <p className="lede short">{t('insightsLede')}</p>
      {error && <p className="error">{error}</p>}
      <div className="panel-grid">
        <div className="panel">
          <h3>{t('service')}</h3>
          <p>
            {t('status')}: {health?.status || '…'}
          </p>
          <p>
            {t('modelReady')}: {String(health?.model_ready)}
          </p>
        </div>
        <div className="panel">
          <h3>{t('activeModel')}</h3>
          {model ? (
            <>
              <p>{model.name}</p>
              <p>
                Acc {((model.accuracy || 0) * 100).toFixed(1)}% · AUC {(model.roc_auc || 0).toFixed(3)}
              </p>
            </>
          ) : (
            <p className="muted">{t('trainHint')}</p>
          )}
        </div>
      </div>

      {techniques?.implemented && (
        <div className="panel">
          <h3>{t('techniques')}</h3>
          <ul className="compact">
            {techniques.implemented.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      )}

      {Object.keys(plots).length > 0 && (
        <div className="plot-grid">
          {Object.entries(plots)
            .filter(([, v]) => v)
            .map(([name, b64]) => (
              <figure key={name} className="panel">
                <figcaption>{name.replace(/_/g, ' ')}</figcaption>
                <img src={`data:image/png;base64,${b64}`} alt={name} />
              </figure>
            ))}
        </div>
      )}
    </section>
  )
}
