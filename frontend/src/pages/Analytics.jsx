import { useEffect, useState } from 'react'
import { api } from '../services/api'

export default function Analytics() {
  const [model, setModel] = useState(null)
  const [health, setHealth] = useState(null)
  const [error, setError] = useState('')
  const [techniques, setTechniques] = useState(null)

  useEffect(() => {
    let raw
    try {
      raw = sessionStorage.getItem('last_prediction')
      if (raw) {
        const r = JSON.parse(raw)
        setTechniques(r.techniques)
      }
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
      <h1>Model insights</h1>
      <p className="lede short">
        Live service status, training metrics, and the methods behind GlycoPath.
      </p>
      {error && <p className="error">{error}</p>}
      <div className="panel-grid">
        <div className="panel">
          <h3>Service</h3>
          <p>Status: {health?.status || '…'}</p>
          <p>Model ready: {String(health?.model_ready)}</p>
        </div>
        <div className="panel">
          <h3>Active model</h3>
          {model ? (
            <>
              <p>{model.name}</p>
              <p>
                Acc {((model.accuracy || 0) * 100).toFixed(1)}% · AUC{' '}
                {(model.roc_auc || 0).toFixed(3)}
              </p>
            </>
          ) : (
            <p className="muted">Train once via API prediction or manage.py train_model</p>
          )}
        </div>
      </div>

      {techniques?.implemented && (
        <div className="panel">
          <h3>Techniques in this build</h3>
          <ul className="compact">
            {techniques.implemented.map((t) => (
              <li key={t}>{t}</li>
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
