export default function ResultsDisplay({ result }) {
  if (!result) {
    return (
      <p className="muted">
        No result yet. Run a prediction from the Predict page.
      </p>
    )
  }

  const guidance = result.clinical_guidance || []
  const cfs = result.counterfactuals || []
  const local = result.local_explanation || []

  return (
    <div className="results-stack">
      {guidance.length > 0 && (
        <section className="panel">
          <h3>Guidance</h3>
          <ul>
            {guidance.map((g, i) => (
              <li key={i}>{g}</li>
            ))}
          </ul>
        </section>
      )}

      {local.length > 0 && (
        <section className="panel">
          <h3>Local drivers</h3>
          <table>
            <thead>
              <tr>
                <th>Feature</th>
                <th>Value</th>
                <th>z</th>
                <th>Role</th>
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
          <h3>What-if scenarios</h3>
          <table>
            <thead>
              <tr>
                <th>Scenario</th>
                <th>Probability</th>
                <th>Δ</th>
                <th>Tier</th>
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
