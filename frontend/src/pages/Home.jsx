import { Link } from 'react-router-dom'

export default function Home() {
  return (
    <section className="home">
      <div className="home-copy">
        <p className="eyebrow">Know your risk. Act early.</p>
        <h1>GlycoPath</h1>
        <p className="lede">
          A clear diabetes risk score from your clinical markers—with
          plain-language guidance and “what if I change…?” scenarios you can
          discuss with a clinician.
        </p>
        <div className="cta-row">
          <Link className="btn primary" to="/predict">
            Assess my risk
          </Link>
          <Link className="btn ghost" to="/analytics">
            How the model works
          </Link>
        </div>
      </div>
      <div className="home-visual" aria-hidden="true" />
    </section>
  )
}
