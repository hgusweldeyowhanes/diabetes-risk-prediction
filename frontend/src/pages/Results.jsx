import { useEffect, useState } from 'react'
import RiskDashboard from '../components/RiskDashboard'
import ResultsDisplay from '../components/ResultsDisplay'

export default function Results() {
  const [result, setResult] = useState(null)

  useEffect(() => {
    try {
      const raw = sessionStorage.getItem('last_prediction')
      if (raw) setResult(JSON.parse(raw))
    } catch {
      setResult(null)
    }
  }, [])

  return (
    <section className="page">
      <h1>Your risk report</h1>
      <p className="lede short">
        Scored probability, risk tier, and next-step style guidance.
      </p>
      <RiskDashboard result={result} />
      <ResultsDisplay result={result} />
    </section>
  )
}
