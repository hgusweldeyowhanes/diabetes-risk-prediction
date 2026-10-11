import { Link } from 'react-router-dom'
import Disclaimer from '../components/Disclaimer'
import { useI18n } from '../i18n/I18nContext'

export default function Home() {
  const { t } = useI18n()
  return (
    <section className="home">
      <div className="home-copy">
        <p className="eyebrow">{t('eyebrow')}</p>
        <h1>{t('homeTitle')}</h1>
        <p className="lede">{t('homeLede')}</p>
        <div className="cta-row">
          <Link className="btn primary" to="/predict">
            {t('ctaAssess')}
          </Link>
          <Link className="btn ghost" to="/analytics">
            {t('ctaInsights')}
          </Link>
        </div>
        <Disclaimer />
      </div>
      <div className="home-visual" aria-hidden="true" />
    </section>
  )
}
