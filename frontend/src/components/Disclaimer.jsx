import { useI18n } from '../i18n/I18nContext'

export default function Disclaimer({ compact = false }) {
  const { t } = useI18n()
  return (
    <aside className={`disclaimer ${compact ? 'compact' : ''}`} role="note">
      <strong>{t('disclaimerTitle')}</strong>
      <p>{t('disclaimerBody')}</p>
    </aside>
  )
}
