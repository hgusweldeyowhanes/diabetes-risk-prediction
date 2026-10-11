import { LANGS } from '../i18n/translations'
import { useI18n } from '../i18n/I18nContext'

export default function LangSwitch() {
  const { lang, setLang } = useI18n()
  return (
    <div className="lang-switch" role="group" aria-label="Language">
      {LANGS.map((item) => (
        <button
          key={item.id}
          type="button"
          className={lang === item.id ? 'is-on' : ''}
          onClick={() => setLang(item.id)}
        >
          {item.label}
        </button>
      ))}
    </div>
  )
}
