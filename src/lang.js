export const LANG_TO_ID = {
  "en-US": [0,   "English (US)"],
  "en-GB": [1,   "English (UK)"],
  "es-ES": [2,   "Español (Spain)"],
  "es":    [3,   "Español (Latin America)"],
  "zh-CN": [4,   "中文"],
  "hi":    [6,   "हिन्दी"],
  "ar":    [7,   "العربية"],
  "fr":    [8,   "Français (France)"],
  "de":    [9,   "Deutsch"],
  "ja":    [10,  "日本語"],
  "ru":    [11,  "Русский"],
  "pt-BR": [12,  "Português (Brazil)"],
  "pt":    [13,  "Português (Portugal)"],
  "ko":    [14,  "한국어"],
  "it":    [15,  "Italiano"],
  "nl":    [16,  "Nederlands"],
  "pl":    [17,  "Polski"],
  "tr":    [18,  "Türkçe"],
  "uk":    [19,  "Українська"],
  "ro":    [20,  "Română"],
  "el":    [21,  "Ελληνικά"],
  "cs":    [22,  "Čeština"],
  "hu":    [23,  "Magyar"],
  "sv":    [24,  "Svenska"],
  "da":    [25,  "Dansk"],
  "fi":    [26,  "Suomi"],
  "sk":    [28,  "Slovenčina"],
  "hr":    [29,  "Hrvatski"],
  "bg":    [30,  "Български"],
  "lt":    [31,  "Lietuvių"],
  "th":    [32,  "ไทย"],
  "vi":    [33,  "Tiếng Việt"],
  "et":    [60,  "Eesti"],
  "lv":    [61,  "Latviešu"],
  "sl":    [62,  "Slovenščina"],
  "he":    [64,  "עברית"],
  "fr-CA": [100, "Français (Canada)"],
  "auto":  [101, "auto"],
  "mt":    [102, "Malti"],
  "nb":    [103, "Norsk Bokmål"],
  "nn":    [104, "Norsk Nynorsk"],
};

export function langName(code) {
  const entry = LANG_TO_ID[code];
  return entry ? entry[1] : null;
}

export function langId(code) {
  const entry = LANG_TO_ID[code];
  return entry ? entry[0] : null;
}
