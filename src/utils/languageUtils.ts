// src/utils/languageUtils.ts

// Define known language codes and their display names
const languageNameMap: Record<string, string> = {
  // Major Philippine Languages
  tgl: 'Tagalog',
  ceb: 'Cebuano',
  ilo: 'Ilocano',
  hil: 'Hiligaynon',
  war: 'Waray',
  bik: 'Bikol', // Central Bikol as default
  pam: 'Kapampangan',
  pag: 'Pangasinan',

  // Specific Philippine Languages from Marayum/Other Sources
  bku: 'Buhid',
  hnn: 'Hanunoo',
  bno: 'Asi', // Updated code for Asi
  onx: 'Onhan',
  ibg: 'Ibanag',
  iro: 'Iranon',
  krj: 'Kinaray-a',
  sgd: 'Surigaonon',
  akl: 'Aklanon',
  msb: 'Masbatenyo',
  cbk: 'Chavacano',
  mbt: 'Manobo', // Default Manobo
  abd: 'Manide',
  mdh: 'Maguindanaon',
  ivv: 'Ivatan',
  itv: 'Itawis',
  isd: 'Isneg',
  ifk: 'Ifugao',
  gad: 'Gaddang',
  cyo: 'Cuyonon',
  bpr: 'Blaan', // Default Blaan

  // Common Foreign Languages
  en: 'English',
  es: 'Spanish',
  la: 'Latin',
  ms: 'Malay',
  ar: 'Arabic',
  zh: 'Chinese',
  ja: 'Japanese',
  sa: 'Sanskrit',
  fr: 'French',
  de: 'German',
  pt: 'Portuguese',

  // Special Codes
  und: 'Undetermined', // Handle 'und' explicitly
  unc: 'Unclassified', // Handle potential 'unc' if it ever appears
};

/**
 * Gets a user-friendly display name for a given language code.
 * @param code - The language code (e.g., 'tgl', 'en', 'und').
 * @returns The display name (e.g., 'Tagalog', 'English', 'Undetermined') or the uppercased code if not found.
 */
export const getLanguageDisplayName = (code: string | null | undefined): string => {
  if (!code) {
    return 'Unknown'; // Return 'Unknown' for null, undefined, or empty string
  }
  const lowerCaseCode = code.toLowerCase();
  return languageNameMap[lowerCaseCode] || code.toUpperCase(); // Fallback to uppercased code
}; 