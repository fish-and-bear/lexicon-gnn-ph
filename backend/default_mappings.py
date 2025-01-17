# Default mappings used when config files don't exist
LANGUAGE_FAMILIES = {
    'Austronesian': {
        'Malayo-Polynesian': {
            'Philippine': {
                'Greater Central Philippine': {
                    'Central Philippine': [
                        'Tagálog', 'Bíkol', 'Sebwáno', 'Hiligaynón', 'Waráy'
                    ],
                    'Mansakan': [
                        'Mandayá', 'Kalagan'
                    ]
                },
                'Northern Luzon': {
                    'Cordilleran': [
                        'Ilokáno', 'Ifugáw', 'Kankanáëy', 'Balangáw', 'Gáddang', 'Isnë́g'
                    ],
                    'Cagayan Valley': [
                        'Ibanág', 'Itáwit', 'Ivatán'
                    ]
                },
                'Central Luzon': [
                    'Kapampángan', 'Sambali'
                ],
                'Manobo': [
                    'Binúkid', 'Agutaynë́n'
                ],
                'Danao': [
                    'Mëranáw', 'Magindanáw'
                ]
            }
        }
    },
    'Indo-European': {
        'Romance': ['Español', 'Portugués', 'French', 'Italian'],
        'Germanic': ['Ingles', 'German', 'Dutch'],
        'Indo-Iranian': ['Sanskrit', 'Hindi'],
        'Hellenic': ['Griego']
    },
    'Sino-Tibetan': {
        'Sinitic': ['Chinese', 'Chino']
    },
    'Afro-Asiatic': {
        'Semitic': ['Arabic', 'Hebrew']
    },
    'Historical': {
        'Classical': ['Latin'],
        'Ancient Philippine': ['Sinaunang Tagalog']
    }
}

REGIONAL_MAPPING = {
    'Luzon': {
        'Metro Manila': ['Tagálog'],
        'Central Luzon': ['Kapampángan', 'Sambali'],
        'Southern Luzon': ['Bíkol'],
        'Cordillera': ['Ifugáw', 'Kankanáëy', 'Balangáw', 'Gáddang', 'Isnë́g'],
        'Ilocos': ['Ilokáno'],
        'Cagayan Valley': ['Ibanág', 'Itáwit', 'Ivatán']
    },
    'Visayas': {
        'Western': ['Hiligaynón', 'Kinaráy-a', 'Aklánon'],
        'Central': ['Sebwáno', 'Waráy'],
        'Eastern': ['Waráy']
    },
    'Mindanao': {
        'Northern': ['Binúkid', 'Mandayá'],
        'Southern': ['Magindanáw', 'Mëranáw'],
        'Western': ['Súg', 'Tausug'],
        'Eastern': ['Mandayá', 'Mansaka']
    }
}

WRITING_SYSTEMS = {
    'Native Scripts': {
        'Baybayin': {
            'Languages': ['Tagálog', 'Sinaunang Tagalog', 'Bíkol', 'Ilokáno'],
            'Period': 'Pre-colonial to Early Spanish Period',
            'Status': 'Historical/Revival'
        },
        'Kulitan': {
            'Languages': ['Kapampángan'],
            'Period': 'Pre-colonial to Early Spanish Period',
            'Status': 'Historical/Limited Use'
        },
        'Tagbanwa': {
            'Languages': ['Tagbanwá'],
            'Period': 'Pre-colonial to Present',
            'Status': 'Limited Use'
        }
    },
    'Adapted Scripts': {
        'Latin': {
            'Languages': ['Tagálog', 'Sebwáno', 'Hiligaynón', 'Waráy', 'Ilokáno'],
            'Period': 'Spanish Period to Present',
            'Status': 'Primary'
        },
        'Arabic': {
            'Languages': ['Mëranáw', 'Magindanáw', 'Tausug'],
            'Period': 'Islamic Period to Present',
            'Status': 'Traditional/Limited'
        }
    }
}

LANGUAGE_STANDARDIZATION = {
    # Historical Tagalog variations
    'ST': 'Sinaunang Tagalog', 'STl': 'Sinaunang Tagalog',
    'old_tagalog': 'Sinaunang Tagalog', 'classical_tagalog': 'Sinaunang Tagalog',
    
    # Philippine languages - major
    'tl': 'Tagálog', 'tag': 'Tagálog', 'Tag': 'Tagálog',
    'bik': 'Bíkol', 'bcl': 'Bíkol', 'Bik': 'Bíkol',
    'war': 'Waráy', 'War': 'Waráy',
    'hil': 'Hiligaynón', 'Hil': 'Hiligaynón', 'Hik': 'Hiligaynón',
    'ceb': 'Sebwáno', 'Seb': 'Sebwáno',
    'klg': 'Kalagan', 'Klg': 'Kalagan',
    'tgb': 'Tagbanwá', 'Tgb': 'Tagbanwá', 'Tbw': 'Tagbanwá',
    
    # Philippine languages - Luzon
    'Ilo': 'Ilokáno', 'Ilk': 'Ilokáno',
    'Ifg': 'Ifugáw', 'Ifu': 'Ifugáw',
    'Isn': 'Isnë́g', 'Isg': 'Isnë́g',
    'Iva': 'Ivatán', 'Itw': 'Itáwit',
    'Kap': 'Kapampángan', 'pam': 'Kapampángan',
    'Gad': 'Gáddang',
    'Kan': 'Kankanáëy', 'Kbn': 'Kankanáëy',
    'Sbl': 'Sambali',
    
    # Philippine languages - Mindanao
    'Mag': 'Magindanáw', 'mdh': 'Magindanáw',
    'Mrw': 'Mëranáw', 'Mrs': 'Mëranáw',
    'Mnd': 'Mandayá', 'Man': 'Mandayá',
    'Bin': 'Binúkid',
    'Agu': 'Agutaynë́n',
    'Tau': 'Tausug', 'Tsg': 'Tausug', 'Súg': 'Tausug',
    
    # Foreign languages
    'en': 'Ingles', 'eng': 'Ingles', 'Ing': 'Ingles',
    'es': 'Español', 'spa': 'Español', 'Esp': 'Español',
    'fr': 'French', 'fra': 'French', 'Fre': 'French',
    'de': 'German', 'deu': 'German', 'Ger': 'German',
    'la': 'Latin', 'lat': 'Latin', 'Lat': 'Latin',
    'el': 'Griego', 'grc': 'Griego', 'Gre': 'Griego',
    'pt': 'Portugués', 'por': 'Portugués', 'Por': 'Portugués',
    'nl': 'Dutch', 'dut': 'Dutch', 'nld': 'Dutch',
    'sa': 'Sanskrit', 'san': 'Sanskrit', 'Skr': 'Sanskrit',
    'hi': 'Hindi', 'hin': 'Hindi', 'Hin': 'Hindi',
    'zh': 'Chinese', 'cmn': 'Chinese', 'Chi': 'Chinese',
    'ar': 'Arabic', 'ara': 'Arabic', 'Ara': 'Arabic',
    'he': 'Hebrew', 'heb': 'Hebrew', 'Heb': 'Hebrew',
    'it': 'Italian', 'ita': 'Italian', 'Ita': 'Italian'
}

VALID_CODES = set((
    # Main Philippine Languages
    'Tagálog', 'Bíkol', 'Sebwáno', 'Hiligaynón', 'Waráy',
    'Kalagan', 'Mandayá', 'Ilokáno', 'Ifugáw', 'Kankanáëy',
    'Balangáw', 'Gáddang', 'Isnë́g', 'Ibanág', 'Itáwit',
    'Ivatán', 'Kapampángan', 'Sambali', 'Binúkid', 'Agutaynë́n',
    'Mëranáw', 'Magindanáw', 'Sinaunang Tagalog', 'Tagbanwá',
    'Tausug', 'Súg', 'Kinaráy-a', 'Aklánon', 'Mansaka',
    
    # Foreign Languages
    'Ingles', 'Español', 'French', 'German', 'Latin',
    'Griego', 'Portugués', 'Dutch', 'Sanskrit', 'Hindi',
    'Chinese', 'Chino', 'Arabic', 'Hebrew', 'Italian',
    
    # Alternative codes - Philippine
    'tl', 'tag', 'Tag', 'bik', 'bcl', 'Bik',
    'ceb', 'Seb', 'hil', 'Hil', 'Hik',
    'war', 'War', 'klg', 'Klg', 'Mnd', 'Man',
    'Ilo', 'Ilk', 'Ifg', 'Ifu', 'Isn', 'Isg',
    'Iva', 'Itw', 'Kap', 'pam', 'Gad',
    'Kan', 'Kbn', 'Sbl', 'Mag', 'mdh',
    'Mrw', 'Mrs', 'Bin', 'Agu', 'Knr',
    'ST', 'STl', 'tgb', 'Tgb', 'Tbw',
    'Tau', 'Tsg', 'Súg', 'Akl', 'Msk',
    
    # Alternative codes - Foreign
    'en', 'eng', 'Ing', 'es', 'spa', 'Esp',
    'fr', 'fra', 'Fre', 'de', 'deu', 'Ger',
    'la', 'lat', 'Lat', 'el', 'grc', 'Gre',
    'pt', 'por', 'Por', 'nl', 'dut', 'nld',
    'sa', 'san', 'Skr', 'hi', 'hin', 'Hin',
    'zh', 'cmn', 'Chi', 'ar', 'ara', 'Ara',
    'he', 'heb', 'Heb', 'it', 'ita', 'Ita'
))

# Add standardization entries for new languages
LANGUAGE_STANDARDIZATION.update({
    # Additional Philippine languages
    'Knr': 'Kinaráy-a', 'Kin': 'Kinaráy-a',
    'Akl': 'Aklánon', 'Akn': 'Aklánon',
    'Msk': 'Mansaka', 'Man': 'Mansaka',
    
    # Ensure all regional languages have standardization
    'Kinaráy-a': 'Kinaráy-a',
    'Aklánon': 'Aklánon',
    'Mansaka': 'Mansaka'
})