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
    
    # English variations
    'en': 'Ingles', 'eng': 'Ingles', 'english': 'Ingles', 
    'lng': 'Ingles', 'Ing': 'Ingles',
    
    # Spanish variations
    'es': 'Español', 'spa': 'Español', 'spanish': 'Español',
    'Esp': 'Español',
    
    # Chinese variations
    'zh': 'Chinese', 'cmn': 'Chinese', 'Chi': 'Chinese',
    'Tsi': 'Chinese', 'Tsino': 'Chinese', 'Chino': 'Chinese',
    
    # Cebuano variations
    'ceb': 'Sebwáno', 'Seb': 'Sebwáno', 'Cebuano': 'Sebwáno',
    'bisaya': 'Sebwáno', 'Bis': 'Sebwáno',
    
    # Russian variations
    'ru': 'Russian', 'Rus': 'Russian', 'rus': 'Russian',
    
    # Hindi variations
    'hi': 'Hindi', 'Hin': 'Hindi', 'hin': 'Hindi',
    
    # Japanese variations
    'ja': 'Japanese', 'jp': 'Japanese', 'jpn': 'Japanese',
    'Jap': 'Japanese',
    
    # Arabic variations
    'ar': 'Arabic', 'ara': 'Arabic', 'Ara': 'Arabic',
    
    # Sanskrit variations
    'sa': 'Sanskrit', 'san': 'Sanskrit', 'San': 'Sanskrit',
    'Skr': 'Sanskrit',
    
    # Latin variations
    'la': 'Latin', 'lat': 'Latin', 'Lat': 'Latin',
    
    # Greek variations
    'el': 'Griego', 'grc': 'Griego', 'Gre': 'Griego',
    'Gri': 'Griego',
    
    # German variations
    'de': 'German', 'deu': 'German', 'ger': 'German',
    'Ger': 'German',
    
    # French variations
    'fr': 'French', 'fra': 'French', 'fre': 'French',
    'Fre': 'French',
    
    # Turkish variations
    'tr': 'Turkish', 'tur': 'Turkish', 'Tur': 'Turkish',
    
    # Hebrew variations
    'he': 'Hebrew', 'heb': 'Hebrew', 'Heb': 'Hebrew',
    
    # Indonesian/Malay variations
    'id': 'Bahása', 'ind': 'Bahása', 'Ind': 'Bahása',
    'ms': 'Malay', 'msa': 'Malay', 'Mal': 'Malay',
    
    # Philippine languages - major
    'tl': 'Tagálog', 'tag': 'Tagálog', 'Tag': 'Tagálog',
    'bik': 'Bíkol', 'bcl': 'Bíkol', 'Bik': 'Bíkol',
    'war': 'Waráy', 'War': 'Waráy',
    'hil': 'Hiligaynón', 'Hil': 'Hiligaynón', 'Hik': 'Hiligaynón',
    'pag': 'Pangasinán', 'Pan': 'Pangasinán', 'Png': 'Pangasinán',
    
    # Philippine languages - Mindanao
    'Mag': 'Magindanáw', 'mdh': 'Magindanáw', 'Mgd': 'Magindanáw', 'Mgn': 'Magindanáw',
    'Mrw': 'Mëranáw', 'Mrs': 'Mëranáw', 'meranao': 'Mëranáw', 'Mar': 'Mëranáw',
    'Kuy': 'Kuyunón', 'kuy': 'Kuyunón',
    'Tbn': 'Tagbanwá', 'Tagbanwá': 'Tagbanwá', 'Tbo': 'Tagbanwá', 'Tbw': 'Tagbanwá',
    'Mnd': 'Mandayá', 'Man': 'Mandayá',
    'Sub': 'Subanen', 'Súg': 'Tausug', 'Tau': 'Tausug',
    'Tgb': 'Tagabanwa', 'Tgk': 'Tagakaulo',
    
    # Philippine languages - Luzon
    'Ilo': 'Ilokáno', 'Ilk': 'Ilokáno',
    'Ifg': 'Ifugáw', 'Ifu': 'Ifugáw',
    'Isn': 'Isnë́g', 'Isg': 'Isnë́g',
    'Iva': 'Ivatán', 'Itw': 'Itáwit',
    'Kap': 'Kapampángan', 'pam': 'Kapampángan', 'Pny': 'Kapampángan',
    'Ibl': 'Ibalóy', 'Ibg': 'Ibanág', 'Ibn': 'Ibanág',
    'Iby': 'Ibatan', 'Itn': 'Itneg', 'Ilt': 'Ilongot',
    'Igo': 'Igorot', 'Iwa': 'Iwatan',
    
    # Philippine languages - Other
    'Agt': 'Agta', 'Agu': 'Agutaynë́n',
    'Akl': 'Aklánon', 'Bal': 'Balangáw',
    'Buh': 'Búhid', 'Gad': 'Gáddang',
    'Kan': 'Kankanáëy', 'Kal': 'Kalíngga', 'Kbn': 'Kankanáëy',
    'Btk': 'Batakan', 'Bat': 'Batak', 'Btg': 'Batangan',
    'Bin': 'Binúkid', 'Bag': 'Bagóbo', 'Buk': 'Bukidnon',
    
    # Manobo variations
    'Mnb': 'Manobo', 'Mny': 'Manobo', 'Mbk': 'Manobo',
    
    # Mangyan variations
    'Mgk': 'Mangyán', 'Mng': 'Mangyán',
    
    # Additional Philippine ethnic groups
    'Aby': 'Abaknon', 'Akn': 'Akaean',
    'Apa': 'Apayao', 'Ayt': 'Ayta',
    'Baj': 'Bajau', 'Ben': 'Benguet',
    'Bil': 'Bilaan', 'Boh': 'Boholano',
    'Bon': 'Bontok', 'Cha': 'Chavacano',
    'Cor': 'Cordilleran', 'Cot': 'Cotabato',
    'Csi': 'Coastal', 'Cuy': 'Cuyunon',
    'Dum': 'Dumagat', 'Dus': 'Dusun',
    'Han': 'Hanunoo', 'Hgn': 'Higaonon',
    'Hwi': 'Hawaiian', 'Kay': 'Kayapa',
    'Kem': 'Keley-i', 'Klg': 'Kalinga',
    'Kol': 'Kolbulano', 'Krw': 'Karaw',
    'Mad': 'Madalum', 'Mam': 'Mambabuid',
    'Min': 'Minangkabau', 'Mmw': 'Mamanwa',
    'Mns': 'Mansaka', 'Mts': 'Matigsalug',
    'Mwr': 'Marwari', 'Pal': 'Palawan',
    'Plw': 'Palawano', 'Sam': 'Samar',
    'Sbl': 'Sambali', 'Sma': 'Samar',
    'Sml': 'Samal', 'Sul': 'Sulu',
    'Tin': 'Tinggian', 'Tir': 'Tiruray',
    'Tng': 'Tinguian', 'Tua': 'Tuali',
    'Yak': 'Yakan', 'Yógad': 'Yogad',
    'Zam': 'Zambal',
    
    # Other languages and variations
    'AmI': 'Ami', 'Eva': 'Evangelical',
    'Exp': 'Expressive', 'Gui': 'Guinaang',
    'Kor': 'Korean', 'Mex': 'Mexican',
    'Noe': 'Northern', 'Nor': 'Norwegian',
    'Por': 'Portugués', 'Rom': 'Romanian',
    'Sco': 'Scottish', 'Snk': 'Sankrityayan',
    'Swa': 'Swahili', 'Tha': 'Thai',
    'Tib': 'Tibetan', 'Zoo': 'Zoological'
}

VALID_CODES = set((
    'Aby','Agt','Agu','Agutaynë́n','Akl','Aklánon','Akn','AmI','Apa','Ara','Arabic',
    'Ata','Ayt','Bag','Bahása','Baj','Bal','Balangáw','Bat','Ben','Bik','Bil',
    'Binúkid','Bis','Boh','Bon','Btg','Btk','Buh','Buk','Bíkol','Búhid','Cha','Chi',
    'Chino','Cor','Cot','Csi','Cuy','Dum','Dus','Dutch','Eso','Esp','Español','Eva',
    'Exp','Fre','French','Gad','Ger','German','Gre','Gri','Griego','Gui','Gáddang',
    'Han','Heb','Hebrew','Hgn','Higa-ónon','Hik','Hil','Hiligaynón','Hin','Hindi',
    'Hwi','Iba','Ibalóy','Ibanág','Ibg','Ibn','Iby','Ifu','Ifugáw','Igo','Ilk','Ilo',
    'Ilokáno','Ilt','Ind','Ing','Ingles','Isg','Isn','Isnë́g','Isp','Ita','Italian',
    'Itn','Itw','Itáwit','Iva','Ivatán','Iwa','Jap','Japanese','Javanese','Kal',
    'Kalíngga','Kan','Kankanáëy','Kap','Kapampángan','Karáw','Kas','Kay','Kbn','Kem',
    'Kinaráy-a','Klg','Kol','Kor','Krw','Kry','Kuy','Kuyunón','Lat','Latin','MOc',
    'Mad','Mag','Magindanáw','Mal','Malay','Mam','Man','Mandayá','Mangyán','Mar',
    'Mbk','Mex','Mexican','Mgd','Mgk','Mgn','Min','Mmw','Mnb','Mnd','Mng','Mns','Mny',
    'Mrs','Mrw','Mts','Mwr','Mëranáw','Noe','Nor','Pal','Paláw-an','Pan','Pangasinán',
    'Plw','Png','Pny','Por','Portugués','Rom','Rus','ST','STl','Sam','San','Sanskrit',
    'Sbl','Sco','Seb','Sebwáno','Sinadánga','Sinaunang Tagalog','Skr','Slavic','Sma',
    'Sml','Snk','Sub','Sul','Swa','Súg','Tag','Tagabáwa','Tagbanwá','Tagálog','Tau',
    'Tbn','Tbo','Tbw','Ted','Tgb','Tgk','Tha','Tib','Tin','Tir','Tng','Tsi','Tua',
    'Tur','Turkish','Tëduráy','War','Waráy','Yak','Yógad','Zam','Zoo','lng',
))