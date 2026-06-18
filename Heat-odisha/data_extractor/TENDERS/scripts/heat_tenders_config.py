
# ---------------------------------------------------------------------------
# Negative keywords: if any of these match, the tender is excluded even if a
# positive keyword also matches (e.g. "Floodlight installation" should not be
# tagged just because "light" overlaps with other terms).
# ---------------------------------------------------------------------------
NEGATIVE_KEYWORDS = [
    'floodlight', 'flood light', 'street light', 'solar street light', 'high mast light',
    'gas pipeline', 'sports complex', 'stadium', 'cricket', 'football', 'volleyball',
    'examination hall', 'cremation ground', 'crematorium', 'burial ground',
    'election', 'covid', 'vaccination', 'wedding hall', 'marriage hall', 'kalyan mandap',

    'boundary wall', 'compound wall', 'staff quarter', 'quarter building',
    'drug warehouse', 'drug ware house', 'godown',
    'bio medical waste', 'biomedical waste', 'bmw management', 'waste management',
]

# ---------------------------------------------------------------------------
# Thematic keyword groups, in priority order. The first group with a match is
# used as the tender's "Response Type". All matched groups (with their
# matched keyword counts) are recorded in "Heat Response - Subhead".
# ---------------------------------------------------------------------------
THEMATIC_KEYWORD_GROUPS = {
    "Heatwave & Emergency Response": [
        'heatwave', 'heat wave', 'heat-wave', 'heat stress', 'heat stroke',
        'heat action plan', 'heat resilience', 'heat resilient',
        'extreme heat', 'high temperature', 'rising temperature', 'heat resistant',
        'climate resilient', 'cooling centre', 'cooling center',
        'emergency response', 'first aid', 'ambulance',
        'ors', 'oral rehydration solution', 'heat stroke treatment',
        # --- HAP2020: heat-specific terms (high precision) ---
        'sunstroke',                 # HAP2020 ex-gratia is paid to "sunstroke victims"
        'sun stroke',                # HAP2020
        'heat related illness',      # HAP2020 (HRI) - written space-form, see header note
        'heat illness',              # HAP2020
        'heat exhaustion',           # HAP2020
        'heat cramp',                # HAP2020
        'heat rash',                 # HAP2020
        'heat alert',                # HAP2020
        'heat warning',              # HAP2020
        'heat advisory',             # HAP2020
        'heat hotline',              # HAP2020 (Fig-8 "promote heat hotline")
        'satark',                    # HAP2020 - OSDMA heatwave alert app (very distinctive)
        'early warning system',      # HAP2020 (EWS)
        'saline',                    # HAP2020 - stored for heat-stroke treatment
        'iv fluid',                  # HAP2020
        'rapid response team',       # HAP2020 (RRT)
        'mobile health unit',        # HAP2020 (MHU)
        'relief chamber',            # HAP2020 ("A/C relief chamber for emergency")
        'heat stroke ward',          # HAP2020 (separate beds for heat-stroke patients)
        'heat stroke bed',           # HAP2020
    ],
    "Drinking Water & Hydration": [
        'drinking water', 'water kiosk', 'water atm',
        'hydration point', 'hydration centre', 'hydration center', 'cold water',
        'tube well', 'tubewell', 'borewell', 'bore well',
        'hand pump', 'handpump', 'overhead tank', 'oh tank',
        'stand post', 'piped water supply', 'water supply scheme',
        # --- HAP2020: Odisha-specific kiosk terms (high precision, low FP risk) ---
        'jalachhatra',               # HAP2020 - water booth opened at bus stands during summer
        'jalachatra',                # HAP2020 (alt. spelling used in BMC section)
        'jala jogana kendra',        # HAP2020 - "water kiosk" in Odia
        'jal jogana kendra',         # HAP2020 (alt. spelling)
        'jal seva shibira',          # HAP2020 (BMC drinking-water camps)
        # --- HAP2020: dual-use water terms (review before relying on) ---
        'water tanker',              # HAP2020 - tankers deployed for summer scarcity
        'water sprinkling',          # HAP2020 (Industry/Mines SOP)
    ],
    "Cooling & Reflective Infrastructure": [
        'cool roof', 'white roof', 'white topping',
        'roof insulation', 'thermal insulation',
        'heat reflective', 'reflective coating',
        'shade net', 'shade structure', 'canopy', 'awning',
        # --- HAP2020: passive-cooling / reflective measures (high precision) ---
        'albedo paint',              # HAP2020 - "albedo/white painting of roof tops"
        'albedo painting',           # HAP2020
        'white painting',            # HAP2020 (roof tops of hospitals/CHCs/PHCs/schools)
        'reflective paint',          # HAP2020
        'green roof',                # HAP2020
        'green roofing',             # HAP2020
        'k glass',                   # HAP2020 ("K-glass, doubly glazed glass")
        'doubly glazed glass',       # HAP2020
        'passive cooling',           # HAP2020 (building-code SOP)
        'air cooler',                # HAP2020 (Nandankanan / cooling facilities)
        'white china mosaic',        # HAP2020 (cool-roof treatment)
        'lime based whitewash',      # HAP2020 (cool-roof treatment)
        'white tarp',                # HAP2020 (cool-roof treatment)
        'acrylic resin coating',     # HAP2020 (cool-roof treatment)
    ],
    "Greening & Urban Forestry": [
        'tree plantation', 'avenue plantation', 'plantation',
        'green belt', 'urban forestry', 'urban forest',
        'park development', 'shelter belt', 'shade tree',
        # --- HAP2020: greening measures (dual-use; "plantation" already over-fires) ---
        'vertical garden',           # HAP2020 (UHI mitigation)
        'green space',               # HAP2020 ("small accessible green spaces")
        'afforestation',             # HAP2020 (Forest dept SOP)
        'green campus',              # HAP2020 (School dept SOP)
    ],
    "Shelter & Rest Infrastructure": [
        'night shelter', 'transit shelter', 'rest shed', 'waiting shed',
        'community hall', 'relief centre', 'relief center',
        'bus stand shelter', 'passenger shelter',
        # --- HAP2020: heat-relief shelter terms ---
        'cool shelter',              # HAP2020 ("special cool shelters", "provide cool shelter")
        'rest shade',                # HAP2020 (used alongside "rest shed" in Water Resources SOP)
        'cool resting space',        # HAP2020 (Transport / HUDD SOPs)
        'cool resting shed',         # HAP2020 (Tourism SOP)
        'passenger shed',            # HAP2020 ("temporary passenger sheds near bus stops")
    ],
    "Health System Strengthening": [
        # Bare facility names REMOVED: they were the single largest source of
        # false positives (e.g. "Bio Medical Waste building at CHC...",
        # "c.c. drain at back side of Perfect Clinic"). Building or repairing a
        # health facility is not a heat-response measure. A genuinely
        # heat-relevant health tender (e.g. "albedo painting of CHC roofs",
        # "heat-stroke ward") is still caught by its heat-specific term, so
        # nothing real is lost by dropping the bare names below:
        #   'health centre', 'health center', 'primary health centre',
        #   'phc', 'chc', 'hospital', 'clinic', 'dispensary',
        'outreach clinic',           # HAP2020 (heat-illness outreach)
        'heat stroke ward',          # HAP2020
        'heat stroke bed',           # HAP2020
    ],
    "Awareness & Capacity Building": [
        'awareness campaign', 'iec activity', 'iec material',
        'capacity building', 'training programme', 'training program',
        'mock drill', 'sensitization',
        # --- HAP2020: IEC / awareness vocabulary ---
        'swasthya kantha',           # HAP2020 - "village health wall" used for do's & don'ts
        'do and dont',               # HAP2020 ("Do's and Don'ts" - clean_text strips apostrophes)
        'dos and donts',             # HAP2020 (alt. normalised form)
        'display board',             # HAP2020 (colour heat-wave alert boards)
        'hoarding',                  # HAP2020 (Do's/Don'ts hoardings)
        'pamphlet',                  # HAP2020
        'leaflet',                   # HAP2020
        'banner',                    # HAP2020
    ],
}

# ---------------------------------------------------------------------------
# Overall "is this tender heat-related at all" keyword list, derived from all
# thematic groups (deduplicated, order-preserving).
# ---------------------------------------------------------------------------
POSITIVE_KEYWORDS = list(dict.fromkeys(
    keyword
    for keywords in THEMATIC_KEYWORD_GROUPS.values()
    for keyword in keywords
))

# ---------------------------------------------------------------------------
# Funding scheme acronyms searched for as standalone tokens in the tender
# title / reference / description.
# ---------------------------------------------------------------------------
SCHEME_KEYWORDS = {
    'ridf', 'sdrf', 'sopd', 'cidf', 'ltif', 'sdmf', 'ndrf', 'jjm', 'amrut', 'sbm',
    # --- HAP2020 / observed in the tender data ---
    'cmrf',     # HAP2020 - Chief Minister's Relief Fund (heat-wave ex-gratia)
    'mgnrega',  # HAP2020 - working-hour restrictions under MGNREGA
    'mnrega',   # HAP2020 (alt. spelling)
    'nhm',      # appears in tender data (NHM PIP)
    'ombadc',   # appears in tender data (OMBADC scheme)
}

# ---------------------------------------------------------------------------
# Departments excluded even if their tenders match a positive keyword.
# NOTE (from audit): the two entries below are Assam departments and never
# match in this Odisha dataset, so the exclusion is currently inert. Replace
# with the Odisha departments that actually generate noise if you want this
# filter to do anything.
# ---------------------------------------------------------------------------
EXCLUDED_DEPARTMENTS = [
    "Directorate of Agriculture and Assam Seed Corporation",
    "Department of Handloom Textile and Sericulture",
]

# ---------------------------------------------------------------------------
# Month -> season label, used to tag tenders by when they were published.
# Odisha's heatwave season runs roughly March-June (HAP: "April to June").
# ---------------------------------------------------------------------------
SEASON_MONTHS = {
    "Pre-Summer (Jan-Feb)": [1, 2],
    "Summer / Heatwave (Mar-Jun)": [3, 4, 5, 6],
    "Monsoon (Jul-Sep)": [7, 8, 9],
    "Post-Monsoon / Winter (Oct-Dec)": [10, 11, 12],
}
