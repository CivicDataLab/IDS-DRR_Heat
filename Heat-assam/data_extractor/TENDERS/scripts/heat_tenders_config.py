# 
# ---------------------------------------------------------------------------
# Negative keywords: if any of these match, the tender is excluded even if a
# positive keyword also matches (note: per heat_tenders.py fix #3, a negative
# no longer overrides a genuinely STRONG heat signal -- it only kills a weak
# match).
# ---------------------------------------------------------------------------
NEGATIVE_KEYWORDS = [
    'flood light', 'street light', 'solar street light', 'high mast light',
    'gas pipeline', 'sports complex', 'stadium', 'cricket', 'football', 'volleyball',
    'examination hall', 'cremation ground', 'crematorium', 'burial ground',
    'election', 'covid', 'vaccination', 'marriage hall',

    'boundary wall', 'compound wall', 'staff quarter', 'quarter building',
    'drug warehouse', 'drug ware house', 'godown',
    'bio medical waste', 'biomedical waste', 'waste management',

    # --- Assam-specific noise (predicted; verify against the corpus) ---------
    # Tea is a commercial crop, tea-estate works would over-fire on the greening term 'plantation'.
    # This also suppresses genuine greening *inside* tea-belt wards
    'tea garden', 'tea estate',
    # Flood / riverbank-erosion works dominate the Assam tender stream and are
    # not heat-response measures. They overlap only weakly with heat positives,
    # so excluding them is mostly defensive. [UHE: Brahmaputra/Barak flood risk]
    'embankment', 'anti erosion', 'anti-erosion', 'riverbank protection',
    'porcupine', 'geo bag', 'geobag', 'sluice gate', 'dyke', 'breach closing',
]

# ---------------------------------------------------------------------------
# Drinking Water & Hydration: ambiguous infra terms + their gate.
#
# Prevent fals positives from Irrigation Dept's PMKSY-HKKP ("Har Khet Ko
# Pani") works, which are for farm irrigation, not drinking water. Genuine
# drinking-water schemes are still being included. PHE's piped-water-supply 
# almost always carry one of the scheme markers below (PWSS / JJM / NWQSM 
# are India's standard rural-drinking-water scheme names), which IS an 
# explicit, unambiguous drinking-water signal.
#
# So these terms only count as a positive hit when they co-occur, in the same
# tender text, with a DRINKING_WATER_QUALIFIER. See heat_tenders.py
# count_keyword_hits() for the gating logic.
# ---------------------------------------------------------------------------
DRINKING_WATER_AMBIGUOUS_KEYWORDS = [
    'tube well', 'tubewell', 'borewell', 'bore well',
    'hand pump', 'handpump', 'overhead tank', 'oh tank',
    'stand post', 'piped water supply', 'water supply scheme',
    # dual-use (deployed for summer scarcity AND year-round) -- review:
    'water tanker',               # [EHW] Tripura/Arunachal deploy tankers in heat
    'water sprinkling',
   
]

DRINKING_WATER_QUALIFIERS = [
    'drinking water', 'potable water', 'human consumption',
    'jal jeevan mission', 'jjm', 'nwqsm', 'pwss', 'rural water supply',
]

# ---------------------------------------------------------------------------
# Thematic keyword groups, in priority order. The first group with a match is
# used as the tender's "Response Type". All matched groups (with their matched
# keyword counts) are recorded in "Heat Response - Subhead".
# ---------------------------------------------------------------------------
THEMATIC_KEYWORD_GROUPS = {
    "Heatwave & Emergency Response": [
        'heatwave', 'heat wave', 'heat-wave', 'heat stress', 'heat stroke',
        'heat action plan', 'heat resilience', 'heat resilient',
        'extreme heat', 'high temperature', 'rising temperature', 'heat resistant',
        'climate resilient', 'cooling centre', 'cooling center',
        'emergency response', 'first aid', 'ambulance',
        'ors', 'oral rehydration solution', 'heat stroke treatment',
        'sunstroke', 'sun stroke',
        'heat related illness', 'heat illness', 'heat exhaustion',
        'heat cramp', 'heat rash',
        'heat alert', 'heat warning', 'heat advisory', 'heat hotline',
        'early warning system', 'saline', 'iv fluid',
        'rapid response team', 'mobile health unit', 'relief chamber',
        'heat stroke ward', 'heat stroke bed',
        # --- Assam additions -------------------------------------------------
        'heat index',                 # [EHW] IMD heat index; NE heat is humidity-driven
        'wet bulb', 'wet bulb temperature',  # [EHW] Dr. Chakraborty: wet-bulb is the NE risk metric
        'apparent temperature',       # [EHW] Godhwani: ambient vs apparent temperature
        'hot and humid',              # [EHW] IMD-Assam's own term for Assam's non-classical heat
        'cooling station',            # [EHW] community cooling station (Jodhpur 12C-relief model)
        'cooling shelter',
        'drims',                      # [EHW] Disaster Reporting & Info Mgmt System (Assam platform)
        'aapda mitra',                # [EHW] Assam volunteer network used for heat response
        # REMOVED 'asdma': audit of the real corpus shows it fires on 48 tenders
        # purely because "ASDMA" is the reference-number prefix on every tender
        # the authority issues (boat ambulances, UPS units, flood shelters, ...),
        # almost none heat-specific. Like the bare facility names, it is org-id
        # noise, not a heat measure.
        # REMOVED (Odisha-only): 'satark' (OSDMA heatwave alert app)
    ],
    # EXPLICIT terms always count; AMBIGUOUS infra terms (defined above) only
    # count when gated by a DRINKING_WATER_QUALIFIER co-occurring in the text.
    "Drinking Water & Hydration": [
        'drinking water', 'water kiosk', 'water atm', 'water booth',
        'hydration point', 'hydration centre', 'hydration center', 'cold water',
    ] + DRINKING_WATER_AMBIGUOUS_KEYWORDS,
    "Cooling & Reflective Infrastructure": [
        'cool roof', 'white roof',
        # REMOVED 'white topping': in the Assam corpus it only ever matched
        # road "whitetopping" (a concrete overlay on flexible pavement), never
        # a cool roof -- it produced the single largest strong false positive.
        'roof insulation', 'thermal insulation',  # NB: 'thermal insulation' is weak-only (matches pipe-insulation tenders)
        'heat reflective', 'reflective coating',
        'shade net', 'shade structure', 'canopy', 'awning',
        'albedo paint', 'albedo painting', 'white painting', 'reflective paint',
        'green roof', 'green roofing', 'passive cooling', 'air cooler',
        # generic physical roof treatments (could appear anywhere, low FP):
        'k glass', 'doubly glazed glass', 'white china mosaic',
        'lime based whitewash', 'white tarp', 'acrylic resin coating',
        # --- Assam additions -------------------------------------------------
        'cool roof campaign',         # [EHW] cool-roof campaign repeatedly recommended for Assam
        'reflective wall', 'cool wall',  # [UHE] annexure 2: cool/reflective walls
        'permeable pavement', 'porous pavement', 'cool pavement',  # [UHE] annexure 2
        'green building', 'green building code',  # [EHW] Min. Mahanta: green building codes for schools/colleges
        'energy conservation building code',      # [EHW] Upadhyay: ECBC
    ],
    "Greening & Urban Forestry": [
        'tree plantation', 'avenue plantation', 'plantation',
        'green belt', 'urban forestry', 'urban forest',
        'park development', 'shelter belt', 'shade tree',
        'vertical garden', 'green space', 'afforestation', 'green campus',
        # --- Assam additions -------------------------------------------------
        'sponge park',                # [UHE] water-retaining cooling park -- a signature Assam-report term
        'pocket park', 'cooling park',  # [UHE]
        'city forest',                # [UHE] urban-forest synonym
        'green corridor',             # [UHE] continuous greening corridor
        'blue green', 'blue-green',   # [UHE] blue-green infrastructure (clean_text -> 'blue green')
        'green buffer', 'riparian buffer',  # [UHE]
        'canopy cover',               # [UHE] 3-30-300 tree-canopy target
        'greened median', 'verge planting', 'bioswale',  # [UHE] annexure 2
        # dual-use wetland terms. AUDIT RESULT: bare 'beel' matched 101 tenders
        # in the Assam corpus, ALL place names (Deepor Beel, Silsako Beel, "road
        # near X beel"...) and 0 genuine conservation tenders, so it is dropped.
        # The qualified forms ('beel rejuvenation', 'wetland conservation') are
        # the right handle -- they currently match 0, kept for when such tenders
        # appear. 'wetland' (31 hits, mixed) is retained as a weak signal only.
        'wetland', 'beel rejuvenation', 'beel conservation', 'wetland conservation',
    ],
    "Shelter & Rest Infrastructure": [
        'night shelter', 'transit shelter', 'rest shed', 'waiting shed',
        'community hall', 'relief centre', 'relief center',
        'bus stand shelter', 'passenger shelter',
        'cool shelter', 'rest shade', 'cool resting space', 'cool resting shed',
        'passenger shed',
        # --- Assam additions -------------------------------------------------
        'shaded bus stop', 'bus shelter', 'shaded shelter',  # [UHE/EHW] shaded transit waiting areas
    ],
    "Health System Strengthening": [
        # Bare facility names (health centre, phc, chc, hospital, clinic,
        # dispensary, anganwadi) are deliberately NOT listed: they are the
        # single largest false-positive source (building/repairing a facility
        # is not a heat measure). A genuine "School / hospital / anganwadi
        # heat-proofing" tender [UHE annexure] is still caught by its
        # heat-specific term (cool roof / shade / drinking water), so nothing
        # real is lost by omitting the bare names.
        'outreach clinic',
        'heat stroke ward', 'heat stroke bed',
    ],
    "Awareness & Capacity Building": [
        'awareness campaign', 'iec activity', 'iec material',
        'capacity building', 'training programme', 'training program',
        'mock drill', 'sensitization',
        'do and dont', 'dos and donts', 'display board', 'hoarding',
        'pamphlet', 'leaflet', 'banner',
        # --- Assam additions -------------------------------------------------
        'heat volunteer',             # [EHW] heat volunteer network (Aapda Mitra / NSS)
        # REMOVED (Odia/Odisha-only): 'swasthya kantha' (village health wall)
    ],
}

# ---------------------------------------------------------------------------
# Overall "is this tender heat-related at all" keyword list, derived from all
# thematic groups (deduplicated, order-preserving). Do not edit by hand.
# ---------------------------------------------------------------------------
POSITIVE_KEYWORDS = list(dict.fromkeys(
    keyword
    for keywords in THEMATIC_KEYWORD_GROUPS.values()
    for keyword in keywords
))

# ---------------------------------------------------------------------------
# HEAT-SPECIFIC keyword set  --  the "strong signal" whitelist.
# (Moved here from heat_tenders.py so the config is the single source of truth.
#  heat_tenders.py now imports this and asserts it is a subset of
#  POSITIVE_KEYWORDS, so a term can never be "strong" without also being a
#  positive.)
#
# A term earns a place here only if its presence is, ON ITS OWN, a reliable
# heat-response signal. Everything else stays a positive (weak) signal.
#
# This list is CURATED, not group-derived: the "Heatwave & Emergency Response"
# group contains generic disaster terms (ambulance, first aid, emergency
# response, early warning system) and dual-use terms (saline, water tanker,
# thermal insulation) that are NOT reliable on their own.
#
# AUDIT against the real Assam corpus (34,747 tenders) drove three demotions:
#   - 'white topping'    -> only matched road concrete-overlay works  (dropped)
#   - 'thermal insulation' -> matched industrial pipe insulation       (weak)
#   - 'beel' / 'asdma'   -> place names / org-id prefix                (dropped)
# It also confirmed the genuine Assam heat tenders are the greening / blue-green
# family (sponge park, urban forest, blue-green infrastructure), so those are
# the backbone of the strong set. Classic terms (cool roof, heatwave, cooling
# centre) currently match 0 tenders but are kept to catch heat procurement as
# Assam's Heat Action Plan rolls out.
# ---------------------------------------------------------------------------
HEAT_SPECIFIC_KEYWORDS = {
    # heat events / illness (unambiguous; mostly future-proofing, ~0 today)
    'heatwave', 'heat wave', 'heat-wave', 'heat stress', 'heat stroke',
    'heat action plan', 'heat resilience', 'heat resilient', 'extreme heat',
    'heat resistant', 'heat related illness', 'heat illness', 'heat exhaustion',
    'heat cramp', 'heat rash', 'heat alert', 'heat warning', 'heat advisory',
    'heat hotline', 'sunstroke', 'sun stroke',
    'heat stroke treatment', 'heat stroke ward', 'heat stroke bed',
    'cooling centre', 'cooling center', 'cooling station', 'cooling shelter',
    'relief chamber',
    # Assam humidity metrics (heat-specific) [EHW]
    'heat index', 'wet bulb', 'wet bulb temperature', 'apparent temperature',
    # reflective / passive cooling (specific phrases only)
    'cool roof', 'white roof', 'cool roof campaign', 'roof insulation',
    'heat reflective', 'reflective coating', 'reflective paint',
    'cool wall', 'reflective wall',
    'albedo paint', 'albedo painting', 'green roof', 'green roofing',
    'passive cooling', 'k glass', 'doubly glazed glass', 'white china mosaic',
    'lime based whitewash', 'white tarp', 'acrylic resin coating',
    'cool pavement', 'permeable pavement', 'porous pavement',
    'green building code', 'energy conservation building code',
    # heat-relief hydration / shelter / shade (specific only)
    'hydration point', 'hydration centre', 'hydration center',
    'cool shelter', 'cool resting space', 'cool resting shed',
    'shaded bus stop', 'shaded shelter', 'shade net', 'shade structure',
    # greening / blue-green -- the genuine Assam interventions in this corpus [UHE]
    'sponge park', 'cooling park', 'pocket park',
    'urban forest', 'city forest', 'urban forestry',
    'blue green', 'blue-green', 'green corridor', 'green buffer',
    'riparian buffer', 'canopy cover', 'vertical garden',
    'greened median', 'bioswale',
    'avenue plantation',
    # NB 'tree plantation' demoted to weak: in the corpus it promoted a
    # fencing-supply tender ("goat proof fencing ... under Tree plantation
    # scheme") and overlaps with compensatory-afforestation (CAMPA) works.
    # 'avenue plantation' stays strong -- it is specifically roadside greening.
    # NB deliberately NOT strong (kept weak as positives): plantation,
    #   afforestation, green space, wetland, beel*, water atm, water kiosk,
    #   water tanker, water sprinkling, drinking water, piped water supply,
    #   tube well, canopy, awning, air cooler, white painting, thermal
    #   insulation, climate resilient, high/rising temperature, ambulance,
    #   first aid, emergency response, ors, saline, iv fluid, early warning
    #   system, rapid response team, mobile health unit, mock drill, asdma*,
    #   drims, aapda mitra, and all awareness/IEC terms.
    #   (* = removed from positives entirely.)
}

# ---------------------------------------------------------------------------
# Funding scheme acronyms searched for as standalone tokens in the tender
# title / reference / description.
#   - Kept national schemes that fund urban/water/health/disaster works.
#   - Kept SOPD (Assam budgets use SOPD = State Own Priority Development).
#   - REMOVED Odisha-only: ombadc (OMBADC), cidf, ltif.
#   - SDMF is the key one: the workshop [EHW] explicitly routes heat seed
#     funding through the State Disaster Mitigation Fund, "regardless of
#     heatwave's disaster notification status".
# ---------------------------------------------------------------------------
SCHEME_KEYWORDS = {
    'sdmf',     # [EHW] State Disaster Mitigation Fund -- primary heat-funding route in Assam
    'ndmf',     # National Disaster Mitigation Fund (paired with SDMF) [EHW]
    'sdrf', 'ndrf',
    'ridf',     # NABARD Rural Infrastructure Development Fund (national)
    'sopd',     # Assam State Own Priority Development (also appears in Odisha)
    'jjm',      # Jal Jeevan Mission (water)
    'amrut',    # urban -- relevant to Guwahati/Dibrugarh/Silchar
    'sbm',      # Swachh Bharat Mission
    'nhm',      # National Health Mission
    'cmrf',     # Chief Minister's Relief Fund (heat ex-gratia)
    'mgnrega', 'mnrega',
    # --- Assam / NE candidates NOT in the source docs -- add only after you
    #     confirm they appear in the Assam tender data:
    #   'nesids'  (North East Special Infrastructure Development Scheme)
    #   'nec'     (North Eastern Council)
    #   'nlcpr'   (Non-Lapsable Central Pool of Resources)
}

# ---------------------------------------------------------------------------
# Departments excluded even if their tenders match a positive keyword.
# ---------------------------------------------------------------------------
EXCLUDED_DEPARTMENTS = [
    "Directorate of Agriculture and Assam Seed Corporation",
    "Department of Handloom Textile and Sericulture",
]

# ---------------------------------------------------------------------------
# Month -> season label, used to tag tenders by when they were published.
# Assam's pre-monsoon heat season runs roughly March-June ([EHW]: "heat waves
# typically occur from March to June"), but two Assam-specific caveats apply:
#   1. The deadly events are increasingly OFF-SEASON: 11 heatwave-related
#      deaths were recorded in SEPTEMBER 2024 ([EHW], AIPSN), during a
#      "hot and humid" spell that did not even meet IMD's heatwave criteria.
#      So do not assume Jul-Sep tenders are heat-irrelevant.
#   2. The Assam study itself uses WINTER satellite scenes (Jan/Feb/Nov; see
#      the report's metadata table) because of persistent monsoon cloud cover,
#      which is why publication-month seasonality is a weak signal here.
# ---------------------------------------------------------------------------
SEASON_MONTHS = {
    "Pre-Heat (Jan-Feb)": [1, 2],
    "Pre-Monsoon / Heat Season (Mar-Jun)": [3, 4, 5, 6],
    "Monsoon (Jul-Sep, incl. Sep hot-humid spells)": [7, 8, 9],
    "Post-Monsoon / Winter (Oct-Dec)": [10, 11, 12],
}
