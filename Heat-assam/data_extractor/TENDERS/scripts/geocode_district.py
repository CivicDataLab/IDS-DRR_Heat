
import pandas as pd
import re
import os
from difflib import SequenceMatcher

root = os.getcwd() + "/Heat-assam/"
tenders_df = pd.read_csv(root + r'/data_extractor/TENDERS/data/heat_tenders_all.csv')

ASSAM_VILLAGES = pd.read_csv(root + r'/data_extractor/Maps/Extra/ASSAM_VILLAGES_RC_DISTRICT_MAPPING.csv',
                              encoding='utf-8').dropna()

# Clean village names
assam_villages = ASSAM_VILLAGES["VILNAM_SOI"]
village_duplicates_df = ASSAM_VILLAGES[assam_villages.isin(assam_villages[assam_villages.duplicated()])].sort_values("VILNAM_SOI")
VILLAGE_CORRECTION_DICT = {
    "SOKARBILA(BOLGARBARI)(DARIAPAR" : "SOKARBILA(BOLGARBARI)(DARIAPAR)",
    "MANGALDAI EXTENDED TOWN (BHEBA" : "MANGALDAI EXTENDED TOWN (BHEBA)",
    "UPPER DIHING R.F. (SOUTH BLOCK" : "UPPER DIHING R.F. (SOUTH BLOCK)",
    "KACHARI MAITHCHAGAON NO.1(BAR" : "KACHARI MAITHCHAGAON NO.1(BAR)",
}
ASSAM_VILLAGES.revenue_ci = ASSAM_VILLAGES.revenue_ci.str.replace('\(Pt\)','')
ASSAM_VILLAGES.revenue_ci = ASSAM_VILLAGES.revenue_ci.str.replace('\(Pt-I\)','')
ASSAM_VILLAGES.revenue_ci = ASSAM_VILLAGES.revenue_ci.str.replace('\(Pt-II\)','')
ASSAM_VILLAGES.revenue_ci = ASSAM_VILLAGES.revenue_ci.str.replace('\n',' ')
ASSAM_VILLAGES.revenue_ci = ASSAM_VILLAGES.revenue_ci.str.strip()

ASSAM_VILLAGES.sdtname_2 = ASSAM_VILLAGES.sdtname_2.str.replace('\(Pt\)','')
ASSAM_VILLAGES.sdtname_2 = ASSAM_VILLAGES.sdtname_2.str.replace('\(Pt-I\)','')
ASSAM_VILLAGES.sdtname_2 = ASSAM_VILLAGES.sdtname_2.str.replace('\(Pt-II\)','')


locations = []
for idx, row in tenders_df.iterrows():
    LOCATION = row['location'].lower()
    LOCATION = LOCATION.replace('village','')
    LOCATION = LOCATION.replace('district','')
    LOCATION = LOCATION.replace('dist','')
    LOCATION = re.sub(r'[^a-zA-Z\n\.]', ' ', LOCATION)
    scores = []
    for revenue_circle in ASSAM_VILLAGES.revenue_ci.dropna().unique():
        score = SequenceMatcher(None, LOCATION, revenue_circle.lower().strip()).ratio()
        scores.append(score)
    if max(scores)>0.8:
        locations.append(ASSAM_VILLAGES.revenue_ci.dropna().unique()[scores.index(max(scores))])
    else:
        locations.append(row['location'])

tenders_df.location = locations

rev_circles = ASSAM_VILLAGES[["revenue_ci",'district_2']].drop_duplicates().dropna()
# These revenue circles are across multiple districts
problematic_rev_circles = rev_circles[rev_circles.duplicated(['revenue_ci'],keep=False)].sort_values('revenue_ci')

sdts = ASSAM_VILLAGES[["sdtname_2",'district_2']].drop_duplicates().dropna()
problematic_sdts = sdts[sdts.duplicated(['sdtname_2'],keep=False)].sort_values('sdtname_2')

# GEOCODE DISTRICTS

# Dictionary of only non-repeated revenue circles, sub-districts, blocks and villages mapped to their districts
assam_revenue_circles_dict = ASSAM_VILLAGES[['revenue_ci','district_2']].dropna().drop_duplicates().drop_duplicates(['revenue_ci'],keep=False).set_index('revenue_ci').to_dict(orient='index')
assam_subdist_dict = ASSAM_VILLAGES[['sdtname_2','district_2']].dropna().drop_duplicates().drop_duplicates(['sdtname_2'],keep=False).set_index('sdtname_2').to_dict(orient='index')
assam_blocks_dict = ASSAM_VILLAGES[['block_name','district_2']].dropna().drop_duplicates().drop_duplicates(['block_name'],keep=False).set_index('block_name').to_dict(orient='index')
assam_villages_dict = ASSAM_VILLAGES[['VILNAM_SOI','district_2']].drop_duplicates(['VILNAM_SOI'],keep=False).set_index('VILNAM_SOI').to_dict(orient='index')

# Force fit duplicate revenue circles in districts
assam_revenue_circles_dict['Baganpara']={'district_2': 'BAKSA'}
assam_revenue_circles_dict['Bagribri']={'district_2': 'DHUBRI'}
assam_revenue_circles_dict['Bajali']={'district_2': 'BAJALI'}
assam_revenue_circles_dict['Barnagar']={'district_2': 'BAKSA'}
assam_revenue_circles_dict['Chapar']={'district_2': 'DHUBRI'}
assam_revenue_circles_dict['Dalgaon']={'district_2': 'DARRANG'}
assam_revenue_circles_dict['Dhakuakhana']={'district_2': 'LAKHIMPUR'}
assam_revenue_circles_dict['Dhekiajuli']={'district_2': 'SONITPUR'}
assam_revenue_circles_dict['Dhubri']={'district_2': 'DHUBRI'}
assam_revenue_circles_dict['Ghograpar']={'district_2': 'NALBARI'}
assam_revenue_circles_dict['Golokganj']={'district_2': 'DHUBRI'}
assam_revenue_circles_dict['Gossaigaon']={'district_2': 'KOKRAJHAR'}
assam_revenue_circles_dict['Jalah']={'district_2': 'BAKSA'}
assam_revenue_circles_dict['Khoirabari']={'district_2': 'UDALGURI'}
assam_revenue_circles_dict['Kokrajhar']={'district_2': 'KOKRAJHAR'}
assam_revenue_circles_dict['Lakhipur']={'district_2': 'GOALPARA'}
assam_revenue_circles_dict['Mangaldoi']={'district_2': 'DARRANG'}
assam_revenue_circles_dict['Pathorighat']={'district_2': 'DARRANG'}
assam_revenue_circles_dict['Sarupeta']={'district_2': 'BAJALI'}
assam_revenue_circles_dict['Sidli']={'district_2': 'CHIRANG'}
assam_revenue_circles_dict['Subansiri']={'district_2': 'LAKHIMPUR'}
assam_revenue_circles_dict['Rangia']={'district_2': 'KAMRUP'}

# Lists of districts, revenue circles, sub-districts and villages with non-repeating names
problematic_rev_circlesUPPERCASE = []
problematic_sdtsUPPERCASE = [sdt.upper().strip() for sdt in problematic_sdts.sdtname_2.unique()]
assam_villages = list(set(assam_villages_dict.keys())-set(problematic_rev_circlesUPPERCASE)-set(problematic_sdtsUPPERCASE))
assam_blocks = list(set(assam_blocks_dict.keys())-set(problematic_rev_circlesUPPERCASE)-set(problematic_sdtsUPPERCASE))
assam_districts = list(set(ASSAM_VILLAGES.district_2.dropna())-set(['KAMRUP','KAMRUP METRO']))

assam_revenue_circles = list(set(assam_revenue_circles_dict.keys()))
assam_sub_districts = list(set(assam_subdist_dict.keys())-set(problematic_rev_circles.revenue_ci.unique())-set(problematic_sdts.sdtname_2.unique()))

# District identifiers parsed from the externalReference column
three_letter_distirct_identifiers_dict = {"bak":"BAKSA", "baksa":"BAKSA",
                                          "bar":"BARPETA", "re-bar": "BARPETA", "barpeta":"BARPETA",
                                          "bongaigoan":"BONGAIGAON",
                                          "tez":"SONITPUR","re-tez":"SONITPUR","tezpur":"SONITPUR","tej":"SONITPUR","re-tej":"SONITPUR",
                                          "silchar":"CACHAR", "re-silchar":"CACHAR","resilchar":"CACHAR","re-sil(mech)":"CACHAR","silchar (mech)":"CACHAR","sil":"CACHAR","sil (mech)":"CACHAR","sil(mech)":"CACHAR",
                                          "dhubri":"DHUBRI", "dhu": "DHUBRI",
                                          "siv":"SIVSAGAR","sivsagar":"SIVSAGAR","re-siv":"SIVSAGAR","sivasagar":"SIVSAGAR",
                                          "chirang":"CHIRANG",
                                          "mang":"DARRANG","re-mang":"DARRANG","mangaldai":"DARRANG","mangaldoi":"DARRANG",
                                          "dhe":"DHEMAJI","dhemaji":"DHEMAJI","dmj":"DHEMAJI","redhemaji":"DHEMAJI",
                                          "hailakandi":"HAILAKANDI","hkd":"HAILAKANDI","re-hailakandi":"HAILAKANDI",
                                          "dib-west":"DIBRUGARH","dib":"DIBRUGARH","dibrugarh":"DIBRUGARH","redib":"DIBRUGARH",
                                          "dima-hasao":"DIMA HASAO","haf":"DIMA HASAO","haflong":"DIMA HASAO",
                                          "goalpara":"GOALPARA","GLP":"GOALPARA",
                                          "diphu":"K.ANGLONG","rediphu":"K.ANGLONG",
                                          "jor":"JORHAT","jorhat":"JORHAT",
                                          "nag":"NAGAON","re-nag":"NAGAON","nagaon":"NAGAON","hatimura":"NAGAON",
                                          "nal":"NALBARI","nalbari":"NALBARI",
                                          "morigaon":"MORIGAON","mor":"MORIGAON","re-mor":"MORIGAON",
                                          "maj":"MAJULI","re-maj":"MAJULI","maju":"MAJULI","majuli":"MAJULI",
                                          "n.lakhimpur":"LAKHIMPUR","dhakuakhana":"LAKHIMPUR","nlp":"LAKHIMPUR","nl":"LAKHIMPUR","dhk":"LAKHIMPUR",
                                          "kar":"KARIMGANJ","rekar":"KARIMGANJ","re-kar":"KARIMGANJ","karimganj":"KARIMGANJ","badarpur":"KARIMGANJ",
                                          "gmda":"KAMRUP METRO","ghy east":"KAMRUP METRO","ghy.east":"KAMRUP METRO","ghy. east":"KAMRUP METRO","ghyeast":"KAMRUP METRO","g.east":"KAMRUP METRO","ghy west":"KAMRUP METRO","ge":"KAMRUP METRO","ghy.west":"KAMRUP METRO","ghy. west":"KAMRUP METRO","ghywest":"KAMRUP METRO",
                                          "kok":"KOKRAJHAR",
                                          "rangia":"KAMRUP",
                                          "gdd":"KAMRUP METRO"
                                         }

# METHOD-2 WEIGHTAGE METHOD
# GET TENDER DISTRICT BASED ON externalReference COLUMN

tenders_df['tender_district_externalReference'] = None
for idx, row in tenders_df.iterrows():

    district_identifier = str(row['tender_externalreference']).split(r'/')[0].lower()
    if 'rgr' in district_identifier:
        district_identifier = district_identifier.split('rgr')[0].strip()[:-1]

    if district_identifier in three_letter_distirct_identifiers_dict:
        tenders_df.loc[idx,'tender_district_externalReference'] = three_letter_distirct_identifiers_dict[district_identifier]

for idx, row in tenders_df.iterrows():
    if row['tender_externalreference'] != None:
        continue
    tender_slug = str(row['tender_externalreference'])
    tender_slug = re.sub(r'[^a-zA-Z0-9 \n\.]', ' ', tender_slug)
    for district in assam_districts:
        if re.findall(r'\b%s\b'%district.lower().strip(), tender_slug.lower()):
            tenders_df.loc[idx,'tender_district_externalReference'] = district
            break

## REVENUE
for idx, row in tenders_df.iterrows():
    if row['tender_externalreference'] != None:
        continue

    tender_slug = str(row['tender_externalreference'])
    tender_slug = re.sub(r'[^a-zA-Z0-9 \n\.]', ' ', tender_slug)

    for revenue_circle in assam_revenue_circles:
        if re.findall(r'\b%s\b'%revenue_circle.lower().strip(), tender_slug.lower()):
            tenders_df.loc[idx,'tender_district_externalReference'] = assam_revenue_circles_dict[revenue_circle]['district_2']
            break

## SUB DISTRICT
for idx, row in tenders_df.iterrows():
    if row['tender_externalreference'] != None:
        continue

    tender_slug = str(row['tender_externalreference'])
    tender_slug = re.sub(r'[^a-zA-Z0-9 \n\.]', ' ', tender_slug)

    for sub_district in assam_sub_districts:
        if re.findall(r'\b%s\b'%sub_district.lower(), tender_slug.lower()):
            tenders_df.loc[idx,'tender_district_externalReference'] = assam_subdist_dict[sub_district]['district_2']
            break

# GET TENDER DISTRICT BASED ON TITLE AND WORK DESCRIPTION

tenders_df['tender_district_title_description'] = None
for idx, row in tenders_df.iterrows():
    tender_slug = str(row['tender_title']) + ' ' + str(row['Work Description'])
    tender_slug = re.sub(r'[^a-zA-Z0-9 \n\.]', ' ', tender_slug)
    for district in assam_districts:
        if re.findall(r'\b%s\b'%district.lower().strip(), tender_slug.lower()):
            tenders_df.loc[idx,'tender_district_title_description'] = district
            break

## REVENUE
for idx, row in tenders_df.iterrows():
    if row['tender_district_title_description'] != None:
        continue

    tender_slug = str(row['tender_title']) + ' ' + str(row['Work Description'])
    tender_slug = re.sub(r'[^a-zA-Z0-9 \n\.]', ' ', tender_slug)

    for revenue_circle in assam_revenue_circles:
        if re.findall(r'\b%s\b'%revenue_circle.lower().strip(), tender_slug.lower()):
            tenders_df.loc[idx,'tender_district_title_description'] = assam_revenue_circles_dict[revenue_circle]['district_2']
            break

## SUB DISTRICT
for idx, row in tenders_df.iterrows():
    if row['tender_district_title_description'] != None:
        continue

    tender_slug = str(row['tender_title']) + ' ' + str(row['Work Description'])
    tender_slug = re.sub(r'[^a-zA-Z0-9 \n\.]', ' ', tender_slug)

    for sub_district in assam_sub_districts:
        if re.findall(r'\b%s\b'%sub_district.lower(), tender_slug.lower()):
            tenders_df.loc[idx,'tender_district_title_description'] = assam_subdist_dict[sub_district]['district_2']
            break

# GET TENDER DISTRICT BASED ON LOCATION COLUMN
tenders_df['tender_district_location'] = None
for idx, row in tenders_df.iterrows():
    tender_slug = str(row['location'])
    tender_slug = re.sub(r'[^a-zA-Z0-9 \n\.]', ' ', tender_slug)
    for district in assam_districts:
        if re.findall(r'\b%s\b'%district.lower().strip(), tender_slug.lower()):
            tenders_df.loc[idx,'tender_district_location'] = district
            break

## REVENUE
for idx, row in tenders_df.iterrows():
    if row['tender_district_location'] != None:
        continue

    tender_slug = str(row['location'])
    tender_slug = re.sub(r'[^a-zA-Z0-9 \n\.]', ' ', tender_slug)

    for revenue_circle in assam_revenue_circles:
        if re.findall(r'\b%s\b'%revenue_circle.lower().strip(), tender_slug.lower()):
            tenders_df.loc[idx,'tender_district_location'] = assam_revenue_circles_dict[revenue_circle]['district_2']
            break

## SUB DISTRICT
for idx, row in tenders_df.iterrows():
    if row['tender_district_location'] != None:
        continue

    tender_slug = str(row['location'])
    tender_slug = re.sub(r'[^a-zA-Z0-9 \n\.]', ' ', tender_slug)

    for sub_district in assam_sub_districts:
        if re.findall(r'\b%s\b'%sub_district.lower(), tender_slug.lower()):
            tenders_df.loc[idx,'tender_district_location'] = assam_subdist_dict[sub_district]['district_2']
            break

# BTC FLAG (Bodoland Territorial Council tenders are administratively
# distinct from regular district departments; kept active, unlike the
# Odisha heat pipeline where this flag is inert.)
tenders_df['BTC_flag'] = None
for idx, row in tenders_df.iterrows():
    BTC_flag = False

    department_slug = str(row["Organisation Chain"]) + ' ' + str(row["Department"])
    department_slug = re.sub(r'[^a-zA-Z0-9 \n\.]', ' ', department_slug)
    if re.findall(r"bodoland", department_slug.lower()):
        BTC_flag = True

    bodoland_dept_slugs = ["BoTC", "BTC"]
    for slug in bodoland_dept_slugs:
        if slug in str(row["Tender ID"]):
            BTC_flag = True

    tenders_df.loc[idx,'BTC_flag'] = BTC_flag

# WEIGHTAGE LOGIC
tenders_df['tender_district_externalReference'].fillna('NA',inplace=True)
tenders_df['tender_district_title_description'].fillna('NA',inplace=True)
tenders_df['tender_district_location'].fillna('NA',inplace=True)

tenders_df['DISTRICT_FINALISED'] = ''

for idx, row in tenders_df.iterrows():
    district1 = row['tender_district_externalReference']
    district2 = row['tender_district_title_description']
    district3 = row['tender_district_location']
    districts = [district1,district2,district3]
    districts = set([x for x in districts if x!='NA'])

    if len(districts)==1:
        DISTRICT_SELECTED = list(districts)[0]
    elif len(districts)==0:
        DISTRICT_SELECTED = 'NA'
    else:
        DISTRICT_SELECTED = 'CONFLICT'

    tenders_df.loc[idx,'DISTRICT_FINALISED'] = DISTRICT_SELECTED

tenders_df.to_csv(root+'/data_extractor/TENDERS/data/heattenders_districtgeotagged.csv',index=False)

total_number_of_heat_tenders = tenders_df.shape[0]
unidentified_heat_tenders = tenders_df[tenders_df['DISTRICT_FINALISED']=='NA'].shape[0]
conflict_heat_tenders = tenders_df[tenders_df['DISTRICT_FINALISED']=='CONFLICT'].shape[0]

percentage = (total_number_of_heat_tenders - unidentified_heat_tenders + conflict_heat_tenders)*100/total_number_of_heat_tenders

print('Total number of heat related tenders: ', total_number_of_heat_tenders)
print('Number of tenders whose district could not be geo-tagged: ',unidentified_heat_tenders)
print('Number of tenders whose district identification is a CONFLICT: ',conflict_heat_tenders)
print(percentage)
