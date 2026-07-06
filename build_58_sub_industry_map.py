"""
build_58_sub_industry_map.py
============================
Maps every ticker in nifty1000_list.csv to one of the 58 official
Sub-Industry Alpha Playbook categories.

Mapping priority:
  1. sector_granular from latest Parquet cache (yfinance industry string)
     → mapped through YFINANCE_TO_PLAYBOOK dict
  2. Fallback: existing Sub_Industry in CSV → mapped through BROAD_TO_PLAYBOOK
  3. Fallback: "Other"

After running, nifty1000_list.csv will have an updated Sub_Industry column
with the 58 playbook names, which backfill_rotation_history.py will use.
"""

import pandas as pd
import glob
import os

# ── The 58 Official Sub-Industries from the Playbook ─────────────────────────
PLAYBOOK_58 = [
    "Aerospace & Defense",
    "Agricultural Food & other Products",
    "Agricultural, Commercial & Construction Vehicles",
    "Auto Components",
    "Automobiles",
    "Banks",
    "Beverages",
    "Capital Markets",
    "Cement & Cement Products",
    "Chemicals & Petrochemicals",
    "Cigarettes & Tobacco Products",
    "Commercial Services & Supplies",
    "Construction",
    "Consumable Fuels",
    "Consumer Durables",
    "Diversified",
    "Diversified FMCG",
    "Diversified Metals",
    "Electrical Equipment",
    "Engineering Services",
    "Entertainment",
    "Ferrous Metals",
    "Fertilizers & Agrochemicals",
    "Finance",
    "Financial Technology (Fintech)",
    "Food Products",
    "Gas",
    "Healthcare Equipment & Supplies",
    "Healthcare Services",
    "Household Products",
    "IT - Hardware",
    "IT - Services",
    "IT - Software",
    "Industrial Manufacturing",
    "Industrial Products",
    "Insurance",
    "Leisure Services",
    "Media",
    "Metals & Minerals Trading",
    "Minerals & Mining",
    "Non - Ferrous Metals",
    "Oil",
    "Other Construction Materials",
    "Other Consumer Services",
    "Other Utilities",
    "Paper, Forest & Jute Products",
    "Personal Products",
    "Petroleum Products",
    "Pharmaceuticals & Biotechnology",
    "Power",
    "Printing & Publication",
    "Realty",
    "Retailing",
    "Telecom - Equipment & Accessories",
    "Telecom - Services",
    "Textiles & Apparels",
    "Transport Infrastructure",
    "Transport Services",
]

# ── yfinance sector_granular → Playbook sub-industry ─────────────────────────
YFINANCE_TO_PLAYBOOK = {
    # Aerospace & Defense
    "Aerospace & Defense": "Aerospace & Defense",

    # Autos
    "Auto Components & Equipments": "Auto Components",
    "Passenger Cars & Utility Vehicles": "Automobiles",
    "2/3 Wheelers": "Automobiles",
    "Commercial Vehicles": "Agricultural, Commercial & Construction Vehicles",
    "Tractors": "Agricultural, Commercial & Construction Vehicles",
    "Construction Vehicles": "Agricultural, Commercial & Construction Vehicles",

    # Banks
    "Private Sector Bank": "Banks",
    "Public Sector Bank": "Banks",
    "Other Bank": "Banks",
    "Microfinance Institutions": "Finance",

    # Beverages
    "Breweries & Distilleries": "Beverages",
    "Other Beverages": "Beverages",

    # Capital Markets / Finance
    "Stockbroking & Allied": "Capital Markets",
    "Asset Management Company": "Capital Markets",
    "Exchange and Data Platform": "Capital Markets",
    "Depositories Clearing Houses and Other Intermediaries": "Capital Markets",
    "Investment Company": "Finance",
    "Non Banking Financial Company (NBFC)": "Finance",
    "Housing Finance Company": "Finance",
    "Financial Institution": "Finance",
    "Financial Services": "Finance",
    "Financial Products Distributor": "Finance",
    "Other Financial Services": "Finance",
    "Financial Technology (Fintech)": "Financial Technology (Fintech)",

    # Cement & Building Materials
    "Cement & Cement Products": "Cement & Cement Products",
    "Ceramics": "Other Construction Materials",
    "Plywood Boards/ Laminates": "Other Construction Materials",
    "Sanitary Ware": "Other Construction Materials",
    "Industrial Minerals": "Other Construction Materials",

    # Chemicals
    "Specialty Chemicals": "Chemicals & Petrochemicals",
    "Commodity Chemicals": "Chemicals & Petrochemicals",
    "Petrochemicals": "Chemicals & Petrochemicals",
    "Paints": "Chemicals & Petrochemicals",
    "Carbon Black": "Chemicals & Petrochemicals",
    "Industrial Gases": "Chemicals & Petrochemicals",

    # Cigarettes & Tobacco
    "Cigarettes & Tobacco Products": "Cigarettes & Tobacco Products",

    # Commercial Services
    "Business Process Outsourcing (BPO)/ Knowledge Process Outsourcing (KPO)": "Commercial Services & Supplies",
    "Diversified Commercial Services": "Commercial Services & Supplies",
    "Stationary": "Commercial Services & Supplies",
    "Logistics Solution Provider": "Transport Services",

    # Construction / Infra
    "Civil Construction": "Construction",
    "Residential Commercial Projects": "Realty",
    "Real Estate": "Realty",

    # Consumer Durables
    "Household Appliances": "Consumer Durables",
    "Consumer Electronics": "Consumer Durables",

    # Diversified
    "Diversified": "Diversified",
    "Holding Company": "Diversified",

    # Diversified FMCG
    "Diversified FMCG": "Diversified FMCG",

    # Electrical Equipment
    "Heavy Electrical Equipment": "Electrical Equipment",
    "Other Electrical Equipment": "Electrical Equipment",
    "Cables - Electricals": "Electrical Equipment",
    "Power - Transmission": "Electrical Equipment",
    "Electrodes & Refractories": "Electrical Equipment",

    # Metals - Ferrous
    "Iron & Steel Products": "Ferrous Metals",
    "Iron & Steel": "Ferrous Metals",

    # Metals - Non Ferrous
    "Aluminium": "Non - Ferrous Metals",
    "Copper": "Non - Ferrous Metals",
    "Zinc": "Non - Ferrous Metals",
    "Diversified Metals": "Diversified Metals",

    # Metals & Minerals Trading
    "Trading - Minerals": "Metals & Minerals Trading",
    "Abrasives & Bearings": "Industrial Products",
    "Castings & Forgings": "Industrial Products",

    # Fertilizers & Agrochemicals
    "Fertilizers": "Fertilizers & Agrochemicals",
    "Pesticides & Agrochemicals": "Fertilizers & Agrochemicals",

    # Food Products & Agri
    "Packaged Foods": "Food Products",
    "Edible Oil": "Food Products",
    "Animal Feed": "Agricultural Food & other Products",
    "Other Food Products": "Agricultural Food & other Products",
    "Other Agricultural Products": "Agricultural Food & other Products",
    "Tea & Coffee": "Agricultural Food & other Products",
    "Sugar": "Agricultural Food & other Products",

    # Gas / Oil
    "Gas Transmission/Marketing": "Gas",
    "LPG/CNG/PNG/LNG Supplier": "Gas",
    "Trading - Gas": "Gas",
    "Oil Exploration & Production": "Oil",
    "Refineries & Marketing": "Petroleum Products",
    "Oil Storage & Transportation": "Petroleum Products",
    "Lubricants": "Petroleum Products",

    # Coal
    "Coal": "Consumable Fuels",

    # Healthcare
    "Hospital": "Healthcare Services",
    "Healthcare Service Provider": "Healthcare Services",
    "Healthcare Research Analytics & Technology": "Healthcare Services",
    "Pharmaceuticals": "Pharmaceuticals & Biotechnology",
    "Medical Equipment & Supplies": "Healthcare Equipment & Supplies",

    # Household Products
    "Household Products": "Household Products",
    "Personal Care": "Personal Products",

    # IT
    "Computers - Software & Consulting": "IT - Software",
    "Software Products": "IT - Software",
    "IT Enabled Services": "IT - Services",
    "Technology": "IT - Services",
    "Internet & Catalogue Retail": "Financial Technology (Fintech)",
    "E-Retail/ E-Commerce": "Retailing",

    # Industrial / Engineering
    "Industrial Products": "Industrial Products",
    "Other Industrial Products": "Industrial Products",
    "Industrial Manufacturing": "Industrial Manufacturing",
    "Compressors Pumps & Diesel Engines": "Industrial Manufacturing",
    "Plastic Products - Industrial": "Industrial Products",

    # Insurance
    "General Insurance": "Insurance",
    "Life Insurance": "Insurance",

    # Leisure / Hospitality
    "Hotels & Resorts": "Leisure Services",
    "Tour Travel Related Services": "Leisure Services",
    "Airline": "Transport Services",

    # Media / Entertainment
    "TV Broadcasting & Software Production": "Media",
    "Film Production Distribution & Exhibition": "Entertainment",
    "Media & Entertainment": "Entertainment",

    # Mining / Minerals
    "Minerals & Mining": "Minerals & Mining",

    # Paper / Forest
    "Paper & Paper Products": "Paper, Forest & Jute Products",

    # Power
    "Power Generation": "Power",
    "Integrated Power Utilities": "Power",
    "Power Distribution": "Other Utilities",

    # Real Estate
    "Port & Port services": "Transport Infrastructure",
    "Airport & Airport services": "Transport Infrastructure",
    "Ship Building & Allied Services": "Transport Services",
    "Shipping": "Transport Services",

    # Retailing
    "Speciality Retail": "Retailing",
    "Diversified Retail": "Retailing",

    # Telecom
    "Telecom - Cellular & Fixed line services": "Telecom - Services",
    "Telecom - Infrastructure": "Telecom - Services",
    "Other Telecom Services": "Telecom - Services",
    "Telecom - Equipment & Accessories": "Telecom - Equipment & Accessories",

    # Textiles
    "Other Textile Products": "Textiles & Apparels",
    "Garments & Apparels": "Textiles & Apparels",
    "Footwear": "Textiles & Apparels",
    "Gems Jewellery And Watches": "Consumer Durables",

    # Transport + Railways
    "Railway Wagons": "Transport Infrastructure",
    "Logistics Solution Provider": "Transport Services",
    "Trading & Distributors": "Commercial Services & Supplies",

    "Restaurants": "Leisure Services",
    "Tyres & Rubber Products": "Auto Components",
    "Explosives": "Chemicals & Petrochemicals",
    "Consumer Durables": "Consumer Durables",
    "Industrials": "Industrial Manufacturing",
    "Basic Materials": "Diversified Metals",
    "Healthcare": "Healthcare Services",
    "Consumer Defensive": "Diversified FMCG",
    "Energy": "Oil",
    "Utilities": "Power",
    "Communication Services": "Telecom - Services",
    "Financial Institution": "Finance",
    # ── Extension: complete coverage of all yfinance labels in nifty1000_list ──
    "Advertising Agencies": "Media",
    "Agricultural Inputs": "Fertilizers & Agrochemicals",
    "Airlines": "Transport Services",
    "Airports & Air Services": "Transport Infrastructure",
    "Aluminum": "Non - Ferrous Metals",
    "Apparel Manufacturing": "Textiles & Apparels",
    "Apparel Retail": "Retailing",
    "Asset Management": "Capital Markets",
    "Auto & Truck Dealerships": "Retailing",
    "Auto Manufacturers": "Automobiles",
    "Auto Parts": "Auto Components",
    "Banks - Regional": "Banks",
    "Beverages - Brewers": "Beverages",
    "Beverages - Non-Alcoholic": "Beverages",
    "Beverages - Wineries & Distilleries": "Beverages",
    "Biotechnology": "Pharmaceuticals & Biotechnology",
    "Broadcasting": "Media",
    "Building Materials": "Other Construction Materials",
    "Building Products & Equipment": "Other Construction Materials",
    "Business Equipment & Supplies": "Commercial Services & Supplies",
    "Chemicals": "Chemicals & Petrochemicals",
    "Coking Coal": "Consumable Fuels",
    "Communication Equipment": "Telecom - Equipment & Accessories",
    "Computer Hardware": "IT - Hardware",
    "Confectioners": "Food Products",
    "Conglomerates": "Diversified",
    "Consulting Services": "Commercial Services & Supplies",
    "Credit Services": "Finance",
    "Department Stores": "Retailing",
    "Diagnostics & Research": "Healthcare Services",
    "Discount Stores": "Retailing",
    "Drug Manufacturers - General": "Pharmaceuticals & Biotechnology",
    "Drug Manufacturers - Specialty & Generic": "Pharmaceuticals & Biotechnology",
    "Education & Training Services": "Other Consumer Services",
    "Electrical Equipment & Parts": "Electrical Equipment",
    "Electronic Components": "Industrial Manufacturing",
    "Electronics & Computer Distribution": "IT - Hardware",
    "Engineering & Construction": "Construction",
    "Entertainment": "Entertainment",
    "Farm & Heavy Construction Machinery": "Agricultural, Commercial & Construction Vehicles",
    "Farm Products": "Agricultural Food & other Products",
    "Financial Conglomerates": "Finance",
    "Financial Data & Stock Exchanges": "Capital Markets",
    "Food Distribution": "Food Products",
    "Footwear & Accessories": "Consumer Durables",
    "Furnishings, Fixtures & Appliances": "Consumer Durables",
    "Gambling": "Leisure Services",
    "Gold": "Minerals & Mining",
    "Grocery Stores": "Retailing",
    "Health Information Services": "Healthcare Services",
    "Home Improvement Retail": "Retailing",
    "Household & Personal Products": "Personal Products",
    "Industrial Distribution": "Commercial Services & Supplies",
    "Information Technology Services": "IT - Services",
    "Infrastructure Operations": "Construction",
    "Insurance - Diversified": "Insurance",
    "Insurance - Life": "Insurance",
    "Insurance - Property & Casualty": "Insurance",
    "Insurance - Reinsurance": "Insurance",
    "Insurance Brokers": "Insurance",
    "Integrated Freight & Logistics": "Transport Services",
    "Internet Content & Information": "Other Consumer Services",
    "Internet Retail": "Retailing",
    "Leisure": "Leisure Services",
    "Lodging": "Leisure Services",
    "Lumber & Wood Production": "Paper, Forest & Jute Products",
    "Luxury Goods": "Consumer Durables",
    "Marine Shipping": "Transport Services",
    "Medical Care Facilities": "Healthcare Services",
    "Medical Devices": "Healthcare Equipment & Supplies",
    "Medical Distribution": "Healthcare Services",
    "Medical Instruments & Supplies": "Healthcare Equipment & Supplies",
    "Metal Fabrication": "Industrial Products",
    "Mortgage Finance": "Finance",
    "Oil & Gas E&P": "Oil",
    "Oil & Gas Equipment & Services": "Oil",
    "Oil & Gas Refining & Marketing": "Petroleum Products",
    "Other Industrial Metals & Mining": "Minerals & Mining",
    "Packaging & Containers": "Industrial Products",
    "Personal Services": "Other Consumer Services",
    "Pharmaceutical Retailers": "Retailing",
    "Pollution & Treatment Controls": "Industrial Manufacturing",
    "Publishing": "Printing & Publication",
    "Railroads": "Transport Services",
    "Real Estate - Development": "Realty",
    "Real Estate - Diversified": "Realty",
    "Real Estate - REITs": "Realty",
    "Real Estate Services": "Realty",
    "Rental & Leasing Services": "Commercial Services & Supplies",
    "Residential Construction": "Realty",
    "Resorts & Casinos": "Leisure Services",
    "Scientific & Technical Instruments": "Healthcare Equipment & Supplies",
    "Security & Protection Services": "Commercial Services & Supplies",
    "Semiconductor Equipment & Materials": "IT - Hardware",
    "Semiconductors": "IT - Hardware",
    "Shell Companies": "Diversified",
    "Software - Application": "IT - Software",
    "Software - Infrastructure": "IT - Software",
    "Solar": "Power",
    "Specialty Business Services": "Commercial Services & Supplies",
    "Specialty Industrial Machinery": "Industrial Manufacturing",
    "Specialty Retail": "Retailing",
    "Staffing & Employment Services": "Commercial Services & Supplies",
    "Steel": "Ferrous Metals",
    "Telecom Services": "Telecom - Services",
    "Textile Manufacturing": "Textiles & Apparels",
    "Thermal Coal": "Consumable Fuels",
    "Tobacco": "Cigarettes & Tobacco Products",
    "Tools & Accessories": "Industrial Products",
    "Travel Services": "Leisure Services",
    "Trucking": "Transport Services",
    "Utilities - Independent Power Producers": "Power",
    "Utilities - Regulated Electric": "Power",
    "Utilities - Regulated Gas": "Gas",
    "Utilities - Regulated Water": "Other Utilities",
    "Utilities - Renewable": "Power",
    "Waste Management": "Other Utilities",
}

# Company-specific overrides where the yfinance label is missing or wrong
TICKER_OVERRIDES = {
    "ELPROINTL.NS": "Realty",
}

# ── Broad CSV sector → Playbook sub-industry (fallback) ──────────────────────
BROAD_TO_PLAYBOOK = {
    "Capital Goods":        "Electrical Equipment",
    "Financial Services":   "Finance",
    "Pharma & Healthcare":  "Pharmaceuticals & Biotechnology",
    "Auto":                 "Automobiles",
    "Consumer Goods":       "Diversified FMCG",
    "IT & Technology":      "IT - Services",
    "Banking":              "Banks",
    "Chemicals":            "Chemicals & Petrochemicals",
    "Metals & Mining":      "Diversified Metals",
    "Infrastructure":       "Construction",
    "Oil & Gas":            "Oil",
    "Power & Utilities":    "Power",
    "Cement & Materials":   "Cement & Cement Products",
    "Retail":               "Retailing",
    "Hospitality":          "Leisure Services",
    "Insurance":            "Insurance",
    "Telecom":              "Telecom - Services",
    "Consumer Durables":    "Consumer Durables",
    "Real Estate":          "Realty",
    "Diversified":          "Diversified",
    "Textiles":             "Textiles & Apparels",
    "Aerospace & Defense":  "Aerospace & Defense",
    "Media":                "Media",
}


def main():
    # 1. Load nifty1000_list.csv
    csv_path = "data/nifty1000_list.csv"
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} tickers from nifty1000_list.csv")

    # 2. Load latest Parquet cache for sector_granular
    parquet_files = sorted(glob.glob("data/cache/market_master_*.parquet"))
    ticker_to_granular = {}
    if parquet_files:
        latest = parquet_files[-1]
        pq = pd.read_parquet(latest, columns=['ticker', 'sector', 'sector_granular'])
        ticker_to_granular = pq.dropna(subset=['sector_granular']).set_index('ticker')['sector_granular'].to_dict()
        ticker_to_broad    = pq.dropna(subset=['sector']).set_index('ticker')['sector'].to_dict()
        print(f"  Loaded {len(ticker_to_granular)} sector_granular entries from {os.path.basename(latest)}")
    else:
        ticker_to_broad = {}
        print("  [WARN] No Parquet cache found.")

    # 3. Map each ticker to a playbook sub-industry
    mapped = []
    unmatched_granular = set()
    unmatched_broad    = set()

    for _, row in df.iterrows():
        ticker = row['Ticker']
        result = None

        # Priority 1: sector_granular from Parquet
        granular = ticker_to_granular.get(ticker, "")
        if granular and granular in YFINANCE_TO_PLAYBOOK:
            result = YFINANCE_TO_PLAYBOOK[granular]
        else:
            if granular and granular not in ("", "Unknown", "Other"):
                unmatched_granular.add(granular)

        # Priority 2: broad sector from Parquet
        if not result:
            broad = ticker_to_broad.get(ticker, "")
            if broad and broad in BROAD_TO_PLAYBOOK:
                result = BROAD_TO_PLAYBOOK[broad]
            else:
                if broad and broad not in ("", "Unknown", "Other"):
                    unmatched_broad.add(broad)

        # Priority 3: existing CSV Sub_Industry → broad fallback
        if not result:
            existing = str(row.get('Sub_Industry', ''))
            if existing in BROAD_TO_PLAYBOOK:
                result = BROAD_TO_PLAYBOOK[existing]
            elif existing in PLAYBOOK_58:
                result = existing

        mapped.append(result or "Diversified")

    df['Sub_Industry'] = mapped

    # 4. Report
    counts = pd.Series(mapped).value_counts()
    coverage = (pd.Series(mapped) != "Diversified").sum()
    print(f"\n  Mapping Results:")
    print(f"  Total tickers     : {len(df)}")
    print(f"  Mapped (non-Other): {coverage}/{len(df)} ({100*coverage/len(df):.1f}%)")
    print(f"  Unique sub-industries: {counts.shape[0]}")
    print(f"\n  Distribution:")
    for si, cnt in counts.items():
        print(f"    {si:<45} {cnt:>4}")

    if unmatched_granular:
        print(f"\n  [INFO] Unmapped granular sectors (add to dict if needed):")
        for s in sorted(unmatched_granular):
            print(f"    '{s}'")

    # 5. Save
    df.to_csv(csv_path, index=False)
    print(f"\n  [DONE] Saved updated nifty1000_list.csv with 58 sub-industries")
    print(f"  Now run: python backfill_rotation_history.py")


if __name__ == "__main__":
    main()
