"""
Sector Consolidation Mapping
Maps granular sectors from Nifty 500 list to broader sector categories for analysis
"""

SECTOR_MAPPING = {
    # === BANKING & FINANCE ===
    "Private Sector Bank": "Banking",
    "Public Sector Bank": "Banking",
    "Other Bank": "Banking",
    "Non Banking Financial Company (NBFC)": "Financial Services",
    "Financial Institution": "Financial Services",
    "Housing Finance Company": "Financial Services",
    "Microfinance Institutions": "Financial Services",
    "Life Insurance": "Insurance",
    "General Insurance": "Insurance",
    "Asset Management Company": "Financial Services",
    "Stockbroking & Allied": "Financial Services",
    "Investment Company": "Financial Services",
    "Holding Company": "Financial Services",
    "Financial Products Distributor": "Financial Services",
    "Financial Technology (Fintech)": "Financial Services",
    "Depositories Clearing Houses and Other Intermediaries": "Financial Services",
    "Exchange and Data Platform": "Financial Services",
    "Other Financial Services": "Financial Services",
    
    # === IT & TECHNOLOGY ===
    "Computers - Software & Consulting": "IT & Technology",
    "IT Enabled Services": "IT & Technology",
    "Software Products": "IT & Technology",
    "Business Process Outsourcing (BPO)/ Knowledge Process Outsourcing (KPO)": "IT & Technology",
    "Internet & Catalogue Retail": "IT & Technology",
    
    # === PHARMA & HEALTHCARE ===
    "Pharmaceuticals": "Pharma & Healthcare",
    "Hospital": "Pharma & Healthcare",
    "Healthcare Service Provider": "Pharma & Healthcare",
    "Healthcare Research Analytics & Technology": "Pharma & Healthcare",
    "Medical Equipment & Supplies": "Pharma & Healthcare",
    
    # === CONSUMER GOODS ===
    "Diversified FMCG": "Consumer Goods",
    "Personal Care": "Consumer Goods",
    "Household Products": "Consumer Goods",
    "Packaged Foods": "Consumer Goods",
    "Other Food Products": "Consumer Goods",
    "Tea & Coffee": "Consumer Goods",
    "Edible Oil": "Consumer Goods",
    "Other Beverages": "Consumer Goods",
    "Breweries & Distilleries": "Consumer Goods",
    "Cigarettes & Tobacco Products": "Consumer Goods",
    "Sugar": "Consumer Goods",
    "Other Agricultural Products": "Consumer Goods",
    
    # === AUTO & ANCILLARY ===
    "Passenger Cars & Utility Vehicles": "Auto",
    "2/3 Wheelers": "Auto",
    "Commercial Vehicles": "Auto",
    "Tractors": "Auto",
    "Auto Components & Equipments": "Auto",
    "Tyres & Rubber Products": "Auto",
    
    # === OIL & GAS ===
    "Oil Exploration & Production": "Oil & Gas",
    "Refineries & Marketing": "Oil & Gas",
    "Gas Transmission/Marketing": "Oil & Gas",
    "LPG/CNG/PNG/LNG Supplier": "Oil & Gas",
    "Oil Storage & Transportation": "Oil & Gas",
    "Trading - Gas": "Oil & Gas",
    "Petrochemicals": "Oil & Gas",
    
    # === METALS & MINING ===
    "Iron & Steel": "Metals & Mining",
    "Iron & Steel Products": "Metals & Mining",
    "Aluminium": "Metals & Mining",
    "Copper": "Metals & Mining",
    "Zinc": "Metals & Mining",
    "Diversified Metals": "Metals & Mining",
    "Industrial Minerals": "Metals & Mining",
    "Coal": "Metals & Mining",
    
    # === POWER & UTILITIES ===
    "Power Generation": "Power & Utilities",
    "Integrated Power Utilities": "Power & Utilities",
    "Power Distribution": "Power & Utilities",
    "Power - Transmission": "Power & Utilities",
    
    # === INFRASTRUCTURE & CONSTRUCTION ===
    "Civil Construction": "Infrastructure",
    "Residential Commercial Projects": "Real Estate",
    "Port & Port services": "Infrastructure",
    "Airport & Airport services": "Infrastructure",
    "Railway Wagons": "Infrastructure",
    "Ship Building & Allied Services": "Infrastructure",
    "Logistics Solution Provider": "Infrastructure",
    "Shipping": "Infrastructure",
    
    # === CEMENT & BUILDING MATERIALS ===
    "Cement & Cement Products": "Cement & Materials",
    "Ceramics": "Cement & Materials",
    "Sanitary Ware": "Cement & Materials",
    "Plywood Boards/ Laminates": "Cement & Materials",
    "Paints": "Cement & Materials",
    
    # === CHEMICALS ===
    "Specialty Chemicals": "Chemicals",
    "Commodity Chemicals": "Chemicals",
    "Fertilizers": "Chemicals",
    "Pesticides & Agrochemicals": "Chemicals",
    "Industrial Gases": "Chemicals",
    "Carbon Black": "Chemicals",
    "Explosives": "Chemicals",
    
    # === CONSUMER DURABLES & ELECTRONICS ===
    "Consumer Electronics": "Consumer Durables",
    "Consumer Durables": "Consumer Durables",
    "Household Appliances": "Consumer Durables",
    
    # === CAPITAL GOODS & INDUSTRIALS ===
    "Heavy Electrical Equipment": "Capital Goods",
    "Other Electrical Equipment": "Capital Goods",
    "Cables - Electricals": "Capital Goods",
    "Industrial Products": "Capital Goods",
    "Other Industrial Products": "Capital Goods",
    "Compressors Pumps & Diesel Engines": "Capital Goods",
    "Abrasives & Bearings": "Capital Goods",
    "Castings & Forgings": "Capital Goods",
    "Electrodes & Refractories": "Capital Goods",
    "Construction Vehicles": "Capital Goods",
    
    # === AEROSPACE & DEFENSE ===
    "Aerospace & Defense": "Aerospace & Defense",
    
    # === TELECOM ===
    "Telecom - Cellular & Fixed line services": "Telecom",
    "Telecom - Infrastructure": "Telecom",
    "Telecom - Equipment & Accessories": "Telecom",
    "Other Telecom Services": "Telecom",
    
    # === RETAIL & CONSUMER SERVICES ===
    "Diversified Retail": "Retail",
    "Speciality Retail": "Retail",
    "E-Retail/ E-Commerce": "Retail",
    "Restaurants": "Retail",
    "Hotels & Resorts": "Hospitality",
    "Tour Travel Related Services": "Hospitality",
    "Airline": "Hospitality",
    
    # === MEDIA & ENTERTAINMENT ===
    "Media & Entertainment": "Media",
    "TV Broadcasting & Software Production": "Media",
    "Film Production Distribution & Exhibition": "Media",
    
    # === TEXTILES ===
    "Other Textile Products": "Textiles",
    "Garments & Apparels": "Textiles",
    
    # === MISC ===
    "Gems Jewellery And Watches": "Consumer Goods",
    "Footwear": "Consumer Goods",
    "Stationary": "Consumer Goods",
    "Plastic Products - Industrial": "Capital Goods",
    "Lubricants": "Oil & Gas",
    "Paper & Paper Products": "Capital Goods",
    "Animal Feed": "Consumer Goods",
    "Diversified": "Diversified",
    "Diversified Commercial Services": "Diversified",
    "Trading & Distributors": "Diversified",
    "Trading - Minerals": "Metals & Mining",
}


def consolidate_sector(granular_sector):
    """
    Maps a granular sector to its broader category.
    Returns 'Other' if not found.
    """
    return SECTOR_MAPPING.get(granular_sector, "Other")


def get_consolidated_sectors():
    """
    Returns list of all consolidated sector names.
    """
    return list(set(SECTOR_MAPPING.values()))
