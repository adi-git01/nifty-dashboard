"""
Sector-Specific Scoring Profiles
================================
Defines sector groups and their custom scoring configurations.
Each sector profile has different metric weights and normalization ranges
tailored to industry-specific business models.
"""

# ============================================
# SECTOR GROUP MAPPING
# ============================================
# Maps each sector from nifty500_list.csv to a scoring profile

SECTOR_PROFILE_MAP = {
    # === YFINANCE SECTOR NAMES (differ from Nifty 500 list) ===
    # yfinance uses broader categories - map them to our profiles
    "Technology": "IT",
    "Financial Services": "NBFC",  # Banks also come under this
    "Basic Materials": "COMMODITY",
    "Energy": "COMMODITY",
    "Consumer Cyclical": "CONSUMER",
    "Consumer Defensive": "CONSUMER",
    "Consumer Goods": "CONSUMER",
    "Consumer Services": "CONSUMER",
    "Utilities": "COMMODITY",
    "Industrials": "CAPGOODS",

    # === CONSOLIDATED SECTOR MAPPINGS (from utils.sector_mapping) ===
    "Banking": "BANK",
    "IT & Technology": "IT",
    "Pharma & Healthcare": "PHARMA",
    "Consumer Goods": "CONSUMER",
    "Consumer Durables": "CONSUMER",
    "Auto": "CONSUMER",  # Consumer Cyclical
    "Metals & Mining": "COMMODITY",
    "Power & Utilities": "COMMODITY",
    "Cement & Materials": "COMMODITY",
    "Chemicals": "COMMODITY",
    "Infrastructure": "CAPGOODS",
    "Capital Goods": "CAPGOODS",
    "Retail": "CONSUMER",
    "Hospitality": "CONSUMER",
    "Media": "CONSUMER",
    "Telecom": "CAPGOODS",  # Infra heavy
    "Textiles": "CONSUMER",
    
    # === NIFTY 500 SECTOR NAMES ===
    # FINANCIALS - BANKS
    "Private Sector Bank": "BANK",
    "Public Sector Bank": "BANK",
    "Other Bank": "BANK",
    
    # FINANCIALS - NBFC/INSURANCE
    "Non Banking Financial Company (NBFC)": "NBFC",
    "Housing Finance Company": "NBFC",
    "Microfinance Institutions": "NBFC",
    "Financial Institution": "NBFC",
    "Life Insurance": "INSURANCE",
    "General Insurance": "INSURANCE",
    "Asset Management Company": "NBFC",
    "Financial Products Distributor": "NBFC",
    "Financial Technology (Fintech)": "NBFC",
    "Stockbroking & Allied": "NBFC",
    "Investment Company": "NBFC",
    "Holding Company": "NBFC",
    
    # IT / SOFTWARE
    "Computers - Software & Consulting": "IT",
    "IT Enabled Services": "IT",
    "Software Products": "IT",
    "Business Process Outsourcing (BPO)/ Knowledge Process Outsourcing (KPO)": "IT",
    
    # CONSUMER / RETAIL (thin margins, high capital efficiency)
    "Gems Jewellery And Watches": "CONSUMER",
    "Diversified Retail": "CONSUMER",
    "Speciality Retail": "CONSUMER",
    "Diversified FMCG": "CONSUMER",
    "Personal Care": "CONSUMER",
    "Household Products": "CONSUMER",
    "Household Appliances": "CONSUMER",
    "Consumer Electronics": "CONSUMER",
    "Packaged Foods": "CONSUMER",
    "Footwear": "CONSUMER",
    "Garments & Apparels": "CONSUMER",
    "Hotels & Resorts": "CONSUMER",
    "Restaurants": "CONSUMER",
    "Other Beverages": "CONSUMER",
    "Breweries & Distilleries": "CONSUMER",
    "Tea & Coffee": "CONSUMER",
    "Edible Oil": "CONSUMER",
    
    # COMMODITY / CAPITAL GOODS (cyclical, capital-intensive)
    "Aluminium": "COMMODITY",
    "Iron & Steel": "COMMODITY",
    "Iron & Steel Products": "COMMODITY",
    "Diversified Metals": "COMMODITY",
    "Zinc": "COMMODITY",
    "Copper": "COMMODITY",
    "Industrial Minerals": "COMMODITY",
    "Cement & Cement Products": "COMMODITY",
    "Power Generation": "COMMODITY",
    "Power Distribution": "COMMODITY",
    "Integrated Power Utilities": "COMMODITY",
    "Power - Transmission": "COMMODITY",
    "Oil Exploration & Production": "COMMODITY",
    "Refineries & Marketing": "COMMODITY",
    "Coal": "COMMODITY",
    "Auto Components & Equipments": "CAPGOODS",
    "Heavy Electrical Equipment": "CAPGOODS",
    "Civil Construction": "CAPGOODS",
    "Construction Vehicles": "CAPGOODS",
    "Aerospace & Defense": "CAPGOODS",
    "Ship Building & Allied Services": "CAPGOODS",
    
    # PHARMA / HEALTHCARE (R&D intensive, high margins)
    "Pharmaceuticals": "PHARMA",
    "Healthcare": "PHARMA",  # yfinance uses this
    "Drug Manufacturers - Specialty & Generic": "PHARMA",
    "Hospital": "PHARMA",
    "Healthcare Service Provider": "PHARMA",
    "Medical Equipment & Supplies": "PHARMA",
    "Healthcare Research Analytics & Technology": "PHARMA",
}

# ============================================
# QUALITY SCORING PROFILES
# ============================================
# Each profile defines:
# - weights: How much each metric contributes
# - ranges: Min/Max for 0-10 normalization

QUALITY_PROFILES = {
    "BANK": {
        "description": "Banks use ROA (not ROE due to leverage), P/B for valuation",
        "weights": {
            "roa": 0.35,      # Key for banks - 1.5-2%+ is excellent
            "npm": 0.25,      # Net Interest Margin proxy
            "roe": 0.15,      # Less important due to leverage
            "gpm": 0.00,      # Not applicable
            "debt": 0.00,     # Banks are leveraged by design
            "eq": 0.15,       # Earnings quality matters
            "pb_bonus": 0.10  # P/B is key for bank valuation
        },
        "ranges": {
            "roa": (0.5, 2.0),    # 0.5% to 2% for banks
            "npm": (5, 25),       # Net profit margin
            "roe": (10, 20),      # Lower bar for banks
        },
        "bonuses": {
            "roa_excellent": {"threshold": 1.5, "bonus": 1.0},  # ROA > 1.5%
            "low_pb": {"threshold": 1.5, "bonus": 0.5}         # P/B < 1.5
        }
    },
    
    "NBFC": {
        "description": "NBFCs and insurance - hybrid between banks and general",
        "weights": {
            "roa": 0.25,
            "npm": 0.20,
            "roe": 0.25,
            "gpm": 0.00,
            "debt": 0.10,
            "eq": 0.20
        },
        "ranges": {
            "roa": (1, 4),
            "npm": (8, 25),
            "roe": (12, 25),
        },
        "bonuses": {}
    },
    
    "INSURANCE": {
        "description": "Insurance companies - focus on combined ratio proxy",
        "weights": {
            "roa": 0.20,
            "npm": 0.25,
            "roe": 0.30,
            "gpm": 0.00,
            "debt": 0.05,
            "eq": 0.20
        },
        "ranges": {
            "roa": (1, 5),
            "npm": (5, 20),
            "roe": (12, 22),
        },
        "bonuses": {}
    },
    
    "IT": {
        "description": "IT/Software - asset-light, high margins expected",
        "weights": {
            "roa": 0.08,
            "npm": 0.28,
            "roe": 0.32,
            "gpm": 0.05,
            "debt": 0.00,
            "eq": 0.15,
            "base": 0.12
        },
        "ranges": {
            "npm": (6, 22),   # Widened more: 15% is 65% of range
            "roe": (8, 35),   # Widened more: 18% ROE = 37% of range = ~3.7/10
            "roa": (5, 22),
        },
        "bonuses": {
            "roe_excellent": {"threshold": 28, "bonus": 1.0},
            "npm_excellent": {"threshold": 18, "bonus": 0.5},
            "npm_decent": {"threshold": 12, "bonus": 1.0}  # 15% NPM + 18% ROE gets boost
        }
    },
    
    "CONSUMER": {
        "description": "Consumer/Retail - thin margins but high capital efficiency",
        "weights": {
            "roa": 0.10,
            "npm": 0.08,      # Expected to be very thin, reduce weight
            "roe": 0.45,      # Key - efficiency matters most
            "gpm": 0.07,      # Very wide range accepted
            "debt": 0.15,
            "eq": 0.15
        },
        "ranges": {
            "npm": (0, 15),   # Widened upper to reward super-premium margins
            "gpm": (10, 45),  # 
            "roe": (8, 25),   # RELAXED: 8% base allows ~15% ROE to score decent (~4-5/10)
        },
        "bonuses": {
            "roe_excellent": {"threshold": 22, "bonus": 1.5}  # ROE >= 22% is king for retail
        }
    },
    
    "COMMODITY": {
        "description": "Commodity/Metals - cyclical, debt important but not killer",
        "weights": {
            "roa": 0.15,
            "npm": 0.18,
            "roe": 0.28,      # ROE more important
            "gpm": 0.14,
            "debt": 0.15,     # Reduced - cyclicals have volatile debt
            "eq": 0.10
        },
        "ranges": {
            "npm": (2, 15),
            "gpm": (12, 38),
            "roe": (5, 18),   # Narrowed top: 12% ROE = 58% of range = 5.8/10
        },
        "bonuses": {
            "low_debt": {"threshold": 80, "bonus": 1.5},   # D/E < 80% gets big bonus
            "roe_excellent": {"threshold": 14, "bonus": 0.5}
        }
    },
    
    "CAPGOODS": {
        "description": "Capital Goods - project-based, working capital intensive",
        "weights": {
            "roa": 0.15,
            "npm": 0.20,
            "roe": 0.25,
            "gpm": 0.15,
            "debt": 0.15,
            "eq": 0.10
        },
        "ranges": {
            "npm": (5, 18),
            "gpm": (20, 50),
            "roe": (12, 25),
        },
        "bonuses": {}
    },
    
    "PHARMA": {
        "description": "Pharma/Healthcare - R&D matters, high margins",
        "weights": {
            "roa": 0.12,
            "npm": 0.22,
            "roe": 0.28,      # ROE more important
            "gpm": 0.18,      # High for pharma
            "debt": 0.10,
            "eq": 0.10
        },
        "ranges": {
            "npm": (3, 20),   # Widened lower: 12% NPM is solid
            "gpm": (30, 65),  # Widened lower
            "roe": (5, 20),   # RELAXED: 5% base. 13% ROE -> ~5.3/10 (Average)
        },
        "bonuses": {
            "gpm_excellent": {"threshold": 55, "bonus": 0.8},
            "roe_excellent": {"threshold": 14, "bonus": 0.5}  # ROE > 14% is good for pharma
        }
    },
    
    "DEFAULT": {
        "description": "General balanced approach for all other sectors",
        "weights": {
            "roa": 0.12,
            "npm": 0.18,
            "roe": 0.30,
            "gpm": 0.10,
            "debt": 0.15,
            "eq": 0.15
        },
        "ranges": {
            "npm": (0, 18),
            "gpm": (15, 50),
            "roe": (5, 25),
            "roa": (2, 12),
        },
        "bonuses": {
            "roe_excellent": {"threshold": 22, "bonus": 1.0},
            "profitability_combo": {"roe_threshold": 18, "npm_threshold": 5, "bonus": 0.5}
        }
    }
}

# ============================================
# VALUE SCORING PROFILES
# ============================================
# Banks use P/B heavily, others use P/E

VALUE_PROFILES = {
    "BANK": {
        "use_pb_primary": True,
        "pb_weight": 0.50,   # P/B is primary for banks
        "pe_weight": 0.20,
        "peg_weight": 0.15,
        "fwd_pe_weight": 0.15,
        "pb_range": (0.5, 3.0)  # 0.5x to 3x P/B
    },
    "NBFC": {
        "use_pb_primary": True,
        "pb_weight": 0.35,
        "pe_weight": 0.30,
        "peg_weight": 0.20,
        "fwd_pe_weight": 0.15,
        "pb_range": (0.8, 4.0)
    },
    "DEFAULT": {
        "use_pb_primary": False,
        "pb_weight": 0.15,
        "pe_weight": 0.40,
        "peg_weight": 0.25,
        "fwd_pe_weight": 0.20,
        "pb_range": (0.5, 5.0)
    }
}


def get_sector_profile(sector: str) -> str:
    """
    Maps a sector name to its scoring profile.
    Returns 'DEFAULT' if sector not in mapping.
    """
    if not sector:
        return "DEFAULT"
    return SECTOR_PROFILE_MAP.get(sector, "DEFAULT")


def get_quality_config(sector: str) -> dict:
    """
    Returns the Quality scoring configuration for a sector.
    """
    profile = get_sector_profile(sector)
    return QUALITY_PROFILES.get(profile, QUALITY_PROFILES["DEFAULT"])


def get_value_config(sector: str) -> dict:
    """
    Returns the Value scoring configuration for a sector.
    Banks and NBFCs use P/B more heavily.
    """
    profile = get_sector_profile(sector)
    
    if profile in ["BANK", "NBFC"]:
        return VALUE_PROFILES.get(profile, VALUE_PROFILES["DEFAULT"])
    else:
        return VALUE_PROFILES["DEFAULT"]


def get_profile_description(sector: str) -> str:
    """
    Returns a human-readable description of the scoring profile.
    For display in stock reports.
    """
    profile = get_sector_profile(sector)
    config = QUALITY_PROFILES.get(profile, QUALITY_PROFILES["DEFAULT"])
    return f"{profile}: {config.get('description', 'Standard scoring')}"
