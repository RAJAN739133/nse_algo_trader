"""Configurable stock universe — NIFTY 50 / 100 / 250."""
import os
NIFTY_50 = ["RELIANCE","HDFCBANK","ICICIBANK","TCS","INFY","BHARTIARTL","SBIN","ITC","LT","KOTAKBANK","AXISBANK","BAJFINANCE","TATAMOTORS","MARUTI","SUNPHARMA","HCLTECH","WIPRO","TATASTEEL","NTPC","POWERGRID","COALINDIA","ONGC","TITAN","ASIANPAINT","ULTRACEMCO","BAJAJFINSV","NESTLEIND","TECHM","HINDALCO","HINDUNILVR","JSWSTEEL","ADANIPORTS","DRREDDY","CIPLA","EICHERMOT","BRITANNIA","DIVISLAB","HEROMOTOCO","BPCL","GRASIM","APOLLOHOSP","TRENT","TATACONSUM","SBILIFE","HDFCLIFE","M&M","BAJAJ-AUTO","SHREECEM","INDUSINDBK","IOC"]
NIFTY_100_EXTRA = ["ADANIENT","ADANIGREEN","AMBUJACEM","BANKBARODA","BEL","BERGEPAINT","CANBK","CHOLAFIN","COLPAL","DABUR","DLF","GAIL","GODREJCP","HAVELLS","HAL","IDFCFIRSTB","INDHOTEL","IRCTC","IRFC","JINDALSTEL","LUPIN","MARICO","MAXHEALTH","MOTHERSON","MUTHOOTFIN","NHPC","OBEROIRLTY","PFC","PNB","POLYCAB","RECLTD","SIEMENS","SRF","TORNTPHARM","TVSMOTOR","VEDL","ETERNAL","ZYDUSLIFE"]
NIFTY_250_EXTRA = ["ABB","ABCAPITAL","ACC","ALKEM","ASHOKLEY","ASTRAL","ATUL","AUBANK","AUROPHARMA","BALKRISIND","BANDHANBNK","BATAINDIA","BHEL","BIOCON","CANFINHOME","COFORGE","CROMPTON","CUMMINSIND","DEEPAKNTR","DIXON","ESCORTS","EXIDEIND","FEDERALBNK","FORTIS","GMRINFRA","GRANULES","HDFCAMC","HINDPETRO","IGL","INDUSTOWER","IPCA","JUBLFOOD","KEI","KPITTECH","LALPATHLAB","LAURUSLABS","LICHSGFIN","LTTS","MANAPPURAM","MGL","MPHASIS","MRF","NAUKRI","NAVINFLUOR","NMDC","OFSS","PAGEIND","PERSISTENT","PETRONET","PIIND","PRESTIGE","RBLBANK","SAIL","STARHEALTH","SUNDARMFIN","SUNTV","TATACOMM","TATAELXSI","TATAPOWER","TIINDIA","TRIDENT","UBL","UNIONBANK","UPL","VOLTAS","WHIRLPOOL"]
ACTIVE_UNIVERSE = os.getenv("UNIVERSE", "nifty250")
def get_universe(name=None):
    name = (name or ACTIVE_UNIVERSE).lower().replace(" ","").replace("_","")
    if "250" in name: return NIFTY_50 + NIFTY_100_EXTRA + NIFTY_250_EXTRA
    elif "100" in name: return NIFTY_50 + NIFTY_100_EXTRA
    return NIFTY_50
DEFAULT_UNIVERSE = get_universe()
ALL_STOCKS = NIFTY_50 + NIFTY_100_EXTRA + NIFTY_250_EXTRA
# Comprehensive sector mapping for diversification
STOCK_SECTORS = {
    # Banking
    "HDFCBANK": "Banking", "ICICIBANK": "Banking", "SBIN": "Banking", "KOTAKBANK": "Banking",
    "AXISBANK": "Banking", "INDUSINDBK": "Banking", "BANKBARODA": "Banking", "PNB": "Banking",
    "CANBK": "Banking", "IDFCFIRSTB": "Banking", "FEDERALBNK": "Banking", "AUBANK": "Banking",
    "BANDHANBNK": "Banking", "RBLBANK": "Banking", "UNIONBANK": "Banking",
    
    # NBFC/Finance
    "BAJFINANCE": "Finance", "BAJAJFINSV": "Finance", "CHOLAFIN": "Finance", 
    "MUTHOOTFIN": "Finance", "LICHSGFIN": "Finance", "MANAPPURAM": "Finance",
    "PFC": "Finance", "RECLTD": "Finance", "CANFINHOME": "Finance", "HDFCAMC": "Finance",
    
    # IT
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT", "TECHM": "IT",
    "LTTS": "IT", "MPHASIS": "IT", "COFORGE": "IT", "PERSISTENT": "IT",
    "KPITTECH": "IT", "TATAELXSI": "IT", "OFSS": "IT", "NAUKRI": "IT",
    
    # Pharma
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma", "DIVISLAB": "Pharma",
    "LUPIN": "Pharma", "TORNTPHARM": "Pharma", "AUROPHARMA": "Pharma", "BIOCON": "Pharma",
    "ALKEM": "Pharma", "ZYDUSLIFE": "Pharma", "GRANULES": "Pharma", "LAURUSLABS": "Pharma",
    "IPCA": "Pharma", "LALPATHLAB": "Pharma", "APOLLOHOSP": "Healthcare", "MAXHEALTH": "Healthcare",
    "FORTIS": "Healthcare", "STARHEALTH": "Healthcare",
    
    # Auto
    "MARUTI": "Auto", "M&M": "Auto", "TATAMOTORS": "Auto", "BAJAJ-AUTO": "Auto",
    "HEROMOTOCO": "Auto", "EICHERMOT": "Auto", "TVSMOTOR": "Auto", "ASHOKLEY": "Auto",
    "ESCORTS": "Auto", "MOTHERSON": "Auto", "EXIDEIND": "Auto", "MRF": "Auto",
    
    # Energy/Oil
    "RELIANCE": "Energy", "ONGC": "Energy", "BPCL": "Energy", "IOC": "Energy",
    "HINDPETRO": "Energy", "GAIL": "Energy", "PETRONET": "Energy", "IGL": "Energy", "MGL": "Energy",
    
    # Power
    "NTPC": "Power", "POWERGRID": "Power", "TATAPOWER": "Power", "NHPC": "Power",
    "ADANIGREEN": "Power", "IRFC": "Power",
    
    # Metals
    "TATASTEEL": "Metals", "JSWSTEEL": "Metals", "HINDALCO": "Metals", "VEDL": "Metals",
    "JINDALSTEL": "Metals", "SAIL": "Metals", "NMDC": "Metals", "COALINDIA": "Metals",
    
    # FMCG
    "ITC": "FMCG", "HINDUNILVR": "FMCG", "NESTLEIND": "FMCG", "BRITANNIA": "FMCG",
    "DABUR": "FMCG", "MARICO": "FMCG", "COLPAL": "FMCG", "GODREJCP": "FMCG",
    "TATACONSUM": "FMCG", "UBL": "FMCG",
    
    # Infrastructure/Construction
    "LT": "Infra", "ADANIPORTS": "Infra", "ULTRACEMCO": "Cement", "SHREECEM": "Cement",
    "AMBUJACEM": "Cement", "ACC": "Cement", "GRASIM": "Cement", "DLF": "Realty",
    "OBEROIRLTY": "Realty", "PRESTIGE": "Realty", "GMRINFRA": "Infra",
    
    # Telecom
    "BHARTIARTL": "Telecom", "INDUSTOWER": "Telecom", "TATACOMM": "Telecom",
    
    # Consumer/Retail
    "TITAN": "Consumer", "ASIANPAINT": "Consumer", "BERGEPAINT": "Consumer",
    "TRENT": "Retail", "ETERNAL": "Tech", "IRCTC": "Consumer", "JUBLFOOD": "Consumer",
    "BATAINDIA": "Consumer", "PAGEIND": "Consumer",
    
    # Insurance
    "SBILIFE": "Insurance", "HDFCLIFE": "Insurance",
    
    # Defence/Engineering
    "HAL": "Defence", "BEL": "Defence", "BHEL": "Engineering",
    "SIEMENS": "Engineering", "ABB": "Engineering", "CUMMINSIND": "Engineering",
    "HAVELLS": "Engineering", "POLYCAB": "Engineering", "CROMPTON": "Engineering",
    "VOLTAS": "Engineering", "DIXON": "Engineering", "KEI": "Engineering",
    
    # Others
    "ADANIENT": "Conglomerate", "SRF": "Chemicals", "PIIND": "Chemicals",
    "NAVINFLUOR": "Chemicals", "DEEPAKNTR": "Chemicals", "ATUL": "Chemicals",
    "ASTRAL": "Chemicals", "TRIDENT": "Textiles", "INDHOTEL": "Hotels",
    "SUNTV": "Media", "TIINDIA": "Auto", "WHIRLPOOL": "Consumer",
    "UPL": "Chemicals", "SUNDARMFIN": "Finance", "BALKRISIND": "Auto",
}
