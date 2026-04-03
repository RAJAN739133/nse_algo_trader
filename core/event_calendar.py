"""
Event Calendar & Gap Trading Module
═══════════════════════════════════════════════════════════════════════════════

Tracks market-moving events and overnight gaps for trading opportunities.

Event Types:
1. Earnings announcements (quarterly results)
2. RBI policy announcements (interest rate decisions)
3. Union Budget (February) - MAJOR
4. State/General Elections - HIGH IMPACT
5. FII/DII flow data (daily) - Sentiment indicator
6. Economic data releases (GDP, IIP, CPI, PMI)
7. Corporate actions (dividends, splits, bonuses)
8. Index rebalancing (Nifty additions/deletions)
9. Global events (Fed meetings, US jobs data)

Gap Trading:
- Pre-market gap scanner with alerts (8:45 AM)
- Identifies overnight gaps from previous close
- Classifies gap type (breakaway, exhaustion, common)
- Generates gap-fill or gap-continuation signals

Post-Earnings Momentum:
- Tracks stocks that announced results yesterday
- Analyzes gap + volume + price action
- Generates momentum continuation signals

Claude Brain Integration:
- Analyzes event impact and sentiment
- Decides gap trade direction
- Sets dynamic targets based on gap size
"""

import json
import logging
import requests
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = Path("data/events")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Alerts directory
ALERTS_DIR = Path("data/alerts")
ALERTS_DIR.mkdir(parents=True, exist_ok=True)


class EventCalendar:
    """
    Tracks market-moving events for NSE stocks.
    """
    
    # Known RBI policy dates (typically bi-monthly)
    RBI_POLICY_MONTHS = [2, 4, 6, 8, 10, 12]  # Feb, Apr, Jun, Aug, Oct, Dec
    
    # High-impact event types and their market impact score (0-1)
    EVENT_IMPACT = {
        # Indian Market Events
        "earnings": 0.9,        # Very high - stock can move 5-15%
        "rbi_policy": 0.8,      # High - affects banking, rate-sensitive
        "budget": 0.95,         # Extreme - sector-wide impact
        "election": 0.9,        # Very high - uncertainty
        "fii_dii": 0.5,         # Medium - sentiment indicator
        "gdp_data": 0.6,        # Medium-high - economic health
        "cpi_inflation": 0.65,  # Medium-high - affects RBI stance
        "iip_data": 0.4,        # Medium - industrial activity
        "pmi_data": 0.5,        # Medium - leading indicator
        "dividend": 0.4,        # Medium - ex-date adjustment
        "bonus": 0.5,           # Medium - stock split effect
        "split": 0.3,           # Low - price adjustment only
        "agm": 0.2,             # Low - unless major announcements
        "index_change": 0.6,    # Medium-high - passive fund flows
        "expiry": 0.7,          # High - F&O expiry volatility
        
        # Global/US Events
        "fed_meeting": 0.75,    # High - affects global sentiment & INR
        "fed_minutes": 0.5,     # Medium - forward guidance
        "us_jobs": 0.6,         # Medium-high - global risk sentiment
        "us_cpi": 0.65,         # Medium-high - Fed rate expectations
        "us_gdp": 0.5,          # Medium - global growth indicator
        "ecb_meeting": 0.4,     # Medium - Euro zone impact
        "boj_meeting": 0.35,    # Medium-low - Yen carry trade
        
        # Geopolitical/Other
        "opec_meeting": 0.5,    # Medium - affects ONGC, Reliance
        "china_data": 0.45,     # Medium - metals, chemicals impact
        "crude_inventory": 0.3, # Low-medium - oil stocks
        "gold_event": 0.3,      # Low-medium - jewelry stocks
        
        # Special Events
        "credit_policy": 0.6,   # Medium-high - quarterly credit policy
        "gst_council": 0.5,     # Medium - sector-specific impacts
        "sebi_circular": 0.4,   # Medium - trading rule changes
        "msci_rebalance": 0.65, # Medium-high - FII flows
        "nifty_rebalance": 0.6, # Medium-high - passive fund flows
    }
    
    # Sector-wise impact mapping for events
    BUDGET_SECTORS = {
        "positive_usually": ["INFRA", "DEFENCE", "RAILWAY", "PSU_BANK"],
        "negative_usually": ["TOBACCO", "ALCOHOL", "LUXURY"],
        "volatile": ["PHARMA", "AUTO", "REALTY", "BANKING"]
    }
    
    ELECTION_SENSITIVE = ["PSU", "INFRA", "DEFENCE", "BANKING", "REALTY"]
    
    def __init__(self, claude_brain=None):
        self.claude_brain = claude_brain
        self.events_cache = self._load_cache()
        self.earnings_calendar = {}
        self.fii_dii_data = {}
    
    def _load_cache(self) -> dict:
        cache_file = CACHE_DIR / f"events_{date.today()}.json"
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text())
            except:
                pass
        return {"events": [], "last_updated": None}
    
    def _save_cache(self):
        cache_file = CACHE_DIR / f"events_{date.today()}.json"
        self.events_cache["last_updated"] = datetime.now().isoformat()
        cache_file.write_text(json.dumps(self.events_cache, indent=2, default=str))
    
    def fetch_earnings_calendar(self, symbols: List[str] = None) -> Dict:
        """
        Fetch upcoming earnings announcements.
        Uses NSE corporate announcements or falls back to static calendar.
        """
        today = date.today()
        
        # Known Q4 FY26 earnings schedule (approximate)
        # In production, this would fetch from NSE API
        q4_earnings = {
            # IT companies (typically early April)
            "TCS": date(2026, 4, 10),
            "INFY": date(2026, 4, 12),
            "WIPRO": date(2026, 4, 15),
            "HCLTECH": date(2026, 4, 17),
            "TECHM": date(2026, 4, 20),
            
            # Banks (mid-April)
            "HDFCBANK": date(2026, 4, 18),
            "ICICIBANK": date(2026, 4, 22),
            "KOTAKBANK": date(2026, 4, 24),
            "SBIN": date(2026, 4, 28),
            "AXISBANK": date(2026, 4, 25),
            
            # FMCG (late April)
            "HINDUNILVR": date(2026, 4, 25),
            "ITC": date(2026, 4, 28),
            "NESTLEIND": date(2026, 4, 29),
            
            # Reliance (late April)
            "RELIANCE": date(2026, 4, 20),
            
            # Pharma
            "SUNPHARMA": date(2026, 5, 5),
            "DRREDDY": date(2026, 5, 8),
            "CIPLA": date(2026, 5, 10),
        }
        
        upcoming = {}
        for sym, earn_date in q4_earnings.items():
            if symbols and sym not in symbols:
                continue
            days_until = (earn_date - today).days
            if -1 <= days_until <= 7:  # Yesterday to 7 days out
                upcoming[sym] = {
                    "date": str(earn_date),
                    "days_until": days_until,
                    "type": "earnings",
                    "impact": self.EVENT_IMPACT["earnings"],
                    "trading_advice": self._get_earnings_advice(days_until)
                }
        
        self.earnings_calendar = upcoming
        return upcoming
    
    def _get_earnings_advice(self, days_until: int) -> str:
        """Get trading advice based on days to earnings."""
        if days_until < 0:
            return "POST_EARNINGS: Trade the reaction, watch for gap"
        elif days_until == 0:
            return "EARNINGS_DAY: High volatility expected, reduce position size"
        elif days_until == 1:
            return "EVE_OF_EARNINGS: Consider reducing exposure"
        elif days_until <= 3:
            return "EARNINGS_WEEK: IV expansion, consider direction"
        else:
            return "APPROACHING: Monitor for pre-earnings drift"
    
    def get_rbi_policy_dates(self, year: int = None) -> List[date]:
        """Get RBI monetary policy announcement dates."""
        year = year or date.today().year
        
        # RBI typically announces on the first week of policy months
        policy_dates = []
        for month in self.RBI_POLICY_MONTHS:
            # Usually first Friday of the month
            first_day = date(year, month, 1)
            # Find first Friday
            days_until_friday = (4 - first_day.weekday()) % 7
            if days_until_friday == 0:
                days_until_friday = 7
            policy_date = first_day + timedelta(days=days_until_friday)
            policy_dates.append(policy_date)
        
        return policy_dates
    
    def check_rbi_impact(self) -> Optional[Dict]:
        """Check if RBI policy is near and which sectors are affected."""
        today = date.today()
        policy_dates = self.get_rbi_policy_dates()
        
        for policy_date in policy_dates:
            days_until = (policy_date - today).days
            if -1 <= days_until <= 3:
                return {
                    "date": str(policy_date),
                    "days_until": days_until,
                    "type": "rbi_policy",
                    "impact": self.EVENT_IMPACT["rbi_policy"],
                    "affected_sectors": ["BANKING", "NBFC", "REALTY", "AUTO"],
                    "affected_stocks": [
                        "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
                        "BAJFINANCE", "HDFC", "LIC", "SBILIFE",
                        "DLF", "GODREJPROP", "OBEROIRLTY",
                        "MARUTI", "TATAMOTORS", "M&M"
                    ],
                    "trading_advice": self._get_rbi_advice(days_until)
                }
        return None
    
    def _get_rbi_advice(self, days_until: int) -> str:
        """Get trading advice for RBI policy."""
        if days_until < 0:
            return "POST_POLICY: Trade sector reaction, banks volatile"
        elif days_until == 0:
            return "POLICY_DAY: Avoid banking stocks until 2:30 PM announcement"
        else:
            return "PRE_POLICY: Reduce banking exposure, expect sideways"
    
    # ════════════════════════════════════════════════════════════
    # BUDGET DAY
    # ════════════════════════════════════════════════════════════
    
    def get_budget_date(self, year: int = None) -> date:
        """Get Union Budget date (typically Feb 1)."""
        year = year or date.today().year
        budget_date = date(year, 2, 1)
        # If Feb 1 is Sunday, budget is on Feb 2
        if budget_date.weekday() == 6:
            budget_date = date(year, 2, 2)
        return budget_date
    
    def check_budget_impact(self) -> Optional[Dict]:
        """Check if Union Budget is near."""
        today = date.today()
        budget_date = self.get_budget_date()
        
        # Also check next year's budget
        if today.month > 2:
            budget_date = self.get_budget_date(today.year + 1)
        
        days_until = (budget_date - today).days
        
        if -1 <= days_until <= 7:
            return {
                "date": str(budget_date),
                "days_until": days_until,
                "type": "budget",
                "impact": self.EVENT_IMPACT["budget"],
                "positive_sectors": self.BUDGET_SECTORS["positive_usually"],
                "negative_sectors": self.BUDGET_SECTORS["negative_usually"],
                "volatile_sectors": self.BUDGET_SECTORS["volatile"],
                "affected_stocks": self._get_budget_sensitive_stocks(),
                "trading_advice": self._get_budget_advice(days_until)
            }
        return None
    
    def _get_budget_sensitive_stocks(self) -> List[str]:
        """Get stocks typically affected by budget."""
        return [
            # Infra/Construction
            "LT", "LTIM", "IRCON", "NBCC", "BEL",
            # PSU Banks
            "SBIN", "PNB", "BANKBARODA", "CANBK",
            # Defence
            "HAL", "BEL", "BHARATFORGE",
            # Railways
            "IRCTC", "RVNL", "IRFC",
            # Auto (EV focus)
            "TATAMOTORS", "M&M", "MARUTI",
            # Realty
            "DLF", "GODREJPROP", "OBEROIRLTY",
            # Tobacco (risk of tax hike)
            "ITC", "GODFRYPHLP",
        ]
    
    def _get_budget_advice(self, days_until: int) -> str:
        """Get trading advice for budget."""
        if days_until < 0:
            return "POST_BUDGET: Trade sector reactions, high volatility all day"
        elif days_until == 0:
            return "BUDGET_DAY: Expect 2-5% swings, reduce positions before 11 AM"
        elif days_until == 1:
            return "BUDGET_EVE: Reduce all positions, IV very high"
        elif days_until <= 3:
            return "BUDGET_WEEK: Position for expected announcements cautiously"
        else:
            return "BUDGET_APPROACHING: Monitor budget leaks and expectations"
    
    # ════════════════════════════════════════════════════════════
    # ELECTIONS
    # ════════════════════════════════════════════════════════════
    
    def get_election_dates(self, year: int = None) -> List[Dict]:
        """Get known election dates and phases."""
        year = year or date.today().year
        
        # Major elections calendar (approximate - update as announced)
        elections = []
        
        # Example: General Elections 2029 (hypothetical)
        if year == 2029:
            elections.append({
                "name": "General Elections 2029",
                "type": "general",
                "phases": [
                    date(2029, 4, 15),
                    date(2029, 4, 22),
                    date(2029, 4, 29),
                    date(2029, 5, 6),
                    date(2029, 5, 13),
                ],
                "result_date": date(2029, 5, 20),
            })
        
        # State elections (add as they're announced)
        # Example for 2026
        if year == 2026:
            elections.append({
                "name": "State Elections 2026",
                "type": "state",
                "states": ["West Bengal", "Tamil Nadu", "Kerala", "Assam"],
                "phases": [date(2026, 4, 1), date(2026, 4, 15)],
                "result_date": date(2026, 5, 2),
            })
        
        return elections
    
    def check_election_impact(self) -> Optional[Dict]:
        """Check if elections are near."""
        today = date.today()
        elections = self.get_election_dates()
        
        for election in elections:
            # Check result date
            result_date = election.get("result_date")
            if result_date:
                days_until_result = (result_date - today).days
                if -1 <= days_until_result <= 7:
                    return {
                        "name": election["name"],
                        "type": election.get("type", "election"),
                        "result_date": str(result_date),
                        "days_until_result": days_until_result,
                        "impact": self.EVENT_IMPACT["election"],
                        "affected_sectors": self.ELECTION_SENSITIVE,
                        "affected_stocks": self._get_election_sensitive_stocks(),
                        "trading_advice": self._get_election_advice(days_until_result, "result")
                    }
            
            # Check polling phases
            for phase_date in election.get("phases", []):
                days_until_phase = (phase_date - today).days
                if -1 <= days_until_phase <= 3:
                    return {
                        "name": election["name"],
                        "type": election.get("type", "election"),
                        "phase_date": str(phase_date),
                        "days_until_phase": days_until_phase,
                        "impact": self.EVENT_IMPACT["election"] * 0.7,  # Phase less impactful than result
                        "affected_sectors": self.ELECTION_SENSITIVE,
                        "trading_advice": self._get_election_advice(days_until_phase, "phase")
                    }
        
        return None
    
    def _get_election_sensitive_stocks(self) -> List[str]:
        """Stocks most affected by election outcomes."""
        return [
            # PSU Banks (govt policy)
            "SBIN", "PNB", "BANKBARODA", "CANBK",
            # Infra (govt spending)
            "LT", "NBCC", "IRCON",
            # Defence PSUs
            "HAL", "BEL", "BHARATFORGE",
            # Power PSUs
            "NTPC", "POWERGRID", "NHPC",
            # Realty (policy uncertainty)
            "DLF", "GODREJPROP",
            # Index heavyweights
            "RELIANCE", "HDFCBANK", "ICICIBANK",
        ]
    
    def _get_election_advice(self, days_until: int, event_type: str) -> str:
        """Get trading advice for elections."""
        if event_type == "result":
            if days_until < 0:
                return "POST_RESULT: Trade the reaction, expect gap + momentum"
            elif days_until == 0:
                return "RESULT_DAY: Extreme volatility, avoid new positions"
            else:
                return "PRE_RESULT: Reduce positions, hedge with options"
        else:
            if days_until == 0:
                return "POLLING_DAY: Moderate volatility expected"
            else:
                return "PRE_POLLING: Watch exit poll leaks"
    
    # ════════════════════════════════════════════════════════════
    # FII/DII DATA
    # ════════════════════════════════════════════════════════════
    
    def fetch_fii_dii_data(self) -> Dict:
        """
        Fetch FII/DII cash market data.
        In production, fetch from NSE or data provider.
        """
        today = date.today()
        
        # Simulated data structure (in production, fetch from NSE API)
        # NSE publishes this around 6 PM daily
        sample_data = {
            "date": str(today - timedelta(days=1)),  # Yesterday's data
            "fii": {
                "buy_value": 8500.0,    # Cr
                "sell_value": 7200.0,   # Cr
                "net_value": 1300.0,    # Cr (positive = buying)
            },
            "dii": {
                "buy_value": 6200.0,
                "sell_value": 5800.0,
                "net_value": 400.0,
            },
            "fii_index_futures": {
                "long_contracts": 125000,
                "short_contracts": 98000,
                "net_oi_change": 5000,
            },
            "interpretation": self._interpret_fii_dii(1300, 400)
        }
        
        self.fii_dii_data = sample_data
        return sample_data
    
    def _interpret_fii_dii(self, fii_net: float, dii_net: float) -> Dict:
        """Interpret FII/DII flows for trading."""
        sentiment = "neutral"
        confidence = 0.5
        advice = ""
        
        # Both buying = strong bullish
        if fii_net > 500 and dii_net > 0:
            sentiment = "bullish"
            confidence = 0.75
            advice = "FII+DII buying: Strong institutional support, favor LONG"
        
        # FII buying, DII selling = moderate bullish (FII leads)
        elif fii_net > 500 and dii_net < 0:
            sentiment = "mildly_bullish"
            confidence = 0.60
            advice = "FII buying, DII profit-booking: Cautious LONG"
        
        # FII selling, DII buying = market bottom attempt
        elif fii_net < -500 and dii_net > 0:
            sentiment = "mildly_bearish"
            confidence = 0.55
            advice = "FII selling, DII supporting: Range-bound, wait for clarity"
        
        # Both selling = bearish
        elif fii_net < -500 and dii_net < 0:
            sentiment = "bearish"
            confidence = 0.70
            advice = "FII+DII selling: Institutional exit, favor SHORT or cash"
        
        # Heavy FII selling
        elif fii_net < -1500:
            sentiment = "very_bearish"
            confidence = 0.80
            advice = "Heavy FII outflow: Expect weakness, SHORT bias"
        
        # Heavy FII buying
        elif fii_net > 2000:
            sentiment = "very_bullish"
            confidence = 0.80
            advice = "Heavy FII inflow: Expect strength, LONG bias"
        
        else:
            advice = "Mixed flows: No clear direction, trade technicals"
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "advice": advice,
            "fii_net_cr": fii_net,
            "dii_net_cr": dii_net,
        }
    
    def get_fii_sentiment(self) -> Dict:
        """Get current FII/DII sentiment."""
        if not self.fii_dii_data:
            self.fetch_fii_dii_data()
        return self.fii_dii_data.get("interpretation", {})
    
    # ════════════════════════════════════════════════════════════
    # F&O EXPIRY
    # ════════════════════════════════════════════════════════════
    
    def get_expiry_dates(self, year: int = None, month: int = None) -> Dict:
        """Get F&O expiry dates."""
        year = year or date.today().year
        month = month or date.today().month
        
        # Weekly expiry: Every Thursday
        # Monthly expiry: Last Thursday of month
        
        # Find all Thursdays in the month
        first_day = date(year, month, 1)
        if month == 12:
            last_day = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            last_day = date(year, month + 1, 1) - timedelta(days=1)
        
        thursdays = []
        current = first_day
        while current <= last_day:
            if current.weekday() == 3:  # Thursday
                thursdays.append(current)
            current += timedelta(days=1)
        
        return {
            "weekly_expiries": thursdays[:-1] if len(thursdays) > 1 else [],
            "monthly_expiry": thursdays[-1] if thursdays else None,
        }
    
    def check_expiry_impact(self) -> Optional[Dict]:
        """Check if F&O expiry is near."""
        today = date.today()
        expiry_info = self.get_expiry_dates()
        
        monthly = expiry_info.get("monthly_expiry")
        if monthly:
            days_until = (monthly - today).days
            if 0 <= days_until <= 2:
                return {
                    "type": "monthly_expiry",
                    "date": str(monthly),
                    "days_until": days_until,
                    "impact": self.EVENT_IMPACT["expiry"],
                    "trading_advice": "EXPIRY_WEEK: High volatility, max pain levels important",
                    "affected_stocks": ["NIFTY", "BANKNIFTY"] + self._get_high_oi_stocks()
                }
        
        # Check weekly expiry
        for weekly in expiry_info.get("weekly_expiries", []):
            days_until = (weekly - today).days
            if days_until == 0:
                return {
                    "type": "weekly_expiry",
                    "date": str(weekly),
                    "days_until": 0,
                    "impact": self.EVENT_IMPACT["expiry"] * 0.6,
                    "trading_advice": "WEEKLY_EXPIRY: Expect pinning near max pain"
                }
        
        return None
    
    def _get_high_oi_stocks(self) -> List[str]:
        """Stocks with high open interest (most affected by expiry)."""
        return [
            "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS",
            "SBIN", "BHARTIARTL", "AXISBANK", "KOTAKBANK", "ITC"
        ]
    
    # ════════════════════════════════════════════════════════════
    # ECONOMIC DATA RELEASES
    # ════════════════════════════════════════════════════════════
    
    def get_economic_calendar(self) -> List[Dict]:
        """Get upcoming economic data releases."""
        today = date.today()
        
        # Monthly economic calendar (approximate dates)
        events = []
        
        # CPI Inflation (12th of each month)
        cpi_date = date(today.year, today.month, 12)
        if cpi_date >= today:
            events.append({
                "name": "CPI Inflation",
                "date": str(cpi_date),
                "type": "cpi_inflation",
                "impact": self.EVENT_IMPACT["cpi_inflation"],
                "affected": ["BANKING", "RATE_SENSITIVE"]
            })
        
        # IIP Data (around 12th, 2 months lag)
        iip_date = date(today.year, today.month, 12)
        if iip_date >= today:
            events.append({
                "name": "IIP Industrial Production",
                "date": str(iip_date),
                "type": "iip_data",
                "impact": self.EVENT_IMPACT["iip_data"],
                "affected": ["INDUSTRIALS", "INFRA"]
            })
        
        # PMI Data (1st-3rd of each month)
        pmi_date = date(today.year, today.month, 1)
        if today.month < 12:
            pmi_date = date(today.year, today.month + 1, 1)
        events.append({
            "name": "PMI Manufacturing",
            "date": str(pmi_date),
            "type": "pmi_data",
            "impact": self.EVENT_IMPACT["pmi_data"],
            "affected": ["INDUSTRIALS", "AUTO"]
        })
        
        # GDP (quarterly, end of Feb/May/Aug/Nov)
        gdp_months = [2, 5, 8, 11]
        for m in gdp_months:
            if today.month == m and today.day <= 28:
                events.append({
                    "name": "GDP Growth Data",
                    "date": str(date(today.year, m, 28)),
                    "type": "gdp_data",
                    "impact": self.EVENT_IMPACT["gdp_data"],
                    "affected": ["BROAD_MARKET"]
                })
        
        return events
    
    # ════════════════════════════════════════════════════════════
    # GLOBAL EVENTS (Fed, US Data, etc.)
    # ════════════════════════════════════════════════════════════
    
    def get_global_events(self) -> List[Dict]:
        """Get upcoming global events that affect Indian markets."""
        today = date.today()
        events = []
        
        # FOMC Meeting dates (8 meetings per year, roughly every 6 weeks)
        # 2026 approximate schedule
        fomc_dates_2026 = [
            date(2026, 1, 29),  # Jan
            date(2026, 3, 19),  # Mar
            date(2026, 5, 7),   # May
            date(2026, 6, 18),  # Jun
            date(2026, 7, 30),  # Jul
            date(2026, 9, 17),  # Sep
            date(2026, 11, 5),  # Nov
            date(2026, 12, 17), # Dec
        ]
        
        for fomc_date in fomc_dates_2026:
            days_until = (fomc_date - today).days
            if -1 <= days_until <= 3:
                events.append({
                    "name": "US Fed FOMC Meeting",
                    "date": str(fomc_date),
                    "type": "fed_meeting",
                    "days_until": days_until,
                    "impact": self.EVENT_IMPACT["fed_meeting"],
                    "time": "11:30 PM IST",
                    "affected_sectors": ["IT", "PHARMA", "BANKING"],
                    "affected_stocks": ["TCS", "INFY", "WIPRO", "HCLTECH", 
                                       "SUNPHARMA", "DRREDDY", "HDFCBANK"],
                    "trading_advice": self._get_fed_advice(days_until)
                })
        
        # US Jobs Report (First Friday of each month)
        # Find first Friday
        first_day = date(today.year, today.month, 1)
        days_until_friday = (4 - first_day.weekday()) % 7
        jobs_date = first_day + timedelta(days=days_until_friday)
        
        days_until = (jobs_date - today).days
        if -1 <= days_until <= 2:
            events.append({
                "name": "US Non-Farm Payrolls",
                "date": str(jobs_date),
                "type": "us_jobs",
                "days_until": days_until,
                "impact": self.EVENT_IMPACT["us_jobs"],
                "time": "6:00 PM IST",
                "affected_sectors": ["IT", "EXPORT_ORIENTED"],
                "trading_advice": "US jobs data affects global risk sentiment and USD/INR"
            })
        
        # US CPI (around 10th-13th of each month)
        us_cpi_date = date(today.year, today.month, 12)
        days_until = (us_cpi_date - today).days
        if -1 <= days_until <= 2:
            events.append({
                "name": "US CPI Inflation",
                "date": str(us_cpi_date),
                "type": "us_cpi",
                "days_until": days_until,
                "impact": self.EVENT_IMPACT["us_cpi"],
                "time": "6:00 PM IST",
                "affected_sectors": ["IT", "BANKING"],
                "trading_advice": "US CPI affects Fed rate expectations, impacts IT stocks"
            })
        
        return events
    
    def _get_fed_advice(self, days_until: int) -> str:
        """Get trading advice for Fed meeting."""
        if days_until < 0:
            return "POST_FED: Trade the reaction, IT/Pharma volatile on USD move"
        elif days_until == 0:
            return "FED_DAY: Expect late volatility after 11:30 PM, reduce overnight exposure"
        else:
            return "PRE_FED: Market may be range-bound, avoid large IT positions"
    
    def get_all_events_today(self, symbols: List[str] = None, 
                             use_claude: bool = False,
                             claude_api_key: str = None) -> Dict:
        """
        Get all events affecting given symbols today.
        
        Args:
            symbols: List of stock symbols to check
            use_claude: If True, also fetch Claude-discovered events
            claude_api_key: API key for Claude (required if use_claude=True)
        """
        events = {
            "earnings": self.fetch_earnings_calendar(symbols),
            "rbi": self.check_rbi_impact(),
            "budget": self.check_budget_impact(),
            "election": self.check_election_impact(),
            "fii_dii": self.get_fii_sentiment(),
            "expiry": self.check_expiry_impact(),
            "economic": self.get_economic_calendar(),
            "global": self.get_global_events(),
            "claude_events": [],
            "high_impact_symbols": [],
            "avoid_symbols": [],
            "opportunity_symbols": [],
            "caution_symbols": [],
        }
        
        # Optionally fetch Claude-discovered events
        if use_claude and claude_api_key:
            try:
                discovery = ClaudeEventDiscovery(api_key=claude_api_key)
                claude_events = discovery.discover_events(symbols)
                events["claude_events"] = claude_events
                
                # Process Claude events for symbol categorization
                for evt in claude_events:
                    if evt.get("impact_score", 0) >= 0.7:
                        affected = evt.get("affected_stocks", [])
                        events["high_impact_symbols"].extend(affected[:5])
                        
                        if evt.get("direction_bias") == "bearish":
                            events["caution_symbols"].extend(affected)
                        elif evt.get("direction_bias") == "bullish":
                            events["opportunity_symbols"].extend(affected)
                            
            except Exception as e:
                logger.warning(f"Failed to fetch Claude events: {e}")
        
        # Categorize symbols based on static events
        
        # Earnings
        for sym, data in events["earnings"].items():
            if data["days_until"] == 0:
                events["avoid_symbols"].append(sym)  # Earnings today - avoid
            elif data["days_until"] < 0:
                events["opportunity_symbols"].append(sym)  # Post-earnings gap play
        
        # RBI impact
        if events["rbi"]:
            events["high_impact_symbols"].extend(events["rbi"]["affected_stocks"][:5])
            events["caution_symbols"].extend(events["rbi"]["affected_stocks"])
        
        # Budget impact
        if events["budget"]:
            events["high_impact_symbols"].extend(events["budget"]["affected_stocks"][:10])
            events["caution_symbols"].extend(events["budget"]["affected_stocks"])
        
        # Election impact
        if events["election"]:
            events["high_impact_symbols"].extend(events["election"].get("affected_stocks", [])[:10])
            events["caution_symbols"].extend(events["election"].get("affected_stocks", []))
        
        # F&O expiry - high OI stocks need caution
        if events["expiry"]:
            events["caution_symbols"].extend(events["expiry"].get("affected_stocks", []))
        
        # Remove duplicates
        events["high_impact_symbols"] = list(set(events["high_impact_symbols"]))
        events["avoid_symbols"] = list(set(events["avoid_symbols"]))
        events["opportunity_symbols"] = list(set(events["opportunity_symbols"]))
        events["caution_symbols"] = list(set(events["caution_symbols"]))
        
        # Summary
        events["summary"] = self._generate_event_summary(events)
        
        return events
    
    def _generate_event_summary(self, events: Dict) -> str:
        """Generate human-readable event summary."""
        lines = []
        
        if events.get("budget"):
            lines.append(f"🏛️ BUDGET: {events['budget']['trading_advice']}")
        
        if events.get("election"):
            lines.append(f"🗳️ ELECTION: {events['election']['trading_advice']}")
        
        if events.get("rbi"):
            lines.append(f"🏦 RBI: {events['rbi']['trading_advice']}")
        
        if events.get("expiry"):
            lines.append(f"📅 EXPIRY: {events['expiry']['trading_advice']}")
        
        if events.get("fii_dii", {}).get("advice"):
            lines.append(f"💰 FII/DII: {events['fii_dii']['advice']}")
        
        # Global events
        for global_evt in events.get("global", []):
            evt_type = global_evt.get("type", "global")
            if evt_type == "fed_meeting":
                lines.append(f"🇺🇸 FED: {global_evt.get('trading_advice', 'Watch Fed meeting')}")
            elif evt_type in ["us_jobs", "us_cpi"]:
                lines.append(f"🇺🇸 US DATA: {global_evt.get('name')} - {global_evt.get('trading_advice', '')}")
        
        # Claude-discovered events
        for claude_evt in events.get("claude_events", []):
            if claude_evt.get("impact_score", 0) >= 0.6:
                emoji = "🔴" if claude_evt.get("direction_bias") == "bearish" else "🟢"
                lines.append(f"{emoji} {claude_evt.get('name', 'Event')}: {claude_evt.get('trading_advice', '')}")
        
        if events.get("avoid_symbols"):
            lines.append(f"⚠️ AVOID: {', '.join(events['avoid_symbols'][:5])}")
        
        if events.get("opportunity_symbols"):
            lines.append(f"💡 OPPORTUNITIES: {', '.join(events['opportunity_symbols'][:5])}")
        
        return "\n".join(lines) if lines else "No major events today"


class ClaudeEventDiscovery:
    """
    Uses Claude AI to discover and analyze market events dynamically.
    
    This class can:
    1. Fetch and analyze recent news for market-moving events
    2. Identify unscheduled events (geopolitical, corporate announcements)
    3. Assess event impact on specific stocks/sectors
    4. Generate trading recommendations based on events
    
    Usage:
        discovery = ClaudeEventDiscovery(api_key="your-key")
        events = discovery.discover_events(symbols=["RELIANCE", "TCS"])
        impact = discovery.analyze_event_impact("RBI announces rate cut", ["HDFCBANK"])
    """
    
    def __init__(self, api_key: str = None, config: dict = None):
        self.api_key = api_key
        self.config = config or {}
        self.cache_file = CACHE_DIR / f"claude_events_{date.today()}.json"
        self._cached_events = self._load_cache()
    
    def _load_cache(self) -> dict:
        """Load cached events from today."""
        if self.cache_file.exists():
            try:
                return json.loads(self.cache_file.read_text())
            except:
                pass
        return {"events": [], "last_updated": None}
    
    def _save_cache(self, events: list):
        """Save discovered events to cache."""
        self._cached_events = {
            "events": events,
            "last_updated": datetime.now().isoformat()
        }
        self.cache_file.write_text(json.dumps(self._cached_events, indent=2, default=str))
    
    def _call_claude(self, system_prompt: str, user_prompt: str, 
                     max_tokens: int = 1000) -> Optional[Dict]:
        """Call Claude API and parse JSON response."""
        if not self.api_key:
            logger.warning("Claude API key not configured for event discovery")
            return None
        
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            text = response.content[0].text
            
            # Extract JSON from response
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            return json.loads(text.strip())
            
        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            return None
    
    def discover_events(self, symbols: List[str] = None, 
                        context: str = None,
                        use_cache: bool = True) -> List[Dict]:
        """
        Discover market-moving events using Claude.
        
        Args:
            symbols: Stocks to focus on (optional)
            context: Additional context like news headlines
            use_cache: Use cached events from today
        
        Returns:
            List of discovered events with trading implications
        """
        # Check cache first
        if use_cache and self._cached_events.get("events"):
            cache_time = self._cached_events.get("last_updated")
            if cache_time:
                cache_dt = datetime.fromisoformat(cache_time)
                if (datetime.now() - cache_dt).seconds < 3600:  # 1 hour cache
                    logger.info("Using cached Claude events")
                    return self._cached_events["events"]
        
        today_str = date.today().strftime("%A, %B %d, %Y")
        symbols_str = ", ".join(symbols[:20]) if symbols else "Nifty 50 stocks"
        
        system_prompt = """You are an expert Indian market analyst who tracks market-moving events.
Your job is to identify events that could impact stock prices TODAY or THIS WEEK.

Focus on:
1. RBI announcements, govt policy changes
2. Corporate earnings, AGMs, dividends
3. Global events (Fed, US data, crude oil)
4. Geopolitical events affecting markets
5. Sector-specific news (pharma approvals, IT deals, etc.)
6. Any breaking news that could move specific stocks

Always return valid JSON."""
        
        user_prompt = f"""Today is {today_str}.

Analyze potential market-moving events for Indian markets, focusing on: {symbols_str}

{f"Additional context: {context}" if context else ""}

Return JSON with discovered events:
{{
    "events": [
        {{
            "name": "Event name",
            "type": "earnings|policy|global|corporate|geopolitical|sector_news",
            "date": "YYYY-MM-DD or 'today' or 'this_week'",
            "time": "HH:MM IST if known, else null",
            "impact_score": 0.0-1.0,
            "affected_sectors": ["BANKING", "IT", etc],
            "affected_stocks": ["SYMBOL1", "SYMBOL2"],
            "direction_bias": "bullish|bearish|neutral|uncertain",
            "trading_advice": "Brief actionable advice",
            "confidence": 0.0-1.0
        }}
    ],
    "market_sentiment": "bullish|bearish|neutral",
    "key_levels_to_watch": "Brief note on Nifty levels if relevant",
    "top_risk": "Main risk factor today"
}}

If no significant events, return empty events list with explanation in market_sentiment."""
        
        result = self._call_claude(system_prompt, user_prompt)
        
        if result and result.get("events"):
            self._save_cache(result["events"])
            return result["events"]
        
        return []
    
    def analyze_event_impact(self, event_description: str, 
                             symbols: List[str]) -> Dict:
        """
        Analyze the impact of a specific event on given stocks.
        
        Args:
            event_description: Description of the event (e.g., "RBI cuts repo rate by 25bps")
            symbols: Stocks to analyze impact for
        
        Returns:
            Dict with impact analysis for each symbol
        """
        system_prompt = """You are an expert at analyzing how market events impact specific stocks.
Consider:
1. Direct impact (sector exposure, revenue impact)
2. Indirect impact (sentiment, valuation multiples)
3. Historical reactions to similar events
4. Current technical setup

Return actionable trading advice with confidence levels."""
        
        user_prompt = f"""EVENT: {event_description}

Analyze impact on these stocks: {', '.join(symbols)}

Return JSON:
{{
    "event_summary": "Brief event summary",
    "overall_market_impact": "bullish|bearish|neutral",
    "impact_magnitude": 0.0-1.0,
    "stock_impacts": {{
        "SYMBOL": {{
            "impact": "positive|negative|neutral",
            "magnitude": 0.0-1.0,
            "expected_move": "+/-X%",
            "timeframe": "intraday|1-3 days|1 week",
            "trading_action": "BUY|SELL|HOLD|AVOID",
            "reasoning": "Brief reasoning"
        }}
    }},
    "best_trades": [
        {{
            "symbol": "SYMBOL",
            "action": "LONG|SHORT",
            "entry_strategy": "How to enter",
            "risk_level": "low|medium|high"
        }}
    ],
    "avoid_list": ["Symbols to avoid with reasons"]
}}"""
        
        return self._call_claude(system_prompt, user_prompt) or {}
    
    def get_morning_brief(self, symbols: List[str] = None) -> Dict:
        """
        Get Claude's morning market brief with event analysis.
        Call this at market open (9:00-9:15 AM).
        
        Returns comprehensive morning analysis including:
        - Overnight global moves
        - Today's key events
        - Stock-specific catalysts
        - Trading recommendations
        """
        today_str = date.today().strftime("%A, %B %d, %Y")
        symbols_str = ", ".join(symbols[:15]) if symbols else "Nifty 50 components"
        
        system_prompt = """You are a senior market strategist providing the morning brief for Indian equity traders.
Your brief should be actionable and focused on TODAY's trading opportunities.

Consider:
1. Overnight US/Asian market moves
2. SGX Nifty indication
3. Today's scheduled events (earnings, data releases)
4. Any overnight news affecting Indian stocks
5. Technical levels and sentiment"""
        
        user_prompt = f"""Provide morning brief for {today_str}.

Focus stocks: {symbols_str}

Return JSON:
{{
    "market_mood": "bullish|bearish|neutral|cautious",
    "confidence": 0.0-1.0,
    "overnight_summary": "Key overnight developments",
    "expected_nifty_open": "gap_up|gap_down|flat",
    "expected_gap_pct": "+/-X%",
    
    "key_events_today": [
        {{
            "event": "Event name",
            "time": "HH:MM IST",
            "impact": "high|medium|low",
            "affected_stocks": ["SYM1", "SYM2"]
        }}
    ],
    
    "stock_catalysts": {{
        "SYMBOL": "Catalyst description"
    }},
    
    "trading_ideas": [
        {{
            "symbol": "SYMBOL",
            "bias": "LONG|SHORT",
            "rationale": "Why",
            "entry_zone": "Price range",
            "stop_loss": "Price or %",
            "target": "Price or %",
            "confidence": 0.0-1.0
        }}
    ],
    
    "avoid_today": ["SYMBOL1", "SYMBOL2"],
    "avoid_reason": "Why to avoid these",
    
    "key_levels": {{
        "nifty_support": 0,
        "nifty_resistance": 0,
        "banknifty_support": 0,
        "banknifty_resistance": 0
    }},
    
    "risk_warning": "Main risk to watch today"
}}"""
        
        return self._call_claude(system_prompt, user_prompt, max_tokens=1500) or {}
    
    def analyze_breaking_news(self, news_headline: str, 
                              news_body: str = None) -> Dict:
        """
        Analyze breaking news for immediate trading implications.
        
        Args:
            news_headline: The news headline
            news_body: Optional full news text
        
        Returns:
            Immediate trading recommendations
        """
        system_prompt = """You are a rapid news analyst for trading desks.
Analyze breaking news and provide IMMEDIATE actionable guidance.
Speed is critical - focus on the most likely market reaction."""
        
        content = news_headline
        if news_body:
            content += f"\n\nDetails: {news_body[:500]}"
        
        user_prompt = f"""BREAKING NEWS: {content}

Provide immediate trading analysis:

{{
    "news_type": "earnings|policy|corporate|macro|geopolitical|other",
    "urgency": "immediate|within_hour|today",
    "market_impact": "major|moderate|minor",
    
    "immediate_actions": [
        {{
            "symbol": "SYMBOL",
            "action": "BUY|SELL|CLOSE_LONG|CLOSE_SHORT|HEDGE",
            "urgency": "now|on_open|wait_for_confirmation",
            "reason": "Brief reason"
        }}
    ],
    
    "sectors_affected": {{
        "positive": ["SECTOR1"],
        "negative": ["SECTOR2"]
    }},
    
    "expected_reaction": {{
        "immediate": "Description of likely immediate reaction",
        "by_eod": "How it might evolve by end of day"
    }},
    
    "key_stocks_to_watch": ["SYM1", "SYM2", "SYM3"],
    
    "false_alarm_probability": 0.0-1.0,
    "confidence": 0.0-1.0
}}"""
        
        return self._call_claude(system_prompt, user_prompt, max_tokens=800) or {}


def get_claude_events(api_key: str, symbols: List[str] = None) -> List[Dict]:
    """
    Convenience function to get Claude-discovered events.
    
    Usage:
        from core.event_calendar import get_claude_events
        events = get_claude_events(api_key, ["RELIANCE", "TCS"])
    """
    discovery = ClaudeEventDiscovery(api_key=api_key)
    return discovery.discover_events(symbols)


def get_morning_brief(api_key: str, symbols: List[str] = None) -> Dict:
    """
    Get Claude's morning market brief.
    
    Usage:
        from core.event_calendar import get_morning_brief
        brief = get_morning_brief(api_key, ["RELIANCE", "TCS"])
    """
    discovery = ClaudeEventDiscovery(api_key=api_key)
    return discovery.get_morning_brief(symbols)


class GapAnalyzer:
    """
    Analyzes overnight gaps for trading opportunities.
    
    Gap Types:
    1. Breakaway Gap: Start of new trend, high volume, DON'T fade
    2. Continuation Gap: Mid-trend, moderate volume, trade with trend
    3. Exhaustion Gap: End of trend, high volume, FADE it
    4. Common Gap: Random noise, low volume, likely to fill
    """
    
    # Gap size thresholds (percentage)
    GAP_THRESHOLDS = {
        "micro": 0.3,      # 0.3% - likely noise
        "small": 0.5,      # 0.5% - tradeable
        "medium": 1.0,     # 1% - significant
        "large": 2.0,      # 2% - major move
        "extreme": 3.0,    # 3%+ - news-driven
    }
    
    def __init__(self, claude_brain=None):
        self.claude_brain = claude_brain
    
    def calculate_gap(self, symbol: str, prev_close: float, today_open: float) -> Dict:
        """Calculate gap percentage and classify it."""
        gap_pct = (today_open - prev_close) / prev_close * 100
        gap_direction = "UP" if gap_pct > 0 else "DOWN"
        abs_gap = abs(gap_pct)
        
        # Classify gap size
        if abs_gap < self.GAP_THRESHOLDS["micro"]:
            gap_size = "micro"
        elif abs_gap < self.GAP_THRESHOLDS["small"]:
            gap_size = "small"
        elif abs_gap < self.GAP_THRESHOLDS["medium"]:
            gap_size = "medium"
        elif abs_gap < self.GAP_THRESHOLDS["large"]:
            gap_size = "large"
        else:
            gap_size = "extreme"
        
        return {
            "symbol": symbol,
            "prev_close": prev_close,
            "today_open": today_open,
            "gap_pct": round(gap_pct, 2),
            "gap_direction": gap_direction,
            "gap_size": gap_size,
            "gap_points": round(today_open - prev_close, 2),
        }
    
    def classify_gap_type(self, gap_info: Dict, 
                          prev_trend: str,
                          open_volume_ratio: float,
                          sector_gap: float = None) -> Dict:
        """
        Classify gap type based on context.
        
        Args:
            gap_info: From calculate_gap()
            prev_trend: "UP", "DOWN", or "SIDEWAYS" (5-day trend)
            open_volume_ratio: Today's open volume / avg volume
            sector_gap: Sector index gap (to check if stock-specific)
        """
        gap_pct = abs(gap_info["gap_pct"])
        direction = gap_info["gap_direction"]
        
        # Default classification
        gap_type = "common"
        trade_direction = None
        confidence = 0.5
        
        # High volume + large gap + trend alignment = Breakaway
        if gap_pct >= 1.5 and open_volume_ratio >= 2.0:
            if (direction == "UP" and prev_trend != "DOWN") or \
               (direction == "DOWN" and prev_trend != "UP"):
                gap_type = "breakaway"
                trade_direction = "WITH_GAP"  # Trade continuation
                confidence = 0.75
        
        # Medium gap + moderate volume + existing trend = Continuation
        elif 0.8 <= gap_pct < 2.0 and 1.2 <= open_volume_ratio < 2.0:
            if (direction == "UP" and prev_trend == "UP") or \
               (direction == "DOWN" and prev_trend == "DOWN"):
                gap_type = "continuation"
                trade_direction = "WITH_GAP"
                confidence = 0.65
        
        # Large gap + extreme volume + against trend = Exhaustion
        elif gap_pct >= 2.0 and open_volume_ratio >= 2.5 and prev_trend != "SIDEWAYS":
            if (direction == "UP" and prev_trend == "UP") or \
               (direction == "DOWN" and prev_trend == "DOWN"):
                gap_type = "exhaustion"
                trade_direction = "FADE_GAP"  # Trade reversal
                confidence = 0.70
        
        # Small gap + low volume = Common (likely to fill)
        elif gap_pct < 1.0 and open_volume_ratio < 1.5:
            gap_type = "common"
            trade_direction = "FADE_GAP"  # Expect gap fill
            confidence = 0.60
        
        # Stock-specific gap (sector didn't gap) = Higher conviction
        if sector_gap is not None:
            stock_vs_sector = gap_pct - abs(sector_gap)
            if stock_vs_sector > 1.0:  # Stock gapped 1%+ more than sector
                confidence += 0.10
                gap_info["stock_specific"] = True
        
        gap_info.update({
            "gap_type": gap_type,
            "trade_direction": trade_direction,
            "confidence": min(confidence, 0.90),
            "volume_ratio": open_volume_ratio,
        })
        
        return gap_info
    
    def generate_gap_signal(self, gap_info: Dict, current_price: float) -> Optional[Dict]:
        """
        Generate trading signal based on gap analysis.
        """
        if gap_info["gap_size"] == "micro":
            return None  # Not tradeable
        
        gap_pct = gap_info["gap_pct"]
        today_open = gap_info["today_open"]
        prev_close = gap_info["prev_close"]
        trade_dir = gap_info.get("trade_direction")
        confidence = gap_info.get("confidence", 0.5)
        
        if confidence < 0.55:
            return None  # Low confidence
        
        signal = {
            "symbol": gap_info["symbol"],
            "strategy": f"GAP_{gap_info['gap_type'].upper()}",
            "gap_info": gap_info,
            "confidence": confidence,
        }
        
        if trade_dir == "WITH_GAP":
            # Trade in gap direction (continuation)
            if gap_pct > 0:
                signal["side"] = "LONG"
                signal["entry"] = current_price
                signal["sl"] = min(prev_close, today_open * 0.995)  # Below gap
                signal["target"] = current_price * 1.015  # 1.5% target
                signal["reason"] = f"Gap-and-go: {gap_pct:+.1f}% {gap_info['gap_type']} gap"
            else:
                signal["side"] = "SHORT"
                signal["entry"] = current_price
                signal["sl"] = max(prev_close, today_open * 1.005)  # Above gap
                signal["target"] = current_price * 0.985
                signal["reason"] = f"Gap-and-go: {gap_pct:+.1f}% {gap_info['gap_type']} gap"
        
        elif trade_dir == "FADE_GAP":
            # Trade gap fill (reversal)
            if gap_pct > 0:
                signal["side"] = "SHORT"
                signal["entry"] = current_price
                signal["sl"] = today_open * 1.005  # Just above open
                signal["target"] = prev_close  # Gap fill target
                signal["reason"] = f"Gap-fill: Fading {gap_pct:+.1f}% {gap_info['gap_type']} gap"
            else:
                signal["side"] = "LONG"
                signal["entry"] = current_price
                signal["sl"] = today_open * 0.995  # Just below open
                signal["target"] = prev_close  # Gap fill target
                signal["reason"] = f"Gap-fill: Fading {gap_pct:+.1f}% {gap_info['gap_type']} gap"
        
        else:
            return None
        
        # Calculate risk/reward
        risk = abs(signal["entry"] - signal["sl"])
        reward = abs(signal["target"] - signal["entry"])
        signal["risk_reward"] = round(reward / risk, 2) if risk > 0 else 0
        
        # Only take trades with good R:R
        if signal["risk_reward"] < 1.5:
            return None
        
        return signal


class EventDrivenTrader:
    """
    Combines event calendar and gap analysis for trading decisions.
    Uses Claude Brain for intelligent analysis.
    """
    
    def __init__(self, config: dict, claude_brain=None):
        self.config = config
        self.claude_brain = claude_brain
        self.event_calendar = EventCalendar(claude_brain)
        self.gap_analyzer = GapAnalyzer(claude_brain)
    
    def morning_scan(self, symbols: List[str], market_data: Dict) -> Dict:
        """
        Run morning scan for event-driven opportunities.
        Called at 9:00 AM before market opens.
        
        Args:
            symbols: List of symbols to scan
            market_data: Dict with prev_close, today_open, volume for each symbol
        
        Returns:
            Dict with trading recommendations
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "events": {},
            "gaps": {},
            "signals": [],
            "avoid_list": [],
            "claude_analysis": None
        }
        
        # 1. Check events
        events = self.event_calendar.get_all_events_today(symbols)
        results["events"] = events
        results["avoid_list"] = events.get("avoid_symbols", [])
        
        # 2. Analyze gaps
        gap_opportunities = []
        for symbol in symbols:
            if symbol in results["avoid_list"]:
                continue  # Skip earnings day stocks
            
            data = market_data.get(symbol, {})
            if not data.get("prev_close") or not data.get("today_open"):
                continue
            
            # Calculate gap
            gap_info = self.gap_analyzer.calculate_gap(
                symbol,
                data["prev_close"],
                data["today_open"]
            )
            
            # Skip micro gaps
            if gap_info["gap_size"] == "micro":
                continue
            
            # Get previous trend (simplified - would use real data)
            prev_trend = data.get("prev_trend", "SIDEWAYS")
            volume_ratio = data.get("volume_ratio", 1.0)
            
            # Classify gap
            gap_info = self.gap_analyzer.classify_gap_type(
                gap_info, prev_trend, volume_ratio
            )
            
            results["gaps"][symbol] = gap_info
            
            # Generate signal
            signal = self.gap_analyzer.generate_gap_signal(
                gap_info, 
                data.get("current_price", data["today_open"])
            )
            if signal:
                gap_opportunities.append(signal)
        
        # 3. Use Claude Brain to analyze and prioritize
        if self.claude_brain and self.claude_brain.enabled and gap_opportunities:
            claude_analysis = self._get_claude_analysis(events, gap_opportunities)
            results["claude_analysis"] = claude_analysis
            
            # Apply Claude's recommendations
            if claude_analysis:
                approved = claude_analysis.get("approved_signals", [])
                results["signals"] = [s for s in gap_opportunities 
                                     if s["symbol"] in approved]
        else:
            # Without Claude, take top 3 by confidence
            gap_opportunities.sort(key=lambda x: x["confidence"], reverse=True)
            results["signals"] = gap_opportunities[:3]
        
        return results
    
    def _get_claude_analysis(self, events: Dict, signals: List[Dict]) -> Optional[Dict]:
        """Get Claude Brain's analysis of gap opportunities."""
        if not self.claude_brain:
            return None
        
        try:
            from strategies.claude_brain_v2 import _call_claude
            
            prompt = f"""MORNING GAP ANALYSIS — {date.today()}

MARKET EVENTS TODAY:
{json.dumps(events, indent=2, default=str)}

GAP OPPORTUNITIES DETECTED:
{json.dumps([{
    "symbol": s["symbol"],
    "gap_pct": s["gap_info"]["gap_pct"],
    "gap_type": s["gap_info"]["gap_type"],
    "trade_direction": s["gap_info"]["trade_direction"],
    "side": s["side"],
    "confidence": s["confidence"],
    "risk_reward": s["risk_reward"]
} for s in signals], indent=2)}

Analyze and return JSON:
{{
    "market_sentiment": "bullish|bearish|neutral",
    "approved_signals": ["SYM1", "SYM2"],
    "rejected_signals": {{"SYM3": "reason"}},
    "priority_order": ["SYM1", "SYM2"],
    "position_size_factor": 1.0,
    "max_gap_trades_today": 2,
    "notes": "brief reasoning"
}}

RULES:
- Reject gaps in stocks with earnings today
- Prefer breakaway gaps over common gaps
- Max 2-3 gap trades per day
- Favor gaps with >2:1 R:R
- Consider sector correlation"""
            
            result = _call_claude(
                self.claude_brain.api_key,
                "You are an expert gap trader. Analyze morning gaps and select the best opportunities.",
                prompt,
                max_tokens=500
            )
            return result
        except Exception as e:
            logger.warning(f"Claude analysis failed: {e}")
            return None
    
    def get_earnings_plays(self, symbols: List[str] = None) -> List[Dict]:
        """
        Get post-earnings trading opportunities.
        Called after market open on earnings announcement days.
        """
        earnings = self.event_calendar.fetch_earnings_calendar(symbols)
        plays = []
        
        for symbol, data in earnings.items():
            if data["days_until"] == -1:  # Yesterday's earnings
                plays.append({
                    "symbol": symbol,
                    "strategy": "POST_EARNINGS_GAP",
                    "event": data,
                    "advice": "Trade the gap reaction - first 30 min most volatile"
                })
        
        return plays


# Utility function for live_paper_v3.py integration
def get_morning_gap_signals(symbols: List[str], config: dict, 
                            market_data: Dict, claude_brain=None) -> List[Dict]:
    """
    Main entry point for gap trading integration.
    Call this at 9:10 AM after getting opening prices.
    """
    trader = EventDrivenTrader(config, claude_brain)
    results = trader.morning_scan(symbols, market_data)
    return results.get("signals", [])


def get_event_warnings(symbols: List[str]) -> Dict:
    """
    Get warnings about upcoming events.
    Call this before selecting stocks.
    """
    calendar = EventCalendar()
    events = calendar.get_all_events_today(symbols)
    
    return {
        "avoid": events.get("avoid_symbols", []),
        "caution": events.get("high_impact_symbols", []),
        "opportunities": events.get("opportunity_symbols", []),
        "rbi_impact": events.get("rbi") is not None
    }


# ════════════════════════════════════════════════════════════════════════════
# PRE-MARKET GAP SCANNER (Run at 8:45 AM)
# ════════════════════════════════════════════════════════════════════════════

class PreMarketGapScanner:
    """
    Scans for overnight gaps before market opens.
    Sends alerts via Telegram for significant gaps.
    
    Run at 8:45-9:00 AM using:
        python -c "from core.event_calendar import run_premarket_scan; run_premarket_scan()"
    
    Or schedule with cron:
        45 8 * * 1-5 cd /path/to/project && python -c "from core.event_calendar import run_premarket_scan; run_premarket_scan()"
    """
    
    # Minimum gap % to alert
    ALERT_THRESHOLD = 0.5  # 0.5%
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.gap_analyzer = GapAnalyzer()
        self.event_calendar = EventCalendar()
    
    def fetch_premarket_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Fetch pre-market / pre-open data.
        Uses NSE pre-open session data (9:00-9:08 AM).
        """
        premarket_data = {}
        
        # In production, fetch from NSE pre-open API
        # NSE pre-open session: 9:00-9:08 AM
        # URL: https://www.nseindia.com/api/market-data-pre-open?key=ALL
        
        try:
            import yfinance as yf
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(f"{symbol}.NS")
                    
                    # Get previous close
                    hist = ticker.history(period="5d")
                    if hist.empty or len(hist) < 2:
                        continue
                    
                    prev_close = hist['Close'].iloc[-2]
                    
                    # Get today's open from fast_info or real-time
                    info = ticker.fast_info
                    today_open = getattr(info, 'open', None) or hist['Open'].iloc[-1]
                    
                    # Previous day's data for trend
                    prev_5d_change = (hist['Close'].iloc[-2] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]
                    prev_trend = "UP" if prev_5d_change > 0.01 else ("DOWN" if prev_5d_change < -0.01 else "SIDEWAYS")
                    
                    # Volume comparison
                    avg_volume = hist['Volume'].mean()
                    
                    premarket_data[symbol] = {
                        "prev_close": float(prev_close),
                        "today_open": float(today_open),
                        "prev_trend": prev_trend,
                        "avg_volume": float(avg_volume),
                        "volume_ratio": 1.0,  # Will be updated when market opens
                    }
                    
                except Exception as e:
                    logger.debug(f"Failed to fetch {symbol}: {e}")
                    continue
            
        except ImportError:
            logger.warning("yfinance not available for pre-market data")
        
        return premarket_data
    
    def scan_gaps(self, symbols: List[str] = None) -> Dict:
        """
        Scan for gaps in given symbols.
        Returns categorized gaps by size and type.
        """
        if symbols is None:
            # Default: FNO stocks
            symbols = [
                "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                "HINDUNILVR", "SBIN", "BHARTIARTL", "KOTAKBANK", "ITC",
                "BAJFINANCE", "LT", "AXISBANK", "ASIANPAINT", "MARUTI",
                "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "HCLTECH",
                "TATAMOTORS", "M&M", "NESTLEIND", "ONGC", "NTPC",
            ]
        
        # Fetch pre-market data
        premarket_data = self.fetch_premarket_data(symbols)
        
        # Get event warnings
        events = self.event_calendar.get_all_events_today(symbols)
        earnings_today = set(events.get("avoid_symbols", []))
        
        # Analyze gaps
        results = {
            "scan_time": datetime.now().isoformat(),
            "total_scanned": len(premarket_data),
            "gaps": {
                "large_up": [],    # >2% gap up
                "medium_up": [],   # 1-2% gap up
                "small_up": [],    # 0.5-1% gap up
                "large_down": [],  # >2% gap down
                "medium_down": [], # 1-2% gap down
                "small_down": [],  # 0.5-1% gap down
            },
            "alerts": [],
            "trade_candidates": [],
            "events": events,
        }
        
        for symbol, data in premarket_data.items():
            gap_info = self.gap_analyzer.calculate_gap(
                symbol,
                data["prev_close"],
                data["today_open"]
            )
            
            gap_pct = gap_info["gap_pct"]
            abs_gap = abs(gap_pct)
            
            if abs_gap < self.ALERT_THRESHOLD:
                continue
            
            # Classify gap
            gap_info = self.gap_analyzer.classify_gap_type(
                gap_info,
                data["prev_trend"],
                data.get("volume_ratio", 1.0)
            )
            
            # Add to appropriate category
            if gap_pct > 2:
                results["gaps"]["large_up"].append(gap_info)
            elif gap_pct > 1:
                results["gaps"]["medium_up"].append(gap_info)
            elif gap_pct > 0.5:
                results["gaps"]["small_up"].append(gap_info)
            elif gap_pct < -2:
                results["gaps"]["large_down"].append(gap_info)
            elif gap_pct < -1:
                results["gaps"]["medium_down"].append(gap_info)
            elif gap_pct < -0.5:
                results["gaps"]["small_down"].append(gap_info)
            
            # Create alert if significant
            if abs_gap >= 1.0:
                alert = {
                    "symbol": symbol,
                    "gap_pct": gap_pct,
                    "gap_type": gap_info.get("gap_type", "unknown"),
                    "prev_close": data["prev_close"],
                    "today_open": data["today_open"],
                    "earnings_today": symbol in earnings_today,
                    "trade_direction": gap_info.get("trade_direction"),
                }
                results["alerts"].append(alert)
                
                # Add to trade candidates if not earnings
                if symbol not in earnings_today and gap_info.get("trade_direction"):
                    results["trade_candidates"].append(gap_info)
        
        # Sort by absolute gap size
        results["alerts"].sort(key=lambda x: abs(x["gap_pct"]), reverse=True)
        results["trade_candidates"].sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        return results
    
    def format_alert_message(self, scan_results: Dict) -> str:
        """Format scan results as Telegram message."""
        alerts = scan_results.get("alerts", [])
        
        if not alerts:
            return "📊 PRE-MARKET SCAN\n\nNo significant gaps detected (≥1%)"
        
        msg = f"📊 PRE-MARKET GAP SCANNER\n"
        msg += f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"⏰ {datetime.now().strftime('%H:%M')} | Stocks scanned: {scan_results['total_scanned']}\n\n"
        
        # Gap Up alerts
        gap_up = [a for a in alerts if a["gap_pct"] > 0]
        if gap_up:
            msg += "📈 GAP UP:\n"
            for alert in gap_up[:5]:
                emoji = "⚠️" if alert["earnings_today"] else "🟢"
                earnings_tag = " [EARNINGS]" if alert["earnings_today"] else ""
                trade_hint = f" → {alert['trade_direction']}" if alert["trade_direction"] else ""
                msg += f"{emoji} {alert['symbol']}: +{alert['gap_pct']:.1f}%{earnings_tag}{trade_hint}\n"
            msg += "\n"
        
        # Gap Down alerts
        gap_down = [a for a in alerts if a["gap_pct"] < 0]
        if gap_down:
            msg += "📉 GAP DOWN:\n"
            for alert in gap_down[:5]:
                emoji = "⚠️" if alert["earnings_today"] else "🔴"
                earnings_tag = " [EARNINGS]" if alert["earnings_today"] else ""
                trade_hint = f" → {alert['trade_direction']}" if alert["trade_direction"] else ""
                msg += f"{emoji} {alert['symbol']}: {alert['gap_pct']:.1f}%{earnings_tag}{trade_hint}\n"
            msg += "\n"
        
        # Trade candidates
        candidates = scan_results.get("trade_candidates", [])[:3]
        if candidates:
            msg += "🎯 TOP TRADE CANDIDATES:\n"
            for c in candidates:
                msg += f"  • {c['symbol']} ({c.get('gap_type', 'gap')}) → {c.get('trade_direction', 'WATCH')}\n"
        
        return msg
    
    def send_alert(self, scan_results: Dict, config: dict = None):
        """Send gap alert via Telegram."""
        config = config or self.config
        
        if not config.get("alerts", {}).get("telegram_enabled"):
            logger.info("Telegram alerts disabled")
            return
        
        message = self.format_alert_message(scan_results)
        
        try:
            from live_paper_v3 import send_telegram
            send_telegram(message, config)
            logger.info("Pre-market gap alert sent")
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            print(message)  # Print to console as fallback
    
    def save_scan_results(self, scan_results: Dict):
        """Save scan results for later analysis."""
        today_str = date.today().strftime("%Y-%m-%d")
        output_file = ALERTS_DIR / f"gap_scan_{today_str}.json"
        
        with open(output_file, 'w') as f:
            json.dump(scan_results, f, indent=2, default=str)
        
        logger.info(f"Scan results saved to {output_file}")


def run_premarket_scan(symbols: List[str] = None, send_alert: bool = True):
    """
    Run pre-market gap scan and send alerts.
    Call this at 8:45-9:00 AM.
    
    Usage:
        python -c "from core.event_calendar import run_premarket_scan; run_premarket_scan()"
    """
    import yaml
    
    # Load config
    config = {}
    config_path = Path("config/config_test.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    
    scanner = PreMarketGapScanner(config)
    
    print(f"🔍 Running pre-market gap scan at {datetime.now().strftime('%H:%M:%S')}...")
    results = scanner.scan_gaps(symbols)
    
    # Print summary
    print(f"\n📊 SCAN RESULTS:")
    print(f"  Stocks scanned: {results['total_scanned']}")
    print(f"  Gaps detected: {len(results['alerts'])}")
    
    for category, gaps in results['gaps'].items():
        if gaps:
            print(f"  {category}: {len(gaps)}")
    
    # Send alert
    if send_alert and results['alerts']:
        scanner.send_alert(results, config)
    
    # Save results
    scanner.save_scan_results(results)
    
    return results


# ════════════════════════════════════════════════════════════════════════════
# POST-EARNINGS MOMENTUM STRATEGY
# ════════════════════════════════════════════════════════════════════════════

class PostEarningsMomentum:
    """
    Strategy for trading stocks the day after earnings.
    
    Key Insights:
    1. Stocks that gap up on good earnings often continue 1-3 days
    2. Stocks that gap down on bad earnings often bounce slightly then fall more
    3. First 30 minutes shows the "true" reaction
    4. Volume confirms conviction
    
    Trade Types:
    1. EARNINGS_CONTINUATION: Gap + strong volume + same direction = ride momentum
    2. EARNINGS_FADE: Gap + weak volume + reversal = fade the move
    3. EARNINGS_BREAKOUT: Consolidation after gap + breakout = new trend
    """
    
    def __init__(self, claude_brain=None):
        self.claude_brain = claude_brain
        self.event_calendar = EventCalendar()
    
    def get_post_earnings_stocks(self) -> List[Dict]:
        """Get stocks that announced earnings yesterday."""
        earnings = self.event_calendar.fetch_earnings_calendar()
        
        post_earnings = []
        for symbol, data in earnings.items():
            if data["days_until"] == -1:  # Yesterday
                post_earnings.append({
                    "symbol": symbol,
                    "earnings_date": data["date"],
                    "event": data
                })
        
        return post_earnings
    
    def analyze_earnings_reaction(self, symbol: str, 
                                   intraday_df: pd.DataFrame,
                                   prev_close: float) -> Dict:
        """
        Analyze the post-earnings price action.
        
        Args:
            symbol: Stock symbol
            intraday_df: Today's intraday OHLCV data
            prev_close: Previous day's close (before earnings)
        
        Returns:
            Dict with analysis and trading recommendation
        """
        if intraday_df.empty or len(intraday_df) < 3:
            return {"error": "Insufficient data"}
        
        # Calculate gap
        today_open = intraday_df['open'].iloc[0]
        gap_pct = (today_open - prev_close) / prev_close * 100
        
        # Current price and position
        current = intraday_df['close'].iloc[-1]
        current_vs_open = (current - today_open) / today_open * 100
        
        # First 30 min analysis (6 candles if 5-min)
        first_30 = intraday_df.iloc[:min(6, len(intraday_df))]
        first_30_high = first_30['high'].max()
        first_30_low = first_30['low'].min()
        first_30_close = first_30['close'].iloc[-1]
        
        # Volume analysis
        avg_volume = intraday_df['volume'].mean()
        first_30_volume = first_30['volume'].sum()
        volume_ratio = first_30_volume / (avg_volume * 6) if avg_volume > 0 else 1.0
        
        # Determine earnings reaction type
        reaction_type = self._classify_reaction(gap_pct, current_vs_open, volume_ratio)
        
        # Generate trading signal
        signal = self._generate_signal(
            symbol, gap_pct, current_vs_open, volume_ratio,
            reaction_type, current, prev_close, today_open,
            first_30_high, first_30_low
        )
        
        return {
            "symbol": symbol,
            "gap_pct": round(gap_pct, 2),
            "current_vs_open": round(current_vs_open, 2),
            "volume_ratio": round(volume_ratio, 2),
            "reaction_type": reaction_type,
            "signal": signal,
            "levels": {
                "prev_close": prev_close,
                "today_open": today_open,
                "first_30_high": first_30_high,
                "first_30_low": first_30_low,
                "current": current,
            }
        }
    
    def _classify_reaction(self, gap_pct: float, 
                           current_vs_open: float,
                           volume_ratio: float) -> str:
        """Classify the earnings reaction type."""
        abs_gap = abs(gap_pct)
        
        # Strong continuation (gap + follow-through)
        if gap_pct > 2 and current_vs_open > 0.5 and volume_ratio > 1.5:
            return "STRONG_BULLISH_CONTINUATION"
        elif gap_pct < -2 and current_vs_open < -0.5 and volume_ratio > 1.5:
            return "STRONG_BEARISH_CONTINUATION"
        
        # Weak continuation (gap but fading)
        if gap_pct > 1 and -0.5 < current_vs_open < 0.5:
            return "BULLISH_CONSOLIDATION"
        elif gap_pct < -1 and -0.5 < current_vs_open < 0.5:
            return "BEARISH_CONSOLIDATION"
        
        # Gap fade (gap reversed)
        if gap_pct > 1 and current_vs_open < -0.5:
            return "BULLISH_FADE"
        elif gap_pct < -1 and current_vs_open > 0.5:
            return "BEARISH_FADE"
        
        # Low volume reaction (unreliable)
        if abs_gap > 1 and volume_ratio < 0.8:
            return "LOW_CONVICTION"
        
        return "NEUTRAL"
    
    def _generate_signal(self, symbol: str, gap_pct: float,
                         current_vs_open: float, volume_ratio: float,
                         reaction_type: str, current: float,
                         prev_close: float, today_open: float,
                         first_30_high: float, first_30_low: float) -> Optional[Dict]:
        """Generate trading signal based on earnings reaction."""
        
        # Strong bullish continuation - ride the momentum
        if reaction_type == "STRONG_BULLISH_CONTINUATION":
            return {
                "direction": "LONG",
                "strategy": "EARNINGS_MOMENTUM_LONG",
                "entry": current,
                "stop_loss": max(prev_close, first_30_low * 0.995),
                "target_1": current * 1.015,  # 1.5%
                "target_2": current * 1.025,  # 2.5%
                "confidence": min(85, 60 + volume_ratio * 10),
                "position_size_factor": 1.0,
                "reason": f"Strong earnings reaction: +{gap_pct:.1f}% gap, {volume_ratio:.1f}x volume"
            }
        
        # Strong bearish continuation - short the weakness
        elif reaction_type == "STRONG_BEARISH_CONTINUATION":
            return {
                "direction": "SHORT",
                "strategy": "EARNINGS_MOMENTUM_SHORT",
                "entry": current,
                "stop_loss": min(prev_close, first_30_high * 1.005),
                "target_1": current * 0.985,
                "target_2": current * 0.975,
                "confidence": min(85, 60 + volume_ratio * 10),
                "position_size_factor": 1.0,
                "reason": f"Weak earnings reaction: {gap_pct:.1f}% gap, {volume_ratio:.1f}x volume"
            }
        
        # Bullish consolidation - wait for breakout
        elif reaction_type == "BULLISH_CONSOLIDATION":
            return {
                "direction": "LONG",
                "strategy": "EARNINGS_BREAKOUT_LONG",
                "entry": first_30_high * 1.002,  # Entry above 30-min high
                "stop_loss": first_30_low * 0.995,
                "target_1": first_30_high * 1.015,
                "target_2": first_30_high * 1.025,
                "confidence": 55,
                "position_size_factor": 0.8,
                "trigger": "BREAKOUT_ABOVE_FIRST_30_HIGH",
                "reason": f"Earnings consolidation: waiting for breakout above {first_30_high:.2f}"
            }
        
        # Bearish consolidation - wait for breakdown
        elif reaction_type == "BEARISH_CONSOLIDATION":
            return {
                "direction": "SHORT",
                "strategy": "EARNINGS_BREAKDOWN_SHORT",
                "entry": first_30_low * 0.998,
                "stop_loss": first_30_high * 1.005,
                "target_1": first_30_low * 0.985,
                "target_2": first_30_low * 0.975,
                "confidence": 55,
                "position_size_factor": 0.8,
                "trigger": "BREAKDOWN_BELOW_FIRST_30_LOW",
                "reason": f"Earnings consolidation: waiting for breakdown below {first_30_low:.2f}"
            }
        
        # Gap fade - counter-trend (higher risk)
        elif reaction_type == "BULLISH_FADE" and volume_ratio > 1.2:
            return {
                "direction": "SHORT",
                "strategy": "EARNINGS_FADE_SHORT",
                "entry": current,
                "stop_loss": first_30_high * 1.003,
                "target_1": today_open * 0.995,  # Back to open
                "target_2": prev_close,  # Gap fill
                "confidence": 50,
                "position_size_factor": 0.6,  # Smaller size for counter-trend
                "reason": f"Fading earnings gap: +{gap_pct:.1f}% gap reversing"
            }
        
        elif reaction_type == "BEARISH_FADE" and volume_ratio > 1.2:
            return {
                "direction": "LONG",
                "strategy": "EARNINGS_FADE_LONG",
                "entry": current,
                "stop_loss": first_30_low * 0.997,
                "target_1": today_open * 1.005,
                "target_2": prev_close,
                "confidence": 50,
                "position_size_factor": 0.6,
                "reason": f"Fading earnings gap: {gap_pct:.1f}% gap reversing"
            }
        
        # Low conviction - skip
        return None
    
    def get_all_signals(self, intraday_data: Dict[str, pd.DataFrame],
                        prev_closes: Dict[str, float]) -> List[Dict]:
        """
        Get trading signals for all post-earnings stocks.
        
        Args:
            intraday_data: Dict of symbol -> intraday DataFrame
            prev_closes: Dict of symbol -> previous close price
        
        Returns:
            List of trading signals
        """
        post_earnings = self.get_post_earnings_stocks()
        signals = []
        
        for stock in post_earnings:
            symbol = stock["symbol"]
            
            if symbol not in intraday_data or symbol not in prev_closes:
                continue
            
            analysis = self.analyze_earnings_reaction(
                symbol,
                intraday_data[symbol],
                prev_closes[symbol]
            )
            
            if analysis.get("signal"):
                signals.append({
                    "symbol": symbol,
                    "analysis": analysis,
                    "signal": analysis["signal"]
                })
        
        # Sort by confidence
        signals.sort(key=lambda x: x["signal"].get("confidence", 0), reverse=True)
        
        return signals


def get_post_earnings_signals(config: dict = None) -> List[Dict]:
    """
    Get post-earnings trading signals.
    Call this at 10:00 AM (after first 30 min).
    
    Usage:
        from core.event_calendar import get_post_earnings_signals
        signals = get_post_earnings_signals()
    """
    strategy = PostEarningsMomentum()
    post_earnings = strategy.get_post_earnings_stocks()
    
    if not post_earnings:
        logger.info("No post-earnings stocks today")
        return []
    
    logger.info(f"Post-earnings stocks today: {[s['symbol'] for s in post_earnings]}")
    
    # In live trading, fetch actual intraday data here
    # For now, return the list of stocks to watch
    return post_earnings
