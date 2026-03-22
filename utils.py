# utils.py
import pandas as pd
import streamlit as st

def format_large_number(num):
    """
    Format large numbers with K, M, B suffixes.
    """
    if num >= 1e9:
        return f"${num/1e9:.2f}B"
    elif num >= 1e6:
        return f"${num/1e6:.2f}M"
    elif num >= 1e3:
        return f"${num/1e3:.2f}K"
    else:
        return f"${num:.2f}"

def validate_dates(start_date, end_date):
    """
    Validate that start date is before end date.
    """
    if start_date >= end_date:
        st.error("Start date must be before end date")
        return False
    return True

def get_market_hours_status():
    """
    Simple check if market is likely open (just for info).
    """
    from datetime import datetime
    import pytz
    
    ny_time = datetime.now(pytz.timezone('US/Eastern'))
    market_open = ny_time.replace(hour=9, minute=30, second=0)
    market_close = ny_time.replace(hour=16, minute=0, second=0)
    
    if ny_time.weekday() >= 5:  # Weekend
        return "Closed (Weekend)"
    elif market_open <= ny_time <= market_close:
        return "Open"
    else:
        return "Closed"