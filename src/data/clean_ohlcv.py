import pandas as pd 
import logging
from typing import Dict, List

# logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_ohlcv(df: pd.DataFrame, issues: List[Dict], ticker:str):
    """
    clean_ohlcv gets called if the validator returns some issues
    in non strict mode. This is a sanitizer that explicitly fixes
    the issues raised by the validate_ohlcv in non-strict mode.

    The issues are related financial sanity not structural. Cleaner 
    will drop those entries and send back the clean df. 

    
    :param df: DataFrame with issues 
    :type df: pd.DataFrame
    :param issues: All the issues in the given df
    :type issues: List[Dict]
    :param ticker: Ticker's name
    :type ticker: str
    
    returns
        df : a cleaned pd.DataFrame that is ready for strict validation
    """
    logging.info(f"{ticker} | Row Count before: {df.shape[0]}")
    bad_indices = {issue['idx'] for issue in issues if issue['idx'] is not None} 
    df = df.drop(index=bad_indices)
    logging.info(f"{ticker} | Row Count after: {df.shape[0]}")
    logging.info(f"{ticker} | Reasons:")
    for issue in issues:
        logging.info(f"{issue['reason']}")
    
    return df    
