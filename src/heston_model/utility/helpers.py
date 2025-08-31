import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any

import pandas as pd
import numpy as np

def remove_cmdstan_files(direc = 'cmdstan_out'):
    print('Clearing cmdstan dir...')
    count = 0
    for file in os.listdir(direc):
        if file.endswith('.csv') or (file.endswith('.txt') and 'stdout' in file):
            os.remove(os.path.join(direc, file))
            print(os.path.join(direc, file))
            count += 1
    print(f'Removed {count} files')
    


def parse_stan_dimensions(stan_code_filepath: str) -> Tuple[Dict[str, List[str]], List[str], Set[str]]:
    """
    Parse Stan code and extract:
    - variable -> dimension mapping
    - posterior predictive variables
    - symbolic dimension labels
    """
    stan_code = Path(stan_code_filepath).read_text()
    
    block_pattern = r"(parameters|transformed parameters|generated quantities)\s*{([^}]*)}"
    matches = re.findall(block_pattern, stan_code, re.DOTALL)

    variables: Dict[str, List[str]] = {}
    posterior_predictive: List[str] = []
    dim_symbols: Set[str] = set()

    vector_pattern = re.compile(r'vector\s*<[^>]*>\s*\[([^\]]+)\]\s+(\w+)')
    vector_simple = re.compile(r'vector\s*\[([^\]]+)\]\s+(\w+)')
    array_vector = re.compile(r'array\s*\[([^\]]+)\]\s+vector\s*\[([^\]]+)\]\s+(\w+)')
    matrix_pattern = re.compile(r'matrix\s*\[([^\]]+),\s*([^\]]+)\]\s+(\w+)')
    simple_pattern = re.compile(r'(?:(real|int)\s+)(\w+)(\s*\[([^\]]+)\])?')

    for block, content in matches:
        lines = content.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('//') or ';' not in line:
                continue
            line = line.split('//')[0]  # remove trailing comments

            var_name, dims = None, []

            if m := array_vector.search(line):
                dims = [m.group(1), m.group(2)]
                var_name = m.group(3)
            elif m := vector_pattern.search(line) or vector_simple.search(line):
                dims = [m.group(1)]
                var_name = m.group(2)
            elif m := matrix_pattern.search(line):
                dims = [m.group(1), m.group(2)]
                var_name = m.group(3)
            elif m := simple_pattern.search(line):
                var_name = m.group(2)
                if m.group(4):
                    dims = [d.strip() for d in m.group(4).split(',')]

            if var_name:
                dim_symbols.update(dims)
                if block == "generated quantities" and (
                    var_name.endswith('_future') or var_name.endswith('_hat') or var_name.endswith('_pred')
                ):
                    posterior_predictive.append(var_name)
                variables[var_name] = dims

    return variables, posterior_predictive, dim_symbols

def map_day_in_quarter(date: pd.Timestamp) -> int:
    """Maps a date to the day in the quarter."""
    start_of_quarter = date - pd.offsets.QuarterBegin(startingMonth=1)
    return (date - start_of_quarter).days + 1

def get_future_days_in_quarter(start_date: pd.Timestamp, num_days: int) -> List[int]:
    """Generates a list of days in the quarter for (only trading days dates)."""
    future_dates = pd.date_range(start=start_date, periods=num_days, freq='B')
    return [map_day_in_quarter(date) for date in future_dates]
    
def create_data_dict(
    log_ret : pd.Series,
    N_days_future : int = 252
    ) -> Dict[str, Any]:

    df = pd.DataFrame(log_ret).dropna()
    
    """
    Create data dictionary for Stan model.
    """
    
    # mapping date to integer day in quarter

    df['day_in_quarter'] = df.index.map(map_day_in_quarter)

    # also getting day in quarter filler for future dates (past the last date in df)
    
    future_days_in_quarter = get_future_days_in_quarter(df.index[-1] + pd.Timedelta(days=1), N_days_future)
    
    data = {
        'T' : len(log_ret),
        'returns' : log_ret.values, #/ np.std(chosen_ticker.values),
        'T_future' : N_days_future,
        'quarter_period' : 63,
        'day_in_quarter' : df['day_in_quarter'].values,
        'day_in_quarter_future' : future_days_in_quarter,
        
    }
    
    return data
    
    