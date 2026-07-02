import json
import numpy as np
import pandas as pd


class FinancialEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles NumPy and Pandas types that the
    standard library cannot serialize (e.g. np.int64, pd.Timestamp).
    Used when writing analysis results to JSON or returning API responses.
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.strftime("%Y-%m-%d")
        return super().default(obj)
