"""Convenience types."""

from datetime import datetime
from typing import TypeVar

import numpy as np
import pandas as pd

#: Datetime-like types (coerceable to pd.Timestamp)
DatetimeLike = TypeVar("DatetimeLike", datetime, pd.Timestamp, np.datetime64, str)
