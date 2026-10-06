import math
from dataclasses import dataclass, field
from typing import Optional, Any, Dict

class DTypes:
    NUMERIC = 'numeric'
    CATEGORICAL = 'categorical'
    BOOLEAN = 'boolean'
    DATE = 'date'

VALID_DTYPES = {DTypes.NUMERIC, DTypes.CATEGORICAL, DTypes.BOOLEAN, DTypes.DATE}

@dataclass
class Column:
    """
    Defines the schema for a single column in the generated dataset.
    """
    name: str
    dtype: str           # 'numeric', 'categorical', 'boolean', 'date'
    generator: str       # 'faker.<provider>' or 'numpy.<distribution>'
    params: dict = field(default_factory=dict)
    churn_weight: float = 0.0  # Coefficient in churn logistic model (0 = no effect)
    category: str = "General"  # Category for grouping in UI (e.g. Demographics, Usage, Billing)
    min_val: Optional[float] = None
    max_val: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Column":
        """
        Robustly instantiate a Column from a dict, validating schema,
        stripping unauthorized 'size' overrides, and rejecting reserved column names.
        """
        raw_name = str(data.get('name', '')).strip()
        if not raw_name:
            raise ValueError("Column name cannot be empty.")
        if raw_name.lower() == 'churn':
            raise ValueError("Column name 'churn' is reserved for the synthetic target label.")
            
        raw_dtype = str(data.get('dtype') or data.get('type') or DTypes.CATEGORICAL).lower()
        if raw_dtype not in VALID_DTYPES:
            raise ValueError(f"Invalid dtype '{raw_dtype}'. Must be one of {sorted(VALID_DTYPES)}.")
            
        generator = str(data.get('generator', 'faker.uuid4')).strip()
        raw_params = data.get('params', {})
        if not isinstance(raw_params, dict):
            raw_params = {}
        # Strictly strip any 'size' parameter to avoid overriding generation row counts
        params = {k: v for k, v in raw_params.items() if k != 'size'}
        
        try:
            churn_weight = float(data.get('churn_weight', 0.0))
            if not math.isfinite(churn_weight):
                churn_weight = 0.0
        except (ValueError, TypeError):
            churn_weight = 0.0
            
        category = str(data.get('category', 'General')).strip()
        
        min_val = data.get('min_val')
        if min_val is not None:
            try:
                min_val = float(min_val)
                if not math.isfinite(min_val):
                    min_val = None
            except (ValueError, TypeError):
                min_val = None
                
        max_val = data.get('max_val')
        if max_val is not None:
            try:
                max_val = float(max_val)
                if not math.isfinite(max_val):
                    max_val = None
            except (ValueError, TypeError):
                max_val = None

        if min_val is not None and max_val is not None and min_val > max_val:
            min_val, max_val = max_val, min_val
        
        return cls(
            name=raw_name,
            dtype=raw_dtype,
            generator=generator,
            params=params,
            churn_weight=churn_weight,
            category=category,
            min_val=min_val,
            max_val=max_val,
        )

