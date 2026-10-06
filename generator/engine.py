import uuid
import numpy as np
import pandas as pd
from faker import Faker
from .presets import PRESETS
from .relationships import RelationshipEngine
from .domain_invariants import apply_domain_invariants, verify_domain_invariants
from .churn_logic import compute_churn_probabilities, assign_churn_labels
from .schema import Column, DTypes

# Pre-cached realistic pools for high-throughput vectorized Faker generation
_CACHED_POOLS = {
    'city': [
        "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia",
        "San Antonio", "San Diego", "Dallas", "Austin", "San Jose", "Seattle",
        "Denver", "Boston", "Atlanta", "Miami", "Portland", "San Francisco"
    ],
    'state': [
        "CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI",
        "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "CO"
    ],
    'country': [
        "United States", "United Kingdom", "Germany", "France", "Canada",
        "Australia", "Netherlands", "Sweden", "Singapore", "Japan"
    ]
}

ALLOWED_GENERATORS = {
    # numpy random distributions
    "numpy.uniform": np.random.uniform,
    "numpy.normal": np.random.normal,
    "numpy.binomial": np.random.binomial,
    "numpy.poisson": np.random.poisson,
    "numpy.exponential": np.random.exponential,
    "numpy.beta": np.random.beta,
    "numpy.gamma": np.random.gamma,
    "numpy.lognormal": np.random.lognormal,
    "numpy.choice": np.random.choice,
    "numpy.randint": np.random.randint,
    "numpy.geometric": np.random.geometric,
    "numpy.weibull": np.random.weibull,
    "numpy.standard_normal": np.random.standard_normal,
    "numpy.random.uniform": np.random.uniform,
    "numpy.random.normal": np.random.normal,
    "numpy.random.binomial": np.random.binomial,
    "numpy.random.poisson": np.random.poisson,
    "numpy.random.exponential": np.random.exponential,
    "numpy.random.beta": np.random.beta,
    "numpy.random.gamma": np.random.gamma,
    "numpy.random.lognormal": np.random.lognormal,
    "numpy.random.choice": np.random.choice,
    "numpy.random.randint": np.random.randint,
    "numpy.random.geometric": np.random.geometric,
    "numpy.random.weibull": np.random.weibull,
    "numpy.random.standard_normal": np.random.standard_normal,
    # faker generators
    "faker.uuid4": "uuid4",
    "faker.random_element": "random_element",
    "faker.city": "city",
    "faker.state": "state",
    "faker.country": "country",
    "faker.zipcode": "zipcode",
    "faker.name": "name",
    "faker.first_name": "first_name",
    "faker.last_name": "last_name",
    "faker.email": "email",
    "faker.phone_number": "phone_number",
    "faker.date_between": "date_between",
    "faker.date_this_year": "date_this_year",
}

def resolve_generator(gen_str: str):
    """
    Safely resolves permitted 'faker.method' or 'numpy.method' to actual callables using an allowlist.
    Prevents unauthorized reflection or execution.
    """
    if gen_str not in ALLOWED_GENERATORS:
        raise ValueError(
            f"Unauthorized or unknown generator: '{gen_str}'. "
            f"Allowed generators: {sorted(ALLOWED_GENERATORS.keys())}"
        )
    target = ALLOWED_GENERATORS[gen_str]
    if callable(target):
        return target
    # Faker attribute
    fake = Faker()
    return getattr(fake, target)

def generate_dataset(config: dict) -> pd.DataFrame:
    """
    Main entry point for dataset generation.
    Honors user-provided columns and relationships if provided.
    Fast vectorized generation supporting high row counts (up to millions).
    """
    seed = config.get('seed')
    if seed is not None:
        np.random.seed(seed)
        Faker.seed(seed)
        
    num_rows = int(config.get('num_rows', 1000))
    preset_name = config.get('preset_name')
    preset = PRESETS.get(preset_name.lower()) if preset_name and preset_name.lower() in PRESETS else None

    # 1. Determine columns: honor user-supplied columns if present
    if 'columns' in config and config['columns'] is not None:
        raw_cols = config['columns']
        columns = [
            Column.from_dict(c) if isinstance(c, dict) else c 
            for c in raw_cols 
            if not isinstance(c, dict) or c.get('enabled', True)
        ]
    elif preset:
        columns = preset['columns']
    else:
        columns = []

    # 2. Determine relationships: honor user-supplied relationships if present
    if 'relationships' in config and config['relationships'] is not None:
        rules = config['relationships']
    elif preset:
        rules = preset['relationships']
    else:
        rules = []

    # 3. Determine target churn rate and noise level
    default_churn = preset['target_churn_rate'] if preset else 0.20
    default_noise = preset['noise_level'] if preset else 0.3
    target_churn_rate = float(config.get('target_churn_rate', default_churn))
    noise_level = float(config.get('noise_level', default_noise))

    df_data = {}
    
    # 4. Generate base columns independently (with vectorization fast-paths)
    for col in columns:
        params = col.params.copy()
        
        # Fast-path for numpy distributions
        if col.generator.startswith('numpy'):
            gen_func = resolve_generator(col.generator)
            # Strictly force generation size to num_rows, ignoring any user-supplied size
            params['size'] = num_rows
            try:
                df_data[col.name] = gen_func(**params)
            except TypeError:
                size = params.pop('size')
                try:
                    df_data[col.name] = gen_func(size=size, **params)
                except TypeError:
                    df_data[col.name] = np.array([gen_func(**params) for _ in range(size)])
        
        # Fast-paths for UUID
        elif col.generator == 'faker.uuid4':
            fake = Faker()
            if seed is not None:
                fake.seed_instance(seed)
            # Vectorized fast string IDs
            df_data[col.name] = [fake.uuid4() for _ in range(num_rows)]
            
        # Fast-path for random element choice
        elif col.generator == 'faker.random_element' and 'elements' in params:
            elements = params['elements']
            p = params.get('p')
            df_data[col.name] = np.random.choice(elements, size=num_rows, p=p)
            
        # Fast-path for common geographic pools
        elif col.generator == 'faker.city':
            df_data[col.name] = np.random.choice(_CACHED_POOLS['city'], size=num_rows)
        elif col.generator == 'faker.state':
            df_data[col.name] = np.random.choice(_CACHED_POOLS['state'], size=num_rows)
        elif col.generator == 'faker.country':
            df_data[col.name] = np.random.choice(_CACHED_POOLS['country'], size=num_rows)
        elif col.generator == 'faker.zipcode':
            df_data[col.name] = np.random.randint(10000, 99999, size=num_rows).astype(str)
            
        elif col.generator.startswith('faker'):
            gen_func = resolve_generator(col.generator)
            df_data[col.name] = [gen_func(**params) for _ in range(num_rows)]
            
    df = pd.DataFrame(df_data)
    
    # 5. Apply relationship engine and record execution telemetry
    engine = RelationshipEngine()
    df, churn_contributions, rule_stats = engine.apply_all(df, rules, noise_level)
    df.attrs['rule_stats'] = rule_stats
    
    # 6. Apply bounds / clipping for numeric columns first
    for col in columns:
        if col.name in df.columns and col.dtype == DTypes.NUMERIC:
            min_val = col.min_val if col.min_val is not None else (0.0 if any(t in col.name for t in ['charge', 'spent', 'balance', 'revenue', 'count', 'amount', 'score', 'days', 'hours', 'points']) else None)
            max_val = col.max_val
            if min_val is not None or max_val is not None:
                df[col.name] = np.clip(df[col.name], a_min=min_val, a_max=max_val)

    # 6.5 Enforce domain invariants & hierarchical service gates on valid bounds
    df = apply_domain_invariants(df, preset_name)
    df.attrs['invariants_report'] = verify_domain_invariants(df, preset_name)
    
    # 7. Round numeric features to 2 decimals BEFORE scoring churn probabilities
    # This guarantees the exported CSV values precisely match the inputs used for scoring.
    for col in columns:
        if col.name in df.columns and col.dtype == DTypes.NUMERIC:
            df[col.name] = df[col.name].apply(lambda x: round(float(x), 2) if pd.notnull(x) else x)

    # 8. Compute churn probabilities and assign labels on exact rounded features
    probs = compute_churn_probabilities(df, columns, churn_contributions, target_churn_rate, noise_level)
    labels = assign_churn_labels(probs)
    df['churn'] = labels
            
    return df

