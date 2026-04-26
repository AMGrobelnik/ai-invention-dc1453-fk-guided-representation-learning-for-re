"""
Relational Deep Learning Dataset Loader
========================================

Loads 15 curated relational datasets with explicit foreign key relationships
for RDL research across e-commerce, healthcare, social networks, information
extraction, behavioral analytics, and tabular domains.

Usage:
    from data import load_dataset, list_datasets

    # List all available datasets
    datasets = list_datasets()

    # Load a specific dataset (full variant)
    df = load_dataset('millat/e-commerce-orders', variant='full')

    # Load mini variant for fast iteration
    df_mini = load_dataset('millat/e-commerce-orders', variant='mini')
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any


def list_datasets() -> List[str]:
    """Return list of available dataset names."""
    datasets_dir = Path(__file__).parent
    dataset_files = list(datasets_dir.glob('full_*.json'))
    return sorted([f.stem.replace('full_', '') for f in dataset_files])


def load_dataset(
    dataset_name: str,
    variant: str = 'full'
) -> List[Dict[str, Any]]:
    """
    Load a dataset variant.

    Args:
        dataset_name: Dataset identifier (e.g., 'millat/e-commerce-orders')
        variant: 'full' (complete), 'mini' (3-10 rows), or 'preview' (10-100 rows)

    Returns:
        List of records (each record is a dict)
    """
    datasets_dir = Path(__file__).parent

    # Normalize dataset name
    safe_name = dataset_name.replace('/', '_').replace('-', '_')
    file_path = datasets_dir / f'{variant}_{safe_name}.json'

    if not file_path.exists():
        raise FileNotFoundError(f'Dataset not found: {file_path}')

    with open(file_path, 'r') as f:
        data = json.load(f)

    return data if isinstance(data, list) else [data]


def dataset_info() -> Dict[str, Dict[str, Any]]:
    """Return metadata about all available datasets."""
    return {
        'e-commerce-orders': {
            'domain': 'E-commerce',
            'rows': 10000,
            'size_mb': 5.1,
            'tables': 3,
            'relationships': ['customer→order', 'product→order'],
        },
        'healthcare-data': {
            'domain': 'Healthcare',
            'rows': 500000,
            'size_mb': 193,
            'tables': 4,
            'relationships': ['patient→admission', 'doctor→admission', 'dept→staff'],
        },
        'social-network-ads': {
            'domain': 'Social Networks',
            'rows': 400,
            'size_mb': 0.04,
            'tables': 1,
            'relationships': ['user→behavior'],
        },
    }


if __name__ == '__main__':
    # Example usage
    print('Available datasets:', list_datasets())
    print('\nLoading sample from e-commerce-orders (mini variant)...')
    sample = load_dataset('millat/e-commerce-orders', variant='mini')
    print(f'Loaded {len(sample)} records')
    if sample:
        print('Sample record:', sample[0])
