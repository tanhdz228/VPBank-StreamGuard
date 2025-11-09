"""
Load entity risk data to DynamoDB.
Reads entity_risk_combined.csv and bulk loads to DynamoDB table.
"""

import boto3
import pandas as pd
from decimal import Decimal
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))


def load_entity_risk(table_name: str, csv_path: str, batch_size: int = 25):
    """
    Load entity risk data to DynamoDB.

    Args:
        table_name: DynamoDB table name
        csv_path: Path to entity_risk_combined.csv
        batch_size: Number of items per batch write (max 25)
    """
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)

    print(f"\n{'='*80}")
    print("Loading Entity Risk to DynamoDB")
    print(f"{'='*80}\n")
    print(f"Table: {table_name}")
    print(f"CSV: {csv_path}\n")

    # Read CSV
    print("Reading CSV...")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df):,} entities from CSV")

    # Show sample
    print(f"\nSample data:")
    print(df.head().to_string(index=False))

    # Prepare items
    print(f"\nPreparing items for DynamoDB...")
    items = []

    for _, row in df.iterrows():
        # Create composite key: entity_type#entity_id
        entity_key = f"{row['entity_type']}#{row['entity_id']}"

        item = {
            'entity_key': entity_key,
            'entity_type': str(row['entity_type']),
            'entity_id': str(row['entity_id']),
            'risk_score': Decimal(str(row['risk_score'])),
            'fraud_rate': Decimal(str(row['fraud_rate'])),
            'anomaly_score': Decimal(str(row['anomaly_score'])),
            'transaction_count': int(row['transaction_count'])
        }

        items.append(item)

    print(f"✓ Prepared {len(items):,} items")

    # Batch write to DynamoDB
    print(f"\nWriting to DynamoDB in batches of {batch_size}...")

    total_batches = (len(items) + batch_size - 1) // batch_size
    successful = 0
    failed = 0

    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        batch_num = i // batch_size + 1

        try:
            with table.batch_writer() as writer:
                for item in batch:
                    writer.put_item(Item=item)

            successful += len(batch)
            print(f"  Batch {batch_num}/{total_batches}: ✓ Wrote {len(batch)} items")

        except Exception as e:
            failed += len(batch)
            print(f"  Batch {batch_num}/{total_batches}: ❌ Failed - {str(e)}")

    # Summary
    print(f"\n{'='*80}")
    print("Load Complete!")
    print(f"{'='*80}")
    print(f"Successful: {successful:,}")
    print(f"Failed: {failed:,}")
    print(f"Total: {len(items):,}\n")

    # Verify
    print("Verifying data...")
    response = table.scan(Select='COUNT')
    item_count = response['Count']
    print(f"✓ Table contains {item_count:,} items")

    # Test query
    print("\nTesting sample queries:")
    test_keys = [
        'id_23#1',
        'DeviceInfo#0',
        'P_emaildomain#0'
    ]

    for entity_key in test_keys:
        try:
            response = table.get_item(Key={'entity_key': entity_key})
            if 'Item' in response:
                item = response['Item']
                print(f"  ✓ {entity_key}: risk_score={float(item['risk_score']):.4f}")
            else:
                print(f"  - {entity_key}: Not found")
        except Exception as e:
            print(f"  ❌ {entity_key}: Error - {str(e)}")

    print(f"\n{'='*80}")
    print("Entity risk data loaded successfully!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load entity risk to DynamoDB")
    parser.add_argument('--table', required=True, help='DynamoDB table name')
    parser.add_argument('--csv', required=True, help='Path to entity_risk_combined.csv')
    parser.add_argument('--batch-size', type=int, default=25, help='Batch size (max 25)')

    args = parser.parse_args()

    load_entity_risk(args.table, args.csv, args.batch_size)
