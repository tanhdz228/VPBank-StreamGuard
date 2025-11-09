"""Quick test script to verify data loading and preprocessing."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_loader import CreditCardLoader, IEEECISLoader
from src.data.creditcard_preprocessor import CreditCardPreprocessor
from src.data.ieee_preprocessor import IEEECISPreprocessor


def test_creditcard_loading():
    """Test Credit Card dataset loading."""
    print("\n" + "="*60)
    print("Testing Credit Card Dataset")
    print("="*60 + "\n")
    
    loader = CreditCardLoader()
    df = loader.load()
    
    stats = loader.get_statistics(df)
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Credit Card loading test passed!")
    return df


def test_creditcard_preprocessing(df):
    """Test Credit Card preprocessing."""
    print("\n" + "="*60)
    print("Testing Credit Card Preprocessing")
    print("="*60 + "\n")
    
    preprocessor = CreditCardPreprocessor()
    
    # Test feature engineering
    df_feat = preprocessor.create_time_features(df)
    df_feat = preprocessor.create_amount_features(df_feat)
    
    print(f"Original columns: {len(df.columns)}")
    print(f"After feature engineering: {len(df_feat.columns)}")
    print(f"New features: {set(df_feat.columns) - set(df.columns)}")
    
    # Test train/val/test split
    print("\nTesting train/val/test split...")
    splits = preprocessor.train_val_test_split(df.sample(10000))  # Sample for speed
    
    print("\n✓ Credit Card preprocessing test passed!")
    return preprocessor


def test_ieee_loading():
    """Test IEEE-CIS dataset loading."""
    print("\n" + "="*60)
    print("Testing IEEE-CIS Dataset")
    print("="*60 + "\n")
    
    loader = IEEECISLoader()
    
    # Load transaction only (faster)
    trans_df = loader.load_transaction(train=True)
    print(f"\nTransaction shape: {trans_df.shape}")
    
    # Load identity
    ident_df = loader.load_identity(train=True)
    print(f"Identity shape: {ident_df.shape}")
    
    # Load full
    print("\nLoading full dataset (transaction + identity)...")
    full_df = loader.load_full(train=True)
    
    # Get V-blocks stats
    print("\nV-blocks statistics (first 20):")
    v_stats = loader.get_v_blocks_stats(full_df)
    print(v_stats.head(20).to_string(index=False))
    
    # Get identity stats
    print("\nIdentity features statistics:")
    id_stats = loader.get_identity_stats(full_df)
    print(id_stats.head(20).to_string(index=False))
    
    print("\n✓ IEEE-CIS loading test passed!")
    return full_df


def test_ieee_preprocessing(df):
    """Test IEEE-CIS preprocessing."""
    print("\n" + "="*60)
    print("Testing IEEE-CIS Preprocessing")
    print("="*60 + "\n")
    
    # Sample for faster testing
    df_sample = df.sample(50000, random_state=42)
    print(f"Testing on sample of {len(df_sample):,} rows\n")
    
    preprocessor = IEEECISPreprocessor()
    
    # Run full preprocessing
    df_processed = preprocessor.preprocess(df_sample, fit=True)
    
    print(f"\nOriginal features: {len(df_sample.columns)}")
    print(f"Processed features: {len(df_processed.columns)}")
    
    # Prepare for training
    X, y, feature_names = preprocessor.prepare_for_training(df_processed)
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Number of features: {len(feature_names)}")
    
    print("\n✓ IEEE-CIS preprocessing test passed!")
    return preprocessor


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("VPBank StreamGuard - Data Pipeline Tests")
    print("="*80)
    
    try:
        # Test Credit Card
        cc_df = test_creditcard_loading()
        test_creditcard_preprocessing(cc_df)
        
        # Test IEEE-CIS
        ieee_df = test_ieee_loading()
        test_ieee_preprocessing(ieee_df)
        
        # Summary
        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)
        print("\nNext steps:")
        print("1. Run: python scripts/train_fast_lane.py")
        print("2. Train Deep Lane models on IEEE-CIS")
        print("3. Build feature store and API")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()