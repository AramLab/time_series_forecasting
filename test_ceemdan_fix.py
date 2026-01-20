#!/usr/bin/env python3
"""
Test script to verify the CEEMDAN import fix
"""
import sys
import os

# Add workspace to path
sys.path.insert(0, '/workspace')

def test_ceemdan_import():
    print("Testing CEEMDAN import fix...")
    
    try:
        from models.ceemdan_models import safe_import_ceemdan
        print("✅ Successfully imported safe_import_ceemdan function")
        
        CEEMDAN_Class = safe_import_ceemdan()
        if CEEMDAN_Class is not None:
            print("✅ CEEMDAN class successfully loaded")
            
            # Test creating an instance (this was causing the original error)
            try:
                ceemdan_instance = CEEMDAN_Class(trials=5, noise_width=0.05)
                print("✅ CEEMDAN instance successfully created")
                
                # Test with some sample data
                import numpy as np
                sample_data = np.random.random(50)
                
                imfs = ceemdan_instance(sample_data)
                print(f"✅ CEEMDAN decomposition successful, got {len(imfs)} IMFs")
                return True
                
            except Exception as e:
                print(f"❌ Error creating or using CEEMDAN instance: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print("❌ CEEMDAN class could not be loaded")
            return False
            
    except Exception as e:
        print(f"❌ Error during CEEMDAN import test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ceemdan_import()
    if success:
        print("\n🎉 CEEMDAN fix test PASSED!")
    else:
        print("\n💥 CEEMDAN fix test FAILED!")
    sys.exit(0 if success else 1)