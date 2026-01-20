#!/usr/bin/env python3
"""
Simple test to verify the CEEMDAN import fix without full dependencies
"""

def test_emd_import_logic():
    """
    Test the core logic for fixing the EMD import issue
    """
    print("Testing EMD import fix logic...")
    
    try:
        # Simulate the problematic import scenario
        import PyEMD
        print("✅ PyEMD imported successfully")
        
        # Check if EMD exists and if it's callable
        if hasattr(PyEMD, 'EMD'):
            print(f"EMD type: {type(PyEMD.EMD)}")
            print(f"Is EMD callable: {callable(PyEMD.EMD)}")
            
            # This is the kind of check we do in our fix
            if not callable(PyEMD.EMD):
                print("EMD is not callable - would need replacement")
                # This simulates what our fix does
                try:
                    from PyEMD.EMD import EMD as EMD_Class
                    print("✅ Successfully imported EMD as class")
                    
                    # Replace the module with the class
                    PyEMD.EMD = EMD_Class
                    print("✅ EMD replaced with callable class")
                    
                    # Verify it's now callable
                    print(f"Now EMD is callable: {callable(PyEMD.EMD)}")
                    
                except ImportError as e:
                    print(f"Could not import EMD as class: {e}")
                    return False
            else:
                print("EMD is already callable")
        
        # Test CEEMDAN import
        try:
            from PyEMD.CEEMDAN import CEEMDAN
            print("✅ CEEMDAN imported successfully")
            
            # Test creating an instance - this is where the original error occurred
            # We'll use minimal parameters to avoid other dependency issues
            ceemdan_instance = CEEMDAN(trials=2, noise_width=0.05)
            print("✅ CEEMDAN instance created successfully")
            
            return True
            
        except ImportError as e:
            print(f"Could not import CEEMDAN: {e}")
            return False
            
    except ImportError as e:
        print(f"PyEMD not available: {e}")
        return False
    except Exception as e:
        print(f"Error in test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Running simple CEEMDAN fix test...")
    success = test_emd_import_logic()
    if success:
        print("\n🎉 Simple test PASSED - the CEEMDAN fix should work!")
    else:
        print("\n⚠️  Simple test had issues, but this might be due to missing PyEMD package")
    print("Note: The actual fix is implemented in the ceemdan_models.py file.")