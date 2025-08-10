#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SMS Spam Detection System - Main Driver
Coordinates the detector and GUI components
"""

import os
from sms_spam_detector import MultiClassSMSDetector
from sms_gui import SMSDetectorTkinterGUI

def main():
    """Main driver function"""
    print("="*60)
    print("      Multi-Class SMS Spam Detection System")
    print("="*60)
    print()
    
    # Data file path
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    csv_file = os.path.join(desktop_path, "balanced_spam_data 2.0.csv")
    
    # Check if data file exists
    if not os.path.exists(csv_file):
        print(f"❌ Error: Data file not found!")
        print(f"Expected location: {csv_file}")
        print("Please ensure 'balanced_spam_data 2.0.csv' is on your Desktop")
        return
    
    print(f"✅ Data file found: {csv_file}")
    
    # Create detector instance
    print("\n" + "="*60)
    print("STEP 1: INITIALIZING SMS DETECTOR")
    print("="*60)
    
    detector = MultiClassSMSDetector(csv_file)
    
    # Load and process data
    print("\nLoading data...")
    if not detector.load_data():
        print("❌ Failed to load data. Exiting.")
        return
    
    # Data cleaning
    text_column = "messages"
    target_column = "multiclass_label"
    detector.data = detector.data.dropna(subset=[text_column, 'label'])
    print(f"✅ Clean data: {len(detector.data)} records")
    
    # Feature engineering
    print("\n" + "="*60)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*60)
    
    if not detector.prepare_features(text_column, target_column):
        print("❌ Feature engineering failed. Exiting.")
        return
    
    # Model training
    print("\n" + "="*60)
    print("STEP 3: MODEL TRAINING")
    print("="*60)
    
    detector.train_multiclass_model()
    
    # Model evaluation
    print("\n" + "="*60)
    print("STEP 4: MODEL EVALUATION")
    print("="*60)
    
    accuracy, _, _ = detector.evaluate_multiclass_model()
    
    print(f"\n✅Multi-class accuracy: {accuracy:.4f}")
    
    # Quick validation with sample predictions
    print("\n" + "="*60)
    print("STEP 5: VALIDATION WITH SAMPLE PREDICTIONS")
    print("="*60)
    
    test_messages = [
        "Congrats!! YOU WON 2 MILLION DOLLARS!! Redeem it NOW",    
        "URGENT! You have won £1000! Call 09061234567 now!",  
        "Hi, can we meet for lunch tomorrow?",     
        "Get your FREE ringtone! Text NOW to 12345",      
    ]
    
    print("Testing with sample messages:")
    for i, message in enumerate(test_messages, 1):
        result = detector.predict_message_type(message)
        
        print(f"\n{i}. Message: \"{message}\"")
        
        if result['is_spam']:
            spam_type_clean = result['spam_type'].replace('_', ' ')
            print(f"   🚨 SPAM: {spam_type_clean} (Confidence: {result['confidence']:.1%})")
        else:
            print(f"   ✅ HAM (Confidence: {result['confidence']:.1%})")
    
    # Interface selection
    print("\n" + "="*60)
    print("STEP 6: CHOOSE INTERFACE")
    print("="*60)
    
    print("🔍 You can now analyze SMS messages with detailed results!")
    print("\nAvailable interfaces:")
    print("1. 🖼️ GUI (Recommended)")
    print("   • Real-time analysis")
    print("   • Detailed results and safety advice")
    print("   • Easy to use")
    print()
    print("2. 💻 Terminal Mode")
    print("   • Command-line interface")
    print("   • Debug capabilities")
    print("   • Text-based interaction")
    print()
    
    try:
        choice = input("Enter your choice (1 or 2, or press Enter for GUI): ").strip()
        
        if not choice or choice == "1":
            print("\nStarting GUI...")
            start_gui_mode(detector)
        elif choice == "2":
            print("\nStarting Terminal Mode...")
            detector.interactive_detection()
        else:
            print("❌ Invalid choice. Starting GUI by default...")
            start_gui_mode(detector)
            
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Exiting...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    print("\n" + "="*60)
    print("Thank you for using SMS Spam Detection System!")
    print("="*60)

def start_gui_mode(detector):
    """Start the GUI interface"""
    try:
        print("🚀 Starting SMS Spam Detection GUI")
        print("\n📝 Instructions:")
        print("   • Type or paste any SMS message in the input box")
        print("   • Click '🔍 Detect' or press Enter")
        print("   • Close the GUI window when done")
        print()
        
        gui = SMSDetectorTkinterGUI(detector)
        gui.show()
        
    except Exception as e:
        print(f"❌ GUI Error: {e}")
        print("Falling back to terminal mode...")
        detector.interactive_detection()

def run_quick_test():
    """Quick test function for development"""
    print("="*40)
    print("    QUICK TEST MODE")
    print("="*40)
    
    # Quick setup for testing
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    csv_file = os.path.join(desktop_path, "balanced_spam_data 2.0.csv")
    
    if not os.path.exists(csv_file):
        print("❌ Test data not found")
        return
    
    detector = MultiClassSMSDetector(csv_file)
    
    print("Loading data for quick test...")
    if detector.load_data():
        detector.data = detector.data.dropna(subset=['messages', 'label'])
        print(f"✅ Loaded {len(detector.data)} records")
        
        # Quick feature prep
        print("Quick feature engineering...")
        if detector.prepare_features('messages', 'multiclass_label'):
            print("Training model...")
            detector.train_multiclass_model()
            
            # Test a message
            test_msg = "URGENT! You WON £1000! Call NOW!"
            result = detector.predict_message_type(test_msg)
            
            print(f"\n📱 Test: '{test_msg}'")
            if result['is_spam']:
                print(f"🚨 Result: {result['spam_type']} ({result['confidence']:.1%})")
            else:
                print(f"✅ Result: Ham ({result['confidence']:.1%})")
            
            print("\n✅ Quick test completed!")
    else:
        print("❌ Quick test failed")

def show_system_info():
    """Show system information"""
    print("="*60)
    print("    SMS SPAM DETECTION SYSTEM INFO")
    print("="*60)
    print("📁 Current Files:")
    current_dir = os.getcwd()
    python_files = [f for f in os.listdir(current_dir) if f.endswith('.py')]
    for f in python_files:
        print(f"  • {f}")
    print()
    print("📊 Features:")
    print("  • Multi-class spam detection (14 spam types + ham)")
    print("  • 25 TF-IDF text features")
    print("  • Random Forest classifier with balanced sampling")
    print("  • Real-time prediction with confidence scores")
    print()
    print("🎯 Optimizations:")
    print("  • Designed for small datasets (~1750 records)")
    print("  • Prevents overfitting with careful parameter tuning")
    print("  • Enhanced spam pattern detection")
    print()
    print("🖥️  Interface Options:")
    print("  • tkinter GUI")
    print("  • Terminal mode with debug capabilities")
    print("  • Both support real-time analysis")
    print()
    print("📁 File Structure:")
    print("  • sms_detector.py - Core detection logic")
    print("  • sms_gui.py - GUI interface")
    print("  • main.py - Driver program")
    print("="*60)

if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            run_quick_test()
        elif sys.argv[1] == "--info":
            show_system_info()
        elif sys.argv[1] == "--help":
            print("SMS Spam Detection System")
            print("Usage:")
            print("  python main.py          - Run full system")
            print("  python main.py --test   - Quick test mode")
            print("  python main.py --info   - Show system info")
            print("  python main.py --help   - Show this help")
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        # Run main program
        main()