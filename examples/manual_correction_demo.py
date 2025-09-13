"""
Manual Correction Interface Demo
===============================

This file demonstrates the manual correction interface for OMR results.
"""

import sys
from pathlib import Path
import json
import time

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.ui.correction_interface import OMRCorrectionUI
from src.omr_pipeline import OMRPipeline

try:
    import cv2
    import numpy as np
    import streamlit as st
except ImportError:
    print("Warning: Required packages not available. This is a demonstration script.")
    cv2 = None
    np = None
    st = None


def create_sample_omr_results():
    """
    Create sample OMR results for demonstration purposes.
    """
    if cv2 is None or np is None:
        print("Cannot create sample results - OpenCV/NumPy not available")
        return None
    
    # Create a sample image
    image = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Add staff lines
    staff_y_positions = [150, 170, 190, 210, 230]
    for y in staff_y_positions:
        cv2.line(image, (50, y), (750, y), (0, 0, 0), 2)
    
    # Add some musical elements
    cv2.putText(image, "♪", (70, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    # Add note heads
    note_positions = [(120, 140), (180, 160), (240, 180), (300, 200), (360, 160)]
    for x, y in note_positions:
        cv2.circle(image, (x, y), 8, (0, 0, 0), -1)
        cv2.line(image, (x + 8, y), (x + 8, y - 40), (0, 0, 0), 2)
    
    # Process with OMR pipeline
    pipeline = OMRPipeline()
    results = pipeline.process_image(image, "demo_sheet_music")
    
    return results, image


def demo_1_basic_correction_interface():
    """
    Demo 1: Basic correction interface usage.
    """
    print("Demo 1: Basic Manual Correction Interface")
    print("=" * 50)
    
    # Create sample data
    results, image = create_sample_omr_results()
    
    if results is None:
        print("❌ Cannot create sample results - dependencies not available")
        return
    
    if not results.success:
        print("❌ OMR processing failed - cannot demonstrate correction interface")
        return
    
    # Save sample results for the interface
    output_dir = Path(__file__).parent / "output" / "correction_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the processed image
    if cv2 is not None:
        cv2.imwrite(str(output_dir / "sample_image.png"), image)
    
    # Save the OMR results
    pipeline = OMRPipeline()
    pipeline.save_json_report(results.json_report, str(output_dir / "sample_results.json"))
    pipeline.save_musicxml(results.musicxml_content, str(output_dir / "sample_output.mxl"))
    
    print(f"✅ Sample OMR results created:")
    print(f"   📁 Output directory: {output_dir}")
    print(f"   🖼️  Sample image: sample_image.png")
    print(f"   📄 JSON results: sample_results.json")
    print(f"   🎵 MusicXML output: sample_output.mxl")
    
    # Instructions for using the correction interface
    print(f"\n🛠️  To launch the manual correction interface:")
    print(f"   1. Navigate to the project root directory")
    print(f"   2. Run: streamlit run src/ui/correction_interface.py")
    print(f"   3. Open your browser to http://localhost:8501")
    print(f"   4. Upload the sample image and JSON results for correction")
    
    # Show what the interface can do
    print(f"\n📋 Manual Correction Interface Features:")
    print(f"   • 📊 Interactive visualization of detected symbols")
    print(f"   • 🖱️  Click-to-edit symbol properties")
    print(f"   • 🔍 Confidence-based filtering")
    print(f"   • ✏️  Add, delete, or modify musical symbols")
    print(f"   • 💾 Export corrected results as MusicXML or JSON")
    print(f"   • 📈 Real-time confidence analysis")


def demo_2_programmatic_interface_usage():
    """
    Demo 2: Programmatic usage of the correction interface components.
    """
    print("\nDemo 2: Programmatic Interface Usage")
    print("=" * 50)
    
    # This demonstrates how to use the correction interface components programmatically
    
    try:
        # Initialize the correction interface
        correction_ui = OMRCorrectionUI()
        
        print("✅ OMR Correction Interface initialized")
        print(f"   📝 Available methods:")
        
        # List the key methods available
        methods = [
            'load_omr_results', 'display_image_with_symbols', 
            'edit_symbol_properties', 'add_new_symbol', 
            'delete_symbol', 'export_corrected_results'
        ]
        
        for method in methods:
            if hasattr(correction_ui, method):
                print(f"   • {method}()")
        
        # Create sample correction workflow
        print(f"\n🔄 Sample Correction Workflow:")
        print(f"   1. Load OMR results from JSON file")
        print(f"   2. Display image with detected symbols overlaid")
        print(f"   3. Filter symbols by confidence level")
        print(f"   4. Edit low-confidence symbols")
        print(f"   5. Add missing symbols")
        print(f"   6. Delete false positive symbols")
        print(f"   7. Export corrected results")
        
    except Exception as e:
        print(f"⚠️  Could not initialize correction interface: {e}")
        print(f"   This is likely due to missing Streamlit dependency")


def demo_3_correction_workflow_simulation():
    """
    Demo 3: Simulate a typical correction workflow.
    """
    print("\nDemo 3: Correction Workflow Simulation")
    print("=" * 50)
    
    # Create sample results
    results, image = create_sample_omr_results()
    
    if results is None:
        print("❌ Cannot create sample results")
        return
    
    # Simulate correction workflow
    print("🔍 Analyzing OMR results for correction needs...")
    
    if results.confidence_scores:
        confidence_scores = results.confidence_scores
        
        print(f"📊 Confidence Analysis:")
        print(f"   Overall confidence: {confidence_scores.get('overall_confidence', 0):.3f}")
        print(f"   High confidence symbols: {confidence_scores.get('high_confidence_count', 0)}")
        print(f"   Medium confidence symbols: {confidence_scores.get('medium_confidence_count', 0)}")
        print(f"   Low confidence symbols: {confidence_scores.get('low_confidence_count', 0)}")
        
        # Identify symbols that need correction
        low_confidence_count = confidence_scores.get('low_confidence_count', 0)
        total_symbols = confidence_scores.get('total_symbols', 0)
        
        if low_confidence_count > 0:
            print(f"\n⚠️  Correction Recommendations:")
            print(f"   • {low_confidence_count} symbols have low confidence")
            print(f"   • Review symbols with confidence < 0.4")
            print(f"   • Check for false positives and missing symbols")
            
            # Simulate correction actions
            correction_actions = [
                f"✏️  Edit note head classification (confidence: 0.25)",
                f"➕ Add missing rest symbol",
                f"🗑️  Delete false positive dot",
                f"🔧 Adjust note position",
                f"✅ Confirm high-confidence symbols"
            ]
            
            print(f"\n🛠️  Simulated Correction Actions:")
            for action in correction_actions:
                print(f"   {action}")
                time.sleep(0.5)  # Simulate user interaction time
        else:
            print(f"\n✅ High quality results - minimal correction needed")
    
    # Show quality improvement
    print(f"\n📈 Expected Quality Improvement:")
    original_quality = results.quality_assessment.get('overall_score', 0)
    estimated_improved_quality = min(original_quality + 0.15, 1.0)  # Simulate improvement
    
    print(f"   Original quality: {original_quality:.3f}")
    print(f"   After correction: {estimated_improved_quality:.3f} (estimated)")
    print(f"   Improvement: +{estimated_improved_quality - original_quality:.3f}")


def demo_4_integration_with_omr_pipeline():
    """
    Demo 4: Integration between OMR pipeline and correction interface.
    """
    print("\nDemo 4: Pipeline Integration")
    print("=" * 50)
    
    print("🔗 OMR Pipeline → Manual Correction → Final Output")
    print()
    
    # Step 1: OMR Processing
    print("1️⃣  OMR Processing Stage:")
    results, image = create_sample_omr_results()
    
    if results and results.success:
        print(f"   ✅ Automatic processing completed")
        print(f"   📊 Confidence: {results.confidence_scores.get('overall_confidence', 0):.3f}")
        print(f"   ⭐ Quality: {results.quality_assessment.get('overall_score', 0):.3f}")
    else:
        print(f"   ❌ Automatic processing failed")
        return
    
    # Step 2: Quality Assessment
    print(f"\n2️⃣  Quality Assessment:")
    quality_score = results.quality_assessment.get('overall_score', 0)
    
    if quality_score < 0.7:
        print(f"   ⚠️  Quality below threshold ({quality_score:.3f} < 0.70)")
        print(f"   🛠️  Manual correction recommended")
        correction_needed = True
    else:
        print(f"   ✅ Quality acceptable ({quality_score:.3f} ≥ 0.70)")
        print(f"   📝 Optional manual review")
        correction_needed = False
    
    # Step 3: Manual Correction (Simulated)
    print(f"\n3️⃣  Manual Correction Stage:")
    
    if correction_needed:
        print(f"   🖥️  Launch correction interface")
        print(f"   👤 User reviews and corrects symbols")
        print(f"   💾 Save corrected results")
        
        # Simulate corrected results
        corrected_quality = min(quality_score + 0.2, 1.0)
        print(f"   📈 Quality improved: {quality_score:.3f} → {corrected_quality:.3f}")
    else:
        print(f"   ⏭️  Skipping manual correction (quality sufficient)")
        corrected_quality = quality_score
    
    # Step 4: Final Output
    print(f"\n4️⃣  Final Output:")
    print(f"   🎵 Generate final MusicXML")
    print(f"   📄 Create quality report")
    print(f"   📊 Final quality score: {corrected_quality:.3f}")
    
    # Save demonstration files
    output_dir = Path(__file__).parent / "output" / "integration_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create integration report
    integration_report = {
        "workflow_stages": {
            "1_omr_processing": {
                "status": "completed",
                "confidence": results.confidence_scores.get('overall_confidence', 0),
                "initial_quality": quality_score
            },
            "2_quality_assessment": {
                "threshold": 0.7,
                "correction_needed": correction_needed
            },
            "3_manual_correction": {
                "performed": correction_needed,
                "quality_improvement": corrected_quality - quality_score if correction_needed else 0
            },
            "4_final_output": {
                "final_quality": corrected_quality,
                "ready_for_use": corrected_quality >= 0.8
            }
        },
        "recommendations": [
            "Manual correction significantly improves quality" if correction_needed else "Automatic processing sufficient",
            "Review low-confidence symbols for accuracy",
            "Validate MusicXML output in MuseScore"
        ]
    }
    
    with open(output_dir / "integration_workflow.json", 'w') as f:
        json.dump(integration_report, f, indent=2)
    
    print(f"\n💾 Integration workflow report saved to: {output_dir}")


def main():
    """Run all manual correction demos."""
    print("🛠️  OMR Manual Correction Interface Demos")
    print("=" * 60)
    
    try:
        demo_1_basic_correction_interface()
        demo_2_programmatic_interface_usage()
        demo_3_correction_workflow_simulation()
        demo_4_integration_with_omr_pipeline()
        
        print(f"\n🎉 All manual correction demos completed!")
        
        # Final instructions
        print(f"\n📋 Next Steps:")
        print(f"   1. Install Streamlit: pip install streamlit")
        print(f"   2. Launch interface: streamlit run src/ui/correction_interface.py")
        print(f"   3. Use sample files in examples/output/ for testing")
        print(f"   4. Explore interactive correction features")
        
    except Exception as e:
        print(f"💥 An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()