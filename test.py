import asyncio
import sys
from pathlib import Path

# add parent directory
sys.path.append(str(Path(__file__).parent))

from test_accuracy import AccuracyTester


async def run_quick_test():
    """run quick accuracy test"""

    print("\n" + "="*80)
    print("QUICK ACCURACY TEST")
    print("="*80 + "\n")

    # setup
    bureau_pdf = "test_data/JEET_ARORA_PARK251217CR671901414.pdf"
    gst_pdf = "test_data/GSTR3B_06AAICK4577H1Z8_012025.pdf"
    num_runs = 2

    print(f"Configuration:")
    print(f"  Bureau PDF: {bureau_pdf}")
    print(f"  GST PDF: {gst_pdf}")
    print(f"  Number of runs: {num_runs}")
    print(f"  API URL: http://localhost:8000\n")

    # check if files exist
    if not Path(bureau_pdf).exists():
        print(f"❌ ERROR: Bureau PDF not found: {bureau_pdf}")
        print(f"   Please update the path or create test_data/ folder\n")
        return

    if not Path(gst_pdf).exists():
        print(f"❌ ERROR: GST PDF not found: {gst_pdf}")
        print(f"   Please update the path or create test_data/ folder\n")
        return

    # create tester
    print("Creating tester...\n")
    tester = AccuracyTester(api_url="http://localhost:8000")

    # run extractions
    print(f"Running {num_runs} extractions...")
    print("(This may take a few minutes)\n")

    try:
        await tester.run_multiple_extractions(
            bureau_path=bureau_pdf,
            gst_path=gst_pdf,
            num_runs=num_runs
        )

        # generate report
        output_file = "quick_test_report.json"
        tester.generate_report(
            output_path=output_file,
            bureau_filename=Path(bureau_pdf).name,
            gst_filename=Path(gst_pdf).name
        )

        print("\n" + "="*80)
        print("✅ TESTING COMPLETE!")
        print("="*80 + "\n")

        print(f"📊 Report saved: {output_file}")
        print(f"\nNext steps:")
        print(f"  1. Review the report JSON file")
        print(f"  2. Check consistency and accuracy metrics")
        print(f"  3. Run full test with --runs 100 for production\n")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}\n")
        print("Troubleshooting:")
        print("  1. Is API running? Check: http://localhost:8000/docs")
        print("  2. Are PDF paths correct?")
        print("  3. Is ground_truth.py configured?\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("QUICK START: Accuracy Testing")
    print("="*80)
    print("\nThis will:")
    print("  1. Run 10 quick extractions")
    print("  2. Measure consistency")
    print("  3. Compare against ground truth")
    print("  4. Generate test report\n")

    print("Prerequisites:")
    print("  ✓ API must be running (python main.py)")
    print("  ✓ Test PDFs must exist")
    print("  ✓ ground_truth.py configured\n")

    input("Press Enter to start...")

    asyncio.run(run_quick_test())
