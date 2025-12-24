import asyncio
import json
import statistics
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from collections import Counter, defaultdict
import argparse

# import modules
import sys
sys.path.append(str(Path(__file__).parent))

from ground_truth import get_ground_truth


class AccuracyTester:
    """test and evaluate extraction accuracy"""

    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
        self.results = []

    async def run_single_extraction(self, bureau_path, gst_path):
        """run single extraction via API"""
        import aiohttp

        try:
            # read files first to avoid closed file error
            with open(bureau_path, 'rb') as f:
                bureau_content = f.read()

            with open(gst_path, 'rb') as f:
                gst_content = f.read()

            async with aiohttp.ClientSession() as session:
                data = aiohttp.FormData()

                # add bureau PDF
                data.add_field('bureau_pdf',
                               bureau_content,
                               filename=Path(bureau_path).name,
                               content_type='application/pdf')

                # add GST PDF
                data.add_field('gst_pdf',
                               gst_content,
                               filename=Path(gst_path).name,
                               content_type='application/pdf')

                async with session.post(f"{self.api_url}/generate-rule", data=data) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        return {"error": f"Status {response.status}: {error_text}"}

        except Exception as e:
            return {"error": str(e)}

    async def run_multiple_extractions(self, bureau_path, gst_path, num_runs=100):
        """run extraction multiple times"""
        print(f"\n{'='*80}")
        print(f"RUNNING {num_runs} EXTRACTIONS")
        print(f"{'='*80}\n")

        print(f"Bureau PDF: {bureau_path}")
        print(f"GST PDF: {gst_path}")
        print(f"Number of runs: {num_runs}\n")

        results = []

        for i in range(num_runs):
            print(f"Run {i+1}/{num_runs}...", end='\r')
            result = await self.run_single_extraction(bureau_path, gst_path)
            results.append({
                "run_number": i + 1,
                "timestamp": datetime.now().isoformat(),
                "result": result
            })

            # small delay to avoid overwhelming API
            await asyncio.sleep(0.1)

        print(f"\nCompleted {num_runs} extractions!\n")
        self.results = results
        return results

    def evaluate_consistency(self):
        """evaluate consistency of values across runs"""
        print(f"\n{'='*80}")
        print("EVALUATING CONSISTENCY")
        print(f"{'='*80}\n")

        # collect values for each parameter
        parameter_values = defaultdict(list)

        for run in self.results:
            if "error" in run["result"]:
                continue

            # bureau parameters
            if "bureau" in run["result"]:
                for param_id, param_data in run["result"]["bureau"].items():
                    # handle both formats
                    if isinstance(param_data, dict):
                        if "value" in param_data and param_data["value"] is not None:
                            parameter_values[param_id].append(param_data["value"])
                    else:
                        if param_data is not None:
                            parameter_values[param_id].append(param_data)

            # GST sales
            if "gst_sales" in run["result"] and run["result"]["gst_sales"]:
                gst_sales_str = json.dumps(run["result"]["gst_sales"], sort_keys=True)
                parameter_values["gst_sales"].append(gst_sales_str)

        # calculate consistency
        consistency_report = {}

        for param_id, values in parameter_values.items():
            total_runs = len(values)
            value_counts = Counter(values)
            most_common = value_counts.most_common(1)[0]
            most_common_value = most_common[0]
            most_common_count = most_common[1]

            consistency_rate = 0
            if total_runs > 0:
                consistency_rate = (most_common_count / total_runs) * 100

            consistency_report[param_id] = {
                "total_extractions": total_runs,
                "unique_values": len(value_counts),
                "most_common_value": most_common_value,
                "most_common_count": most_common_count,
                "consistency_rate": consistency_rate,
                "all_values": dict(value_counts)
            }

            # print info
            print(f"Parameter: {param_id}")
            print(f"  Total extractions: {total_runs}")
            print(f"  Unique values: {len(value_counts)}")
            print(f"  Most common: {most_common_value} ({most_common_count}/{total_runs} = {consistency_rate:.1f}%)")

            if len(value_counts) > 1:
                print(f"  ⚠️  WARNING: Inconsistent values!")
                print(f"  All values: {dict(value_counts)}")
            else:
                print(f"  ✅ 100% consistent")

            print()

        return consistency_report

    def evaluate_accuracy(self, bureau_filename, gst_filename):
        """evaluate accuracy against ground truth"""
        print(f"\n{'='*80}")
        print("EVALUATING ACCURACY")
        print(f"{'='*80}\n")

        # get most common values
        consistency_report = self.evaluate_consistency()

        accuracy_report = {}
        correct_params = []
        incorrect_params = []
        missing_params = []

        # collect ground truth params
        all_ground_truth_params = set()

        from ground_truth import GROUND_TRUTH_BUREAU, GROUND_TRUTH_GST

        if bureau_filename in GROUND_TRUTH_BUREAU:
            for key in GROUND_TRUTH_BUREAU[bureau_filename].keys():
                all_ground_truth_params.add(key)

        if gst_filename in GROUND_TRUTH_GST:
            for key in GROUND_TRUTH_GST[gst_filename].keys():
                all_ground_truth_params.add(key)

        # check each parameter
        for param_id in all_ground_truth_params:
            ground_truth = get_ground_truth(bureau_filename, param_id)
            if not ground_truth:
                ground_truth = get_ground_truth(gst_filename, param_id)

            if not ground_truth:
                continue

            expected_value = ground_truth["expected_value"]

            # get extracted value
            if param_id in consistency_report:
                extracted_value = consistency_report[param_id]["most_common_value"]

                # parse GST sales JSON
                if param_id == "gst_sales":
                    try:
                        extracted_value = json.loads(extracted_value)
                    except:
                        pass

                consistency_rate = consistency_report[param_id]["consistency_rate"]

                # compare values
                is_correct = False
                if expected_value is None:
                    is_correct = extracted_value is None or extracted_value == "not_found"
                elif isinstance(expected_value, list):
                    expected_json = json.dumps(expected_value, sort_keys=True)
                    extracted_json = json.dumps(extracted_value, sort_keys=True)
                    is_correct = expected_json == extracted_json
                else:
                    is_correct = extracted_value == expected_value

                accuracy_report[param_id] = {
                    "expected": expected_value,
                    "extracted": extracted_value,
                    "correct": is_correct,
                    "consistency_rate": consistency_rate
                }

                if is_correct:
                    correct_params.append(param_id)
                    print(f"✅ {param_id}")
                    print(f"   Expected: {expected_value}")
                    print(f"   Extracted: {extracted_value}")
                    print(f"   Consistency: {consistency_rate:.1f}%")
                else:
                    incorrect_params.append(param_id)
                    print(f"❌ {param_id}")
                    print(f"   Expected: {expected_value}")
                    print(f"   Extracted: {extracted_value}")
                    print(f"   Consistency: {consistency_rate:.1f}%")
            else:
                # parameter not extracted
                if expected_value is None:
                    # correct - not found and expected None
                    correct_params.append(param_id)
                    accuracy_report[param_id] = {
                        "expected": None,
                        "extracted": None,
                        "correct": True,
                        "consistency_rate": 100.0
                    }
                    print(f"✅ {param_id}")
                    print(f"   Expected: None")
                    print(f"   Extracted: None")
                    print(f"   Consistency: 100.0%")
                else:
                    # missing - expected but not found
                    missing_params.append(param_id)
                    accuracy_report[param_id] = {
                        "expected": expected_value,
                        "extracted": None,
                        "correct": False,
                        "consistency_rate": 0
                    }
                    print(f"⚠️  {param_id}")
                    print(f"   Expected: {expected_value}")
                    print(f"   Extracted: NOT FOUND")

            print()

        # calculate overall accuracy
        total_params = len(all_ground_truth_params)
        correct_count = len(correct_params)
        overall_accuracy = 0
        if total_params > 0:
            overall_accuracy = (correct_count / total_params) * 100

        print(f"\n{'='*80}")
        print("ACCURACY SUMMARY")
        print(f"{'='*80}\n")

        print(f"Total parameters: {total_params}")
        print(f"Correct: {correct_count} ({overall_accuracy:.1f}%)")

        incorrect_pct = 0
        if total_params > 0:
            incorrect_pct = (len(incorrect_params) / total_params) * 100
        print(f"Incorrect: {len(incorrect_params)} ({incorrect_pct:.1f}%)")

        missing_pct = 0
        if total_params > 0:
            missing_pct = (len(missing_params) / total_params) * 100
        print(f"Missing: {len(missing_params)} ({missing_pct:.1f}%)")

        print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")

        return {
            "total_parameters": total_params,
            "correct": correct_count,
            "incorrect": len(incorrect_params),
            "missing": len(missing_params),
            "overall_accuracy": overall_accuracy,
            "per_parameter": accuracy_report,
            "correct_params": correct_params,
            "incorrect_params": incorrect_params,
            "missing_params": missing_params
        }

    def generate_report(self, output_path, bureau_filename, gst_filename):
        """generate comprehensive test report"""
        print(f"\n{'='*80}")
        print("GENERATING REPORT")
        print(f"{'='*80}\n")

        # evaluate metrics
        consistency_report = self.evaluate_consistency()
        accuracy_report = self.evaluate_accuracy(bureau_filename, gst_filename)

        # build report
        report = {
            "test_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_runs": len(self.results),
                "bureau_file": bureau_filename,
                "gst_file": gst_filename
            },
            "consistency_metrics": consistency_report,
            "accuracy_metrics": accuracy_report,
            "all_runs": self.results
        }

        # save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"✅ Report saved: {output_path}\n")

        return report


async def main():
    """main testing function"""
    parser = argparse.ArgumentParser(description="Test extraction accuracy")
    parser.add_argument("--runs", type=int, default=100, help="Number of test runs")
    parser.add_argument("--bureau", required=True, help="Path to bureau PDF")
    parser.add_argument("--gst", required=True, help="Path to GST PDF")
    parser.add_argument("--output", default="test_report.json", help="Output report path")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API URL")

    args = parser.parse_args()

    # validate files
    if not Path(args.bureau).exists():
        print(f"❌ Bureau PDF not found: {args.bureau}")
        return

    if not Path(args.gst).exists():
        print(f"❌ GST PDF not found: {args.gst}")
        return

    # create tester
    tester = AccuracyTester(api_url=args.api_url)

    # run extractions
    await tester.run_multiple_extractions(
        bureau_path=args.bureau,
        gst_path=args.gst,
        num_runs=args.runs
    )

    # generate report
    tester.generate_report(
        output_path=args.output,
        bureau_filename=Path(args.bureau).name,
        gst_filename=Path(args.gst).name
    )

    print("\n✅ Testing complete!")
    print(f"📊 Report: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
