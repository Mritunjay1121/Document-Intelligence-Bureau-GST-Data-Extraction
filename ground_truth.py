
import json
from pathlib import Path

GROUND_TRUTH_BUREAU = {
    "JEET_ARORA_PARK251217CR671901414.pdf": {
        "bureau_credit_score": {
            "expected_value": 627,
            "value_type": "number",
            "notes": "CRIF Report – PERFORM CONSUMER 2.2 Score Section"
        },
        "bureau_ntc_accepted": {
            "expected_value": None,
            "value_type": "boolean",
            "notes": "Not found in this bureau report"
        },
        "bureau_overdue_threshold": {
            "expected_value": 53270046,  # Note: This is 53,270,046 not 5,327,046
            "value_type": "number",
            "notes": "Account Summary - Total Amount Overdue"
        },
        "bureau_dpd_30": {
            "expected_value": None,
            "value_type": "number",
            "notes": "DPD (Days Past Due) buckets not found in this report format"
        },
        "bureau_dpd_60": {
            "expected_value": None,
            "value_type": "number",
            "notes": "DPD (Days Past Due) buckets not found in this report format"
        },
        "bureau_dpd_90": {
            "expected_value": None,
            "value_type": "number",
            "notes": "DPD (Days Past Due) buckets not found in this report format"
        },
        "bureau_settlement_writeoff": {
            "expected_value": "0",
            "value_type": "text",
            "notes": "Account Information Table - Settlement Amt column (all accounts show blank/0)"
        },
        "bureau_no_live_pl_bl": {
            "expected_value": None,
            "value_type": "boolean",
            "notes": "Not found in this bureau report format"
        },
        "bureau_suit_filed": {
            "expected_value": False,
            "value_type": "boolean",
            "notes": "Account Information - Account 8 & 15 Remarks show 'No Suit filed'"
        },
        "bureau_wilful_default": {
            "expected_value": None,
            "value_type": "boolean",
            "notes": "Not found in this bureau report"
        },
        "bureau_written_off_debt_amount": {
            "expected_value": "0",
            "value_type": "text",
            "notes": "Account Information Table - Total Writeoff Amt column shows 0 for all accounts"
        },
        "bureau_max_loans": {
            "expected_value": None,
            "value_type": "number",
            "notes": "Not found - would need to be calculated from active accounts"
        },
        "bureau_loan_amount_threshold": {
            "expected_value": 42300000,  # 4,23,00,000
            "value_type": "number",
            "notes": "Account Information, Account 15 (GECL LOAN SECURED), Collateral/Security Details - Security Value"
        },
        "bureau_credit_inquiries": {
            "expected_value": 13,
            "value_type": "number",
            "notes": "Additional Summary - NUM-GRANTORS"
        },
        "bureau_max_active_loans": {
            "expected_value": 25,
            "value_type": "number",
            "notes": "Account Summary - Active Accounts"
        }
    }
}

# Ground truth for GST reports
# GST sales are returned as an array with month and sales data
GROUND_TRUTH_GST = {
    "GSTR3B_06AAICK4577H1Z8_012025.pdf": {
        "gst_sales": {
            "expected_value": [
                {
                    "month": "January 2025",
                    "sales": 951381
                }
            ],
            "value_type": "array",
            "notes": "GSTR-3B Table 3.1(a) - Outward taxable supplies total taxable value"
        }
    }
}


def get_ground_truth(filename: str, parameter_id: str):
    
    # Check bureau ground truth
    if filename in GROUND_TRUTH_BUREAU:
        if parameter_id in GROUND_TRUTH_BUREAU[filename]:
            return GROUND_TRUTH_BUREAU[filename][parameter_id]
    
    # Check GST ground truth
    if filename in GROUND_TRUTH_GST:
        if parameter_id in GROUND_TRUTH_GST[filename]:
            return GROUND_TRUTH_GST[filename][parameter_id]
    
    return None


def save_ground_truth_template(output_path: str):
 
    template = {
        "YOUR_DOCUMENT.pdf": {
            "parameter_id_1": {
                "expected_value": "FILL_THIS",
                "value_type": "number|boolean|text|array",
                "notes": "Optional: Add notes about this parameter"
            },
            "parameter_id_2": {
                "expected_value": "FILL_THIS",
                "value_type": "number|boolean|text|array"
            }
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"Ground truth template saved to: {output_path}")
    print("Fill in the expected values for your test documents!")


if __name__ == "__main__":
    # Generate template for users
    save_ground_truth_template("ground_truth_template.json")