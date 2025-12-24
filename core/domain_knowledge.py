
BUREAU_TERMINOLOGY = [
    "Bureau credit score: Numerical representation of creditworthiness ranging from 300 (poor) to 900 (excellent). Higher scores indicate better credit history and lower risk.",
    "DPD (Days Past Due): Number of days a payment is overdue beyond the due date. Common thresholds monitored are 30+, 60+, and 90+ DPD.",
    "30+ DPD: Count of accounts with payments overdue by 30 or more days in the specified monitoring period. Indicates early stage delinquency.",
    "60+ DPD: Count of accounts with payments overdue by 60 or more days in the specified monitoring period. Indicates moderate delinquency.",
    "90+ DPD: Count of accounts with payments overdue by 90 or more days in the specified monitoring period. Indicates serious delinquency.",
    "Settlement: Debt resolved by borrower paying less than the full amount owed, typically after negotiation with creditor. Marked negatively on credit report.",
    "Write-off: Debt declared unrecoverable by lender and removed from active accounts. Severely impacts credit score and indicates non-payment.",
    "NTC (No-Track-Case): Credit applicants with insufficient credit history or no previous credit accounts in bureau database. Also called 'New to Credit'.",
    "Suit Filed: Legal action initiated by creditor for debt recovery through courts. Indicates serious delinquency and unwillingness to pay.",
    "Wilful Default: Deliberate non-payment of debt despite having the financial ability to pay. Considered fraudulent behavior and severely impacts creditworthiness.",
    "Live PL/BL: Active Personal Loan or Business Loan currently being serviced by the borrower with regular payments.",
    "Overdue amount: Total unpaid amount across all accounts that is past the due date. Sum of all overdue balances.",
    "Credit inquiry: Request made by lender to check credit report when applicant applies for credit. Too many inquiries indicate credit hunger.",
    "Active loans: Loans currently being serviced by borrower, not yet closed or settled. Indicates current credit obligations.",
    "Loan exposure: Total outstanding amount across all loans. Also called total debt or credit exposure.",
]

# GST and GSTR-3B Terminology
GST_TERMINOLOGY = [
    "GSTR-3B: Monthly return filing summarizing outward supplies, input tax credit claimed, and net tax liability for the tax period.",
    "Table 3.1(a): Section in GSTR-3B reporting outward taxable supplies (other than zero rated, nil rated and exempted). This is the main sales figure.",
    "Outward supplies: Goods or services provided by the registered GST taxpayer to customers. This is the sales/revenue of the business.",
    "Taxable supplies: Supplies on which GST is levied at applicable rates (5%, 12%, 18%, or 28%). Excludes exempted and nil-rated supplies.",
    "Taxable value: The base value on which GST is calculated, excluding the GST amount itself. This is the pre-tax revenue.",
    "Outward taxable supplies: Sales of goods/services on which GST is applicable. Found in GSTR-3B Table 3.1, row (a).",
    "GSTR-3B structure: Contains multiple tables - Table 3.1 for outward supplies, Table 3.2 for inter-state supplies, Table 4 for input tax credit.",
    "Tax period: The month and year for which the GST return is filed. Format is usually 'Month YYYY' (e.g., January 2025).",
    "GSTIN: GST Identification Number, unique 15-digit alphanumeric code assigned to each registered taxpayer.",
]

# Validation and Business Rules
VALIDATION_RULES = [
    "Valid bureau credit scores: Must be between 300 and 900 inclusive. Scores outside this range are invalid.",
    "Credit score interpretation: 300-579 is Poor, 580-669 is Fair, 670-739 is Good, 740-799 is Very Good, 800-900 is Excellent.",
    "DPD hierarchy rule: 90+ DPD count ≤ 60+ DPD count ≤ 30+ DPD count. If this is violated, data may be incorrect.",
    "GST sales validation: Taxable value should be non-negative numbers. Negative sales indicate data entry error.",
    "Suspicious GST amounts: Values over 10 crore (100,000,000 rupees) should be flagged for verification as potentially incorrect.",
    "Written-off debt amount: Should be non-negative. Negative values indicate error in extraction or data.",
    "Loan counts validation: Max loans and max active loans should be non-negative integers. Cannot have negative loan counts.",
    "Overdue threshold: Maximum allowable overdue amount, typically ranging from 0 to several lakhs. Depends on risk appetite.",
    "Credit inquiry limits: Excessive inquiries (>5 in 6 months) indicate credit hunger and should be flagged.",
    "Zero values interpretation: Zero or null values may indicate either absence of the attribute or that the parameter is not applicable.",
]

# Extraction Hints and Location Guidance
EXTRACTION_HINTS = [
    "Bureau credit score location: Typically appears near terms like 'PERFORM', 'CONSUMER', 'Score', 'CIBIL', or in a dedicated score section on first page.",
    "Credit score format: Usually displayed as a 3-digit number between 300-900, sometimes with a gauge or range indicator.",
    "DPD information location: Often found in payment history tables, delinquency sections, or account performance summary.",
    "Settlement and write-off status: Usually marked explicitly in account status columns with keywords 'Settled', 'Written Off', or status codes.",
    "Live loan indicators: Marked with 'Active', 'Current', 'Live', or similar status in account listings.",
    "GSTR-3B sales extraction: Sales figures are in Table 3.1, row labeled '(a) Outward taxable supplies', second column shows taxable value.",
    "GSTR-3B month extraction: Month information appears as 'Period' followed by month name (January, February, etc.).",
    "GSTR-3B year extraction: Year appears in 'Year' field in format 'YYYY-YY' (e.g., 2024-25) or in filename as MMYYYY (e.g., 012025).",
    "Table structure in PDFs: Tables may span multiple pages. Look for continuation rows and merged cells.",
    "Multiple bureau reports: When processing multiple reports, extract parameters separately for each person/entity.",
    "NTC acceptance: Check for explicit mentions of 'No Track Case', 'NTC', 'New to Credit' status in summary or remarks.",
    "Suit filed indicators: Look for keywords 'Suit Filed', 'Legal Action', 'Court Case' in account remarks or status.",
]

# Common Patterns and Formats
COMMON_PATTERNS = [
    "Date formats in bureau reports: DD-MM-YYYY, DD/MM/YYYY, or MMM-YYYY for month-year format.",
    "Currency representation: Indian Rupees shown as '₹', 'Rs.', 'INR', or just numbers with commas (e.g., 1,50,000).",
    "Percentage formats: Shown with '%' symbol or as decimals (0.15 = 15%).",
    "Boolean values: Yes/No, True/False, Y/N, 1/0, or Present/Absent for presence/absence of attributes.",
    "Account types: PL (Personal Loan), BL (Business Loan), CC (Credit Card), HL (Home Loan), AL (Auto Loan).",
    "Status codes in bureau: STD (Standard), SMA (Special Mention Account), SUB (Sub-standard), DBT (Doubtful), LSS (Loss).",
]

# All knowledge combined for easy iteration
ALL_KNOWLEDGE = (
    BUREAU_TERMINOLOGY +
    GST_TERMINOLOGY +
    VALIDATION_RULES +
    EXTRACTION_HINTS +
    COMMON_PATTERNS
)

# Category mapping for retrieval filtering
KNOWLEDGE_CATEGORIES = {
    "bureau_terminology": BUREAU_TERMINOLOGY,
    "gst_terminology": GST_TERMINOLOGY,
    "validation_rules": VALIDATION_RULES,
    "extraction_hints": EXTRACTION_HINTS,
    "common_patterns": COMMON_PATTERNS,
}