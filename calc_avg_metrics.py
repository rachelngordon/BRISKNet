import pandas as pd
from pathlib import Path

# Path to your CSV
csv_path = Path("output/ei_warp_large_Lkernel/inference_20260113_142855/metrics.csv")

# Load CSV
df = pd.read_csv(csv_path)

# Drop the 'sample' column (non-numeric identifier)
df_numeric = df.drop(columns=["sample"], errors="ignore")

# ----------------------------
# 1. Overall average per metric
# ----------------------------
overall_means = df_numeric.mean(numeric_only=True)
print("=== Overall mean per metric ===")
print(overall_means)
print()

# -----------------------------------
# 2. Average metrics per category
#    (based on column name prefixes)
# -----------------------------------
def get_category(col_name):
    """
    Extract category as the prefix before the first underscore.
    e.g. 'dl_ssim' -> 'dl'
         'grasp_psnr' -> 'grasp'
         'raw_dc_mse' -> 'raw'
    """
    return col_name.split("_")[0]

# Build mapping: category -> list of columns
categories = {}
for col in df_numeric.columns:
    cat = get_category(col)
    categories.setdefault(cat, []).append(col)

# Compute mean per category
category_means = {}
for cat, cols in categories.items():
    category_means[cat] = df_numeric[cols].mean(numeric_only=True)

# Convert to DataFrame for nicer display
category_means_df = pd.DataFrame(category_means)

print("=== Mean metrics per category ===")
print(category_means_df)
print()

# -----------------------------------
# 3. (Optional) Save results to CSV
# -----------------------------------
overall_means.to_csv("overall_metric_means.csv", header=["mean"])
category_means_df.to_csv("category_metric_means.csv")

print("Saved overall_metric_means.csv and category_metric_means.csv")
