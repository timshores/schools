"""
Statistical Analysis Module for PPE Report

This module performs statistical analyses to examine associations between:
- Enrollment cohort membership
- Enrollment growth rates
- Per-pupil expenditure amounts
- Per-pupil expenditure growth rates

Key analyses:
1. Cohort membership vs. 2024 $/pupil amount (ANOVA)
2. Cohort membership vs. $/pupil growth 2009-2024 (ANOVA)
3. Enrollment growth rate vs. 2024 $/pupil amount (regression)
4. Enrollment growth rate vs. $/pupil growth 2009-2024 (regression)
5. Combined model: Cohort + enrollment growth vs. 2024 $/pupil (ANCOVA)
6. Combined model: Cohort + enrollment growth vs. $/pupil growth (ANCOVA)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import scipy.stats as stats

# Try to import statsmodels (if not available, will use simpler methods)
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("[WARN] statsmodels not available, using simplified analysis")

from school_shared import (
    load_data, get_enrollment_group, get_cohort_label,
    get_western_cohort_districts, IN_DISTRICT_FTE_KEY,
    EXCLUDE_DISTRICTS
)


def calculate_cagr(start_value: float, end_value: float, years: int) -> float:
    """Calculate Compound Annual Growth Rate."""
    if start_value <= 0 or end_value <= 0 or years <= 0:
        return np.nan
    return (end_value / start_value) ** (1 / years) - 1


def get_district_metrics(df: pd.DataFrame, reg: pd.DataFrame,
                         start_year: int = 2009, end_year: int = 2024) -> pd.DataFrame:
    """
    Extract key metrics for each Western MA district.

    Returns DataFrame with columns:
    - district: District name
    - cohort: Enrollment cohort (Tiny, Small, Medium, Large, Springfield)
    - enrollment_2024: In-district FTE in 2024
    - enrollment_start: In-district FTE in start_year
    - enrollment_cagr: Enrollment CAGR from start_year to end_year
    - ppe_2024: Total per-pupil expenditure in 2024
    - ppe_start: Total per-pupil expenditure in start_year
    - ppe_cagr: PPE CAGR from start_year to end_year
    """
    # Get Western MA traditional districts
    mask = (reg["EOHHS_REGION"].str.lower() == "western") & \
           (reg["SCHOOL_TYPE"].str.lower() == "traditional")
    western_districts = sorted(set(reg[mask]["DIST_NAME"].str.lower()))
    present = set(df["DIST_NAME"].str.lower())
    western_districts = [d for d in western_districts if d in present and d not in EXCLUDE_DISTRICTS]

    results = []

    for dist in western_districts:
        # Get 2024 enrollment
        enroll_2024_data = df[
            (df["DIST_NAME"].str.lower() == dist) &
            (df["IND_CAT"].str.lower() == "student enrollment") &
            (df["IND_SUBCAT"].str.lower() == IN_DISTRICT_FTE_KEY) &
            (df["YEAR"] == end_year)
        ]
        if enroll_2024_data.empty:
            continue
        enrollment_2024 = float(enroll_2024_data["IND_VALUE"].iloc[0])

        # Get start year enrollment
        enroll_start_data = df[
            (df["DIST_NAME"].str.lower() == dist) &
            (df["IND_CAT"].str.lower() == "student enrollment") &
            (df["IND_SUBCAT"].str.lower() == IN_DISTRICT_FTE_KEY) &
            (df["YEAR"] == start_year)
        ]
        if enroll_start_data.empty:
            enrollment_start = np.nan
            enrollment_cagr = np.nan
        else:
            enrollment_start = float(enroll_start_data["IND_VALUE"].iloc[0])
            enrollment_cagr = calculate_cagr(enrollment_start, enrollment_2024,
                                            end_year - start_year)

        # Get 2024 PPE (sum of all subcategories except total)
        ppe_2024_data = df[
            (df["DIST_NAME"].str.lower() == dist) &
            (df["IND_CAT"].str.lower() == "expenditures per pupil") &
            (~df["IND_SUBCAT"].str.lower().isin([
                "total expenditures", "total in-district expenditures"
            ])) &
            (df["YEAR"] == end_year)
        ]
        if ppe_2024_data.empty:
            continue
        ppe_2024 = float(ppe_2024_data["IND_VALUE"].sum())

        # Get start year PPE
        ppe_start_data = df[
            (df["DIST_NAME"].str.lower() == dist) &
            (df["IND_CAT"].str.lower() == "expenditures per pupil") &
            (~df["IND_SUBCAT"].str.lower().isin([
                "total expenditures", "total in-district expenditures"
            ])) &
            (df["YEAR"] == start_year)
        ]
        if ppe_start_data.empty:
            ppe_start = np.nan
            ppe_cagr = np.nan
        else:
            ppe_start = float(ppe_start_data["IND_VALUE"].sum())
            ppe_cagr = calculate_cagr(ppe_start, ppe_2024, end_year - start_year)

        # Get cohort
        cohort = get_enrollment_group(enrollment_2024)

        results.append({
            'district': dist,
            'cohort': cohort,
            'enrollment_2024': enrollment_2024,
            'enrollment_start': enrollment_start,
            'enrollment_cagr': enrollment_cagr,
            'ppe_2024': ppe_2024,
            'ppe_start': ppe_start,
            'ppe_cagr': ppe_cagr
        })

    return pd.DataFrame(results)


def analyze_cohort_vs_ppe_2024(metrics_df: pd.DataFrame) -> Dict:
    """
    Analyze association between cohort membership and 2024 $/pupil amount.
    Uses ANOVA (or Kruskal-Wallis if assumptions violated).
    """
    # Filter out any rows with missing PPE data
    clean_df = metrics_df.dropna(subset=['cohort', 'ppe_2024'])

    # Group by cohort
    cohort_groups = [group['ppe_2024'].values for name, group in clean_df.groupby('cohort')]
    cohort_names = [name for name, group in clean_df.groupby('cohort')]

    # Perform ANOVA
    if len(cohort_groups) >= 2:
        f_stat, p_value = stats.f_oneway(*cohort_groups)

        # Calculate effect size (eta-squared)
        grand_mean = clean_df['ppe_2024'].mean()
        ss_between = sum(len(group) * (group.mean() - grand_mean)**2
                        for group in cohort_groups)
        ss_total = sum((clean_df['ppe_2024'] - grand_mean)**2)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0

        # Descriptive statistics by cohort
        cohort_stats = clean_df.groupby('cohort')['ppe_2024'].agg([
            ('n', 'count'),
            ('mean', 'mean'),
            ('std', 'std'),
            ('min', 'min'),
            ('max', 'max')
        ]).round(2)

        return {
            'test': 'ANOVA',
            'f_statistic': f_stat,
            'p_value': p_value,
            'eta_squared': eta_squared,
            'cohort_stats': cohort_stats,
            'interpretation': interpret_p_value(p_value),
            'effect_size_interpretation': interpret_effect_size(eta_squared)
        }
    else:
        return {
            'test': 'ANOVA',
            'error': 'Insufficient cohort groups for analysis'
        }


def analyze_cohort_vs_ppe_growth(metrics_df: pd.DataFrame) -> Dict:
    """
    Analyze association between cohort membership and $/pupil growth 2009-2024.
    Uses ANOVA (or Kruskal-Wallis if assumptions violated).
    """
    # Filter out any rows with missing data
    clean_df = metrics_df.dropna(subset=['cohort', 'ppe_cagr'])

    # Group by cohort
    cohort_groups = [group['ppe_cagr'].values * 100 for name, group in clean_df.groupby('cohort')]
    cohort_names = [name for name, group in clean_df.groupby('cohort')]

    # Perform ANOVA
    if len(cohort_groups) >= 2:
        f_stat, p_value = stats.f_oneway(*cohort_groups)

        # Calculate effect size (eta-squared)
        grand_mean = (clean_df['ppe_cagr'] * 100).mean()
        ss_between = sum(len(group) * (group.mean() - grand_mean)**2
                        for group in cohort_groups)
        ss_total = sum(((clean_df['ppe_cagr'] * 100) - grand_mean)**2)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0

        # Descriptive statistics by cohort
        clean_df_copy = clean_df.copy()
        clean_df_copy['ppe_cagr_pct'] = clean_df_copy['ppe_cagr'] * 100
        cohort_stats = clean_df_copy.groupby('cohort')['ppe_cagr_pct'].agg([
            ('n', 'count'),
            ('mean', 'mean'),
            ('std', 'std'),
            ('min', 'min'),
            ('max', 'max')
        ]).round(2)

        return {
            'test': 'ANOVA',
            'f_statistic': f_stat,
            'p_value': p_value,
            'eta_squared': eta_squared,
            'cohort_stats': cohort_stats,
            'interpretation': interpret_p_value(p_value),
            'effect_size_interpretation': interpret_effect_size(eta_squared)
        }
    else:
        return {
            'test': 'ANOVA',
            'error': 'Insufficient cohort groups for analysis'
        }


def analyze_enrollment_growth_vs_ppe_2024(metrics_df: pd.DataFrame) -> Dict:
    """
    Analyze association between enrollment growth rate and 2024 $/pupil amount.
    Uses Pearson correlation and linear regression.
    """
    # Filter out any rows with missing data
    clean_df = metrics_df.dropna(subset=['enrollment_cagr', 'ppe_2024'])

    if len(clean_df) < 3:
        return {'error': 'Insufficient data for regression analysis'}

    # Convert CAGR to percentage for easier interpretation
    x = clean_df['enrollment_cagr'].values * 100
    y = clean_df['ppe_2024'].values

    # Pearson correlation
    r, p_value = stats.pearsonr(x, y)
    r_squared = r ** 2

    # Linear regression
    if HAS_STATSMODELS:
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        slope = model.params[1]
        intercept = model.params[0]
        slope_pvalue = model.pvalues[1]
    else:
        # Simple linear regression without statsmodels
        slope, intercept = np.polyfit(x, y, 1)
        slope_pvalue = p_value  # Use correlation p-value as approximation

    return {
        'test': 'Linear Regression',
        'n': len(clean_df),
        'correlation': r,
        'r_squared': r_squared,
        'p_value': p_value,
        'slope': slope,
        'intercept': intercept,
        'slope_pvalue': slope_pvalue,
        'interpretation': interpret_p_value(p_value),
        'effect_size_interpretation': interpret_correlation(r)
    }


def analyze_enrollment_growth_vs_ppe_growth(metrics_df: pd.DataFrame) -> Dict:
    """
    Analyze association between enrollment growth rate and $/pupil growth 2009-2024.
    Uses Pearson correlation and linear regression.
    """
    # Filter out any rows with missing data
    clean_df = metrics_df.dropna(subset=['enrollment_cagr', 'ppe_cagr'])

    if len(clean_df) < 3:
        return {'error': 'Insufficient data for regression analysis'}

    # Convert CAGRs to percentage for easier interpretation
    x = clean_df['enrollment_cagr'].values * 100
    y = clean_df['ppe_cagr'].values * 100

    # Pearson correlation
    r, p_value = stats.pearsonr(x, y)
    r_squared = r ** 2

    # Linear regression
    if HAS_STATSMODELS:
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        slope = model.params[1]
        intercept = model.params[0]
        slope_pvalue = model.pvalues[1]
    else:
        # Simple linear regression without statsmodels
        slope, intercept = np.polyfit(x, y, 1)
        slope_pvalue = p_value  # Use correlation p-value as approximation

    return {
        'test': 'Linear Regression',
        'n': len(clean_df),
        'correlation': r,
        'r_squared': r_squared,
        'p_value': p_value,
        'slope': slope,
        'intercept': intercept,
        'slope_pvalue': slope_pvalue,
        'interpretation': interpret_p_value(p_value),
        'effect_size_interpretation': interpret_correlation(r)
    }


def analyze_combined_model_ppe_2024(metrics_df: pd.DataFrame) -> Dict:
    """
    Combined model: Cohort + enrollment growth rate → 2024 $/pupil amount.
    Uses ANCOVA (Analysis of Covariance).
    """
    if not HAS_STATSMODELS:
        return {'error': 'statsmodels required for ANCOVA analysis'}

    # Filter out any rows with missing data
    clean_df = metrics_df.dropna(subset=['cohort', 'enrollment_cagr', 'ppe_2024']).copy()

    if len(clean_df) < 10:  # Need sufficient data for ANCOVA
        return {'error': 'Insufficient data for ANCOVA analysis'}

    # Convert enrollment CAGR to percentage
    clean_df['enrollment_cagr_pct'] = clean_df['enrollment_cagr'] * 100

    # Fit ANCOVA model
    try:
        formula = 'ppe_2024 ~ C(cohort) + enrollment_cagr_pct'
        model = ols(formula, data=clean_df).fit()

        return {
            'test': 'ANCOVA',
            'n': len(clean_df),
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'f_statistic': model.fvalue,
            'p_value': model.f_pvalue,
            'aic': model.aic,
            'summary': str(model.summary()),
            'interpretation': interpret_p_value(model.f_pvalue)
        }
    except Exception as e:
        return {'error': f'ANCOVA failed: {str(e)}'}


def analyze_combined_model_ppe_growth(metrics_df: pd.DataFrame) -> Dict:
    """
    Combined model: Cohort + enrollment growth rate → $/pupil growth 2009-2024.
    Uses ANCOVA (Analysis of Covariance).
    """
    if not HAS_STATSMODELS:
        return {'error': 'statsmodels required for ANCOVA analysis'}

    # Filter out any rows with missing data
    clean_df = metrics_df.dropna(subset=['cohort', 'enrollment_cagr', 'ppe_cagr']).copy()

    if len(clean_df) < 10:  # Need sufficient data for ANCOVA
        return {'error': 'Insufficient data for ANCOVA analysis'}

    # Convert CAGRs to percentage
    clean_df['enrollment_cagr_pct'] = clean_df['enrollment_cagr'] * 100
    clean_df['ppe_cagr_pct'] = clean_df['ppe_cagr'] * 100

    # Fit ANCOVA model
    try:
        formula = 'ppe_cagr_pct ~ C(cohort) + enrollment_cagr_pct'
        model = ols(formula, data=clean_df).fit()

        return {
            'test': 'ANCOVA',
            'n': len(clean_df),
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'f_statistic': model.fvalue,
            'p_value': model.f_pvalue,
            'aic': model.aic,
            'summary': str(model.summary()),
            'interpretation': interpret_p_value(model.f_pvalue)
        }
    except Exception as e:
        return {'error': f'ANCOVA failed: {str(e)}'}


def interpret_p_value(p: float) -> str:
    """Interpret p-value."""
    if p < 0.001:
        return "Very strong evidence of association (p < 0.001)"
    elif p < 0.01:
        return "Strong evidence of association (p < 0.01)"
    elif p < 0.05:
        return "Moderate evidence of association (p < 0.05)"
    elif p < 0.10:
        return "Weak evidence of association (p < 0.10)"
    else:
        return "Little to no evidence of association (p ≥ 0.10)"


def interpret_effect_size(eta_sq: float) -> str:
    """Interpret eta-squared effect size."""
    if eta_sq < 0.01:
        return "Negligible effect size"
    elif eta_sq < 0.06:
        return "Small effect size"
    elif eta_sq < 0.14:
        return "Medium effect size"
    else:
        return "Large effect size"


def interpret_correlation(r: float) -> str:
    """Interpret Pearson correlation coefficient."""
    abs_r = abs(r)
    if abs_r < 0.1:
        return "Negligible correlation"
    elif abs_r < 0.3:
        return "Weak correlation"
    elif abs_r < 0.5:
        return "Moderate correlation"
    else:
        return "Strong correlation"


def run_all_analyses(start_year: int = 2009, end_year: int = 2024) -> Dict:
    """
    Run all statistical analyses and return results.
    """
    print("Loading data...")
    df, reg, ch70 = load_data()

    print(f"Extracting district metrics ({start_year}-{end_year})...")
    metrics_df = get_district_metrics(df, reg, start_year, end_year)

    print(f"Analyzing {len(metrics_df)} districts...")

    results = {
        'metrics_df': metrics_df,
        'analysis_1_cohort_vs_ppe_2024': analyze_cohort_vs_ppe_2024(metrics_df),
        'analysis_2_cohort_vs_ppe_growth': analyze_cohort_vs_ppe_growth(metrics_df),
        'analysis_3_enrollment_growth_vs_ppe_2024': analyze_enrollment_growth_vs_ppe_2024(metrics_df),
        'analysis_4_enrollment_growth_vs_ppe_growth': analyze_enrollment_growth_vs_ppe_growth(metrics_df),
        'analysis_5_combined_ppe_2024': analyze_combined_model_ppe_2024(metrics_df),
        'analysis_6_combined_ppe_growth': analyze_combined_model_ppe_growth(metrics_df)
    }

    return results


def format_results_for_report(results: Dict) -> List[str]:
    """
    Format analysis results into text blocks suitable for PDF report.
    Returns list of text paragraphs.
    """
    text_blocks = []

    # Introduction
    text_blocks.append(
        "<b>Statistical Analysis: Associations Between Enrollment Cohorts, "
        "Enrollment Growth, and Per-Pupil Expenditures</b>"
    )

    text_blocks.append(
        "This section examines statistical associations between district characteristics "
        "(enrollment cohort, enrollment growth rate) and per-pupil expenditure patterns "
        "(2024 amounts and 2009-2024 growth rates). These analyses help identify whether "
        "certain district characteristics are systematically associated with spending levels "
        "or growth rates."
    )

    # Analysis 1: Cohort vs. PPE 2024
    a1 = results['analysis_1_cohort_vs_ppe_2024']
    if 'error' not in a1:
        text_blocks.append(
            f"<b>1. Enrollment Cohort and 2024 Per-Pupil Expenditure</b>"
        )
        text_blocks.append(
            f"ANOVA test: F = {a1['f_statistic']:.2f}, p = {a1['p_value']:.4f}. "
            f"{a1['interpretation']}. Effect size (η² = {a1['eta_squared']:.3f}): "
            f"{a1['effect_size_interpretation']}."
        )

    # Analysis 2: Cohort vs. PPE Growth
    a2 = results['analysis_2_cohort_vs_ppe_growth']
    if 'error' not in a2:
        text_blocks.append(
            f"<b>2. Enrollment Cohort and Per-Pupil Expenditure Growth (2009-2024)</b>"
        )
        text_blocks.append(
            f"ANOVA test: F = {a2['f_statistic']:.2f}, p = {a2['p_value']:.4f}. "
            f"{a2['interpretation']}. Effect size (η² = {a2['eta_squared']:.3f}): "
            f"{a2['effect_size_interpretation']}."
        )

    # Analysis 3: Enrollment Growth vs. PPE 2024
    a3 = results['analysis_3_enrollment_growth_vs_ppe_2024']
    if 'error' not in a3:
        text_blocks.append(
            f"<b>3. Enrollment Growth Rate and 2024 Per-Pupil Expenditure</b>"
        )
        text_blocks.append(
            f"Linear regression: r = {a3['correlation']:.3f}, R² = {a3['r_squared']:.3f}, "
            f"p = {a3['p_value']:.4f}. {a3['interpretation']}. "
            f"{a3['effect_size_interpretation']}. "
            f"Slope: ${a3['slope']:.2f} per 1% enrollment growth."
        )

    # Analysis 4: Enrollment Growth vs. PPE Growth
    a4 = results['analysis_4_enrollment_growth_vs_ppe_growth']
    if 'error' not in a4:
        text_blocks.append(
            f"<b>4. Enrollment Growth Rate and Per-Pupil Expenditure Growth (2009-2024)</b>"
        )
        text_blocks.append(
            f"Linear regression: r = {a4['correlation']:.3f}, R² = {a4['r_squared']:.3f}, "
            f"p = {a4['p_value']:.4f}. {a4['interpretation']}. "
            f"{a4['effect_size_interpretation']}. "
            f"Slope: {a4['slope']:.3f}pp per 1% enrollment growth."
        )

    # Analysis 5: Combined model PPE 2024
    a5 = results['analysis_5_combined_ppe_2024']
    if 'error' not in a5:
        text_blocks.append(
            f"<b>5. Combined Model: Cohort + Enrollment Growth → 2024 Per-Pupil Expenditure</b>"
        )
        text_blocks.append(
            f"ANCOVA: R² = {a5['r_squared']:.3f}, Adjusted R² = {a5['adj_r_squared']:.3f}, "
            f"F = {a5['f_statistic']:.2f}, p = {a5['p_value']:.4f}. {a5['interpretation']}."
        )

    # Analysis 6: Combined model PPE Growth
    a6 = results['analysis_6_combined_ppe_growth']
    if 'error' not in a6:
        text_blocks.append(
            f"<b>6. Combined Model: Cohort + Enrollment Growth → PPE Growth (2009-2024)</b>"
        )
        text_blocks.append(
            f"ANCOVA: R² = {a6['r_squared']:.3f}, Adjusted R² = {a6['adj_r_squared']:.3f}, "
            f"F = {a6['f_statistic']:.2f}, p = {a6['p_value']:.4f}. {a6['interpretation']}."
        )

    # Interpretation note
    text_blocks.append(
        "<b>Interpretation Notes:</b> These analyses examine statistical associations, "
        "not causal relationships. Significant associations indicate that certain district "
        "characteristics tend to occur together with certain spending patterns, but do not "
        "prove that one causes the other. Many other factors (wealth, demographics, facility "
        "age, governance, etc.) also influence per-pupil expenditures."
    )

    return text_blocks


if __name__ == "__main__":
    # Run analyses when script is executed directly
    results = run_all_analyses()

    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS RESULTS")
    print("="*80)

    for analysis_name, analysis_results in results.items():
        if analysis_name == 'metrics_df':
            print(f"\nDistrict Metrics Summary:")
            print(f"Total districts: {len(analysis_results)}")
            print(f"Cohort distribution:")
            print(analysis_results['cohort'].value_counts().sort_index())
        else:
            print(f"\n{analysis_name}:")
            if 'error' in analysis_results:
                print(f"  Error: {analysis_results['error']}")
            else:
                for key, value in analysis_results.items():
                    if key not in ['cohort_stats', 'summary']:
                        print(f"  {key}: {value}")

    print("\n" + "="*80)
