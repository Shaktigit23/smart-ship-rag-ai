import pandas as pd


def load_and_prepare_data():

    # ============================================
    # STEP 1 - LOAD CSV
    # ============================================

    csv_path = r"D:\Data Science\Smart_Ship_Project\Vessel_Project\Data\engine_failure_dataset.csv"

    df = pd.read_csv(csv_path)

    print("CSV Loaded Successfully")

    # ============================================
    # STEP 2 - ADD NEW COLUMNS
    # ============================================

    df["Vessel_ID"] = ""
    df["Vessel_Type"] = ""
    df["Engine_Model"] = ""
    df["Expected_Temp"] = 120
    df["Defect_ID"] = ""
    df["Root_Cause"] = ""
    df["Troubleshooting_Steps"] = ""
    df["Severity"] = ""
    df["Remediation_Actions"] = ""
    
    # ============================================
    # STEP 3 - ADD VESSEL CONTEXT
    # ============================================

    # Vessel IDs

    df.loc[0:300, "Vessel_ID"] = "VSL-101"
    df.loc[301:600, "Vessel_ID"] = "VSL-102"
    df.loc[601:999, "Vessel_ID"] = "VSL-103"

    # Vessel Types

    df.loc[0:300, "Vessel_Type"] = "Cargo"

    df.loc[301:600, "Vessel_Type"] = "Oil Tanker"

    df.loc[601:999, "Vessel_Type"] = "Container Ship"

    # Engine Models

    df.loc[0:300, "Engine_Model"] = "Wartsila-X12"

    df.loc[301:600, "Engine_Model"] = "MAN-B&W 6S50"

    df.loc[601:999, "Engine_Model"] = "Sulzer RT-flex96C"

# STEP 4 - INTELLIGENT RCA RULES
# RULE 1 - TEMPERATURE BASED ANOMALY DETECTION
# ============================================

# Normal LEVEL
    warning_temp_mask = (
        (df["Temperature (°C)"] > 50) &
        (df["Temperature (°C)"] <= 80)
    )

    df.loc[
        warning_temp_mask,
        "Root_Cause"
    ] = "Cooling efficiency is good."

    df.loc[
        warning_temp_mask,
        "Severity"
    ] = "Normal"

    df.loc[
        warning_temp_mask,
        "Defect_ID"
    ] = "DEF-TEMP-NORM-001"

    df.loc[
        warning_temp_mask,
        "Troubleshooting_Steps"
    ] = (
        "Coolant level is normal. "
    )

    df.loc[
        warning_temp_mask,
        "Remediation_Actions"
    ] = (
        "No action taken. Continue regular monitoring."
        
    )
    # ============================================
    # Warning LEVEL
    
    warning_temp_mask = (
        (df["Temperature (°C)"] > 80) &
        (df["Temperature (°C)"] <= 100)
    )

    df.loc[
        warning_temp_mask,
        "Root_Cause"
    ] = "Cooling efficiency degradation"

    df.loc[
        warning_temp_mask,
        "Severity"
    ] = "Warning"

    df.loc[
        warning_temp_mask,
        "Defect_ID"
    ] = "DEF-TEMP-WARN-001"

    df.loc[
        warning_temp_mask,
        "Troubleshooting_Steps"
    ] = (
        "Inspect coolant level and radiator airflow"
    )

    df.loc[
        warning_temp_mask,
        "Remediation_Actions"
    ] = (
        "Monitor engine temperature continuously"
    )

    # ============================================
    # HIGH LEVEL
    # ============================================

    high_temp_mask = (
        (df["Temperature (°C)"] > 100) &
        (df["Temperature (°C)"] <= 120)
    )

    df.loc[
        high_temp_mask,
        "Root_Cause"
    ] = "Possible thermostat malfunction"

    df.loc[
        high_temp_mask,
        "Severity"
    ] = "High"

    df.loc[
        high_temp_mask,
        "Defect_ID"
    ] = "DEF-TEMP-HIGH-002"

    df.loc[
        high_temp_mask,
        "Troubleshooting_Steps"
    ] = (
        "Inspect thermostat and cooling pump"
    )

    df.loc[
        high_temp_mask,
        "Remediation_Actions"
    ] = (
        "Reduce engine load and inspect cooling system"
    )

    # ============================================
    # CRITICAL LEVEL
    # ============================================

    critical_temp_mask = (
        df["Temperature (°C)"] > 120
    )

    df.loc[
        critical_temp_mask,
        "Root_Cause"
    ] = (
        "Critical cooling system or thermostat failure"
    )

    df.loc[
        critical_temp_mask,
        "Severity"
    ] = "Critical"

    df.loc[
        critical_temp_mask,
        "Defect_ID"
    ] = "DEF-TEMP-CRITICAL-003"

    df.loc[
        critical_temp_mask,
        "Troubleshooting_Steps"
    ] = (
        "Inspect full cooling loop, radiator, thermostat, and pump"
    )

    df.loc[
        critical_temp_mask,
        "Remediation_Actions"
    ] = (
        "Immediate engine shutdown and emergency maintenance"
    )

    # ============================================
    # STEP 5 - NORMAL CONDITIONS
    # ============================================

    df.loc[
        df["Root_Cause"] == "",
        "Root_Cause"
    ] = "Normal operation"

    df.loc[
        df["Severity"] == "",
        "Severity"
    ] = "Normal"

    df.loc[
        df["Defect_ID"] == "",
        "Defect_ID"
    ] = "NONE"

    df.loc[
        df["Troubleshooting_Steps"] == "",
        "Troubleshooting_Steps"
    ] = "No troubleshooting required"

    df.loc[
        df["Remediation_Actions"] == "",
        "Remediation_Actions"
    ] = "No remediation required"

    # ============================================
    # STEP 6 - FILTER ONLY ISSUES
    # ============================================

    issue_df = df[
        (df["Severity"] == "Warning") |
        (df["Severity"] == "High") |
        (df["Severity"] == "Critical")
    ]

    # ============================================
    # STEP 7 - CREATE ISSUE TABLE
    # ============================================

    issue_table = issue_df[
        [
            "Time_Stamp",
            "Vessel_ID",
            "Vessel_Type",
            "Engine_Model",
            "Temperature (°C)",
            "Expected_Temp",
            "RPM",
            "Fault_Condition",
            "Defect_ID",
            "Root_Cause",
            "Troubleshooting_Steps",
            "Severity",
            "Remediation_Actions"
        ]
    ]

    print("\n========== ISSUE TABLE ==========\n")

    print(issue_table.head(20))

    # ============================================
    # STEP 8 - SAVE CLEAN DATASET
    # ============================================

    df.to_csv(
        "modified_ship_data.csv",
        index=False
    )

    print(f"\nModified issue dataset saved successfully")

    print(f"\nTotal Issue Records: {len(issue_df)}")

    # ============================================
    # STEP 9 - RETURN ISSUE DATAFRAME
    # ============================================

    return issue_df

# ============================================
# RUN FILE DIRECTLY
# ============================================

if __name__ == "__main__":

    load_and_prepare_data()
    