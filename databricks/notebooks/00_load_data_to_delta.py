# Databricks notebook source

# COMMAND ----------

# MAGIC %md # Load CSV Data into Delta Tables
# MAGIC
# MAGIC Reads CSV files from the UC Volume and creates Delta tables in `workspace.dataccion`.

# COMMAND ----------

VOLUME_PATH = "/Volumes/workspace/dataccion/raw_data"
CATALOG = "workspace"
SCHEMA = "dataccion"

def clean_columns(df):
    """Rename columns to remove special characters for Delta compatibility."""
    import re
    for col in df.columns:
        new_col = col.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
        new_col = re.sub(r'[^a-zA-Z0-9_]', '_', new_col)
        new_col = re.sub(r'_+', '_', new_col).strip('_').lower()
        if new_col != col:
            df = df.withColumnRenamed(col, new_col)
    return df

# COMMAND ----------

# MAGIC %md ## 1. Employment Status by Gender, Skill Level & Sector (ILO)

# COMMAND ----------

df = spark.read.csv(f"{VOLUME_PATH}/EMP_STAT_SEX_SKL_ECO_NB_A-remodelado-2026-04-05.csv", header=True, inferSchema=True)
df = clean_columns(df)
df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.emp_stat_sex_skill_eco")
print(f"emp_stat_sex_skill_eco: {df.count()} rows, columns: {df.columns}")

# COMMAND ----------

# MAGIC %md ## 2. Employment Type by Gender & Sector (ILO)

# COMMAND ----------

df = spark.read.csv(f"{VOLUME_PATH}/EMP_TEMP_SEX_ECO_NB_A-remodelado-2026-04-04.csv", header=True, inferSchema=True)
df = clean_columns(df)
df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.emp_temp_sex_eco")
print(f"emp_temp_sex_eco: {df.count()} rows, columns: {df.columns}")

# COMMAND ----------

# MAGIC %md ## 3. Hours Worked by Gender, Household Type & Children

# COMMAND ----------

df = spark.read.csv(f"{VOLUME_PATH}/GED_PHOW_SEX_HHT_CHL_NB_A-remodelado-2026-04-05.csv", header=True, inferSchema=True)
df = clean_columns(df)
df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.hours_worked_family_type")
print(f"hours_worked_family_type: {df.count()} rows, columns: {df.columns}")

# COMMAND ----------

# MAGIC %md ## 4. Hours Worked by Gender & Sector (ILO)

# COMMAND ----------

df = spark.read.csv(f"{VOLUME_PATH}/HOW_TEMP_SEX_ECO_NB_A-remodelado-2026-04-05.csv", header=True, inferSchema=True)
df = clean_columns(df)
df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.hours_worked_sex_eco")
print(f"hours_worked_sex_eco: {df.count()} rows, columns: {df.columns}")

# COMMAND ----------

# MAGIC %md ## 5. World Bank — Women in LatAm (Long Format)

# COMMAND ----------

from pyspark.sql.functions import col
df = spark.read.csv(f"{VOLUME_PATH}/wb_women_latam_long.csv", header=True, inferSchema=False)
df = df.withColumn("year", col("year").cast("int")).withColumn("value", col("value").cast("double"))
df = clean_columns(df)
df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.wb_women_latam_long")
print(f"wb_women_latam_long: {df.count()} rows, columns: {df.columns}")

# COMMAND ----------

# MAGIC %md ## 6. World Bank — Women in LatAm (Wide Format)

# COMMAND ----------

df = spark.read.csv(f"{VOLUME_PATH}/wb_women_latam_wide.csv", header=True, inferSchema=False)
df = clean_columns(df)
df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.wb_women_latam_wide")
print(f"wb_women_latam_wide: {df.count()} rows, columns: {df.columns}")

# COMMAND ----------

# MAGIC %md ## 7. World Bank — Correlation Matrix

# COMMAND ----------

df = spark.read.csv(f"{VOLUME_PATH}/world_bank_corrmatrix.csv", header=True, inferSchema=False)
df = clean_columns(df)
df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.wb_correlation_matrix")
print(f"wb_correlation_matrix: {df.count()} rows, columns: {df.columns}")

# COMMAND ----------

# MAGIC %md ## Summary

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES IN workspace.dataccion
