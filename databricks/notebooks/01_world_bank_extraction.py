# Databricks notebook source

# COMMAND ----------

# MAGIC %md # World Bank Data: Women's Barriers in the Labor Market — Latin America

# COMMAND ----------

# MAGIC %pip install wbgapi
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import wbgapi as wb
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.2f}".format)

# COMMAND ----------

# MAGIC %md ## Countries and Indicators

# COMMAND ----------

LATAM = [
    "ARG", "BOL", "BRA", "CHL", "COL", "CRI", "CUB", "DOM",
    "ECU", "SLV", "GTM", "HND", "MEX", "NIC", "PAN", "PRY",
    "PER", "URY", "VEN",
]

# Years of interest
YEARS = range(2020, 2025)

# Indicators grouped by theme
INDICATORS = {
    # --- Labor force participation ---
    "SL.TLF.CACT.FE.ZS": "Labor force participation rate, female (% of female population 15+)",
    "SL.TLF.CACT.MA.ZS": "Labor force participation rate, male (% of male population 15+)",
    "SL.TLF.CACT.FM.ZS": "Ratio female-to-male labor force participation rate (%)",

    # --- Unemployment ---
    "SL.UEM.TOTL.FE.ZS": "Unemployment rate, female (% of female labor force)",
    "SL.UEM.TOTL.MA.ZS": "Unemployment rate, male (% of male labor force)",

    # --- Employment quality ---
    "SL.EMP.VULN.FE.ZS": "Vulnerable employment, female (% of female employment)",
    "SL.EMP.VULN.MA.ZS": "Vulnerable employment, male (% of male employment)",
    "SL.EMP.SELF.FE.ZS": "Self-employed, female (% of female employment)",
    "SL.EMP.MPYR.FE.ZS": "Employers, female (% of female employment)",
    "SL.EMP.WORK.FE.ZS": "Wage & salaried workers, female (% of female employment)",

    # --- Sectoral employment ---
    "SL.AGR.EMPL.FE.ZS": "Employment in agriculture, female (% of female employment)",
    "SL.IND.EMPL.FE.ZS": "Employment in industry, female (% of female employment)",
    "SL.SRV.EMPL.FE.ZS": "Employment in services, female (% of female employment)",

    # --- Education (access barrier) ---
    "SE.ADT.LITR.FE.ZS": "Literacy rate, adult female (% of females 15+)",
    "SE.SEC.ENRR.FE":    "School enrollment, secondary, female (gross %)",
    "SE.TER.ENRR.FE":    "School enrollment, tertiary, female (gross %)",

    # --- Reproductive & household burden ---
    "SP.DYN.TFRT.IN":    "Fertility rate, total (births per woman)",
    "SH.STA.MMRT":       "Maternal mortality ratio (per 100,000 live births)",

    # --- Legal & institutional ---
    "SG.GEN.PARL.ZS":    "Proportion of seats held by women in national parliaments (%)",
    "IC.FRM.FEMM.ZS":    "Firms with female top manager (% of firms)",

    # --- Agriculture & Rural ---
    "AG.LND.AGRI.ZS":    "Agricultural land (% of land area)",
    # --- Human Capital Index Plus (HCI+) ---
    "HD_HCIP_EDUC_FE": "Human capital index plus (HCI+): education pillar score, female (scale 0-188)",
    "HD_HCIP_EDUC_MA": "Human capital index plus (HCI+): education pillar score, male (scale 0-188)",
    "HD_HCIP_EDUC_TO": "Human capital index plus (HCI+): education pillar score, total (scale 0-188)",
    "HD_HCIP_HLTH_FE": "Human capital index plus (HCI+): health pillar score, female (scale 0-50)",
    "HD_HCIP_HLTH_MA": "Human capital index plus (HCI+): health pillar score, male (scale 0-50)",
    "HD_HCIP_HLTH_TO": "Human capital index plus (HCI+): health pillar score, total (scale 0-50)",
    "HD_HCIP_OTJL_FE": "Human capital index plus (HCI+): on-the-job learning pillar score, female (scale -30-87)",
    "HD_HCIP_OTJL_MA": "Human capital index plus (HCI+): on-the-job learning pillar score, male (scale -30-87)",
    "HD_HCIP_OTJL_TO": "Human capital index plus (HCI+): on-the-job learning pillar score, total (scale -30-87)",
    "HD_HCIP_OVRL_FE": "Human capital index plus (HCI+): overall score, female (scale 0-325)",
    "HD_HCIP_OVRL_MA": "Human capital index plus (HCI+): overall score, male (scale 0-325)",
    "HD_HCIP_OVRL_TO": "Human capital index plus (HCI+): overall score, total (scale 0-325)",

    # --- Firms & Ownership ---
    "IC.FRM.FEMO.ZS":    "Firms with female participation in ownership (% of firms)",

    # --- Internet access ---
    "IT.NET.USER.FE.ZS": "Individuals using the Internet, female (% of female population)",
    "IT.NET.USER.MA.ZS": "Individuals using the Internet, male (% of male population)",

    # --- Literacy ---
    "SE.ADT.LITR.MA.ZS": "Literacy rate, adult male (% of males ages 15 and above)",

    # --- Learning poverty ---
    "SE.LPV.PRIM.FE":    "Learning poverty: Share of Female Children at End-of-Primary below minimum reading proficiency (%)",
    "SE.LPV.PRIM.MA":    "Learning poverty: Share of Male Children at End-of-Primary below minimum reading proficiency (%)",
    "SE.LPV.PRIM.SD.FE": "Female primary school age children out-of-school (%)",
    "SE.LPV.PRIM.SD.MA": "Male primary school age children out-of-school (%)",

    # --- Pre-primary enrollment ---
    "SE.PRE.ENRR":       "School enrollment, preprimary (% gross)",
    "SE.PRE.ENRR.FE":    "School enrollment, preprimary, female (% gross)",

    # --- Primary education ---
    "SE.PRM.CMPT.FE.ZS": "Primary completion rate, female (% of relevant age group)",
    "SE.PRM.CMPT.MA.ZS": "Primary completion rate, male (% of relevant age group)",
    "SE.PRM.CUAT.FE.ZS": "Educational attainment, at least completed primary, population 25+, female (%) (cumulative)",
    "SE.PRM.CUAT.MA.ZS": "Educational attainment, at least completed primary, population 25+, male (%) (cumulative)",
    "SE.PRM.PRSL.FE.ZS": "Persistence to last grade of primary, female (% of cohort)",
    "SE.PRM.PRSL.MA.ZS": "Persistence to last grade of primary, male (% of cohort)",
    "SE.PRM.UNER.FE.ZS": "Children out of school, female (% of female primary school age)",
    "SE.PRM.UNER.MA.ZS": "Children out of school, male (% of male primary school age)",

    # --- Secondary education ---
    "SE.SEC.CUAT.UP.FE.ZS": "Educational attainment, at least completed upper secondary, population 25+, female (%) (cumulative)",
    "SE.SEC.CUAT.UP.MA.ZS": "Educational attainment, at least completed upper secondary, population 25+, male (%) (cumulative)",
    "SE.SEC.DURS":           "Secondary education, duration (years)",
    "SE.SEC.UNER.LO.FE.ZS": "Adolescents out of school, female (% of female lower secondary school age)",
    "SE.SEC.UNER.LO.MA.ZS": "Adolescents out of school, male (% of male lower secondary school age)",

    # --- Tertiary education ---
    "SE.TER.CUAT.BA.FE.ZS": "Educational attainment, at least Bachelor's or equivalent, population 25+, female (%) (cumulative)",
    "SE.TER.CUAT.BA.MA.ZS": "Educational attainment, at least Bachelor's or equivalent, population 25+, male (%) (cumulative)",
    "SE.TER.CUAT.DO.FE.ZS": "Educational attainment, Doctoral or equivalent, population 25+, female (%) (cumulative)",
    "SE.TER.CUAT.DO.MA.ZS": "Educational attainment, Doctoral or equivalent, population 25+, male (%) (cumulative)",
    "SE.TER.CUAT.MS.FE.ZS": "Educational attainment, at least Master's or equivalent, population 25+, female (%) (cumulative)",
    "SE.TER.CUAT.MS.MA.ZS": "Educational attainment, at least Master's or equivalent, population 25+, male (%) (cumulative)",

    # --- Unpaid care work ---
    "SG.TIM.UWRK.FE":    "Proportion of time spent on unpaid domestic and care work, female (% of 24 hour day)",
    "SG.TIM.UWRK.MA":    "Proportion of time spent on unpaid domestic and care work, male (% of 24 hour day)",

    # --- Child employment ---
    "SL.AGR.0714.FE.ZS": "Child employment in agriculture, female (% of female economically active children ages 7-14)",
    "SL.AGR.0714.MA.ZS": "Child employment in agriculture, male (% of male economically active children ages 7-14)",
    "SL.FAM.0714.FE.ZS": "Children in employment, unpaid family workers, female (% of female children in employment, ages 7-14)",
    "SL.FAM.0714.MA.ZS": "Children in employment, unpaid family workers, male (% of male children in employment, ages 7-14)",

    # --- Employment (additional) ---
    "SL.EMP.MPYR.MA.ZS": "Employers, male (% of male employment) (modeled ILO estimate)",
    "SL.EMP.SELF.MA.ZS": "Self-employed, male (% of male employment) (modeled ILO estimate)",
    "SL.EMP.SMGT.FE.ZS": "Female share of employment in senior and middle management (%)",
    "SL.EMP.WORK.MA.ZS": "Wage and salaried workers, male (% of male employment) (modeled ILO estimate)",
    "SL.FAM.WORK.FE.ZS": "Contributing family workers, female (% of female employment) (modeled ILO estimate)",
    "SL.FAM.WORK.MA.ZS": "Contributing family workers, male (% of male employment) (modeled ILO estimate)",
    "SL.IND.EMPL.MA.ZS": "Employment in industry, male (% of male employment) (modeled ILO estimate)",
    "SL.SRV.EMPL.MA.ZS": "Employment in services, male (% of male employment) (modeled ILO estimate)",

    # --- Labor force (additional) ---
    "SL.TLF.ACTI.FE.ZS": "Labor force participation rate, female (% of female population ages 15-64) (modeled ILO estimate)",
    "SL.TLF.ACTI.MA.ZS": "Labor force participation rate, male (% of male population ages 15-64) (modeled ILO estimate)",
    "SL.TLF.ADVN.FE.ZS": "Labor force with advanced education, female (% of female working-age population with advanced education)",
    "SL.TLF.ADVN.MA.ZS": "Labor force with advanced education, male (% of male working-age population with advanced education)",
    "SL.TLF.BASC.FE.ZS": "Labor force with basic education, female (% of female working-age population with basic education)",
    "SL.TLF.BASC.MA.ZS": "Labor force with basic education, male (% of male working-age population with basic education)",
    "SL.TLF.INTM.FE.ZS": "Labor force with intermediate education, female (% of female working-age population with intermediate education)",
    "SL.TLF.INTM.MA.ZS": "Labor force with intermediate education, male (% of male working-age population with intermediate education)",
    "SL.TLF.PART.FE.ZS": "Part time employment, female (% of total female employment)",
    "SL.TLF.PART.MA.ZS": "Part time employment, male (% of total male employment)",

    # --- Household & family ---
    "SP.HOU.FEMA.ZS":    "Female headed households (% of households with a female head)",

    # --- Reproductive health ---
    "SH.FPL.SATM.ZS":    "Demand for family planning satisfied by modern methods (% of married women with demand for family planning)",
    "SP.ADO.TFRT":        "Adolescent fertility rate (births per 1,000 women ages 15-19)",
    "SP.MTR.1519.ZS":     "Teenage mothers (% of women ages 15-19 who have had children or are currently pregnant)",

    # --- Legal & institutional (additional) ---
    "SG.LAW.INDX":        "Women Business and the Law Index Score (scale 1-100)",
}

# COMMAND ----------

# MAGIC %md ## Fetch data from World Bank API

# COMMAND ----------

def fetch_indicators(indicators: dict, countries: list, years) -> pd.DataFrame:
    """Fetch multiple WB indicators and return a tidy long-format DataFrame."""
    frames = []
    for code, label in indicators.items():
        try:
            df = wb.data.DataFrame(
                code,
                economy=countries,
                time=years,
                labels=True,      # adds human-readable country names
                skipBlanks=True,
            )
            # wb returns wide format: rows = countries, columns = years
            df = df.reset_index()
            df = df.melt(
                id_vars=["economy", "Country"],
                var_name="year",
                value_name="value",
            )
            df["indicator_code"] = code
            df["indicator_name"] = label
            frames.append(df)
        except Exception as e:
            print(f"Could not fetch {code}: {e}")

    result = pd.concat(frames, ignore_index=True)
    result["year"] = result["year"].str.replace("YR", "").astype(int)
    result = result.rename(columns={"economy": "country_code", "Country": "country_name"})
    return result[["country_code", "country_name", "year", "indicator_code", "indicator_name", "value"]]


df_raw = fetch_indicators(INDICATORS, LATAM, YEARS)
print(f"Shape: {df_raw.shape}")
df_raw.head(10)

# COMMAND ----------

# MAGIC %md ## Explore coverage

# COMMAND ----------

# Non-null observations per indicator
coverage = (
    df_raw.dropna(subset=["value"])
    .groupby(["indicator_code", "indicator_name"])
    .agg(non_null_obs=("value", "count"), countries=("country_code", "nunique"))
    .reset_index()
    .sort_values("non_null_obs", ascending=False)
)
coverage

# COMMAND ----------

# MAGIC %md ## Wide-format snapshot — latest available year per country

# COMMAND ----------

# For each country + indicator, keep the most recent non-null value
latest = (
    df_raw.dropna(subset=["value"])
    .sort_values("year", ascending=False)
    .groupby(["country_code", "country_name", "indicator_code"], as_index=False)
    .first()
)

# Pivot so each indicator is a column
df_wide = latest.pivot_table(
    index=["country_code", "country_name"],
    columns="indicator_code",
    values="value",
)
df_wide.columns.name = None
df_wide = df_wide.reset_index()
df_wide

# COMMAND ----------

# MAGIC %md ## Save to CSV

# COMMAND ----------

import os

output_dir = "/Volumes/workspace/dataccion/raw_data"

df_raw.to_csv(os.path.join(output_dir, "wb_women_latam_long.csv"), index=False)
df_wide.to_csv(os.path.join(output_dir, "wb_women_latam_wide.csv"), index=False)

print("Saved:")
print(f"  Long format : {df_raw.shape}  → data/wb_women_latam_long.csv")
print(f"  Wide format : {df_wide.shape} → data/wb_women_latam_wide.csv")

# COMMAND ----------

# MAGIC %md ## Visualizations

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = {"female": "#c2527a", "male": "#4a7fb5", "gap": "#e08c3a"}

# Helper: get latest value for a single indicator across all countries
def get_latest(indicator_code: str) -> pd.DataFrame:
    return (
        df_raw[df_raw["indicator_code"] == indicator_code]
        .dropna(subset=["value"])
        .sort_values("year", ascending=False)
        .groupby(["country_code", "country_name"], as_index=False)
        .first()
    )

# COMMAND ----------

# MAGIC %md ### 1. Female vs Male Labor Force Participation — by country (latest year)

# COMMAND ----------

lfp_f = get_latest("SL.TLF.CACT.FE.ZS").set_index("country_name")["value"].rename("female")
lfp_m = get_latest("SL.TLF.CACT.MA.ZS").set_index("country_name")["value"].rename("male")
lfp = pd.concat([lfp_f, lfp_m], axis=1).dropna().sort_values("female")

fig, ax = plt.subplots(figsize=(10, 7))
y = range(len(lfp))
ax.barh([i - 0.2 for i in y], lfp["female"], height=0.4, color=COLORS["female"], label="Female")
ax.barh([i + 0.2 for i in y], lfp["male"],   height=0.4, color=COLORS["male"],   label="Male")
ax.set_yticks(list(y))
ax.set_yticklabels(lfp.index)
ax.set_xlabel("Labor force participation rate (%)")
ax.set_title("Labor Force Participation Rate by Gender\n(latest available year, Latin America)", fontweight="bold")
ax.legend()
ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md ### 2. Gender gap in labor force participation over time — regional trend

# COMMAND ----------

HIGHLIGHT = ["Brazil", "Mexico", "Argentina", "Colombia", "Guatemala", "Bolivia"]

gap_ts = (
    df_raw[df_raw["indicator_code"] == "SL.TLF.CACT.FM.ZS"]
    .dropna(subset=["value"])
    .groupby(["country_name", "year"])["value"]
    .mean()
    .reset_index()
)

# Regional average
regional_avg = gap_ts.groupby("year")["value"].mean().reset_index()

fig, ax = plt.subplots(figsize=(12, 6))
for country in HIGHLIGHT:
    data = gap_ts[gap_ts["country_name"] == country]
    if not data.empty:
        ax.plot(data["year"], data["value"], marker="o", markersize=3, linewidth=1.5, label=country)

ax.plot(regional_avg["year"], regional_avg["value"], color="black",
        linewidth=2.5, linestyle="--", label="Regional average")
ax.axhline(100, color="grey", linewidth=0.8, linestyle=":")
ax.set_xlabel("Year")
ax.set_ylabel("Ratio female-to-male LFP rate (%)")
ax.set_title("Gender Gap in Labor Force Participation Over Time\n(100 = parity)", fontweight="bold")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md ### 3. Female vs Male Unemployment rate — scatter plot

# COMMAND ----------

unem_f = get_latest("SL.UEM.TOTL.FE.ZS").set_index("country_name")["value"].rename("female")
unem_m = get_latest("SL.UEM.TOTL.MA.ZS").set_index("country_name")["value"].rename("male")
unem = pd.concat([unem_f, unem_m], axis=1).dropna()

fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(unem["male"], unem["female"], color=COLORS["female"], s=80, zorder=3)
for country, row in unem.iterrows():
    ax.annotate(country, (row["male"], row["female"]),
                textcoords="offset points", xytext=(5, 3), fontsize=8)

# Parity line
lim = max(unem.max()) + 2
ax.plot([0, lim], [0, lim], color="grey", linewidth=1, linestyle="--", label="Parity (f = m)")
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_xlabel("Male unemployment rate (%)")
ax.set_ylabel("Female unemployment rate (%)")
ax.set_title("Female vs Male Unemployment Rate\n(points above the line = women more unemployed)", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md ### 4. Vulnerable employment — female vs male by country

# COMMAND ----------

vuln_f = get_latest("SL.EMP.VULN.FE.ZS").set_index("country_name")["value"].rename("female")
vuln_m = get_latest("SL.EMP.VULN.MA.ZS").set_index("country_name")["value"].rename("male")
vuln = pd.concat([vuln_f, vuln_m], axis=1).dropna()
vuln["gap"] = vuln["female"] - vuln["male"]
vuln = vuln.sort_values("female")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: grouped bars
y = range(len(vuln))
axes[0].barh([i - 0.2 for i in y], vuln["female"], height=0.4, color=COLORS["female"], label="Female")
axes[0].barh([i + 0.2 for i in y], vuln["male"],   height=0.4, color=COLORS["male"],   label="Male")
axes[0].set_yticks(list(y))
axes[0].set_yticklabels(vuln.index)
axes[0].set_xlabel("Vulnerable employment (%)")
axes[0].set_title("Vulnerable Employment by Gender", fontweight="bold")
axes[0].legend()

# Right: gap (female − male)
gap_sorted = vuln["gap"].sort_values()
colors = [COLORS["female"] if v > 0 else COLORS["male"] for v in gap_sorted]
axes[1].barh(gap_sorted.index, gap_sorted.values, color=colors)
axes[1].axvline(0, color="black", linewidth=0.8)
axes[1].set_xlabel("Gap (female − male, pp)")
axes[1].set_title("Vulnerable Employment Gender Gap\n(positive = women more vulnerable)", fontweight="bold")

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md ### 5. Female sectoral employment composition — stacked bar

# COMMAND ----------

sector_codes = {
    "SL.AGR.EMPL.FE.ZS": "Agriculture",
    "SL.IND.EMPL.FE.ZS": "Industry",
    "SL.SRV.EMPL.FE.ZS": "Services",
}
sector_frames = {label: get_latest(code).set_index("country_name")["value"]
                 for code, label in sector_codes.items()}
sectors = pd.DataFrame(sector_frames).dropna()
sectors = sectors.sort_values("Services", ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
bottom = pd.Series(0, index=sectors.index)
palette = ["#7bba8a", "#f0a868", "#7eb0d5"]
for col, color in zip(["Agriculture", "Industry", "Services"], palette):
    ax.barh(sectors.index, sectors[col], left=bottom, color=color, label=col)
    bottom += sectors[col]

ax.set_xlabel("Share of female employment (%)")
ax.set_title("Female Employment by Sector\n(latest available year)", fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md ### 6. Women in parliament — evolution over time

# COMMAND ----------

parl = (
    df_raw[df_raw["indicator_code"] == "SG.GEN.PARL.ZS"]
    .dropna(subset=["value"])
)
parl_avg = parl.groupby("year")["value"].mean().reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: regional average over time
axes[0].fill_between(parl_avg["year"], parl_avg["value"], alpha=0.3, color=COLORS["female"])
axes[0].plot(parl_avg["year"], parl_avg["value"], color=COLORS["female"], linewidth=2)
axes[0].axhline(50, color="grey", linewidth=0.8, linestyle="--", label="Parity (50%)")
axes[0].set_xlabel("Year")
axes[0].set_ylabel("% seats held by women")
axes[0].set_title("Women in Parliament — Regional Average", fontweight="bold")
axes[0].legend()

# Right: latest value per country
latest_parl = (
    parl.sort_values("year", ascending=False)
    .groupby("country_name", as_index=False)
    .first()
    .sort_values("value")
)
bar_colors = [COLORS["female"] if v >= 30 else "#e0a8bc" for v in latest_parl["value"]]
axes[1].barh(latest_parl["country_name"], latest_parl["value"], color=bar_colors)
axes[1].axvline(30, color="grey", linewidth=1, linestyle="--", label="30% threshold")
axes[1].axvline(50, color="black", linewidth=0.8, linestyle=":", label="Parity (50%)")
axes[1].set_xlabel("% seats held by women")
axes[1].set_title("Women in Parliament — Latest Year", fontweight="bold")
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md ### 7. Correlation heatmap — all indicators (latest year, cross-country)

# COMMAND ----------

# Short names for readability in the heatmap
short_names = {
    "SL.TLF.CACT.FE.ZS": "LFP female",
    "SL.TLF.CACT.MA.ZS": "LFP male",
    "SL.TLF.CACT.FM.ZS": "LFP f/m ratio",
    "SL.UEM.TOTL.FE.ZS": "Unempl. female",
    "SL.UEM.TOTL.MA.ZS": "Unempl. male",
    "SL.EMP.VULN.FE.ZS": "Vuln. empl. female",
    "SL.EMP.VULN.MA.ZS": "Vuln. empl. male",
    "SL.EMP.SELF.FE.ZS": "Self-empl. female",
    "SL.EMP.MPYR.FE.ZS": "Employers female",
    "SL.EMP.WORK.FE.ZS": "Wage workers female",
    "SL.AGR.EMPL.FE.ZS": "Agric. empl. female",
    "SL.IND.EMPL.FE.ZS": "Industry empl. female",
    "SL.SRV.EMPL.FE.ZS": "Services empl. female",
    "SE.ADT.LITR.FE.ZS": "Literacy female",
    "SE.SEC.ENRR.FE":    "Secondary enroll. female",
    "SE.TER.ENRR.FE":    "Tertiary enroll. female",
    "SP.DYN.TFRT.IN":    "Fertility rate",
    "SH.STA.MMRT":       "Maternal mortality",
    "SG.GEN.PARL.ZS":    "Women in parliament",
    "IC.FRM.FEMM.ZS":    "Firms w/ female mgr",
}

short_names = INDICATORS

# Use df_wide which already has the latest-year pivot
corr_df = df_wide.set_index("country_code").drop(columns=["country_name"])
corr_df = corr_df.rename(columns=short_names)
corr_matrix = corr_df.corr()

mask = corr_matrix.isnull()  # hide pairs with no overlap
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    linewidths=0.4,
    ax=ax,
    annot_kws={"size": 7},
    vmin=-1, vmax=1,
)
ax.set_title("Correlation Matrix — Women's Labor Market Indicators\n(cross-country, latest year)", fontweight="bold")
plt.tight_layout()
plt.show()

# COMMAND ----------

corr_matrix.to_csv(
    "/Volumes/workspace/dataccion/raw_data/world_bank_corrmatrix.csv"
)
