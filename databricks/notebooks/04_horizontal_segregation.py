# Databricks notebook source

# COMMAND ----------

# MAGIC %md # Analyze Distribution

# COMMAND ----------

import pandas as pd

df = pd.read_csv(
    '/Volumes/workspace/dataccion/raw_data/EMP_TEMP_SEX_ECO_NB_A-remodelado-2026-04-04.csv'
)

# COMMAND ----------

df

# COMMAND ----------

# MAGIC %md ## Segregación horizontal por género y actividad económica

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np

# --- Prep ---
# Drop "Total" rows and rows with all-NaN values
df_act = df[df['Actividad económica'] != 'Total'].dropna(subset=['Mujeres', 'Hombres']).copy()

# Short label: letter code only (A, B, C, …)
df_act['Código'] = df_act['Actividad económica'].str.extract(r'^([A-Z](?:\d+)?)\.')

# Full short label for tooltips/legend
df_act['Etiqueta'] = df_act['Actividad económica'].str.slice(0, 55)

# Female share (%) out of total employed in that sector
df_act['Total_sector'] = df_act['Mujeres'] + df_act['Hombres']
df_act['Pct_mujeres'] = df_act['Mujeres'] / df_act['Total_sector'] * 100
df_act['Pct_hombres'] = df_act['Hombres'] / df_act['Total_sector'] * 100

countries = df_act['Área de referencia'].unique()
print(f"Países: {list(countries)}")
print(f"Actividades únicas: {df_act['Código'].nunique()}")
df_act.head()

# COMMAND ----------

# MAGIC %md ### 1. Gráfico de mariposa: distribución por género y actividad (por país)
# MAGIC %md
# MAGIC %md Cada barra muestra el porcentaje de mujeres (izquierda, rosa) y hombres (derecha, azul) en cada sector. La línea punteada en 50 % indica paridad.

# COMMAND ----------

COLOR_F = '#E07B8A'   # women – pink/red
COLOR_M = '#4A7BB5'  # men  – blue

ncols = 2
nrows = int(np.ceil(len(countries) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 6))
axes = axes.flatten()

for ax, country in zip(axes, countries):
    sub = df_act[df_act['Área de referencia'] == country].copy()
    # Sort by female share so most female-dominated sectors appear at top
    sub = sub.sort_values('Pct_mujeres', ascending=True)
    y = np.arange(len(sub))

    ax.barh(y, sub['Pct_mujeres'], color=COLOR_F, label='Mujeres', height=0.6)
    ax.barh(y, sub['Pct_hombres'], left=sub['Pct_mujeres'], color=COLOR_M, label='Hombres', height=0.6)

    ax.axvline(50, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_xlim(0, 100)
    ax.set_yticks(y)
    ax.set_yticklabels(sub['Código'], fontsize=9)
    ax.set_xlabel('Porcentaje (%)', fontsize=9)
    ax.set_title(country, fontsize=12, fontweight='bold')
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    if ax == axes[0]:
        ax.legend(loc='lower right', fontsize=8)

# Hide unused subplots
for ax in axes[len(countries):]:
    ax.set_visible(False)

# Build legend for activity codes (use first country as reference)
ref = df_act[df_act['Área de referencia'] == countries[0]][['Código', 'Etiqueta']].drop_duplicates().sort_values('Código')
legend_text = '\n'.join(f"{row['Código']}: {row['Etiqueta']}" for _, row in ref.iterrows())
fig.text(0.5, -0.01, legend_text, ha='center', va='top', fontsize=7,
         family='monospace', wrap=True)

fig.suptitle('Segregación horizontal: % de mujeres y hombres por actividad económica',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('/Volumes/workspace/dataccion/raw_data/horizontal_segregation_butterfly.png', dpi=150, bbox_inches='tight')
plt.show()

# COMMAND ----------

# MAGIC %md ### 2. Mapa de calor: proporción de mujeres (%) por país y actividad
# MAGIC %md
# MAGIC %md Valores > 50 % (tonos verdes) indican sectores feminizados; valores < 50 % (tonos rojos) indican sectores masculinizados.

# COMMAND ----------

# Pivot: rows = activity code, columns = country
pivot = df_act.pivot_table(
    index='Código', columns='Área de referencia', values='Pct_mujeres', aggfunc='mean'
)
# Sort rows by median female share across countries
pivot = pivot.loc[pivot.median(axis=1).sort_values().index]

fig, ax = plt.subplots(figsize=(max(8, len(countries) * 1.5), len(pivot) * 0.55 + 1))

sns.heatmap(
    pivot,
    annot=True, fmt='.0f',
    cmap='RdYlGn',        # red = male-dominated, green = female-dominated
    center=50,
    vmin=0, vmax=100,
    linewidths=0.4,
    linecolor='white',
    cbar_kws={'label': '% mujeres', 'shrink': 0.6},
    ax=ax
)

ax.set_xlabel('')
ax.set_ylabel('Actividad económica (código)', fontsize=10)
ax.set_title('% de mujeres en el empleo por sector y país\n(verde > 50 % = feminizado · rojo < 50 % = masculinizado)',
             fontsize=12, fontweight='bold', pad=12)
ax.tick_params(axis='x', labelsize=10, rotation=30)
ax.tick_params(axis='y', labelsize=9)

plt.tight_layout()
plt.savefig('/Volumes/workspace/dataccion/raw_data/horizontal_segregation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# COMMAND ----------

# MAGIC %md ### 3. Índice de segregación por actividad (promedio regional)
# MAGIC %md
# MAGIC %md Distancia respecto a la paridad (50 %): valores positivos = sectores feminizados, negativos = masculinizados.

# COMMAND ----------

# Mean female share per activity across all countries
seg = (
    df_act.groupby(['Código', 'Etiqueta'])['Pct_mujeres']
    .mean()
    .reset_index()
    .rename(columns={'Pct_mujeres': 'Pct_mujeres_media'})
)
seg['Desviación_paridad'] = seg['Pct_mujeres_media'] - 50
seg = seg.sort_values('Desviación_paridad')

colors = [COLOR_F if v > 0 else COLOR_M for v in seg['Desviación_paridad']]

fig, ax = plt.subplots(figsize=(10, len(seg) * 0.45 + 1.5))
bars = ax.barh(seg['Código'], seg['Desviación_paridad'], color=colors, height=0.65, edgecolor='white')

ax.axvline(0, color='black', linewidth=1)
ax.set_xlabel('Desviación respecto a la paridad (pp)', fontsize=10)
ax.set_title('Índice de segregación horizontal por actividad (promedio regional)\n'
             'Rosa = feminizado · Azul = masculinizado', fontsize=12, fontweight='bold')
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%+.0f pp'))

# Annotate with mean % value
for bar, val, pct in zip(bars, seg['Desviación_paridad'], seg['Pct_mujeres_media']):
    xpos = val + (1.2 if val >= 0 else -1.2)
    ha = 'left' if val >= 0 else 'right'
    ax.text(xpos, bar.get_y() + bar.get_height() / 2,
            f'{pct:.0f}%', va='center', ha=ha, fontsize=8)

plt.tight_layout()
plt.savefig('/Volumes/workspace/dataccion/raw_data/horizontal_segregation_index.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nActividades más feminizadas (promedio regional):")
print(seg.tail(5)[['Código', 'Etiqueta', 'Pct_mujeres_media']].to_string(index=False))
print("\nActividades más masculinizadas (promedio regional):")
print(seg.head(5)[['Código', 'Etiqueta', 'Pct_mujeres_media']].to_string(index=False))

# COMMAND ----------

# MAGIC %md ### 4. Variabilidad entre países: ¿qué sectores muestran más dispersión?
# MAGIC %md
# MAGIC %md Un sector con alta desviación estándar entre países indica que la segregación no es homogénea en la región.

# COMMAND ----------

stats = (
    df_act.groupby('Código')['Pct_mujeres']
    .agg(media='mean', std='std', min='min', max='max')
    .reset_index()
    .sort_values('media')
)

fig, ax = plt.subplots(figsize=(10, len(stats) * 0.45 + 1.5))

# Range bar (min–max)
ax.barh(stats['Código'], stats['max'] - stats['min'],
        left=stats['min'], color='#BBCCEE', height=0.6, label='Rango (mín–máx)')

# Mean marker
ax.scatter(stats['media'], stats['Código'],
           color='#2C4770', zorder=5, s=50, label='Media regional')

# Std error bars
ax.errorbar(stats['media'], stats['Código'],
            xerr=stats['std'], fmt='none', color='#2C4770',
            capsize=3, linewidth=1.2, label='± 1 DE')

ax.axvline(50, color='black', linestyle='--', linewidth=0.8, alpha=0.6, label='Paridad 50%')
ax.set_xlim(0, 100)
ax.set_xlabel('% mujeres en el sector', fontsize=10)
ax.set_title('Dispersión del % de mujeres por sector entre países\n(rango completo, media y desviación estándar)',
             fontsize=12, fontweight='bold')
ax.xaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(fontsize=9, loc='lower right')

plt.tight_layout()
plt.savefig('/Volumes/workspace/dataccion/raw_data/horizontal_segregation_dispersion.png', dpi=150, bbox_inches='tight')
plt.show()

print(stats.sort_values('std', ascending=False).to_string(index=False))
