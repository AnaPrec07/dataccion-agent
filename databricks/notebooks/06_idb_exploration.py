# Databricks notebook source

# COMMAND ----------

# MAGIC %md # IDB

# COMMAND ----------

# MAGIC %pip install idbsocialdatapy
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import idbsocialdatapy as idb
poverty = idb.query_indicator(indicator = 'pobreza') # define indicator to consult 
                          #categories = 'sex', # define category/ies I want to see in the indicator
                          #countries = 'COL,BRA,MEX', #define countries you want data from
                          #yearstart = '2005', # starting period
                          #yearend = '2021') # ending period
