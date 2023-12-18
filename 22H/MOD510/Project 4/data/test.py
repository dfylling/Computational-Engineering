# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
cwd = os.getcwd()
cwd = cwd.replace("\\", "/")

path = f"{cwd}/NOR-USA-FIN_Data.csv"

df = pd.read_csv(path)
# %%
useful_data_df = df.drop(
    df.columns.difference(['iso_code',
                           'date',
                           'new_cases_smoothed',
                           'new_deaths_smoothed',
                           'new_vaccinations_smoothed',
                           'population'
                           ]), 1)
NOR_useful_data_df = useful_data_df[useful_data_df['iso_code'] == "NOR"]
FIN_useful_data_df = useful_data_df[useful_data_df['iso_code'] == "FIN"]
USA_useful_data_df = useful_data_df[useful_data_df['iso_code'] == "USA"]
# %%
#   "new_cases_smoothed"
# | "new_deaths_smoothed"
# | "new_vaccinations_smoothed"
y = "new_cases_smoothed"
NOR_useful_data_df.plot(x="date", y=y)
plt.title(f"NOR | {y} \nFrom: {path}")
FIN_useful_data_df.plot(x="date", y=y)
plt.title(f"FIN | {y} \nFrom: {path}")
USA_useful_data_df.plot(x="date", y=y)
plt.title(f"USA | {y} \nFrom: {path}")
# %%
