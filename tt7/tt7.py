import pandas as pd
import numpy as np

# TASK 1
morg_df = pd.read_csv("data/morg_d07_strings.csv", index_col="h_id")

# TASK 2
print(morg_df[["age"]])

#TASK 3
print(morg_df.loc["1_2_2"])

#TASK 4
print(morg_df[0:4])

#TASK 5
columns_with_missing_values = {}

for column in morg_df.columns:
    if any(morg_df.loc[:, column].isna()):
        columns_with_missing_values[column] = 0

print(columns_with_missing_values)

#TASK 6
print(morg_df.fillna(value=columns_with_missing_values))


#TASK 7
TO_CATEGORICALS = ["gender", "race", "ethnicity", "employment_status"]

for column in TO_CATEGORICALS:
    morg_df.loc[:, column] = morg_df.loc[:, column].astype("category")

# Example use of cut()
boundaries = range(16, 89, 8)
morg_df.loc[:, "age_bin"] = pd.cut(morg_df.loc[:, "age"],
                                   bins=boundaries,
                                   labels=range(len(boundaries)-1),
                                   include_lowest=True, right=False)

#TASK 8
boundaries = range(0, 100, 10)
morg_df.loc[:, "hwpw_bin"] = pd.cut(morg_df.loc[:, "hours_worked_per_week"],
                                    bins = boundaries,
                                    labels=range(len(boundaries)-1),
                                    include_lowest=True, right=False)

print("Morg columns types after Task 8")
print(morg_df.dtypes)

#TASK 9
filter = (morg_df.hours_worked_per_week >= 35)
print(morg_df[filter])

#TASK 10
filter = (morg_df.employment_status != "Working")
print(morg_df[filter])

#TASK 11
filter = (morg_df.hours_worked_per_week >= 35) & (morg_df.earnings_per_week >= 1000)
print(morg_df[filter])

#TASK 12
race_counts = morg_df.loc[:, "race"].value_counts()
print(race_counts[0:5])

#TASK 13
print(morg_df.groupby("race").size())

#TASK 14
students = pd.read_csv("data/students.csv")
extended_grades = pd.read_csv("data/extended_grades.csv")

dataframe = pd.merge(students, extended_grades, on="UCID", how="inner")
print(dataframe.groupby(["Grade", "Major"]).size().reset_index(name="Count"))