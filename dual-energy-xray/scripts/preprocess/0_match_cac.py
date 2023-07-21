import pandas as pd

df_cac = pd.read_excel(
    "/neodata/oxr/innocare/raw/111-060-F臨床試驗(睿生光電).xlsx",
    dtype={"病歷號": "str"},  # Otherwise some would be nan
)

# Strip 病歷號
df_cac["病歷號"] = df_cac["病歷號"].str.strip()

# Manual drop, remove instances with problem
remove = ["H510308"]
df_cac = df_cac[~df_cac["病歷號"].isin(remove)]

# Mapping of AN and UID
df_uid = pd.read_csv("/neodata/oxr/innocare/raw/收案表單.csv", usecols=["AN", "UID"])

# Manual fix, match = {AN: UID}
match = {"H196333": "145_20230221"}
for an, uid in match.items():
    row_an = df_uid["AN"] == an
    if sum(row_an) != 1:
        raise ValueError(f"Invalid AN of {an} found.")
    df_uid.loc[row_an, "UID"] = uid

# Manual fix, update = {old_AN: new_AN}
update = {"HG89271": "HG89217"}
for old_an, new_an in update.items():
    row_an = df_uid["AN"] == old_an
    if sum(row_an) != 1:
        raise ValueError(f"Invalid AN of {old_an} found.")
    df_uid.loc[row_an, "AN"] = new_an

# Manual add, update = {AN: UID}
update = {"H641371": "119_20230210"}
for an, uid in update.items():
    df_uid = df_uid._append({"AN": an, "UID": uid}, ignore_index=True)

# Merge UID into df_cac based on 病歷號 and AN
df_cac = pd.merge(df_cac, df_uid, left_on="病歷號", right_on="AN", how="left")
df_cac = df_cac.drop(columns="AN")

# Save the mapped CAC scores
df_cac.to_csv("/neodata/oxr/innocare/CAC_scores.csv", index=False)
