import pandas as pd

import config

def read_all():
    train_income_df = pd.read_csv(config.data_dir + "train.csv")
    test_income_df = pd.read_csv(config.data_dir + "test.csv")
    demographic_df = pd.read_csv(config.data_dir + "demographics.csv")
    cc_df = pd.read_csv(config.data_dir + "cc.csv")
    kplus_df = pd.read_csv(config.data_dir + "kplus.csv")

    train_income_df = train_income_df.set_index("id")
    test_income_df = test_income_df.set_index("id")

    merged_df = make_merged_df(train_income_df, test_income_df, demographic_df, cc_df, kplus_df)

    train_df = pd.concat([train_income_df, merged_df], axis=1)
    train_df = train_df.dropna()

    test_df = pd.concat([test_income_df, merged_df], axis=1)
    test_df = test_df.dropna()

    return train_income_df, test_income_df, demographic_df, cc_df, kplus_df, train_df, test_df

def make_merged_df(train_income_df, test_income_df, demographic_df, cc_df, kplus_df):
    # count cc
    n_cc_df = demographic_df.groupby("id").count()[["cc_no"]].copy()
    n_cc_df.columns = ["cc_count"]

    # merge cc_df
    merged_df = demographic_df.merge(cc_df, on="cc_no", how="outer")

    # sum cc
    sum_cc_txn_amt_df = merged_df[["id", "cc_txn_amt"]].groupby("id").sum()

    # count cc
    count_cc_txn_amt_df = merged_df[["id", "cc_txn_amt"]].groupby("id").count()
    
    # add sum cc and count cc
    merged_df = merged_df.groupby("id").last()
    merged_df["cc_txn_amt"] = sum_cc_txn_amt_df["cc_txn_amt"]
    merged_df["cc_txn_cnt"] = count_cc_txn_amt_df["cc_txn_amt"]

    # group kplus
    group_kplus_df = kplus_df.groupby("id").sum()
    group_kplus_df["kp_txn_mean"] = group_kplus_df["kp_txn_amt"] / group_kplus_df["kp_txn_count"]

    # merge
    merged_df = merged_df.merge(n_cc_df, on="id", how="outer")
    merged_df = merged_df.merge(group_kplus_df, on="id", how="outer")

    # fill nan with 0 
    merged_df = merged_df.fillna(0)
    return merged_df