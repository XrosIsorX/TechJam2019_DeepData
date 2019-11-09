import pandas as pd
import pickle
import matplotlib.pyplot as plt

import config
import utils.DataMaker as dm
from Model import Model

train_income_df, test_income_df, demographic_df, cc_df, kplus_df, train_df, test_df = dm.read_all()

model = Model()

model.train(train_df)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

test_df[config.target] = model.predict(test_df)
test_df[[config.target]].to_csv("q45_group_age_ocp_kp_txn_mean_group_-q45_group_age_ocp-_-q45_group_age-26000.csv")