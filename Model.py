class Model:
    
    def __init__(self):
        self.step_list_dict = {}
        self.step_list_dict["kp_txn_mean"] = [i for i in range(5,50, 5)] + [i for i in range(1000,12000, 1000)]
        self.step_list_dict["cc_txn_cnt"] = [i for i in range(3,50, 3)]
        self.step_list_dict["cc_txn_mean"] = [i for i in range(500,8000, 500)]
        self.step_list_dict["kp_txn_count"] = [i for i in range(10,140, 10)]

        self.discrete_list = ["kp_txn_mean", "cc_txn_cnt", "cc_txn_mean", "kp_txn_count"]
    
        self.feature_list_list = [
            # ["age", "ocp_cd", "cc_count", "cc_txn_cnt_group", "kp_txn_mean_group"],
            # ["age", "ocp_cd", "cc_count", "kp_txn_mean_group"],
            ["age", "ocp_cd", "kp_txn_mean_group"],
            ["age", "ocp_cd", ],
            ["age"],
        ]

        self.group_mean_df_list = []

    def discretize(self, row, step_list, column):
        for i, step in enumerate(step_list):
            if row[column] < step:
                return i
        return len(step_list)

    def discretize_all(self, df):
        temp_df = df.copy()
        for column in self.discrete_list:
            temp_df[column + "_group"] = temp_df.apply(self.discretize, args=[self.step_list_dict[column], column], axis=1)
        return temp_df

    def get_income(self, row):
        income = 26000
        for feature_list, group_mean_df in zip(self.feature_list_list, self.group_mean_df_list):
            key = []
            for feature in feature_list:
                key.append(row[feature])
            key = tuple(key)

            if key in group_mean_df:
                income = int(group_mean_df[key])
                return income
        print("No matching")
        return income

    def train(self, df):
        df = self.discretize_all(df)

        for feature_list in self.feature_list_list:
            self.group_mean_df_list.append(df.groupby(feature_list).quantile(0.45)["income"])
        
    def predict(self, df):
        df = self.discretize_all(df)
        return df.apply(self.get_income, axis=1)