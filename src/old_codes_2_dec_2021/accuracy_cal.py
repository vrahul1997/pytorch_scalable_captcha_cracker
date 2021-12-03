import pandas as pd

df = pd.read_csv("preds_epfo_tan_pan_oltas.csv")
pred_len = len(df[df["values"] == df["preds"]])
total_len = len(df)

acc = pred_len/total_len * 100
print("acc:", acc)

print(df[df["values"] != df["preds"]]["values"])
