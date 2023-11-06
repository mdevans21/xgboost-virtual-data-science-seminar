import math

import pandas as pd
import durkon as du

#Setup

df = pd.read_csv("r_sim_data.csv")

df["cat2"] = df["cat2"].astype(str)

trainDf = df[df["split"]=="train"].reset_index(drop=True)
testDf = df[df["split"]=="test"].reset_index(drop=True)

#Specify columns

cats = ["cat1","cat2","cat3"]
conts = ["num4"]

#Starting model

model = du.wraps.prep_model(trainDf, "combined", cats, conts)

model = du.wraps.train_tweedie_model(trainDf, "combined", 200, 0.07, model, pTweedie=1.7, prints="silent")

#Detect interactions, add, retrain.

print("Interaction potentials:")
du.wraps.interxhunt_tweedie_model(trainDf, "combined", cats, conts, model) #Results suggest the biggest interaction is between cat2 and cat3

model = du.prep.add_catcat_to_model(model, trainDf, "cat2", "cat3", replace=True)

model = du.wraps.train_tweedie_model(trainDf, "combined", 500, 0.07, model, pTweedie=1.7, prints="silent")

#This is the entire model in code form (much shorter and more readable than xgb equivalent!)

du.misc.save_model(model)
print("model:")
print(model)

#Visualize result

du.wraps.viz_multiplicative_model(model)

#Predict on test data

testDf["preds"] = du.misc.predict(testDf, model)

#Eval!

RMSE = math.sqrt(sum((testDf["preds"]-testDf["combined"])**2)/len(testDf))
print("RMSE: ", RMSE)