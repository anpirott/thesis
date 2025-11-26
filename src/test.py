import pandas as pd
from IPython.display import display

if __name__ == "__main__":
    a = {"metal" : [0, 0, 0, 0, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5], 
         "phase":[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4], 
         "log_Teff" : [10, 20, 30, 40, 13, 23, 33, 43, 16, 26, 36, 46],
         "log_L" : [10, 20, 30, 40, 13, 23, 33, 43, 16, 26, 36, 46]}

    df = pd.DataFrame.from_dict(a)

    display(df)

    df.loc[((df["metal"] == 0) & 
            (df["log_Teff"] >= 15) & 
            (df["phase"] == 3)), 
            ["phase"]] = 555


    # data_dff.loc[[data_dff["metallicity"] == metallicity, data_dff["phase"] == 3, 
    #               data_dff["log_L"] <= log_L_lim[0], (data_dff["log_Teff"] >= log_Teff_lim[0])]] = 6



    display(df)