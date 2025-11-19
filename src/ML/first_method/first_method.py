# prendre le sous-ensemble ['star_mass', 'log_Teff', 'log_L', 'log_g', 'log_R'] et exécuter la méthode avec les paramètres qu'il faut.
# selon ces paramètres, je regarde dans le sous-ensemble les valeurs que je devrais avoir et je les compare avec ce que la méthode me renvoie.

import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import pickle
import joblib
import sys

from src.ML.utils.Iso_data_handler import Iso_data_handler


def first_method(age, metallicity, log_Teff, log_g, q, primary_mdl_path="model/primary_XGB.pkl", secondary_mdl_path="model/secondary_XGB.pkl"):
    """
    First method to compute Teff, log_g and log_R of a secondary star given the age, metallicity, T_eff and log_g of the primary star as well as a mass ratio.

    Parameters:
        age (float) : log10 of the age of the system in years
        metallicity (float) : metallicity of the system
        log_Teff (float) : log10 of the effective temperature of the primary star
        log_g (float) : log10 of the surface gravity of the primary star
        q (float) : mass ratio of the secondary star to the primary star (M2/M1)
        primary_mdl_path (str) : path to the trained primary model
        secondary_mdl_path (str) : path to the trained secondary model

    Returns:
        log_R1 (float) : log10 of the radius of the primary star
        log_Teff2 (float) : log10 of the effective temperature of the secondary star
        log_g2 (float) : log10 of the surface gravity of the secondary star
        log_R2 (float) : log10 of the radius of the secondary star
    """
    # Load the models
    primary_model = joblib.load(primary_mdl_path)
    secondary_model = joblib.load(secondary_mdl_path)

    # Predict the mass and radius of the primary star
    star_mass1, log_R1 = primary_model.predict([[age, metallicity, log_Teff, log_g]]).flatten()
    star_mass2 = star_mass1 * q

    # Predict the effective temperature, surface gravity and radius of the secondary star
    log_Teff2, log_g2, log_R2 = secondary_model.predict([[age, metallicity, star_mass2, log_R1]]).flatten()

    return log_R1, log_Teff2, log_g2, log_R2



if __name__ == "__main__":
    iso_handler = Iso_data_handler("data/MIST_v1.2_vvcrit0.0_basic_isos/", 
                              ['log10_isochrone_age_yr', 'log_Teff', 'log_g', 'star_mass', 'phase', 'metallicity', 'log_R'])
    test_df = iso_handler.full_iso_data_to_panda(override=False)

    index = int(sys.argv[1])
    q = float(sys.argv[2])
    row = test_df.loc[index]

    print("predicting values...")
    log_R1, log_Teff2, log_g2, log_R2 = first_method(row["log10_isochrone_age_yr"], row["metallicity"], row["log_Teff"], row["log_g"], q)
    print(f"Primary star mass: {row['star_mass']}, log_Teff: {row['log_Teff']}, log_g: {row['log_g']}, log_R: {row['log_R']}")
    print(f"Primary star predicted log_R: {log_R1}")
    print(f"Secondary star mass: {row['star_mass'] * q}, log_Teff: {log_Teff2}, log_g: {log_g2}, log_R: {log_R2}")

    # test_sample = test_df.sample(100000)
    # # print(test_sample)
    # for index, row in test_sample.iterrows():
    #     # print(row["log_g"])
    #     q = random.uniform(0.1, 0.9)
    #     secondary_star_mass = row["star_mass"] * q
    #     print(test_df.loc[test_df['star_mass'] == secondary_star_mass])
    #     print(type(test_df.loc[test_df['star_mass'] == secondary_star_mass]))
    #     break

    #     log_R1, log_Teff2, log_g2, log_R2 = first_method(row["log10_isochrone_age_yr"], row["metallicity"], row["log_Teff"], row["log_g"], q)
