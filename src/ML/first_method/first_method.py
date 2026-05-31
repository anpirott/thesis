"""
Script used to get the parameters of a secondary star from a binary system given some parameters of the primary star.

Inputs:
    age (float) : if input value < 100, log10 of the age of the system in years, value between 5.3 and 10.25 for PARSEC and 5.0 and 10.3 for MIST
                  else, age of the system in years, value between 200,000 and 17,782,794,100 for PARSEC and 100,000 and 19,952,623,149 for MIST
    metallicity (float) : metallicity of the system, value between -4 and 0.5 for both
    log_Teff (float) : log10 of the effective temperature of the primary star, value between 3.36 and 4.5177 for PARSEC and 3.34 and 5.3788 for MIST
    log_g (float) : log10 of the surface gravity of the primary star, value between -0.38 and 5.345 for PARSEC and -1.08 and 6.2594 for MIST
    q (float) : mass ratio of the secondary star to the primary star (M2/M1), value between 0 and 1 for both
    model_A_path (str) : path to the trained primary model
    model_B_path (str) : path to the trained secondary model

Outputs:
    star_mass1 (float) : log10 of the mass in solar masses
    log_R1 (float) : log10 of the radius of the primary star
    log_Teff2 (float) : log10 of the effective temperature of the secondary star
    log_g2 (float) : log10 of the surface gravity of the secondary star
    log_R2 (float) : log10 of the radius of the secondary star
"""

# prendre le sous-ensemble ['star_mass', 'log_Teff', 'log_L', 'log_g', 'log_R'] et exécuter la méthode avec les paramètres qu'il faut.
# selon ces paramètres, je regarde dans le sous-ensemble les valeurs que je devrais avoir et je les compare avec ce que la méthode me renvoie.

import joblib
import sys
import math


def first_method(log_age, metallicity, log_Teff, log_g, q, model_A_path, model_B_path):
    """
    First method to compute Teff, log_g and log_R of a secondary star given the log_age, metallicity, T_eff and log_g of the primary star as well as a mass ratio.

    Parameters:
        log_age (float) : log10 of the age of the system in years
        metallicity (float) : metallicity of the system
        log_Teff (float) : log10 of the effective temperature of the primary star
        log_g (float) : log10 of the surface gravity of the primary star
        q (float) : mass ratio of the secondary star to the primary star (M2/M1)
        model_A_path (str) : path to the trained primary model
        model_B_path (str) : path to the trained secondary model

    Returns:
        star_mass1 (float) : log10 of the mass in solar masses
        log_R1 (float) : log10 of the radius of the primary star
        log_Teff2 (float) : log10 of the effective temperature of the secondary star
        log_g2 (float) : log10 of the surface gravity of the secondary star
        log_R2 (float) : log10 of the radius of the secondary star
    """
    # Load the models
    model_A = joblib.load(model_A_path)
    model_B = joblib.load(model_B_path)
    # TODO dans un deuxième temps, pour les modèles, donner l'erreur associé quand on l'utilise

    # Predict the mass and radius of the primary star
    star_mass1, log_R1 = model_A.predict([[log_age, metallicity, log_Teff, log_g]]).flatten()
    star_mass2 = star_mass1 * q # TODO! gros souci si on demande de donner une valeur qui n'est pas possible dans une isochrone

    # Predict the effective temperature, surface gravity and radius of the secondary star
    log_Teff2, log_g2, log_R2 = model_B.predict([[log_age, metallicity, star_mass2]]).flatten()

    return star_mass1, log_R1, log_Teff2, log_g2, log_R2

def interactive_first_method():
    """
    First method to compute Teff, log_g and log_R of a secondary star given the log_age, metallicity, T_eff and log_g of the primary star as well as a mass ratio.
    Is interactive through the shell terminal, loads the model once and can predict multiple values one after the other
    """
    model_A_path = "to_change"
    model_B_path = "to_change"
    print(f"Model A : {model_A_path}")
    print(f"Model B : {model_B_path}")
    print("Loading models...")
    model_A = joblib.load(model_A_path)
    model_B = joblib.load(model_B_path)
    
    print("To quit, press ctrl+C.")
    while True:
        age, metallicity, Teff1, log_g1, q = input("Star parameters (age (in Gy), metallicity (in dex), T_eff primary (in K), log_g primary (in log(cm/s^2)), q): ").split(" ")
        log_age, metallicity, log_Teff1, log_g1, q = math.log(float(age)), float(metallicity), math.log(float(Teff1)), float(log_g1), float(q)

        print("Predicting values...")
        # Predict the mass and radius of the primary star
        star_mass1, log_R1 = model_A.predict([[log_age, metallicity, log_Teff1, log_g1]]).flatten()
        star_mass2 = star_mass1 * q

        # Predict the effective temperature, surface gravity and radius of the secondary star
        log_Teff2, log_g2, log_R2 = model_B.predict([[log_age, metallicity, star_mass2]]).flatten()

        print(f"Primary star parameters : age : {log_age}, metallicity : {metallicity}, mass : {star_mass1}, log_Teff : {log_Teff1}, log_g : {log_g1}, radius : {log_R1}")
        print(f"Secondary star parameters : age : {log_age}, metallicity : {metallicity}, mass : {star_mass2}, log_Teff : {log_Teff2}, log_g : {log_g2}, radius : {log_R2}")


if __name__ == "__main__":
    if sys.argv[1] == "interactive":
        interactive_first_method()
    else:
        age = float(sys.argv[1])
        if age > 100:
            age = math.log10(age)
        metallicity = float(sys.argv[2])
        log_Teff1 = float(sys.argv[3])
        log_g1 = float(sys.argv[4])
        q = float(sys.argv[5])
        model_A_path = str(sys.argv[6])
        model_B_path = str(sys.argv[7])

        print("Predicting values...")
        star_mass1, log_R1, log_Teff2, log_g2, log_R2 = first_method(age, metallicity, log_Teff1, log_g1, q, model_A_path, model_B_path)
        print(f"Primary star parameters : age : {age}, metallicity : {metallicity}, mass : {star_mass1}, log_Teff : {log_Teff1}, log_g : {log_g1}, radius : {log_R1}")
        print(f"Secondary star parameters : age : {age}, metallicity : {metallicity}, mass : {star_mass1*q}, log_Teff : {log_Teff2}, log_g : {log_g2}, radius : {log_R2}")


    # faire une version où un prompt avec des input() demande les paramètres mais où on ne load le modèle qu'une seule fois


    # from physics.Iso_data_handler import Iso_data_handler
    # iso_handler = Iso_data_handler("data/MIST_v1.2_vvcrit0.0_basic_isos/", 
    #                           ['log10_isochrone_age_yr', 'log_Teff', 'log_g', 'star_mass', 'phase', 'metallicity', 'log_R'],
    #                           "MIST")
    # test_df = iso_handler.get_isochrone_dataframe(override=False)

    # index = int(sys.argv[1])
    # q = float(sys.argv[2])
    # row = test_df.loc[index]

    # print("predicting values...")
    # star_mass1, log_R1, log_Teff2, log_g2, log_R2 = first_method(row["log10_isochrone_age_yr"], row["metallicity"], row["log_Teff"], row["log_g"], q)
    # print(f"Primary star mass: {row['star_mass']}, log_Teff: {row['log_Teff']}, log_g: {row['log_g']}, log_R: {row['log_R']}")
    # print(f"Primary star predicted log_R: {log_R1}")
    # print(f"Secondary star mass: {row['star_mass'] * q}, log_Teff: {log_Teff2}, log_g: {log_g2}, log_R: {log_R2}")

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
