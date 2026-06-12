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


def first_method(log_age, metallicity, log_Teff, log_g, q, model_A_path, model_B_path, scaler_A_path=None, scaler_B_path=None):
    """
    First method to compute Teff, log_g and log_R of a secondary star given the log_age, metallicity, T_eff and log_g of the primary star as well as a mass ratio.
    The parameters are not checked, we expect the user to input parameters whihc are possible.

    Parameters:
        log_age (float) : log10 of the age of the system in years
        metallicity (float) : metallicity of the system
        log_Teff (float) : log10 of the effective temperature of the primary star
        log_g (float) : log10 of the surface gravity of the primary star
        q (float) : mass ratio of the secondary star to the primary star (M2/M1)
        model_A_path (str) : path to the trained primary model
        model_B_path (str) : path to the trained secondary model
        scaler_A_path (str) : path to the scaler of the primary model, if provided
        scaler_B_path (str) : path to the scaler of the secondary model, if provided

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
    # a model's associated errors depending on the input can be found in the "/results/model_(A_partial|B)/final_models/{model_name}/{output}_input_err_plot" and ".../{output}_input_max_err_plot"

    model_A_inputs = [[log_age, log_Teff, log_g, metallicity]]
    if scaler_A_path is not None: # apply a scaling to the data if provided
        model_A_scaler = joblib.load(scaler_A_path)
        model_A_inputs = model_A_scaler.transform(model_A_inputs)
    # Predict the mass and radius of the primary star
    star_mass1, log_R1 = model_A.predict(model_A_inputs).flatten()
    star_mass2 = star_mass1 * q

    model_B_inputs = [[log_age, metallicity, star_mass2]]
    if scaler_B_path is not None: # apply a scaling to the data if provided
        model_B_scaler = joblib.load(scaler_B_path)
        model_B_inputs = model_B_scaler.transform(model_B_inputs)
    # Predict the effective temperature, surface gravity and radius of the secondary star
    log_Teff2, log_g2, log_R2 = model_B.predict(model_B_inputs).flatten()

    return star_mass1, log_R1, log_Teff2, log_g2, log_R2

def interactive_first_method():
    """
    First method to compute Teff, log_g and log_R of a secondary star given the log_age, metallicity, T_eff and log_g of the primary star as well as a mass ratio.
    Is interactive through the shell terminal, loads the model once and can predict multiple values one after the other.
    The parameters are not checked, we expect the user to input parameters whihc are possible.
    """
    model_A_path = "to_change"
    model_B_path = "to_change"
    print(f"Model A : {model_A_path}")
    print(f"Model B : {model_B_path}")
    print("Loading models...")
    model_A = joblib.load(model_A_path)
    model_B = joblib.load(model_B_path)
    # a model's associated errors depending on the input can be found in the "/results/model_(A_partial|B)/final_models/{model_name}/{output}_input_err_plot" and ".../{output}_input_max_err_plot"

    scaler_A_path = "to_change"
    scaler_B_path = "to_change"
    if scaler_A_path is not None:
        print(f"Scaler A : {scaler_A_path}")
        model_A_scaler = joblib.load(scaler_A_path)
    if scaler_B_path is not None:
        print(f"Scaler B : {scaler_B_path}")
        model_B_scaler = joblib.load(scaler_B_path)    
    
    print("To quit, press ctrl+C.")
    while True:
        age, metallicity, Teff1, log_g1, q = input("Star parameters (age (in Gy), metallicity (in dex), T_eff primary (in K), log_g primary (in log(cm/s^2)), q): ").split(" ")
        log_age, metallicity, log_Teff1, log_g1, q = math.log(float(age)), float(metallicity), math.log(float(Teff1)), float(log_g1), float(q)

        print("Predicting values...")
        model_A_inputs = [[log_age, log_Teff1, log_g1, metallicity]]
        if scaler_A_path is not None:
            model_A_inputs = model_A_scaler.transform(model_A_inputs)
        # Predict the mass and radius of the primary star
        star_mass1, log_R1 = model_A.predict(model_A_inputs).flatten()
        star_mass2 = star_mass1 * q

        model_B_inputs = [[log_age, metallicity, star_mass2]]
        if scaler_B_path is not None:
            model_B_inputs = model_B_scaler.transform(model_B_inputs)
        # Predict the effective temperature, surface gravity and radius of the secondary star
        log_Teff2, log_g2, log_R2 = model_B.predict(model_B_inputs).flatten()

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
