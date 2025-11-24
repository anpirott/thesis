import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import sys
import os

# xlabel_HR_Kiel = "$\log(T_{\mathrm{eff}}) [\mathrm{K}]$"

# ylabel_HR = "$\log(L/L_{\odot})$"

# ylabel_Kiel = "$\log(g) [\mathrm{cm/s^2}]$"

# legend_title_age = "$\log(Age) [\mathrm{yr}]$"
# legend_title_phase = "Phases"

# quelles colonnes on veut : ['log10_isochrone_age_yr', 'log_Teff', 'log_g', 'phase', 'metallicity', 'star_mass', 'log_R', "log_L"]
# quelles colonnes on a    : ['logAge',                 'logTe'     'logg',  'label', 'metallicity', 'Mass',      'Rpol',  "logL"]

class Data_visualiser():
    def __init__(self, iso_df : pd.DataFrame, physical_model : str):
        self.iso_df = iso_df
        if physical_model == "MIST" or physical_model == "PARSEC":
            self.physical_model = physical_model
        else:
            print("Error: physical_model should be either 'MIST' or 'PARSEC'")
            sys.exit(1)

        self.c_dict_MIST = {-1 : "orange", 0 : "blue", 2 : "green", 3 : "red", 4 : "purple", 5 : "yellow", 6 : "cyan", 9 : "grey"}
        self.phase_dict_MIST = {-1 : "PMS", 0 : "MS", 2 : "RGB", 3 : "CHeB", 4 : "EAGB", 5 : "TPAGB", 6 : "postAGB", 9 : "WR"}
        self.all_metallicities_MIST = [-4, -3.5, -3, -2.5, -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5]

        self.c_dict_PARSEC = {0 : "orange", 1 : "blue", 2 : "grey", 3 : "green", 4 : "red",  5 : "red",  6 : "red", 7 : "purple", 8 : "yellow", 9 : "cyan"}
        self.phase_dict_PARSEC = {0 : "PMS", 1 : "MS", 2 : "SGB", 3 : "RGB", 4 : "CHeB",  5 : "CHeB_blue",  6 : "CHeB_red", 7 : "EAGB", 8 : "TPAGB", 9 : "postAGB"}
        self.all_metallicities_PARSEC = [-2.75, -2.5, -2.25, -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5]

    # TODO? deux fonctions en une avec un paramètre pour savoir si je prends le log_L ou le log_g
    def plot_HR(self, ages : list[float], metallicities : list[float], iso_df : pd.DataFrame=None, physical_model : str=None):
        if iso_df is None:
            iso_df = self.iso_df
        if physical_model is None:
            physical_model = self.physical_model
        if len(ages) == 0:
            if physical_model == "MIST":
                ages = iso_df["log10_isochrone_age_yr"].unique()
            elif physical_model == "PARSEC":
                ages = iso_df["logAge"].unique()
        if len(metallicities) == 0:
            if physical_model == "MIST":
                metallicities = self.all_metallicities_MIST
            elif physical_model == "PARSEC":
                metallicities = self.all_metallicities_PARSEC
        
        if physical_model == "MIST":
            unique_phases = iso_df["phase"].unique()
            phase_dict = {key: self.phase_dict_MIST[key] for key in unique_phases} # only keeps the values for the phases the dataframe actually contains
            c_dict = {key: self.c_dict_MIST[key] for key in unique_phases}
        elif physical_model == "PARSEC":
            unique_phases = iso_df["label"].unique()
            phase_dict = {key: self.phase_dict_PARSEC[key] for key in unique_phases} # only keeps the values for the phases the dataframe actually contains
            c_dict = {key: self.c_dict_PARSEC[key] for key in unique_phases}

        for metallicity in metallicities:
            for phase in unique_phases:
                for age in ages:
                    if physical_model == "MIST":
                        logTeff = iso_df[(iso_df["phase"]==phase) & (iso_df["metallicity"]==metallicity) & (iso_df["log10_isochrone_age_yr"]==age)]["log_Teff"]
                        logL = iso_df[(iso_df["phase"]==phase) & (iso_df["metallicity"]==metallicity) & (iso_df["log10_isochrone_age_yr"]==age)]["log_L"]
                        plt.plot(logTeff, logL, c=c_dict[phase])
                    elif physical_model == "PARSEC":
                        logTeff = iso_df[(iso_df["label"]==phase) & (iso_df["metallicity"]==metallicity) & (iso_df["logAge"]==age)]["logTe"]
                        logL = iso_df[(iso_df["label"]==phase) & (iso_df["metallicity"]==metallicity) & (iso_df["logAge"]==age)]["logL"]
                        plt.plot(logTeff, logL, c=c_dict[phase])
                
            plt.xlim(5.7, 3.3)
            plt.xlabel("$\log(T_{\mathrm{eff}}) [\mathrm{K}]$")
            plt.ylabel("$\log(L/L_{\odot})$")
            plt.legend(title="Phases", fontsize="small", 
                    handles = [mlines.Line2D([], [], color=c_dict[key], label=f"{phase_dict[key]}") for key in c_dict.keys()])
            plt.title(f"Metallicity = {metallicity}")
            plt.show()

    def plot_Kiel(self, ages : list[float], metallicities : list[float], iso_df : pd.DataFrame=None, physical_model : str=None):
        if iso_df is None:
            iso_df = self.iso_df
        if physical_model is None:
            physical_model = self.physical_model
        if len(ages) == 0:
            if physical_model == "MIST":
                ages = iso_df["log10_isochrone_age_yr"].unique()
            elif physical_model == "PARSEC":
                ages = iso_df["logAge"].unique()
        if len(metallicities) == 0:
            if physical_model == "MIST":
                metallicities = self.all_metallicities_MIST
            elif physical_model == "PARSEC":
                metallicities = self.all_metallicities_PARSEC
        
        if physical_model == "MIST":
            unique_phases = iso_df["phase"].unique()
            phase_dict = {key: self.phase_dict_MIST[key] for key in unique_phases} # only keeps the values for the phases the dataframe actually contains
            c_dict = {key: self.c_dict_MIST[key] for key in unique_phases}
        elif physical_model == "PARSEC":
            unique_phases = iso_df["label"].unique()
            phase_dict = {key: self.phase_dict_PARSEC[key] for key in unique_phases} # only keeps the values for the phases the dataframe actually contains
            c_dict = {key: self.c_dict_PARSEC[key] for key in unique_phases}

        for metallicity in metallicities:
            for phase in unique_phases:
                for age in ages:
                    if physical_model == "MIST":
                        logTeff = iso_df[(iso_df["phase"]==phase) & (iso_df["metallicity"]==metallicity) & (iso_df["log10_isochrone_age_yr"]==age)]["log_Teff"]
                        logg = iso_df[(iso_df["phase"]==phase) & (iso_df["metallicity"]==metallicity) & (iso_df["log10_isochrone_age_yr"]==age)]["log_g"]
                        plt.plot(logTeff, logg, c=c_dict[phase])
                    elif physical_model == "PARSEC":
                        logTeff = iso_df[(iso_df["label"]==phase) & (iso_df["metallicity"]==metallicity) & (iso_df["logAge"]==age)]["logTe"]
                        logg = iso_df[(iso_df["label"]==phase) & (iso_df["metallicity"]==metallicity) & (iso_df["logAge"]==age)]["logg"]
                        plt.plot(logTeff, logg, c=c_dict[phase])
                
            plt.xlim(5.7, 3.3)
            plt.gca().invert_yaxis()
            plt.xlabel("$\log(T_{\mathrm{eff}}) [\mathrm{K}]$")
            plt.ylabel("$\log(g) [\mathrm{cm/s^2}]$")
            plt.legend(title="Phases", fontsize="small", 
                    handles = [mlines.Line2D([], [], color=c_dict[key], label=f"{phase_dict[key]}") for key in c_dict.keys()])
            plt.title(f"Metallicity = {metallicity}")
            plt.show()

if __name__ == "__main__":
    pass