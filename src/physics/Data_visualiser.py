import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import sys

# quelles colonnes on veut : ['log10_isochrone_age_yr', 'log_Teff', 'log_g', 'phase', 'metallicity', 'star_mass', 'log_R', "log_L"]
# quelles colonnes on a    : ['logAge',                 'logTe'     'logg',  'label', 'metallicity', 'Mass',      'Rpol',  "logL"]

# TODO mettre les docstrings
class Data_visualiser():
    def __init__(self, iso_df : pd.DataFrame, physical_model : str):
        """
        Initializes the Data_visualiser class with the isochrone dataframe and the physical model used to generate it.

        Parameters:
            iso_df (pd.DataFrame): The isochrone dataframe containing the stellar evolution data.
            physical_model (str): The physical model used to generate the isochrone data. Should be either "MIST" or "PARSEC".
        """
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

    def plot_isochrone(self, ages : list[float], metallicities : list[float], x_values : str, y_values : str, x_label : str, y_label : str,
                       iso_df : pd.DataFrame=None, physical_model : str=None, x_lim : tuple[float]=None, y_lim : tuple[float]=None,
                       x_log : bool=False, y_log : bool=False):
        """
        Plots the isochrones for the given ages and metallicities, using the specified x and y values for the axes. The plot is colored according to the different evolutionary phases of the stars.

        Parameters:
            ages (list[float]): A list of ages (in log years) for which to plot the isochrones. If empty, all ages in the dataframe will be plotted.
            metallicities (list[float]): A list of metallicities for which to plot the isochrones. If empty, all metallicities in the dataframe will be plotted.
            x_values (str): The column name in the dataframe to use for the x-axis values
            y_values (str): The column name in the dataframe to use for the y-axis values
            x_label (str): The label for the x-axis
            y_label (str): The label for the y-axis
            iso_df (pd.DataFrame, optional): The isochrone dataframe to use for plotting. If None, the dataframe provided during initialization will be used.
            physical_model (str, optional): The physical model to use for plotting. Should be either "MIST" or "PARSEC". If None, the model provided during initialization will be used.
            x_lim (tuple[float], optional): The limits for the x-axis. If None, the limits get preset values if not provided.
            y_lim (tuple[float], optional): The limits for the y-axis. If None, the limits get preset values if not provided.
            x_log (bool, optional): Whether to use a logarithmic scale for the x-axis. Default is False.
            y_log (bool, optional): Whether to use a logarithmic scale for the y-axis. Default is False.
        """
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
        if x_lim is None:
            x_lim = (5.7, 3.3)
        
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
                        x_axis = iso_df[(iso_df["phase"]==phase) & (iso_df["metallicity"]==metallicity) & (iso_df["log10_isochrone_age_yr"]==age)][x_values]
                        y_axis = iso_df[(iso_df["phase"]==phase) & (iso_df["metallicity"]==metallicity) & (iso_df["log10_isochrone_age_yr"]==age)][y_values]                        
                        plt.plot(x_axis, y_axis, c=c_dict[phase])
                    elif physical_model == "PARSEC":
                        x_axis = iso_df[(iso_df["label"]==phase) & (iso_df["metallicity"]==metallicity) & (iso_df["logAge"]==age)][x_values]
                        y_axis = iso_df[(iso_df["label"]==phase) & (iso_df["metallicity"]==metallicity) & (iso_df["logAge"]==age)][y_values]
                        plt.plot(x_axis, y_axis, c=c_dict[phase])
                
            plt.xlim(x_lim[0], x_lim[1])
            if y_lim is not None:
                plt.ylim(y_lim[0], y_lim[1])
            if x_log:
                plt.xscale('log')            
            if y_log:
                plt.yscale('log')
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.legend(title="Phases", fontsize="small", 
                    handles = [mlines.Line2D([], [], color=c_dict[key], label=f"{phase_dict[key]}") for key in c_dict.keys()])
            plt.title(f"Metallicity = {metallicity}")
            plt.show()
    
    # TODO? deux fonctions en une avec un paramètre pour savoir si je prends le log_L ou le log_g
    def plot_HR(self, ages : list[float], metallicities : list[float], iso_df : pd.DataFrame=None, physical_model : str=None, x_lim : tuple[float]=None, y_lim : tuple[float]=None):
        """
        Plots the HR diagram for the given ages and metallicities, using log_Teff for the x-axis and log_L for the y-axis. The plot is colored according to the different evolutionary phases of the stars.

        Parameters:
            ages (list[float]): A list of ages (in log years) for which to plot the isochrones. If empty, all ages in the dataframe will be plotted.
            metallicities (list[float]): A list of metallicities for which to plot the isochrones. If empty, all metallicities in the dataframe will be plotted.
            iso_df (pd.DataFrame, optional): The isochrone dataframe to use for plotting. If None, the dataframe provided during initialization will be used.
            physical_model (str, optional): The physical model to use for plotting. Should be either "MIST" or "PARSEC". If None, the model provided during initialization will be used.
            x_lim (tuple[float], optional): The limits for the x-axis. If None, the limits get preset values if not provided.
            y_lim (tuple[float], optional): The limits for the y-axis. If None, the limits get preset values if not provided.
        """
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
        if x_lim is None:
            x_lim = (5.7, 3.3)
        
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
                
            plt.xlim(x_lim[0], x_lim[1])
            if y_lim is not None:
                plt.ylim(y_lim[0], y_lim[1])
            plt.xlabel("$\log(T_{\mathrm{eff}}) [\mathrm{K}]$")
            plt.ylabel("$\log(L/L_{\odot})$")
            plt.legend(title="Phases", fontsize="small", loc='lower left',
                    handles = [mlines.Line2D([], [], color=c_dict[key], label=f"{phase_dict[key]}") for key in c_dict.keys()])
            if len(ages) == 1:
                plt.title(f"Metallicity = {metallicity}, $\log(Age)$ = {ages[0]}")
            else:
                plt.title(f"Metallicity = {metallicity}")
            plt.show()

    def plot_Kiel(self, ages : list[float], metallicities : list[float], iso_df : pd.DataFrame=None, physical_model : str=None, x_lim : tuple[float]=None, y_lim : tuple[float]=None):
        """
        Plots the Kiel diagram for the given ages and metallicities, using log_Teff for the x-axis and log_L for the y-axis. The plot is colored according to the different evolutionary phases of the stars.

        Parameters:
            ages (list[float]): A list of ages (in log years) for which to plot the isochrones. If empty, all ages in the dataframe will be plotted.
            metallicities (list[float]): A list of metallicities for which to plot the isochrones. If empty, all metallicities in the dataframe will be plotted.
            iso_df (pd.DataFrame, optional): The isochrone dataframe to use for plotting. If None, the dataframe provided during initialization will be used.
            physical_model (str, optional): The physical model to use for plotting. Should be either "MIST" or "PARSEC". If None, the model provided during initialization will be used.
            x_lim (tuple[float], optional): The limits for the x-axis. If None, the limits get preset values if not provided.
            y_lim (tuple[float], optional): The limits for the y-axis. If None, the limits get preset values if not provided.
        """
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
        if x_lim is None:
            x_lim = (5.7, 3.3)
        
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
                
            plt.xlim(x_lim[0], x_lim[1])
            if y_lim is not None:
                plt.ylim(y_lim[0], y_lim[1])
            plt.gca().invert_yaxis()
            plt.xlabel("$\log(T_{\mathrm{eff}}) [\mathrm{K}]$")
            plt.ylabel("$\log(g) [\mathrm{cm/s^2}]$")
            plt.legend(title="Phases", fontsize="small", loc='upper left',
                    handles = [mlines.Line2D([], [], color=c_dict[key], label=f"{phase_dict[key]}") for key in c_dict.keys()])
            plt.title(f"Metallicity = {metallicity}")
            plt.show()

    def show_distribution(self, col_names : list[str]):
        """
        Plots the distribution of the values in the specified columns as histograms.

        Parameters:
            col_names (list[str]): A list of column names in the dataframe for which to plot the distributions.
        """
        for cols in col_names:
            distribution_plot = plt.figure(figsize=(6,6))
            plt.hist(self.iso_df[cols], bins=60)
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title(f'Distribution of the {cols}')
            plt.show()
        return

if __name__ == "__main__":
    pass