# -*- coding: utf-8 -*-
"""
Report analysing historical AQHI (Air Quality Health Index) produced by Hong Kong governement.

http://www.aqhi.gov.hk/en/aqhi/statistics-of-aqhi/past-aqhi-records.html

@ Author : David PICON
@ mail : davpicon@gmail.com

"""

import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
import logging
import plotly
import plotly.graph_objs as go
from plotly import tools
import numpy as np

# =============================================================================
# Input parameters to run the script
# =============================================================================
DEFAULT_PATH = os.path.join('~', 'Downloads')
ALLOWED_LOCATIONS = ["Central", "Tseung Kwan O", "Causeway Bay", "Eastern"]


# =============================================================================
# Class
# =============================================================================
class AirQualityFileReader():
    '''
    #--------------------------------------------------------------------------
    # Function used to read one file from HK Air Quality Website
    #--------------------------------------------------------------------------
    '''

    def __init__(self, source_path=DEFAULT_PATH, file_names=None, allowed_locations=None):

        self.source_path = source_path
        self.df = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging.INFO)
        self.files = file_names
        self.allowed_locations = allowed_locations

    '''
    #--------------------------------------------------------------------------
     Simple function performing a multiple file reading with concatenation
    #--------------------------------------------------------------------------
    '''
    def read_files(self):
        _ = []
        for _file in self.files :
            try:
                _.append(self.read_file(file_name=_file, inplace=False))
            except Exception as e:
                self.logger.error("Error reading file {}".format(_file))
                print(e)
        self.df = pd.concat(_, join='outer', axis=0)

    '''
    #--------------------------------------------------------------------------
    # Function used to read one file from HK Air Quality Website
    #--------------------------------------------------------------------------
    '''

    def read_file(self, file_name, source_path=None, inplace=True):

        # Check if we specified a source path
        if source_path is None:
            source_path = self.source_path

        self.logger.info("Reading file {} in folder {}".format(file_name, source_path))

        # Open the csv file
        df = pd.read_csv(os.path.join(source_path, file_name), skiprows=7)

        # Cl1 - Cleaning the date format
        for idx in df.index:
            # If the value is empty --> we need to assign the correct date which is the last str item in the column
            if pd.isnull(df.loc[idx, "Date"]):
                df.loc[idx, "Date"] = _currDate
            elif type(df.loc[idx, "Date"]) is str:
                _currDate = df.loc[idx, "Date"]
            else:
                raise("Error : index in dataframe no handled")

        df = df[df["Hour"] != "Daily Max"]
        df["Date"] = pd.to_datetime(df["Date"])
        df["Hour"] = pd.to_numeric(df["Hour"])
        df["Date"] = df[["Date", "Hour"]].apply(lambda x: x["Date"] + datetime.timedelta(hours=x["Hour"]), axis=1)
        df = df.drop(["Hour"], axis=1)
        df = df.set_index("Date")

        # Cl2 - Cleaning the * measures
        df = df.fillna(0)
        for col in df.columns:
            df[col] = df[col].astype(str)
            for x in range(0,11,1):
                df[col] = df[col].replace("{}*".format(x),str(x))
            df[col] = df[col].replace("10+","11")
            df[col] = df[col].replace("10+*", "11")
            df[col] = df[col].replace("*","0")
            df[col] = pd.to_numeric(df[col])


        # Cl3 - If we have specified a list of locations
        if self.allowed_locations is not None:
            df = df.drop([x for x in df.columns if x not in self.locations], axis=1)

        if inplace:
            self.df = df

        return df

    ''' 
    ------------------------------------------------------------
     Function returning a graph on HTML format
    ------------------------------------------------------------
    '''
    def plot(self, use_plotly=True, use_matplotlib=True, period="hourly", display_min_max=(False, False)):

        # First we need to work on a copy of our core dataFrame
        _df = self.df.copy()

        # By default we stays in hourly mode (following self.df structure), but other options are available
        if period != "hourly" :
            _df = _df.reset_index()
            if period == "daily":
                _df["Date"] = _df["Date"].apply(lambda x: datetime.datetime(x.year, x.month, x.day))
            elif period == "monthly":
                _df["Date"] = _df["Date"].apply(lambda x: datetime.datetime(x.year, x.month, 1))
            elif period == "bi-monthly":
                _df["Date"] = _df["Date"].apply(lambda x: datetime.datetime(x.year, ((x.month-1)//2)+1, 1))
            elif period == "quarterly":
                _df["Date"] = _df["Date"].apply(lambda x: datetime.datetime(x.year, ((x.month-1)//3)+1, 1))
            elif period == "semi-annually":
                _df["Date"] = _df["Date"].apply(lambda x: datetime.datetime(x.year, (1 if x.month < 6 else 6),1))
            elif period == "yearly":
                _df["Date"] = _df["Date"].apply(lambda x: datetime.datetime(x.year, 1,1))
            #_df_min = _df.groupby("Date").min(skipna=True)
            #_df_max = _df.groupby("Date").max(skipna=True)
            _df = _df.groupby("Date").mean()

        # Adding Hong Kong average
        _df["Hong Kong (Avg)"] = _df.apply(lambda x: np.nanmean(x), axis=1)

        # Optionally adding min/max obtained from daily results
        if display_min_max[0]:
            #_df["Hong Kong (Min)"] = _df_min.apply(lambda x: np.nanmean(x), axis=1)
            _df["Hong Kong (Min)"] = _df.apply(lambda x: np.nanmin(x), axis=1)
        if display_min_max[1]:
            #_df["Hong Kong (Max)"] = _df_max.apply(lambda x: np.nanmean(x), axis=1)
            _df["Hong Kong (Max)"] = _df.apply(lambda x: np.nanmax(x), axis=1)

        # Graphing output with plotly
        if use_plotly:

            fig = tools.make_subplots(1,2, column_width=[0.15, 0.85])

            # Plotly representation
            for col in _df.columns:
                fig.add_trace(go.Scatter(x=_df.index,
                                         y=_df[col],
                                         name=col,
                                         ), row=1, col=2
                              )

            _df = _df.mean(skipna=True).sort_values()

            # Adding a bart chart with classified items
            fig.add_trace(go.Bar(y=_df.index, x=_df.values, orientation='h',
                                 name="Ranking", legendgroup='group2',
                                 text=_df.index, textposition='auto',
                                 ),
                          row=1,col=1)

            fig.layout.update({"title" : "AQHI - Hong Kong Air quality evolution - [Jan2014-Jul2019] ({})".format(period)})
            plotly.offline.plot(fig, filename='output/AQHI - Air quality history {}.html'.format(period),)



        if use_matplotlib:

            # Matplotlib representation
            fig = plt.Figure(figsize=(10, 5))
            for col in _df.columns:
                plt.bar(_df.index, _df[col], label=col)
            plt.legend()


# =============================================================================
#   Main part
# =============================================================================


if __name__ == "__main__":
    file_names = ["{}{:02d}_Eng.csv".format(y, m) for y in range(2014,2020,1) for m in range(1,13,1)]
    #file_names = file_names[-10:]
    f = AirQualityFileReader(file_names=file_names, source_path=os.path.join(os.path.dirname(__file__),"data"))
    #f.read_file("201902_Eng.csv")
    f.read_files()

    # Graph the various frequencies
    f.plot(use_plotly=True, use_matplotlib=False, period="monthly", display_min_max=(True, True))
    f.plot(use_plotly=True, use_matplotlib=False, period="daily", display_min_max=(True, True))
    f.plot(use_plotly=True, use_matplotlib=False, period="bi-monthly", display_min_max=(True, True))
    f.plot(use_plotly=True, use_matplotlib=False, period="quarterly", display_min_max=(True, True))
    f.plot(use_plotly=True, use_matplotlib=False, period="semi-annually", display_min_max=(True, True))
    f.plot(use_plotly=True, use_matplotlib=False, period="yearly", display_min_max=(True, True))
