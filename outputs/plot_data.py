import matplotlib.pyplot as plt
import json

import os
import argparse

import pandas as pd

import ast

class Plot_Graph:

    def __init__(self):

        # get the folder containing the data files
        prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                      description="""Diamond Net Data Plotter""")
        prs.add_argument("-folder", dest="folder", type=str, help="Folder containing all the data files.\n")

        folder = prs.parse_args().folder

        # the files that contain only one value per column
        self.single_valued = ['pressure.csv', 'total_queued.csv']

        # list all the files in the folder
        self.all_files =  [ f'{folder}/{file}' for file in os.listdir(folder) if file[-4:] == '.csv']


    def make_plots(self):
        for file in self.all_files:
            # opens the csv
            df = self._open_csvfile( file )

            file_name = file.split('/')[-1]

            # calls the method employed of doing the plot
            if file_name in self.single_valued:
                self.plot_single_valued( df, file, True )
            else:
                self.plot_more_valued( df, file )


    def _open_csvfile(self, file):
        df = pd.read_csv( file )
        return df


    def plot_single_valued(self, df, file_name, every):
        count = df.count()
        if not all( count[0] == c for c in count ):
            raise Exception( 'Some column(s) has(ve) different length' )

        y = df.columns

        df['steps'] = range( count[0] )

        if every:
            for col in df.columns[:-1]:
                df.plot( kind='line', x='steps', y=col )
                os.makedirs( f'{file_name[:-4]}',exist_ok=True )
                plt.savefig( f'{file_name[:-4]}/{col}_{file_name.split("/")[-1][:-4]}.png', dpi=500 )
                plt.close()

        df.plot( kind='line', x='steps', y=y, lw=1 )
        plt.title(file_name.split("/")[-1][:-4])
        # plt.subplots(figsize=(8, 6))
        plt.savefig( f'{file_name[:-4]}_plot.png', dpi=500 )
        plt.close()

    def plot_more_valued(self, df, file_name):

        for col in df.columns:

            lanes_data = []

            for arr in df[col]:
                new_arr = ast.literal_eval( arr )
                lanes_data.append( new_arr )
            lanes_df = pd.DataFrame(lanes_data)

            dirname = os.path.dirname( file_name )
            n_file_name = f'{dirname}/{file_name.split("/")[-1][:-4]}/{file_name.split("/")[-1][:-4]}_{col}{file_name.split("/")[-1][-4:]}'
            os.makedirs( os.path.dirname( f'{dirname}/{file_name.split("/")[-1][:-4]}/{col}{file_name.split("/")[-1][-4:]}' ) , exist_ok=True )

            self.plot_single_valued( lanes_df, n_file_name, False )




if __name__ == '__main__':
    Plot_Graph().make_plots()
