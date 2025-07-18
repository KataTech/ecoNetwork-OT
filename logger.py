"""
Logger class for logging experiments. 
"""

import datetime 
import os 
import pickle

class Logger(): 

    def __init__(self, log_folder_path, experiment_name): 
        self.log_folder_path = log_folder_path
        self.experiment_name = experiment_name
        # we will use time as the identifier for this experiment
        self.time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.exp_folder = self.experiment_name + "_" + self.time
        self.exp_folder_path = os.path.join(self.log_folder_path, self.exp_folder) + "/"
    
    def create_log_folder(self): 
        """
        Create a log folder for this experiment. 
        """
        # if the folder already exists, do nothing
        if os.path.exists(self.exp_folder_path): 
            return
        # check that the log_folder exists. If not, create it.
        if not os.path.exists(self.log_folder_path): 
            os.makedirs(self.log_folder_path)
        # create the log folder for this experiment
        os.mkdir(self.exp_folder_path)

    def log_params(self, params): 
        """
        Log the parameters of the experiment. 
        """
        self.create_log_folder()
        # write the parameters to a file
        param_path = os.path.join(self.exp_folder_path, "params.txt")
        with open(param_path, 'w') as f: 
            for key, value in params.items(): 
                f.write(f"{key}: {value}\n")
        # save the parameters to a pickle file
        with open(os.path.join(self.exp_folder_path, "params.pkl"), 'wb') as f: 
            pickle.dump(params, f)

    def get_exp_folder_path(self): 
        """
        Return the path to the experiment folder. 
        """
        self.create_log_folder()
        return self.exp_folder_path
        
