import os
current_working_directory = os.getcwd()
home_name = current_working_directory.split('/')[2]

Project_directory = str(current_working_directory)
if (home_name=='lveithen'):
    Project_directory = "/scratch/lveithen/SourceCode_25_06_2024"
    tudat_path = r"/home/lveithen/tudat-bundle/build/tudatpy"
elif (home_name=='lorenz_veithen'):
    Project_directory = "/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python"
    tudat_path = r"/Users/lorenz_veithen/tudat-bundle/build/tudatpy"
else:
    raise Exception('does not know which system is being used')


AMS_directory = Project_directory + "/0_GeneratedData/AMS_Data"

c_sol = 299792458   # [m/s]
W = 1400    # [W/m^2] - roughly
R_E = 6371e3    # [m]
acc0 = 0.045 * 1E-3   # [m/s/s] characteristic sail acceleration - original ACS3 value

# Constant with no impact on the simulation really, therefore kept here
default_ellipse_bounding_box_margin = 2