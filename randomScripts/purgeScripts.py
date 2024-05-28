# delete files with a certain naming
import os


def delete_file_under_condition(complete_dir, f):
    if f.count("9") > 6:
        os.remove(complete_dir + "/" + f)
        print("deleted")
    return True

main_directory = "/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/AMS"
for subdir1 in os.listdir(main_directory):
    if (subdir1[0] != "."):
        for subdir2 in os.listdir(main_directory + "/" + subdir1):
            if (subdir2[0] != "."):
                for subdir3 in os.listdir(main_directory + "/" + subdir1 + "/" + subdir2):
                    if (subdir3[0] != "."):
                        for dataset_or_plotFile in os.listdir(main_directory + "/" + subdir1 + "/" + subdir2 + "/" + subdir3):
                            if (subdir1 == "Datasets"):
                                print(dataset_or_plotFile)
                                delete_file_under_condition(main_directory + "/" + subdir1 + "/" + subdir2 + "/" + subdir3, dataset_or_plotFile)
                            else:
                                if (dataset_or_plotFile[0] != "."):
                                    for plotFile in os.listdir(main_directory + "/" + subdir1 + "/" + subdir2 + "/" + subdir3 + "/" + dataset_or_plotFile):
                                        print(plotFile)
                                        delete_file_under_condition(
                                            main_directory + "/" + subdir1 + "/" + subdir2 + "/" + subdir3 + "/" + dataset_or_plotFile,
                                            plotFile)

