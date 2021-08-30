import os

# Helper function to create of list of all files joined with their directories
def listdir_fullpath(dir):
    #init variable
    Full_paths = []
    #create list of files in dir
    dirlist = os.listdir(dir)
    #For each item in listdir, join it with its directory
    for file in dirlist:
        fullpath = (os.path.join(dir,file))
        Full_paths.append(fullpath)
    return Full_paths
