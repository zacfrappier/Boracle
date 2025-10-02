# folder location  (old)
# cd "c:\Users\zacfr\OneDrive\Desktop\docus\SCHOOL\CSUN\Arcs - Boracle\postboarding\CombinedSportsDataDaily ML"

#----------------------------  venv ---------------------------------
#       venv - home pc (the one i built)
#       venv2 - laptop 

#----------- ------------       Pip      --------------------
#   upgrade pip
#       python.exe -m pip install --upgrade pip

#   ------------------------ Environment Start/End/-------------------------------
#to create environment
#  1) go to folder containing project 
#  2) run in terminal 
#        python -m venv venv

# to start environment
#   venv\Scripts\activate

#clsoe environment 
#   deactivate

#  ------------------------ install libraries---------------------------
#               run this in terminal for (3) libraries
#               pip install pandas scikit-learn altair

# -----------------------Error Handling------------------

#1)--ERROR-- incase error "cannot be loaded b/c running scripts is disabled on this system"
#  
# 1.1)solutionSolution- run (doesnt work for me)
#       venv\Scripts\activate.bat

# 1.2)Solution -change security settings (works)
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

#2) --ERROR-- ModuleNotFoundError: No module named 'tabulate'
#   2.1)solution run
#           pip install tabulate


# --------------------- run file --------------------------------
# in terminal: python *filename  
# example: python KNN.py
#       will create html file, 'run with live server'