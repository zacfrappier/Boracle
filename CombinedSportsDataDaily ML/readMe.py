""" 

------------------------      venv     ---------------------------------
       venv - home pc (the one i built)
       venv2 - laptop 

----------- ------------       Pip      --------------------------------
   upgrade pip
       python.exe -m pip install --upgrade pip

   --------------------     Environment Start/End     ------------------
to create environment
  1) go to folder containing project 
  2) run in terminal 
        python -m venv venv

 to start environment
   venv\Scripts\activate

close environment 
   deactivate

----------------------    install libraries    -----------------------
               run this in terminal for (3) libraries
               pip install pandas scikit-learn altair

-------------------    run file    -----------------------------------
 in terminal: python *filename  
 example: python KNN.py
       will create html file, 'run with live server'

-------------------    git add    ------------------------------------ 

add all files 
    git add .
add single file 
    git add <filename>
add descriptor to upload 
    git commit -m "super duper importants fancy pant code"
push (final step)
    git push
    
 -----------------------Error Handling--------------------------------

1)--ERROR-- incase error "cannot be loaded b/c running scripts is disabled on this system"
  
 1.1)solutionSolution- run (doesnt work for me)
       venv\Scripts\activate.bat

 1.2)Solution -change security settings (works)
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

2) --ERROR-- ModuleNotFoundError: No module named 'tabulate'
   
   2.1)solution run
           pip install tabulate

3)  --ERROR-- when running 'git add .' in git bash 
   
    msg in terminal ---

    error: 'CombinedSportsDataDaily ML/' does not have a commit checked out
    fatal: adding files failed

    3.1) Navigate into problematic folder: 
            cd <foldername>
         remove hidden .get folder:
            rm -rf .git

        Why was this .git file made? when adding folders to main branch, git will
        treat them as a new submodule and auto generate this file
       


"""