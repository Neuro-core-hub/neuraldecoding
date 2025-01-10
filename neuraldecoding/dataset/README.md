# code-style to remember

- function names are lower case with _ (e.g. function_name)
- every function need the comment defining the scope of the function, the inputs, and the ouputs. For each input/output variable if possible define the type and an intelligible description. Like the following:
```bash
      def function_name(var_name1, var_name2, ..):
          """
          This function is going to do this and that...
    
          Inputs:
              var_name1: str
                  Variable description
              var_name1: int
                  Variable description
              ....
          Outputs:
              var_name1: str
                  Variable description
              var_name1: int
                  Variable description
              ....
          """
```


# implementation details that needs to be discussed

- is the storing/saving of each run NWB file automatic or should it be done only if requested?

# To-do list dataset package

- [ ] when loading a run of a dataset, add a check if for that run the NWB file was already created and stored in the server, if yes just load it instead of re-creating it
- [ ] definition for the __str__ method for the dataset representation: it should return a meaningfull string representation of the content of the dataset (like the type of data, list of runs, etc..)
- [ ] writing the code for pre_processing function: this function takes as input a dictionary of params. The params indicate which pre_processing steps needs to be applied and the values (i.e. CAR, BP, notch)
- [ ] writing the code for extracting_features function. This function extract the features and take as input the features to extract and params; params contains bin_size, behav_lag, overall_lag, overlap (take a look into the pybmi getZfeats for this).
- [ ] writing the code for synching data functions. Basically we need to convert the functions from the Matlab package (svn_repository/Utility code/SynchZNeural) (needs a bit of understanding of why the definition of synching cpd and cerebus needs to be different)
- [ ] completing the README.md with the inclusion of a package description, list of functions, and references to examples
- [ ] adding examples for the different functions of the dataset package and the different ways of using them: from loading a single day data, to data normalizationa, features extraction, to creating a training/test dataset, etc..
