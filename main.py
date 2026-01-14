from test_case import *

if __name__ == "__main__":
   
    choix = 1 

    if choix == 1 : 
        # Validation 1D rapide
        validation_DF()
    else : 
        # Validation de l'équation adjointe
        test_inverse_source_localization()


   