from test_case import *

if __name__ == "__main__":
   
    choix = 8

    if choix == 1 : 
        # Validation 1D rapide
        validation_DF()
    elif choix ==2 : 
        # Validation de l'équation adjointe
        #test_inverse_source_localization()
        test_inverse_one_source_vectorial()
    elif choix ==3 :
        test_inverse_source_localization_two_sources_complete()
    elif choix==4:
        test_inverse_three_sources_vectorial()
    elif choix ==5 :
        test_inverse_complex_five_sources()
    elif choix ==6 :
        test_inverse_extreme_ten_sources()
    elif choix ==7 :
        test_inverse_complex_pipeline_vectorial()
    else :
        test_inverse_realistic_pipeline_vectorial()