import os
import logging
import json

import numpy as np

from find_agn import ezgal_wrapper, notebook_utils

class Template():

    def __init__(
        self,
        star_history,
        # current_z,
        # current_age,
        modelset='bc03',
        metallicity=0.05,
        imf='salp',
    ):

        self.star_history = star_history
        # self.current_z = current_z
        # check type of current age
        # self.current_age = current_age
        self.modelset = modelset
        self.metallicity = metallicity
        self.imf = imf

        # self.formation_z = notebook_utils.formation_redshift_for_fixed_age(
        #     self.current_z,
        #     self.current_age)

        self.calculated = False

    
    def __repr__(self):
        return 'EzGal {} model Model with metallicity {}, imf {}, custom star history'.format(
            self.modelset, 
            self.metallicity,
            self.imf
        )





    def calculate(self, model_dir):
        """Calculate galaxy template based on desired properties e.g. star history, metallicity, etc
        
        Args:
            model_dir (str): directory containing bc03 models
        """
        try:
            assert self.modelset == 'bc03'
        except AssertionError:
            logging.critical('Template currently only supports bc03 calculation')
            raise NotImplementedError

        model_loc = os.path.join(
            model_dir, 
            ezgal_wrapper.bc03_model_name(
                metallicity=self.metallicity, 
                imf=self.imf)
        )
        # delegate calculation of template to ezgal
        self.model = ezgal_wrapper.get_model(model_loc, self.star_history)
        self.calculated = True


    def save(save_dir):
        json.dump()
        template.model.save_model(filter_info=True)


# def save_template(self, save_dir):
#     """Save galaxy and continuum templates from EZGAL model at defined formation_z, current_z.
#     Model SF history MUST match formation_z, current_z. TODO: enforce consistency

#     Args:
#         model (ezgal.ezgal.ezgal): EZGAL model stellar population of 1 Msun
#         save_dir (str): path to directory into which to save templates
#         formation_z (float): formation redshift of template. Must match model SF history.
#         current_z (float): current redshift of template. Must match model SF history
#         mass (float): mass of template to save. Default: 1. Later rescaling of galaxy is trivial.
#     """
#     if not os.path.isdir(save_dir):
#         os.mkdir(save_dir)

#     galaxy_loc, sed_loc = get_saved_template_locs(
#         save_dir, 
#         self.formation_z, 
#         self.current_z, 
#         mass=1.)  # template always has normalised mass?

#     with open(galaxy_loc, 'w') as f:
#         json.dump(self.fake_galaxy.to_dict(), f)

#     # cannot serialise np array, only simple lists
#     for key, data in self.sed.items():
#         if isinstance(data, np.ndarray):
#             self.sed[key] = list(data)
#     with open(sed_loc, 'w') as f:
#         json.dump(self.sed, f)



# def save_model(model_loc, star_history, model_params):
#     model = ezgal_wrapper.get_model(model_loc, dual_burst)
#     mass = 1.  # do not rescale, leave as 1 Msun stellar population
#     # TODO refactor magnitude mass rescaling as separate notebook_utils method
#     ezgal_wrapper.save_template(model, SAVED_TEMPLATE_DIR, formation_z, current_z, mass)

# # new_model = True
# # if new_model:
# #     save_model()



# def load_saved_template(save_dir, formation_z, current_z, mass):
#     """Load saved galaxy and continuum templates from EZGAL model at defined formation_z, current_z.

#     Args:
#         model (ezgal.ezgal.ezgal): EZGAL model stellar population of 1 Msun
#         save_dir (str): path to directory from which to load templates
#         formation_z (float): formation redshift of template. Must match model SF history.
#         current_z (float): current redshift of template. Must match model SF history
#         mass (float): mass of template to save. Default: 1. Later rescaling of galaxy is trivial.

#     Return:
#         pd.Series: template galaxy in standard format
#         dict: of form {frequency, energy at frequency} for template continuum
#     """
#     galaxy_loc, sed_loc = get_saved_template_locs(save_dir, formation_z, current_z, mass)

#     with open(galaxy_loc, 'r') as galaxy_f:
#         galaxy = json.load(galaxy_f)

#     with open(sed_loc, 'r') as continuum_f:
#         continuum = json.load(continuum_f)

#     return galaxy, continuum


# def get_saved_template_locs(save_dir, formation_z, current_z, mass):
#     param_string = 'fz_{:.3}_cz_{:.3}_m_{}'.format(formation_z, current_z, mass)
#     galaxy_loc = os.path.join(save_dir, param_string + '_galaxy.txt')
#     sed_loc = os.path.join(save_dir, param_string + '_sed.txt')
#     return galaxy_loc, sed_loc
