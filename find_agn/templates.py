import os
import time
import logging
import json

import numpy as np
import easyGalaxy

from find_agn import ezgal_wrapper, notebook_utils, continuums


class Template():

    def __init__(
            self,
            star_history,
            # current_z,
            # current_age,
            modelset='bc03',
            metallicity=0.02,
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
        # EZGAl requires a formation redshift in order to save a calculated model
        # Accepts multiple redshifts if desired
        self.formation_z = 1.5  # TODO temporary

        self.calculated = False
        self.model = None


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
            logging.critical(
                'Template currently only supports bc03 calculation')
            raise NotImplementedError

        model_loc = os.path.join(
            model_dir,
            ezgal_wrapper.bc03_model_name(
                metallicity=self.metallicity,
                imf=self.imf)
        )
        # delegate calculation of template to ezgal
        self.model = ezgal_wrapper.get_model(
            model_loc, 
            self.star_history.history,
            self.formation_z
            )

        self.calculated = True

    def save(self, db, save_dir):

        assert self.calculated
        assert self.model  # exists
        current_timestamp = int(time.time() * 1000000)  # in nanoseconds
        model_loc = os.path.join(save_dir, str(current_timestamp) + '.model')

        self.model.save_model(model_loc, filter_info=True)

        cursor = db.cursor()
        cursor.execute(
            '''
            INSERT INTO dual_burst_models(
                id,
                model_loc,
                modelset,
                metallicity,
                imf,
                current_z,
                formation_z,
                first_duration,
                second_duration
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                current_timestamp,
                model_loc,
                self.modelset,
                self.metallicity,
                self.imf,
                self.star_history.current_z,
                self.star_history.formation_z,
                self.star_history.first_duration,
                self.star_history.second_duration
            )
        )
        db.commit()


    def load(self, db):
        cursor = db.cursor()
        cursor.execute(  # TODO also check for star history matches
            '''
            SELECT model_loc FROM dual_burst_models 
            WHERE modelset=?;
            ''', 
            (self.modelset,)
        )
        matching_model = cursor.fetchone()
        model_loc = matching_model[0]  
        self.model = easyGalaxy.ezgal.model(model_loc)
        self.calculated = True


            # '''
            # SELECT model_loc FROM dual_burst_models 
            # WHERE modelset=? AND metallicity=? AND imf=?
            # ''', 

        
    def get_continuum(self, z, normalisation=1.):
        """[summary]
        
        Args:
            galaxy (dict): with calculated ezgal model, formation_z, z, 
            observed (bool, optional): Defaults to False. If True, return continuum in observed frame.
        
        Returns:
            [type]: [description]
        """
        freq, sed = self.model.get_sed_z(  # flux density Fv (energy/area/time/photon for each frequency v)
            self.formation_z,
            z,
            units='Fv',  # in Jy (I think) i.e. 10^-23 erg / s / cm^2 / Hz
            observed=False,
            return_frequencies=True
            )

        # TODO disable normalisation - should operate only on template for now, and only with mass later
        return continuums.Continuum(freq, sed * normalisation)


