import sqlite3

import pandas as pd

from find_agn import star_formation, get_template_grid, notebook_utils


def make_new_grid(metallicities, star_history, model_dir, save_dir, db_loc):

    db = sqlite3.connect(db_loc)
    cursor = db.cursor()
    cursor.execute('DROP TABLE dual_burst_models')
    cursor.execute(
        '''
        CREATE TABLE
        dual_burst_models(
                    id INTEGER PRIMARY KEY,
                    model_loc TEXT,
                    modelset TEXT,
                    metallicity FLOAT,
                    imf FLOAT,
                    current_z FLOAT,
                    formation_z FLOAT, 
                    first_duration FLOAT, 
                    second_duration FLOAT)
                        ''')
    db.commit()

    get_template_grid.make_metallicity_grid(
        metallicities, 
        star_history,
        db, 
        model_dir,
        save_dir
    )

    db.close()
        

if __name__ == '__main__':

    model_dir = 'population_models'
    save_dir = 'template_database'
    db_loc = 'template_database/templates.db'
    metallicities = [0.008, 0.02, 0.05]

    current_z = 0.1  # let's just assume and ignore for now
    formation_z = 1.5  # let's just assume and ignore for now
    first_duration = 1.  # Gyr
    second_duration = 1.  # Gyr
    star_history = star_formation.DualBurstHistory(
        current_z=current_z,
        formation_z=formation_z,
        first_duration=first_duration,
        second_duration=second_duration)

    new_grid = False
    if new_grid:
        make_new_grid(metallicities, star_history, model_dir, save_dir, db_loc)

    db = sqlite3.connect(db_loc)
    templates = get_template_grid.load_metallicity_grid(metallicities, star_history, db)

    all_bpt_lines = []

    visualise = False
    redshifts = [0.05, 0.5, 1.]
    for template in templates:
        for redshift in redshifts:

            continuum = template.get_continuum(redshift, normalisation=1e24)

            continuum.get_bpt_lines()
            lines = continuum.measured_bpt_lines()
            lines['redshift'] = redshift
            lines['metallicity'] = template.metallicity
            all_bpt_lines.append(lines)
        

            if visualise:
                fig_name = '{}_{}'.format(redshift, template.metallicity)

                fig, _ = star_formation.visualise_model_sfh(
                    template.model, 
                    fig_name,
                    template.formation_z)
                fig.savefig('temp/history/' + fig_name + '_history.png')

                fig, _ = continuum.visualise_spectrum()
                fig.savefig('temp/continuum/' + fig_name + '_continuum.png')

                fig = continuum.visualise_fit()
                fig.savefig('temp/fit/' + fig_name + '_fit.png')

    df = pd.DataFrame(data=all_bpt_lines)
    df = notebook_utils.add_bpt_parameters(df)
    df.to_csv('temp/bpt_lines.csv', index=False)
