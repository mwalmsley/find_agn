from find_agn import templates


# TODO extend to make grids with more dimensions of model assumptions
def make_metallicity_grid(metallicities, star_history, db, model_dir, save_dir):
    """Make templates over a range of metallicities, save to disk, record in database

    Args:
        star_history (star_formation.DualBurstHistory): star history of templates
        metallicities (list): Calculate models with these metallicities
        db ([type]): sqlite database with dual_burst_models table
        model_dir (str): directory with ezgal stellar population models
        save_dir (str): directory into which to save new templates
    """
    for metallicity in metallicities:
        template_with_met = templates.Template(metallicity=metallicity, star_history=star_history)
        template_with_met.calculate(model_dir)
        template_with_met.save(db, save_dir)


def load_metallicity_grid(metallicities, star_history, db):
    template_grid = []
    for metallicity in metallicities:
        template_with_met = templates.Template(metallicity=metallicity, star_history=star_history)
        template_with_met.load(db)
        template_grid.append(template_with_met)
    return template_grid
