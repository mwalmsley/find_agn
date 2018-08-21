import numpy as np
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('notebook')
from astropy.cosmology import WMAP9 as cosmo

import holoviews as hv
hv.extension('bokeh')
from holoviews.operation.datashader import datashade, dynspread
import datashader as ds
import datashader.transfer_functions as tf
from colorcet import fire
from matplotlib import cm
viridis = cm.get_cmap('viridis')


def bpt_kauffmann(x):  # x is log NII HA
    y = 0.61 / (x - 0.05) + 1.3
    y[x > - 0.2] = -2.  # mask anything near the asympote, but use a real value so that galaxies can be 'above and to the right'
    return y


def bpt_kewley(x):  # x is log NII HA
    y = 0.61 / (x - 0.47) + 1.19
    y[x > 0.25] = -1.9
    return y


def normalize_to_hbeta(df):
    lines = [
        'OIII_5006',
        'HA_6562',
        'NII_6583',
        'SII_6716',
        'OI_6300'
    ]
    for line in lines:
        df[line] = df[line] / df['HB_4861']
    df['HB_4861'] = 1
    df = add_bpt_parameters(df)
    return df


def add_bpt_parameters(df):
    # works on both series and dataframe
    df['log_OIII_HB'] = np.log10(df['OIII_5006'] / df['HB_4861'])
    df['log_NII_HA'] = np.log10(df['NII_6583'] / df['HA_6562'])
    df['log_SII_HA'] = np.log10(df['SII_6716'] / df['HA_6562'])
    df['log_OI_HA'] = np.log10(df['OI_6300'] / df['HA_6562'])
    return df




# which galaxies are starforming?
def is_starforming(data):  # Below Ka03
    # basic version with no SN consideration, works on rows
    return data['log_OIII_HB'] < bpt_kauffmann(data['log_NII_HA'])  # diagram 1 only for now


# which galaxies are definitely AGN? For the sake of argument, include liner
def is_agn(data):  # Above Ke01
    if 'is_starforming' in data.columns.values:
        starforming = data['is_starforming']
    else:
        starforming = is_starforming(data)
    return (data['log_OIII_HB'] > bpt_kewley(data['log_NII_HA'])) & ~starforming  # must be above BOTH lines


def liner_if_agn(data):
    return data['log_OIII_HB'] < 1.89 * data['log_SII_HA'] + 0.76  # S2 diagnostic only for now


def is_liner(data):
    if 'is_agn' in set(data.columns.values):
        return data['is_agn'] & liner_if_agn(data)
    else:
        return is_agn(data) & liner_if_agn(data)
    

def is_seyfert(data):
    if ('is_agn', 'is_liner') in set(data.columns.values):
        return ~data['is_liner'] & data['is_agn']
    else:
        return ~is_liner(data) & is_agn(data)


# which galaxies are in-between?
def is_composite(data):
    # if columns already calculated, return
    if {'is_agn', 'is_starforming'} in set(data.columns.values):
        return ~data['is_agn'] & ~data['is_starforming']
    else:
        return ~is_starforming(data) & ~is_agn(data)
    


def get_galaxy_types(data):
    data['starforming'] = is_starforming(data)
    data['agn'] = is_agn(data)  # intermediate type, includes liner and seyfert
    data['composite'] = is_composite(data)
        
    data['liner'] = is_liner(data)
    data['seyfert'] = is_seyfert(data)
    del data['agn']  # to avoid confusion
    
    # check every galaxy is exactly one type
    type_columns = ['starforming', 'composite', 'liner', 'seyfert']
    assert all(np.sum(data[type_columns], axis=1) == 1)
    
    # can probably do more neatly with pandas
    data['galaxy_type'] = list(map(lambda x: type_columns[x], np.argmax(data[type_columns].values, axis=1)))

    return data




def zoom_bpt_diagram(df):
    return df[
    (df['log_NII_HA'] > -1.5) &
    (df['log_NII_HA'] < 0.5) &
    (df['log_OIII_HB'] > -1.2) &
    (df['log_OIII_HB'] < 1.2)
]


def all_permutations(*vectors):
    sample_vectors = np.meshgrid(*vectors)
    # flat_sample_vectors = [v.flatten() for v in sample_vectors]
    grouped_sample_vectors = np.stack(sample_vectors, axis=-1)
    return grouped_sample_vectors


def bpt_sample_grid(n_samples):
    samples_per_dim = int(np.sqrt(n_samples)) 
    axes = {
        'log_NII_HA': np.linspace(-1.5, 0.5, samples_per_dim),
        'log_OIII_HB': np.linspace(-1.2, 1.2, samples_per_dim)
    }
    x_samples, y_samples = np.meshgrid(axes['log_NII_HA'], axes['log_OIII_HB'])
    flat_x, flat_y = x_samples.flatten(), y_samples.flatten()
    paired_samples = np.array(list(zip(flat_x, flat_y)))
    return axes, (x_samples, y_samples), paired_samples


def show_bpt_diagram_static(df, ax, shade=False, cmap=None, legend=False):
    """Visualise BPT diagram static rendering (matplotlib), scaled with datashader
    
    Args:
        bpt_df ([type]): [description]
        legend (bool, optional): Defaults to False. [description]
    """
    df_zoomed = zoom_bpt_diagram(df)
    sns.kdeplot(data=df_zoomed['log_NII_HA'], data2=df_zoomed['log_OIII_HB'], shade=shade, cmap=cmap, ax=ax)

    x = np.linspace(-1.5, 0.5, 1000)
    bpt_curves = pd.DataFrame(data={'log_NII_HA': x, 'Kewley': bpt_kewley(x), 'Kauffmann': bpt_kauffmann(x)})
    
    ax.plot(bpt_curves['log_NII_HA'], bpt_curves['Kewley'], 'k-', linewidth=2.0, label='Kewley 2001 (Max)')
    ax.plot(bpt_curves['log_NII_HA'], bpt_curves['Kauffmann'], 'k--', linewidth=2.0, label='Kauffmann 2003')
    
    ax.set_xlim((-1.5, 0.5))
    ax.set_ylim((-1.2, 1.2))
    
    ax.set_xlabel('log ([NII] 6583 / Ha)')
    ax.set_ylabel('log ([OIII] 5006 / Hb')

    return ax


def show_bpt_diagram(bpt_df, legend=False):
    
    from datashader.colors import Sets1to3 # default datashade() and shade() color cycle
    
    galaxy_types = bpt_df['galaxy_type'].unique()
    
    points_dict = {}
    for n, galaxy_type in enumerate(galaxy_types):
        sample = bpt_df[bpt_df['galaxy_type'] == galaxy_type]
        sample = zoom_bpt_diagram(sample)
        points_dict[n] = hv.Points(sample, ['log_NII_HA', 'log_OIII_HB'])

    points = dynspread(datashade(hv.NdOverlay(points_dict, kdims='galaxy_type'), aggregator=ds.count_cat('galaxy_type')))

    x = np.linspace(-1.5, 0.5, 1000)
    bpt_curves = pd.DataFrame(data={'log_NII_HA': x, 'Kewley': bpt_kewley(x), 'Kauffmann': bpt_kauffmann(x)})
    bpt_curves_ds = hv.Dataset(bpt_curves, kdims='log_NII_HA', vdims=['Kewley', 'Kauffmann'])
    
    kewley = hv.Curve(bpt_curves_ds, 'log_NII_HA', 'Kewley').options(line_width=1.0)
    kewley = kewley.redim.label(Kewley='bpt_y')  # align labels

    kauffmann = hv.Curve(bpt_curves_ds, 'log_NII_HA', 'Kauffmann').options(line_width=1.0)
    kauffmann = kauffmann.redim.label(Kauffmann='bpt_y')  # align labels
    
    if legend:
        kewley = kewley.relabel('Kewley 2001 (Max)')
        kauffmann = kauffmann.relabel('Kauffmann 2003')
    
#     plot = datashade(points, cmap=viridis) * kewley
    plot = points * kewley * kauffmann
    plot = plot.redim.range(bpt_x=(-1.5, 0.5), bpt_y=(-1.2, 1.2))
    
    # finally, relabel axis
    plot = plot.redim.label(log_NII_HA='log ([NII] 6583 / Ha)')
    plot = plot.redim.label(log_OIII_HB='log ([OIII] 5006 / Hb')
    
    return plot

