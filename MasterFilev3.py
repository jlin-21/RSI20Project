#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from tqdm import tqdm
tqdm.pandas()
import matplotlib.colors
import scipy.stats
from ucsc_genomes_downloader import Genome
hg38 = Genome(assembly="hg38")
import os
import pysam
import argparse

parser = argparse.ArgumentParser(description='Process histograms, scatter plots and metaplots')

parser.add_argument('cell_type')
parser.add_argument('tabix_file')
parser.add_argument('fragments')

args = parser.parse_args()
cell_type = args.cell_type

tabix_file = pysam.TabixFile(args.tabix_file)

os.system('gunzip -c {} | bedtools intersect -sorted -c -a /home/John/JohnProject/reference/DHS_adjusted_6mer_bias_adjustedby_30_sorted_no_blacklist.unique.bed -b - > {}/index_cuts_{}_intersect.bed'.format(args.fragments, cell_type, cell_type))

from reference.tools import exp_profile, tabix_profile

dhs = pd.read_table("{}/index_cuts_{}_intersect.bed".format(cell_type, cell_type), names = 'dhs_chr adjusted_dhs_start adjusted_dhs_end index_cuts'.split(), header=None , low_memory=False)

dhs_bias = pd.read_table('/home/John/JohnProject/reference/DHS_with_footprints_and_biases_6mer_adjustedby30.txt.gz', low_memory = False)
size_footprint_frac = (dhs_bias.footprint_end - dhs_bias.footprint_start) / (dhs_bias.adjusted_dhs_end - dhs_bias.adjusted_dhs_start)
dhs_bias['adjusted_bias_frac'] = 0.9 * dhs_bias.footprint_bias_fraction + 0.1 * size_footprint_frac


dhs=pd.merge(dhs, dhs_bias, on = ['dhs_chr','adjusted_dhs_start', 'adjusted_dhs_end'])
dhs['log_cuts_per_base'] = np.arcsinh(dhs.index_cuts/(dhs.adjusted_dhs_end - dhs.adjusted_dhs_start))

dhs_unique = dhs.drop_duplicates(subset = ('dhs_chr', 'adjusted_dhs_start', 'adjusted_dhs_end', 'motif_cluster'))

threshold = 0.6
positive_footprints = dhs_unique.query('log_cuts_per_base >= @threshold')
negative_footprints = dhs_unique.query('log_cuts_per_base <= @threshold')
pos_enrich = pd.value_counts(positive_footprints.motif_cluster).to_frame().reset_index()
pos_enrich = pd.DataFrame(pos_enrich)
pos_enrich.columns = ['motif_cluster','pos_frequency']
neg_enrich = pd.value_counts(negative_footprints.motif_cluster).to_frame().reset_index()
neg_enrich = pd.DataFrame(neg_enrich)
neg_enrich.columns = ['motif_cluster','neg_frequency']
intersection = pd.merge(pos_enrich, neg_enrich, how='inner', on='motif_cluster')
intersection['pos_enrich'] = intersection.pos_frequency/(intersection.pos_frequency + intersection.neg_frequency)
intersection.sort_values(by = ['pos_enrich'], ascending=False, inplace=True)
enriched = intersection.head(10)
enriched.to_csv("{}_enriched_footprints.txt".format(cell_type))

def histogram_with_motif(motif):
    dhs_unique['has_cur_motif'] = (motif == dhs_unique.motif_cluster)
    dhs_grouped = dhs_unique.groupby('dhs_chr dhs_start dhs_end'.split())['has_cur_motif', 'log_cuts_per_base'].max()
    with_motif_vals = dhs_grouped[dhs_grouped.has_cur_motif > 0]
    return with_motif_vals.log_cuts_per_base
    
def histogram_without_motif(motif):
    dhs_unique['has_cur_motif'] = (motif == dhs_unique.motif_cluster)
    dhs_grouped = dhs_unique.groupby('dhs_chr dhs_start dhs_end'.split())['has_cur_motif', 'log_cuts_per_base'].max()
    without_motif_vals = dhs_grouped[dhs_grouped.has_cur_motif == 0]
    return without_motif_vals.log_cuts_per_base

pseudo_count = 10
def observed_transformed(observe):
    footprint_index_cuts2 = observe + pseudo_count
    footprint_index_cuts2_transformed = 2*np.sqrt(footprint_index_cuts2 + 3/8)
    current_footprints['transformed_observed'] = footprint_index_cuts2_transformed
    return footprint_index_cuts2_transformed

def expected_transformed(expect):
    footprint_index_expected_footprint_cuts = expect + pseudo_count
    footprint_index_expected_footprint_cuts_transformed = 2*np.sqrt(footprint_index_expected_footprint_cuts + 3/8)
    current_footprints['transformed_expected'] = footprint_index_expected_footprint_cuts_transformed
    return footprint_index_expected_footprint_cuts_transformed

def calc_p_value(observed, expected):
    z_scores = observed - expected 
    current_footprints['z_scores'] = z_scores
    current_footprints_unique = current_footprints.sort_values("z_scores").groupby("footprint_chr footprint_start footprint_end".split()).first()
    p_values = scipy.stats.norm.sf(-current_footprints_unique.z_scores)
    current_footprints_unique['p_values_correction'] = p_values * len(current_footprints_unique)
    log_p_value = current_footprints_unique['log_p_value'] = -np.log10(current_footprints_unique.p_values_correction)
    current_footprints_unique = pd.DataFrame.reset_index(current_footprints_unique)
    return current_footprints_unique


def metaplot(protected):
    expected_profiles = []
    observed_profiles = []
    for idx, row in tqdm(protected.iterrows(), total=len(protected)):
        dhs_observed_profiles = tabix_profile(row.dhs_chr, row.adjusted_dhs_start, row.adjusted_dhs_end, tabix_file)
        dhs_expected_profile = exp_profile(row.dhs_chr, row.adjusted_dhs_start, row.adjusted_dhs_end, dhs_observed_profiles.sum())
        footprint_observed = dhs_observed_profiles[row.footprint_start - row.adjusted_dhs_start-10:row.footprint_end - row.adjusted_dhs_start+10]
        footprint_expected = dhs_expected_profile[row.footprint_start - row.adjusted_dhs_start-10:row.footprint_end - row.adjusted_dhs_start+10]
        if row.motif_strand == '-':
            footprint_expected = footprint_expected[::-1]
            footprint_observed = footprint_observed[::-1]
        expected_profiles.append(footprint_expected)
        observed_profiles.append(footprint_observed)
    expected_profiles = np.array(expected_profiles)
    observed_profiles = np.array(observed_profiles)
    expected_profiles_transformed = 2 * np.sqrt(expected_profiles + 3/8)
    observed_profiles_transformed = 2 * np.sqrt(observed_profiles + 3/8)
    difference_transformed = observed_profiles_transformed - expected_profiles_transformed
    return difference_transformed

for footprint in enriched.motif_cluster:
    current_motif = footprint
    if "/" in current_motif:
        current_motif_file = current_motif.replace('/','')
    else:
        current_motif_file = current_motif
    footprint = dhs_unique.loc[dhs_unique['motif_cluster'] == current_motif]
    with_motif = histogram_with_motif(current_motif)
    without_motif = histogram_without_motif(current_motif)
    seaborn.kdeplot(with_motif, label = 'with motif', bw =0.25)
    seaborn.kdeplot(without_motif, label = 'without motif', bw =0.25)
    plt.xlabel('arcsinh transformed cuts per base')
    plt.ylabel('frequency')
    plt.title("{} Histogram in {}".format(current_motif, cell_type))
    plt.axvline(x=1, c='r')
    plt.savefig('{}/{}_{}_histogram.png'.format(cell_type, cell_type, current_motif_file))
    plt.show()
    plt.close()

for current_motif in enriched.motif_cluster:
    current_motif_file = current_motif.replace('/','')
    current_footprints = dhs.loc[dhs['motif_cluster'] == current_motif].copy()
    print(current_motif)
    for index, row in tqdm(current_footprints.iterrows(), total=current_footprints.shape[0]):
        dhs_observed_profile = tabix_profile(row.dhs_chr, row.adjusted_dhs_start, row.adjusted_dhs_end, tabix_file)
        footprint_cuts = dhs_observed_profile[row.footprint_start - row.adjusted_dhs_start:row.footprint_end - row.adjusted_dhs_start]
        current_footprints.loc[index, 'footprint_cuts'] = footprint_cuts.sum()
        current_footprints.loc[index, 'expected_footprint_cuts'] = dhs_observed_profile.sum() * row.adjusted_bias_frac
    transformed_observed = observed_transformed(current_footprints.footprint_cuts)
    transformed_expected = expected_transformed(current_footprints.expected_footprint_cuts)
    current_footprints = calc_p_value(transformed_observed, transformed_expected)
    scatter = plt.scatter(current_footprints.transformed_expected, current_footprints.transformed_observed , alpha =0.75, c=current_footprints.log_p_value, vmin = 0, vmax =10)
    plt.axis('equal')
    plt.xlabel('expected cuts')
    plt.ylabel('observed cuts')
    plt.colorbar(scatter, label = 'log_p_value')
    plt.title('{} Footprint Accessibility in {}'.format(current_motif, cell_type))
    plt.savefig("{}/{}_{}_scatter.png".format(cell_type, cell_type, current_motif_file))
    plt.show()
    plt.close()
    protected = current_footprints.query('p_values_correction < 0.05')
    protected.to_csv("{}/{}_{}_protected.txt".format(cell_type, cell_type, current_motif_file))
    differences_transformed = metaplot(protected)
    footprint_len = row.footprint_end - row.footprint_start
    plt.plot(differences_transformed.mean(axis=0), color = 'k')
    plt.plot(differences_transformed.mean(axis=0) + differences_transformed.std(), color = 'b', alpha = 0.5)
    plt.plot(differences_transformed.mean(axis=0) - differences_transformed.std(), color = 'b', alpha = 0.5)
    plt.title("{} Metaplot in {}(n={})".format(current_motif, cell_type, len(protected)))
    plt.xlabel('distance from start')
    plt.ylabel('observed_expected_difference')
    plt.ylim(-15,10)
    plt.axhline(y=0, c='k')
    plt.axvspan(10, footprint_len + 10, color='red', alpha=0.3)
    plt.savefig("{}/{}_{}_metaplot.png".format(cell_type, cell_type, current_motif_file))
    plt.show()
    plt.close()
    unprotected = current_footprints.query("p_values_correction > 0.05")
    total_protected = len(protected)
    unprotected_random = [0] * total_protected
    for i in range(0, len(unprotected)):
        if i < total_protected:
            unprotected_random[i] = i
        if i >= total_protected:
            if len(protected) >= i * np.random.uniform():
                unprotected_random[np.random.randint(0, total_protected)] = i
    unprotected_sample = unprotected.iloc[unprotected_random]
    differences_transformed_unprotected = metaplot(unprotected_sample)
    plt.plot(differences_transformed_unprotected.mean(axis=0), color = 'k')
    plt.plot(differences_transformed_unprotected.mean(axis=0) + differences_transformed_unprotected.std(), color = 'b', alpha = 0.5)
    plt.plot(differences_transformed_unprotected.mean(axis=0) - differences_transformed_unprotected.std(), color = 'b', alpha = 0.5)
    plt.title("{} Unprotected Metaplot in {}(n={})".format(current_motif, cell_type, len(unprotected_sample)))
    plt.xlabel('distance from start')
    plt.ylabel('observed_expected_difference')
    plt.ylim(-15,10)
    plt.axhline(y=0, c='k')
    plt.axvspan(10, footprint_len + 10, color='red', alpha=0.3)
    plt.savefig("{}/{}_{}_unprotected_metaplot.png".format(cell_type, cell_type, current_motif_file))
    plt.close()
    unprotected_sort = unprotected.sort_values(by = ['p_values_correction'])
    least_protected = unprotected_sort.tail(total_protected)
    differences_transformed_least_protected = metaplot(least_protected)
    plt.plot(differences_transformed_least_protected.mean(axis=0), color = 'k')
    plt.plot(differences_transformed_least_protected.mean(axis=0) + differences_transformed_least_protected.std(), color = 'b', alpha = 0.5)
    plt.plot(differences_transformed_least_protected.mean(axis=0) - differences_transformed_unprotected.std(), color = 'b', alpha = 0.5)
    plt.title("{} Least Protected Metaplot in {}(n={})".format(current_motif, cell_type, len(unprotected_sample)))
    plt.xlabel('distance from start')
    plt.ylabel('observed_expected_difference')
    plt.ylim(-15,10)
    plt.axhline(y=0, c='k')
    plt.axvspan(10, footprint_len + 10, color='red', alpha=0.3)
    plt.savefig("{}/{}_{}_least_protected_metaplot.png".format(cell_type, cell_type, current_motif_file))
    plt.show()
    plt.close()
    



  




