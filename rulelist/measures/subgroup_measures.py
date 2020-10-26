# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:10:10 2019

@author: gathu
"""
from itertools import combinations
from math import exp, log2

import numpy as np
from gmpy2 import xmpz, mpz, popcount

from rulelist.mdl.mdl_base_codes import log2_0


def jaccard_index_model(list_bitsets):
    nrules = len(list_bitsets)
    if nrules < 2:
        return 0, None
    else:
        intersect = np.zeros([nrules,nrules],dtype = np.uint)
        for r1 in range(nrules):
            tid_rule1 = list_bitsets[r1]
            for r2 in range(nrules):
                tid_rule2 = list_bitsets[r2]
                intersect[r1,r2] = popcount(tid_rule1 &  tid_rule2)
        jaccard = np.zeros([nrules,nrules])
        for rr in combinations(range(nrules), 2):
            inter = intersect[rr]
            supp1 = intersect[(rr[0],rr[0])]
            supp2 = intersect[(rr[1],rr[1])]
            jaccard[rr]= inter/(supp1+supp2-inter)
        uptm = np.triu_indices(nrules, 1)
        jacc_avg = np.sum(jaccard) / len(uptm[0])
        return jacc_avg, jaccard

def wracc_numeric(data_mean,data_var, values):
    usage = len(values)
    if usage > 0:
        sg_mean = np.mean(values)
        wracc = usage * np.absolute(sg_mean - data_mean)
    else:
        wracc = 0
    return wracc

def kullbackleibler_gaussian_paramters(data_mean,data_var, values):
    usage = len(values)
    RSS = sum([(val - data_mean) ** 2 for val in values])
    variance = np.var(values) if usage > 2 else 0
    l_e = log2(exp(1))
    if usage > 2 and variance != 0:
        kl_aux1 = 0.5 * log2_0(data_var) + \
                  0.5 * RSS / usage / data_var*l_e
        kl_aux2 = 0.5*log2_0(variance)+0.5*l_e
        kl = kl_aux1 - kl_aux2
        wkl = usage*kl
    else:
        wkl = 0
    return wkl

def numeric_single2multitargets_function(function2multi, data_mean, data_var,values, number_targets):
    sum_score_targets = 0
    for ntarget in range(number_targets):
        columnvalues = values[:,ntarget] if number_targets > 1 else values
        sum_score_targets += function2multi(data_mean[ntarget],data_var[ntarget], columnvalues)
    return sum_score_targets


def numeric_discovery_measures(rulelist,X,Y):
    nrules= rulelist.number_rules
    if nrules == 0:
        measures = dict()
        measures["avg_supp"] = measures["wkl_supp"] = measures["avg_usg"] = measures["wkl_usg"] = measures["wacc_supp"] =\
        measures["wacc_usg"] = measures["jacc_avg"] =measures["n_rules"] = measures["avg_items"] = measures["wkl_sum"] = \
        measures["wkl_sum_norm"] = measures["wacc_supp_sum"] = measures["wacc_usg_sum"] = measures["std_rules"] =\
        measures["top1_std"] = measures["length_orig"] = measures["length_final"] = measures["length_ratio"] = 0
    #nrows= len(rulelist.target_values)
    wkl_supp,wkl_usg,wkl_sum = np.zeros(nrules), np.zeros(nrules), np.zeros(nrules)
    wacc_supp, wacc_usg = np.zeros(nrules),np.zeros(nrules)
    support, usage = np.zeros(nrules),np.zeros(nrules)
    std_rules = [var1target ** 0.5 for sg in rulelist.subgroups for var1target in sg.statistics.variance]
    std_rulesalternative = []
    data_mean = rulelist.default_rule_statistics.mean
    data_var = rulelist.default_rule_statistics.variance
    tid_covered =  mpz()
    list_bitsets = []
    number_targets = len(rulelist.subgroups[0].statistics.mean)
    for r in range(nrules):
        tid_support = rulelist.subgroups[r].bitarray
        list_bitsets.append(tid_support)
        tid_usage = tid_support & ~ tid_covered
        tid_covered = tid_covered | tid_support
        aux_bitset = xmpz(tid_support)
        idx_bits = list(aux_bitset.iter_set())
        values_support  = Y.iloc[idx_bits, :].values
        aux_bitset = xmpz(tid_usage)
        idx_bits = list(aux_bitset.iter_set())
        values_usage = Y.iloc[idx_bits, :].values
        support[r] = values_support.shape[0]
        usage[r] = values_usage.shape[0]
        wkl_supp[r] = numeric_single2multitargets_function(kullbackleibler_gaussian_paramters,data_mean, data_var,
                                                               values_support,number_targets)
        wkl_usg[r] = numeric_single2multitargets_function(kullbackleibler_gaussian_paramters,data_mean, data_var,
                                                               values_usage,number_targets)
        std_rulesalternative.append(np.std(values_usage))
        wacc_supp[r] = numeric_single2multitargets_function(wracc_numeric,data_mean, data_var,values_support,number_targets)
        wacc_usg[r] = numeric_single2multitargets_function(wracc_numeric,data_mean, data_var,values_usage,number_targets)

    
    wkl_sum = sum(wkl_usg)
    #  Average them all!!!!
    measures = dict()
    measures["avg_supp"] = np.mean(support)
    measures["wkl_supp"]  = np.mean(wkl_supp)
    measures["avg_usg"] = np.mean(usage)
    measures["wkl_usg"]  = np.mean(wkl_usg)
    measures["wacc_supp"]  = np.mean(wacc_supp)
    measures["wacc_usg"]   = np.mean(wacc_usg)
    measures["jacc_avg"], jaccard_matrix = jaccard_index_model(list_bitsets)
    measures["n_rules"] = rulelist.number_rules
    measures["avg_items"] = sum([len(sg.pattern) for sg in rulelist.subgroups]) / rulelist.number_rules
    measures["wkl_sum"] = wkl_sum
    measures["wkl_sum_norm"] = wkl_sum/X.shape[0]
    measures["wacc_supp_sum"] = np.sum(wacc_supp)
    measures["wacc_usg_sum"] = np.sum(wacc_usg)
    measures["std_rules"] = np.mean(std_rules)
    measures["top1_std"] =std_rules[0]
    measures["length_orig"] = rulelist.length_original
    measures["length_final"] = rulelist.length_data + rulelist.length_model
    measures["length_ratio"] = rulelist.length_ratio
    
    return measures 


def wkl_wracc(data_probs,values,number_instances, number_targets):
    sum_wracc_targets = 0
    sum_wkl_targets = 0
    usage = values.shape[0]
    if usage:
        for ntarget, target_variable in enumerate(data_probs):
            columnvalues = values[:,ntarget] if number_targets > 1 else values
            number_classes = len(data_probs[target_variable])
            aux_wacc_score = 0
            for category, prob_default in data_probs[target_variable].items():
                #wracc
                counts_category = sum(columnvalues == category)
                prob_rule = counts_category/usage
                aux_wacc_score += (usage / number_instances) * abs(prob_rule - prob_default)
                #wkl
                sum_wkl_targets += usage*prob_rule*log2_0(prob_rule/prob_default)

            sum_wracc_targets += aux_wacc_score/number_classes
    return sum_wkl_targets, sum_wracc_targets


def nominal_discovery_measures(rulelist,X,Y):
    nrules= rulelist.number_rules
    nrows= X.shape[0]
    data_prob_class = rulelist.default_rule_statistics.prob_per_classes
    wkl_supp,wkl_usg,wkl_sum = np.zeros(nrules), np.zeros(nrules), np.zeros(nrules)
    wacc_supp, wacc_usg = np.zeros(nrules),np.zeros(nrules)
    support, usage = np.zeros(nrules),np.zeros(nrules)
    tid_covered =  mpz()
    list_bitsets = []
    number_targets = len(rulelist.default_rule_statistics.prob_per_classes)
    for r in range(nrules):
        tid_support = rulelist.subgroups[r].bitarray
        list_bitsets.append(tid_support)
        tid_usage = tid_support & ~ tid_covered
        tid_covered = tid_covered | tid_support
        aux_bitset = xmpz(tid_support)
        idx_bits = list(aux_bitset.iter_set())
        values_support  = Y.iloc[idx_bits, :].values
        aux_bitset = xmpz(tid_usage)
        idx_bits = list(aux_bitset.iter_set())
        values_usage = Y.iloc[idx_bits, :].values
        support[r] = values_support.shape[0]
        usage[r] = values_usage.shape[0]
        wkl_supp[r], wacc_supp[r] = wkl_wracc(data_prob_class,values_support,nrows, number_targets)
        wkl_usg[r], wacc_usg[r] = wkl_wracc(data_prob_class,values_usage,nrows, number_targets)

    wkl_sum = sum(wkl_usg)
    #  Average them all!!!!
    measures = dict()
    measures["avg_supp"] = np.mean(support)
    measures["wkl_supp"] = np.mean(wkl_supp)

    measures["avg_usg"] = np.mean(usage)
    measures["wkl_usg"] = np.mean(wkl_usg)

    measures["wacc_supp"] = np.mean(wacc_supp)
    measures["wacc_usg"] = np.mean(wacc_usg)



    measures["jacc_avg"], jaccard_matrix = jaccard_index_model(list_bitsets)
    measures["n_rules"] = rulelist.number_rules
    measures["avg_items"] = sum([len(sg.pattern) for sg in rulelist.subgroups]) / rulelist.number_rules
    measures["wkl_sum"] = wkl_sum
    measures["wkl_sum_norm"] = wkl_sum/X.shape[0]

    measures["wacc_supp_sum"] = np.sum(wacc_supp)
    measures["wacc_usg_sum"] = np.sum(wacc_usg)

    measures["length_orig"] = rulelist.length_original
    measures["length_final"] = rulelist.length_data + rulelist.length_model
    measures["length_ratio"] = rulelist.length_ratio
    return measures 




def discovery_itemset(data,model):
    nrules = model.number_rules
    cl = model.class_codes
    rules_supp = {nr: {c: int(0) for c in cl} for nr in range(nrules)}
    rules_usg = {nr: {c: int(0) for c in cl} for nr in range(nrules)}
    count_cl = {c: int(0) for c in cl}
    pred = []
    prob = []
    RULEactivated = []
    intersect = np.zeros([nr,nr],dtype = np.uint)
    jaccard = np.zeros([nr,nr])
    # Find majority class
    for t in data:
        active_r = list()
        first = True 
        for r in range(nrules):
            if model[r]['p'] <= t and first:
                pred.append(model[r]['cl'])
                prob.append(model[r][model[r]['cl']])
                RULEactivated.append(r)
                active_r.append(r)
                intersect[r,r] +=1
                for ic, c in enumerate(cl):
                    if c <= t:
                        rules_supp[r][c] +=1
                        rules_usg[r][c] +=1
                        count_cl[c] +=1
                first = False
            elif model[r]['p'] <= t and not first:
                active_r.append(r)
                intersect[r,r] +=1
                for ic, c in enumerate(cl):
                    if c <= t:
                        rules_supp[r][c] +=1 
        for rr in combinations(active_r, 2):
            intersect[rr] +=1

    for rr in combinations(range(nr), 2):
        inter = intersect[rr]
        supp1 = intersect[(rr[0],rr[0])]
        supp2 = intersect[(rr[1],rr[1])]
        jaccard[rr]= inter/(supp1+supp2-inter)  
    
    # remove empty rule column and row
    jaccard = np.delete(jaccard, -1, 0)
    jaccard = np.delete(jaccard, -1, 1)
    # average over all possible cases 
    uptm = np.triu_indices(nr-1,1)
    jacc_avg = np.sum(jaccard)/len(uptm[0])
    jacc_consecutive_avg = np.mean(np.diagonal(jaccard,1))
    avg_supp = np.mean([sum([rules_supp[r][c] for c in cl]) for r in range(nr-1)])
    avg_usg = np.mean([sum([rules_usg[r][c] for c in cl]) for r in range(nr-1)])
    
    return pred, prob, RULEactivated,rules_supp,rules_usg,count_cl,jacc_avg,avg_supp,avg_usg