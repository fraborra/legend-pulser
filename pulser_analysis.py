import os, json
import argparse
import time
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pickle

from multihist import Hist1d, Histdd
from datetime import datetime, timezone

import matplotlib.colors as mcolors
base_colors = [c for c in mcolors.BASE_COLORS.keys()]
import matplotlib.dates as mdates
from scipy.optimize import curve_fit

import pygama.math.histogram as pgh
import pygama.math.histogram as pgh

from dspeed.processors import bl_subtract, linear_slope_fit
from dspeed.processors import zac_filter
from dspeed.processors import cusp_filter
from dspeed.processors import presum

import lgdo.lh5_store as lh5
from lgdo import Array

from legendmeta import LegendMetadata
meta_path = '/lfs/l1/legend/users/dandrea/software/legend-metadata'
lmeta = LegendMetadata(path=meta_path)
chmap = lmeta.hardware.configuration.channelmaps.on("20230310T175624Z")


def plot_pulser_waveforms(ge_key, det_name, raw_list, puls_chn = 'ch1027201',
                          nwf = 10, apply_mask = False, fc_bl = False,
                          us = 1000/16, xlim=(3060,3090), ylim=(1e4,1e9), plot_dir = None ):
    rawid = chmap[ge_key]['daq']['rawid']
    chn = f'ch{rawid}'
    store = lh5.LH5Store()
    wfs, n_rows = store.read_object(f"{chn}/raw/waveform/values", raw_list)
    timestamp, n_rows = store.read_object(f"{chn}/raw/timestamp", raw_list)
    baseline, n_rows = store.read_object(f"{chn}/raw/baseline", raw_list)
    wfs, timestamp, baseline = wfs.nda, timestamp.nda, baseline.nda
    
    daq_puls, n_rows = store.read_object(f"{puls_chn}/raw/daqenergy", raw_list)
    daq_puls = daq_puls.nda
    wfs = wfs[daq_puls>1000]
    n_ev, wsize = len(wfs), len(wfs[0])
    dts = np.linspace(0,wsize/us,wsize)
    print(chn,'n. pulser events',n_ev,'size',wsize)
    
    fig, axis = plt.subplots(figsize=(12,6.75), facecolor='white')
    for j in range(nwf):
        wf = wfs[j,:]
        if j < 10: axis.plot(wf,label=f'{j} {(timestamp[j]-timestamp[0])*1e3:.2f} ms')
        else: axis.plot(wf)
    axis.set_xlabel('samples')
    axis.legend(title=f'{ge_key} - {det_name}',loc='upper left')
    axin = axis.inset_axes([0.55, 0.1, 0.4, 0.55])
    for j in range(nwf):
        axin.plot(wfs[j])
        axin.set_xlim(xlim[0],xlim[1])
        axin.set_yticklabels('')
        

def pulser_waveform(raw_list, chmap, dsp_config, hit_config_prod, test_dict_file, n_test = 1, n_acq = 0, nwf = 10,
                    fc_bl=False, nstring = None, ns = 1/16, us = 1000/16, wcut = 2100, sconv = 200,
                    plot_dir = None, lh5_file = None ):
    with open(dsp_config, "r") as file:
        db = json.load(file)
    with open(test_dict_file,'r') as file:
        test_dict = json.load(file)
    fig0, axis0 = plt.subplots(figsize=(20,8), facecolor='white')
    fig1, axis1 = plt.subplots(figsize=(20,8), facecolor='white')
    fig2, axis2 = plt.subplots(nrows=4, ncols=3,figsize=(24,16), facecolor='white')
    fig3, axis3 = plt.subplots(nrows=4, ncols=3,figsize=(24,16), facecolor='white')
    chn_off = (26,33,46,47,48,50,84,85,94,112,116,117,120,122)
    ge_off = ['V07298B', 'P00665A', 'V01386A', 'V01403A', 'V01404A', 'B00091D', 'P00537A', 'B00091B', 'P00538B', 'P00661A', 'P00665B', 'P00698B']
    ge_drift = ['V01406A', 'V01415A', 'V01387A', 'P00665C','P00748B', 'P00748A']
    strings = np.array([int(chmap[ch]['location']['string']) for ch in chmap.keys() if chmap[ch]['system']=='geds'])
    if nstring is None: nstring = strings.max()
    run = int(raw_list[0].split('/')[-1].split('-')[2].split('r')[1])
    for string in range(1,nstring+1):
        print('string',string)
        ax2 = axis2.flat[string-1]
        ax3 = axis3.flat[string-1]
        ge_keys = [ch for ch in chmap.keys() if chmap[ch]['system']=='geds' and chmap[ch]['location']['string']==string]
        if run < 20:
            ge_keys = [ch for n, ch in zip(ge_numbers,det_names) if n not in chn_off ]
        else:
            ge_keys = [ch for ch in ge_keys if ch not in ge_off]
            ge_keys = [ch for ch in ge_keys if ch not in ge_drift]
        ge_numbers = [chmap[ch]['daq']['fcid'] for ch in ge_keys]
        chns = [f'ch{n:03}' for n in ge_numbers]
        ge_rawid = [chmap[ch]['daq']['rawid']  for ch in ge_keys]
        if run < 20: ge_table = chns
        else: ge_table = [f'ch{id}' for id in ge_rawid]
        
        fig, axis = plt.subplots(nrows=5, ncols=3,figsize=(24,16), facecolor='white')
        figc, axisc = plt.subplots(nrows=5, ncols=3,figsize=(24,16), facecolor='white')
        figg, axiss = plt.subplots(figsize=(12,6.75), facecolor='white')
        
        fwhm, fwhm_err = np.zeros(len(ge_table)), np.zeros(len(ge_table))
        sigma_cusp, flat_cusp = np.ones(len(ge_table))*10, np.ones(len(ge_table))*10
        pos, pos_err = np.zeros(len(ge_table)), np.zeros(len(ge_table))
        for i, chn in enumerate(ge_table[:]):
            chn_old = chns[i]
            det = ge_keys[i]
            ax = axis.flat[i]
            axc = axisc.flat[i]
            try:
                for ii, ff in enumerate(raw_list):
                    #print('Open raw file',ff)
                    f = h5py.File(ff,'r')
                    if ii == 0:
                        wfs = f[f'{chn}/raw/waveform/values'][:]
                        timestamp = f[f'{chn}/raw/timestamp'][:]
                        baseline = f[f'{chn}/raw/baseline'][:]
                    else:
                        wfs = np.append(wfs, f[f'{chn}/raw/waveform/values'][:], axis=0)
                        timestamp = np.append(timestamp, f[f'{chn}/raw/timestamp'][:], axis=0)
                        baseline = np.append(baseline, f[f'{chn}/raw/baseline'][:], axis=0)
                #min_thr, max_thr = baseline.mean()-1000, baseline.mean()+1500
                #mask = (wfs.min(axis=1)>min_thr)# & (wfs.max(axis=1)>max_thr)
                bl_mean, bl_std, bl_slope, bl_intercept = linear_slope_fit(wfs[:,:2500])
                
                #min_thr, max_thr = bl_mean.mean()-10*bl_std.mean(), bl_mean.mean()+5*bl_std.mean()
                #mask = (wfs[:,2000:].max(axis=1)>max_thr)
                wf_max = wfs[:,:].max(axis=1)
                min_thr = bl_mean.mean()-(wf_max.mean()-bl_mean.mean())
                max_thr = bl_mean.mean()+(wf_max.mean()-bl_mean.mean())/3
                mask = (wfs[:,2000:].max(axis=1)>max_thr) & (wfs[:,:].min(axis=1)>min_thr) & (np.abs(bl_slope)<0.2)
                wfs, timestamp, baseline = wfs[mask], timestamp[mask], baseline[mask]
                n_ev, wsize = len(wfs), len(wfs[0])
                nsize, fsize = round(wsize-wcut), round(wsize-wcut-sconv)
                wfs, dts = wfs[:,:nsize], np.linspace(0,nsize/us,nsize)
                print(chn,det,'n. pulser events',n_ev,'original size',wsize,'cut size',nsize,'->',nsize/us,'us')
                for j in range(nwf):
                    ax.plot(dts,wfs[j],label=f'{j} {(timestamp[j]-timestamp[0])*1e3:.2f} ms')
                ax.set_xlabel('time ($\mu$s)')
                ax.legend(title=f'S{string} - {chn} - {det}',loc='upper left')
                axin = ax.inset_axes([0.55, 0.1, 0.4, 0.55])
                for j in range(nwf):
                    axin.plot(dts,wfs[j])
                axin.set_xlim(nsize/2/us-2,nsize/2/us+2.5)# 30.4,35.2)
                axin.set_yticklabels('')
            except:
                wfs = f[f'{chn}/raw/waveform/values'][:]
                timestamp = f[f'{chn}/raw/timestamp'][:]
                for j in range(nwf):
                    ax.plot(wfs[j],label=f'{j} {(timestamp[j]-timestamp[0])*1e3:.2f} ms')
                ax.set_xlabel('time ($\mu$s)')
                ax.legend(title=f'S{string} - {chn} - {det}',loc='upper left')
                continue
            
            # energy calculation
            if fc_bl:
                wfs = bl_subtract(wfs, baseline)
            else:
                bl_mean, bl_std, bl_slope, bl_intercept = linear_slope_fit(wfs[:,:1000])
                wfs = bl_subtract(wfs, bl_mean)
            # cusp
            print('cusp parameters from config',db[chn_old]['cusp'],db[chn_old]['pz'])
            t_start = time.time()
            tau = float(db[chn_old]['pz']['tau'].split('*ns')[0])
            s_cusp, f_cusp = db[chn_old]['cusp']['sigma'], db[chn_old]['cusp']['flat']
            #sigma_cusp[i], flat_cusp[i] = float(s_cusp.split('*us')[0]), float(f_cusp.split('*us')[0])
            #wf_cusp = np.zeros(( n_ev, sconv+1 ))
            #cusp_func = cusp_filter(fsize, sigma_cusp[i]*us, int(flat_cusp[i]*us), tau*ns)
            #cusp_func(wfs[:,:fsize],wf_cusp)
            #cusp_ene = np.max(wf_cusp,axis=1)
            sigma_prod, flat_prod = float(s_cusp.split('*us')[0]), float(f_cusp.split('*us')[0])
            
            fwhm_rel = 10
            #for sigma in [10,20,50,100,150,200,250,300,400,500]:
            for sigma in [sigma_prod]:#,20,100,500]:
                if sigma == sigma_prod: flat = flat_prod
                else: flat = 0.5
                wf_cusp = np.zeros(( n_ev, sconv+1 ))
                cusp_func = cusp_filter(fsize, sigma*us, int(flat*us), tau*ns)
                cusp_func(wfs[:,:nsize],wf_cusp)
                temp_ene = np.max(wf_cusp,axis=1)
                #print(f"Time to process w/ cusp: {(time.time() - t_start):.2f} sec")
                # energy results
                try:
                    temp_centr, temp_fwhm, temp_fwhm_err, hc, bc = fit_gaussian_peak(temp_ene, relative = 1)
                    print(f'sigma = {sigma} us, flat = {flat}, FWHM = {temp_fwhm:.3f} +/- {temp_fwhm_err:.3f} %')
                except: continue
                if temp_fwhm < fwhm_rel:
                    fwhm_rel = temp_fwhm
                    cusp_ene = temp_ene
                    sigma_cusp[i] = sigma
                    flat_cusp[i] = flat
            #cuspd = au.cusp_filter_loc(fsize, sigma*us, int(flat*us), tau*ns)
            for j in range(nwf):
                #conv = np.convolve(wfs[j,:nsize], cuspd, "valid")
                axc.plot(wf_cusp[j],label=f'{j} {(timestamp[j]-timestamp[0])*1e3:.2f} ms')
            axc.set_xlabel('time ($\mu$s)')
            axc.legend(title=f'S{string} - {chn} - {det}',loc='upper left')
            #cal_dir = '/data1/users/marshall/prod-ref/v06.00/generated/tier/raw/cal/p02/r017/'
            #m_cal, q_cal = process_calibration(chn, cal_dir, sigma_cusp[i], flat_cusp[i], tau, axis = axis3.flat[i] )
            #if sigma_cusp[i] != sigma_prod: hit_config = hit_config_prod.split('.json')[0] + '_' + str(int(sigma_cusp[i])) + '.json'
            #else:
            hit_config = hit_config_prod
            print('Using hit config',hit_config)
            with open(hit_config, "r") as file:
                hit = json.load(file)
            m_cal = hit[chn_old]['operations']['cuspEmax_ctc_cal']['parameters']['a']
            q_cal = hit[chn_old]['operations']['cuspEmax_ctc_cal']['parameters']['b']
            cusp_ene = m_cal * cusp_ene + q_cal
            
            try:
                pos[i], fwhm[i], fwhm_err[i], hc, bc = fit_gaussian_peak(cusp_ene)
                print(f'Result: sigma = {sigma_cusp[i]} us, flat = {flat_cusp[i]}, FWHM = {fwhm[i]:.3f} +/- {fwhm_err[i]:.3f} keV')
                ax2.plot(bc,hc,label=f'{det}: {pos[i]:.2f} keV, FWHM={fwhm[i]:.2f} keV')
            except: continue
            test_dict[f'test{n_test}'][f'{n_acq}'][chn] = {'position':pos[i],
                                                           'fwhm':fwhm[i],
                                                           'fwhm_err':fwhm_err[i],
                                                           'sigma_cusp':sigma_cusp[i],
                                                           'flat_cusp':flat_cusp[i]}
            bsize = 2500
            fsizeb = round(bsize-sconv)
            wf_cusp = np.zeros(( n_ev, sconv+1 ))
            cusp_func = cusp_filter(fsizeb, sigma_cusp[i]*us, int(flat_cusp[i]*us), tau*ns)
            cusp_func(wfs[:,:bsize],wf_cusp)
            bl_ene = np.max(wf_cusp,axis=1)
            bl_ene = m_cal * bl_ene + q_cal
            pos_bl, fwhm_bl, fwhm_err_bl, hc, bc = fit_gaussian_peak(bl_ene, nb = 80, ran = 5)
            print(f'Result baseline: pos = {pos_bl:.2f} keV, FWHM = {fwhm_bl:.2f} +/- {fwhm_err_bl:.2f} keV')
            axiss.plot(bc,hc,label=f'{det}: {pos_bl:.2f} keV, FWHM={fwhm_bl:.2f} keV')
            #freq, power = au.calculate_fft(wfs[:,:bsize], nbls = 1000)
            ax3.plot(freq[1:], power[1:],label=det)
                
            print()
            if lh5_file is not None:
                store = lh5.LH5Store()
                try:
                    store.gimme_file(lh5_file, "r")
                    print('lh5 file already existing')
                except:
                    print('new lh5 file')
                store.write_object(Array(cusp_ene), name="pulser_energy", lh5_file=lh5_file,
                                   wo_mode='overwrite',group=chn)
                store.write_object(Array(timestamp), name="timestamp", lh5_file=lh5_file,
                                   wo_mode='overwrite',group=chn)
        ax2.legend(title=f'String {string}',loc='upper left')
        ax2.set_xlabel('energy (keV)')
        ax3.set_xlabel('frequency (MHz)')
        ax3.set_ylabel('power spectral density')
        ax3.set_ylim(1e4,1e10)
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.legend(title=f'String {string}',loc='upper left')
        axiss.legend(title=f'String {string}',loc='upper left')
        axiss.set_xlabel('energy (keV)')
        axiss.set_ylabel('counts')
        if string == 1: dets, fwhms, fwhm_errs, poss = ge_keys, fwhm, fwhm_err, pos
        else: dets,fwhms,fwhm_errs,poss =np.append(dets,ge_keys),np.append(fwhms,fwhm),np.append(fwhm_errs,fwhm_err),np.append(poss,pos)
        if plot_dir is not None:
            fig.savefig(f'{plot_dir}/test{n_test}_acq{n_acq}_wfs_string{string}.png',dpi=300, bbox_inches='tight')
            #fig3.savefig(f'{plot_dir}/test{n_test}_acq{n_acq}_cal_string{string}.png',dpi=300, bbox_inches='tight')
        axis0.errorbar(ge_keys,fwhm,yerr=fwhm_err,marker='o',ls='',label=f'String {string}')
        axis1.plot(ge_keys,pos,marker='o',ls='',label=f'String {string}')
    axis0.set_ylabel('FWHM (keV)')
    axis0.set_xticklabels(dets, rotation = 90, ha="right")
    axis0.legend(loc='upper right')
    axis0.grid()
    axis1.set_ylabel('pulser position (keV)')
    axis1.set_xticklabels(dets, rotation = 90, ha="right")
    axis1.legend(loc='upper right')
    axis1.grid()
    print('Updating dict file',test_dict_file)
    with open(test_dict_file,'w') as f:
        json.dump(test_dict, f, indent=4)
    if plot_dir is not None:
        fig0.savefig(f'{plot_dir}/test{n_test}_acq{n_acq}_fwhms.png',dpi=300, bbox_inches='tight')
        fig1.savefig(f'{plot_dir}/test{n_test}_acq{n_acq}_positions.png',dpi=300, bbox_inches='tight')
        fig2.savefig(f'{plot_dir}/test{n_test}_acq{n_acq}_fwhms_strings.png',dpi=300, bbox_inches='tight')
        fig3.savefig(f'{plot_dir}/test{n_test}_acq{n_acq}_fft.png',dpi=300, bbox_inches='tight')


def pulser_z_scan(raw_list, chmap, dsp_config, hit_config, dict_file, n_pos = 0, nwf = 10,
                  fc_bl=True, nstring = None, ns = 1/16, us = 1000/16, wcut = 2100, sconv = 200,
                  plot_dir = None, lh5_file = None ):
    with open(dsp_config, "r") as file:
        db = json.load(file)
    with open(dict_file,'r') as file:
        test_dict = json.load(file)
    fig0, axis0 = plt.subplots(figsize=(20,8), facecolor='white')
    fig1, axis1 = plt.subplots(figsize=(20,8), facecolor='white')
    fig2, axis2 = plt.subplots(nrows=4, ncols=3,figsize=(24,16), facecolor='white')
    ge_off = (26,33,46,47,48,50,84,85,94,112,116,117,120,122)
    strings = np.array([int(chmap[ch]['location']['string']) for ch in chmap.keys() if chmap[ch]['system']=='geds'])
    if nstring is None: nstring = strings.max()
    for string in range(1,nstring+1):
        print('string',string)
        ax2 = axis2.flat[string-1]
        det_names = [ch for ch in chmap.keys() if chmap[ch]['system']=='geds' and chmap[ch]['location']['string']==string]
        ge_numbers = [chmap[ch]['daq']['fcid'] for ch in chmap.keys() if chmap[ch]['system']=='geds' and chmap[ch]['location']['string']==string]
        chns = [f'ch{n:03}' for n in ge_numbers if n not in ge_off ]
        det_names = [ch for n, ch in zip(ge_numbers,det_names) if n not in ge_off ]
        fig, axis = plt.subplots(nrows=4, ncols=3,figsize=(24,16), facecolor='white')
        figg, axiss = plt.subplots(nrows=4, ncols=3,figsize=(24,16), facecolor='white')
        fwhm, fwhm_err = np.zeros(len(chns)), np.zeros(len(chns))
        sigma_cusp, flat_cusp = np.ones(len(chns))*10, np.ones(len(chns))*10
        pos, pos_err = np.zeros(len(chns)), np.zeros(len(chns))
        for i, chn in enumerate(chns[:]):
            det = det_names[i]
            ax, axx = axis.flat[i], axiss.flat[i]
            try:
                for ii, ff in enumerate(raw_list):
                    #print('Open raw file',ff)
                    f = h5py.File(ff,'r')
                    if ii == 0:
                        wfs = f[f'{chn}/raw/waveform/values'][:]
                        timestamp = f[f'{chn}/raw/timestamp'][:]
                        baseline = f[f'{chn}/raw/baseline'][:]
                    else:
                        wfs = np.append(wfs, f[f'{chn}/raw/waveform/values'][:], axis=0)
                        timestamp = np.append(timestamp, f[f'{chn}/raw/timestamp'][:], axis=0)
                        baseline = np.append(baseline, f[f'{chn}/raw/baseline'][:], axis=0)
                #min_thr, max_thr = baseline.mean()-1000, baseline.mean()+1500
                #mask = (wfs.min(axis=1)>min_thr)# & (wfs.max(axis=1)>max_thr)
                bl_mean, bl_std, bl_slope, bl_intercept = linear_slope_fit(wfs[:,:2500])
                
                #min_thr, max_thr = bl_mean.mean()-10*bl_std.mean(), bl_mean.mean()+5*bl_std.mean()
                #mask = (wfs[:,2000:].max(axis=1)>max_thr)
                wf_max = wfs[:,:].max(axis=1)
                min_thr = bl_mean.mean()-(wf_max.mean()-bl_mean.mean())
                max_thr = bl_mean.mean()+(wf_max.mean()-bl_mean.mean())/3
                mask = (wfs[:,2000:].max(axis=1)>max_thr) & (wfs[:,:].min(axis=1)>min_thr) & (np.abs(bl_slope)<0.2)
                wfs, timestamp, baseline = wfs[mask], timestamp[mask], baseline[mask]
                n_ev, wsize = len(wfs), len(wfs[0])
                nsize, fsize = round(wsize-wcut), round(wsize-wcut-sconv)
                wfs, dts = wfs[:,:nsize], np.linspace(0,nsize/us,nsize)
                print(chn,det,'n. pulser events',n_ev,'original size',wsize,'cut size',nsize,'->',nsize/us,'us')
                for j in range(nwf):
                    ax.plot(dts,wfs[j],label=f'{j} {(timestamp[j]-timestamp[0])*1e3:.2f} ms')
                ax.set_xlabel('time ($\mu$s)')
                ax.legend(title=f'S{string} - {chn} - {det}',loc='upper left')
                axin = ax.inset_axes([0.55, 0.1, 0.4, 0.55])
                for j in range(nwf):
                    axin.plot(dts,wfs[j])
                axin.set_xlim(nsize/2/us-2,nsize/2/us+2.5)# 30.4,35.2)
                axin.set_yticklabels('')
            except:
                wfs = f[f'{chn}/raw/waveform/values'][:]
                timestamp = f[f'{chn}/raw/timestamp'][:]
                for j in range(nwf):
                    ax.plot(wfs[j],label=f'{j} {(timestamp[j]-timestamp[0])*1e3:.2f} ms')
                ax.set_xlabel('time ($\mu$s)')
                ax.legend(title=f'S{string} - {chn} - {det}',loc='upper left')
                continue
            
            # energy calculation
            if fc_bl:
                wfs = bl_subtract(wfs, baseline)
            else:
                bl_mean, bl_std, bl_slope, bl_intercept = linear_slope_fit(wfs[:,:1000])
                wfs = bl_subtract(wfs, bl_mean)
            # cusp
            print('cusp parameters from config',db[chn]['cusp'],db[chn]['pz'])
            t_start = time.time()
            tau = float(db[chn]['pz']['tau'] .split('*ns')[0])
            s_cusp, f_cusp = db[chn]['cusp']['sigma'], db[chn]['cusp']['flat']
            sigma_cusp[i], flat_cusp[i] = float(s_cusp.split('*us')[0]), float(f_cusp.split('*us')[0])
            wf_cusp = np.zeros(( n_ev, sconv+1 ))
            cusp_func = cusp_filter(fsize, sigma_cusp[i]*us, int(flat_cusp[i]*us), tau*ns)
            cusp_func(wfs[:,:fsize],wf_cusp)
            cusp_ene = np.max(wf_cusp,axis=1)
            with open(hit_config, "r") as file:
                hit = json.load(file)
            m_cal = hit[chn]['operations']['cuspEmax_ctc_cal']['parameters']['a']
            q_cal = hit[chn]['operations']['cuspEmax_ctc_cal']['parameters']['b']
            cusp_ene = m_cal * cusp_ene + q_cal
            
            try:
                pos[i], fwhm[i], fwhm_err[i], hc, bc = fit_gaussian_peak(cusp_ene)
                print(f'Result: sigma = {sigma_cusp[i]} us, flat = {flat_cusp[i]}, FWHM = {fwhm[i]:.3f} +/- {fwhm_err[i]:.3f} keV')
                ax2.plot(bc,hc,label=f'{det}: {pos[i]:.2f} keV, FWHM={fwhm[i]:.2f} keV')
            except: continue
            test_dict[f'{n_pos}'][chn] = {'position':pos[i],
                                          'fwhm':fwhm[i],
                                          'fwhm_err':fwhm_err[i],
                                          'sigma_cusp':sigma_cusp[i],
                                          'flat_cusp':flat_cusp[i]}
            # stability plot
            xm = (np.percentile(cusp_ene, 50))
            xlo, xhi, nb = xm - 20, xm + 20, 500
            dt = np.linspace(0,len(cusp_ene),len(cusp_ene))
            tspace = np.linspace(0,len(cusp_ene),100)
            pspace = np.linspace(xm - 15, xm + 15,100)
            plt.sca(axx)
            php = Histdd(dt, cusp_ene, bins=(tspace, pspace))
            php.plot(log_scale=True,cmap='viridis',colorbar=True)
            havg = php.average(axis=1)
            hstd = [php[ii:].std() for ii in range(php[:,:].shape[0])]
            avg, std = havg[:].mean(), havg[:].std()/np.sqrt(len(havg[:]))
            plt.plot(tspace[1:],havg[:],color='r',label=f'avg {avg:.2f} $\pm$ {std:.2f} keV')
            #plt.plot(tspace[1:],havg[:]-hstd,color='b',ls='-')
            #plt.plot(tspace[1:],havg[:]+hstd,color='b',ls='-')
            axx.set_title(f'{det} - {chn} - pulser')
            axx.legend(title=f'S{string} - {chn} - {det}',loc='upper left')
            axx.set_xlabel('time')
            axx.set_ylabel('energy (keV)')
            print()
            if lh5_file is not None:
                store = lh5.LH5Store()
                try:
                    store.gimme_file(lh5_file, "r")
                    print('lh5 file already existing')
                except:
                    print('new lh5 file')
                store.write_object(Array(cusp_ene), name="pulser_energy", lh5_file=lh5_file,
                                   wo_mode='overwrite',group=chn)
                store.write_object(Array(timestamp), name="timestamp", lh5_file=lh5_file,
                                   wo_mode='overwrite',group=chn)
        ax2.legend(title=f'String {string}',loc='upper left')
        ax2.set_xlabel('energy (keV)')
        if string == 1: dets, fwhms, fwhm_errs, poss = det_names, fwhm, fwhm_err, pos
        else: dets,fwhms,fwhm_errs,poss =np.append(dets,det_names),np.append(fwhms,fwhm),np.append(fwhm_errs,fwhm_err),np.append(poss,pos)
        if plot_dir is not None:
            fig.savefig(f'{plot_dir}/pos{n_pos}_wfs_string{string}.png',dpi=300, bbox_inches='tight')
        axis0.errorbar(det_names,fwhm,yerr=fwhm_err,marker='o',ls='',label=f'String {string}')
        axis1.plot(det_names,pos,marker='o',ls='',label=f'String {string}')
    axis0.set_ylabel('FWHM (keV)')
    axis0.set_xticklabels(dets, rotation = 90, ha="right")
    axis0.legend(loc='upper right')
    axis0.grid()
    axis1.set_ylabel('pulser position (keV)')
    axis1.set_xticklabels(dets, rotation = 90, ha="right")
    axis1.legend(loc='upper right')
    axis1.grid()
    print('Updating dict file',dict_file)
    with open(dict_file,'w') as f:
        json.dump(test_dict, f, indent=4)
    if plot_dir is not None:
        fig0.savefig(f'{plot_dir}/pos{n_pos}_fwhms.png',dpi=300, bbox_inches='tight')
        fig1.savefig(f'{plot_dir}/pos{n_pos}_positions.png',dpi=300, bbox_inches='tight')
        fig2.savefig(f'{plot_dir}/pos{n_pos}_fwhms_strings.png',dpi=300, bbox_inches='tight')


def pulser_z_scan_list(det_names, raw_list, chmap, dsp_config, hit_config_prod, dict_file, n_pos = 0, nwf = 10,sigma_table=None,
                       fc_bl=True, ns = 1/16, us = 1000/16, down_sample = None, wcut = 2100, sconv = 200, plot_dir = None, lh5_file = None ):
    if down_sample is not None: ns, us = ns/down_sample, us/down_sample
    with open(dsp_config, "r") as file:
        db = json.load(file)
    with open(dict_file,'r') as file:
        test_dict = json.load(file)
    fig0, axis0 = plt.subplots(figsize=(20,8), facecolor='white')
    fig1, axis1 = plt.subplots(figsize=(20,8), facecolor='white')
    fig2, axis2 = plt.subplots(figsize=(20,8), facecolor='white')
    fig, axis = plt.subplots(nrows=4, ncols=3,figsize=(24,16), facecolor='white')
    figr, axisr = plt.subplots(nrows=4, ncols=3,figsize=(24,16), facecolor='white')
    figg, axiss = plt.subplots(nrows=4, ncols=3,figsize=(24,16), facecolor='white')
    
    ge_numbers = [chmap[ch]['daq']['fcid'] for ch in det_names]
    strings = [chmap[ch]['location']['string'] for ch in det_names]
    chns = [f'ch{n:03}' for n in ge_numbers]
    fwhm, fwhm_err = np.zeros(len(chns)), np.zeros(len(chns))
    sigma_cusp, flat_cusp = np.ones(len(chns))*10, np.ones(len(chns))*10
    pos, pos_err = np.zeros(len(chns)), np.zeros(len(chns))
    for i, chn in enumerate(chns[:]):
        det, string = det_names[i], strings[i]
        ax, axr, axx = axis.flat[i], axisr.flat[i], axiss.flat[i]
        try:
            for ii, ff in enumerate(raw_list):
                #print('Open raw file',ff)
                f = h5py.File(ff,'r')
                if ii == 0:
                    wfs = f[f'{chn}/raw/waveform/values'][:]
                    timestamp = f[f'{chn}/raw/timestamp'][:]
                    baseline = f[f'{chn}/raw/baseline'][:]
                    daqenergy = f[f'{chn}/raw/daqenergy'][:]
                else:
                    wfs = np.append(wfs, f[f'{chn}/raw/waveform/values'][:], axis=0)
                    timestamp = np.append(timestamp, f[f'{chn}/raw/timestamp'][:], axis=0)
                    baseline = np.append(baseline, f[f'{chn}/raw/baseline'][:], axis=0)
                    daqenergy = np.append(daqenergy, f[f'{chn}/raw/daqenergy'][:], axis=0)
            bl_mean, bl_std, bl_slope, bl_intercept = linear_slope_fit(wfs[:,:2500])
            wf_max = wfs[:,:].max(axis=1)
            min_thr = bl_mean.mean()-(wf_max.mean()-bl_mean.mean())
            max_thr = bl_mean.mean()+(wf_max.mean()-bl_mean.mean())/3
            mask = (wfs[:,2000:].max(axis=1)>max_thr) & (wfs[:,:].min(axis=1)>min_thr) & (np.abs(bl_slope)<0.2)
            mask_cal = (wfs[:,2000:].max(axis=1)<max_thr) & (daqenergy > 0)
            wfs_cal, timestamp_cal, baseline_cal, daqenergy_cal = wfs[mask_cal], timestamp[mask_cal], baseline[mask_cal], daqenergy[mask_cal]
            wfs, timestamp, baseline = wfs[mask], timestamp[mask], baseline[mask]
            n_ev, n_cal, wsize = len(wfs), len(wfs_cal), len(wfs[0])
            nsize, fsize = round(wsize-wcut), round(wsize-wcut-sconv)
            wfs, wfs_cal = wfs[:,:nsize], wfs_cal[:,:nsize]
            print(chn,det,'n. pulser events',n_ev,'calibration events',n_cal,'original size',wsize,'cut size',nsize,'->',nsize/us,'us')
            if down_sample is not None:
                nsize, fsize = round(nsize / down_sample), round(fsize / down_sample)
                wfs_presum = np.zeros(( len(wfs), nsize ))
                presum(wfs,1,down_sample,wfs_presum)
                wfs = wfs_presum
                wfs_presum_cal = np.zeros(( len(wfs_cal), nsize ))
                presum(wfs_cal,1,down_sample,wfs_presum_cal)
                wfs_cal = wfs_presum_cal
                print('down sampling: new waveform size',nsize,'filter size',fsize)
            dts = np.linspace(0,nsize/us,nsize)
            for j in range(nwf):
                ax.plot(dts,wfs[j],label=f'{j} {(timestamp[j]-timestamp[0])*1e3:.2f} ms')
            ax.set_xlabel('time ($\mu$s)')
            ax.legend(title=f'S{string} - {chn} - {det}',loc='upper left')
            axin = ax.inset_axes([0.55, 0.1, 0.4, 0.55])
            for j in range(nwf):
                axin.plot(dts,wfs[j])
            axin.set_xlim(nsize/2/us-2,nsize/2/us+2.5)# 30.4,35.2)
            axin.set_yticklabels('')
            # plot calibration events
            if n_cal > 0:
                for j in range(nwf):
                    if j < n_cal: axr.plot(dts,wfs_cal[j],label=f'{j} {daqenergy_cal[j]}')
                axr.set_xlabel('time ($\mu$s)')
                axr.legend(title=f'S{string} - {chn} - {det}',loc='upper left')
                axinr = axr.inset_axes([0.55, 0.1, 0.4, 0.55])
                for j in range(nwf):
                    if j < n_cal: axinr.plot(dts,wfs_cal[j])
                axinr.set_xlim(nsize/2/us-2,nsize/2/us+2.5)# 30.4,35.2)
                axinr.set_yticklabels('')
        except:
            wfs = f[f'{chn}/raw/waveform/values'][:]
            timestamp = f[f'{chn}/raw/timestamp'][:]
            for j in range(nwf):
                ax.plot(wfs[j],label=f'{j} {(timestamp[j]-timestamp[0])*1e3:.2f} ms')
            ax.set_xlabel('time ($\mu$s)')
            ax.legend(title=f'S{string} - {chn} - {det}',loc='upper left')
            continue
            
        # energy calculation
        if fc_bl:
            wfs = bl_subtract(wfs, baseline)
        else:
            bl_mean, bl_std, bl_slope, bl_intercept = linear_slope_fit(wfs[:,:400])
            wfs = bl_subtract(wfs, bl_mean)
        # cusp
        print('cusp parameters from config',db[chn]['cusp'],db[chn]['pz'])
        t_start = time.time()
        tau = float(db[chn]['pz']['tau'] .split('*ns')[0])
        s_cusp, f_cusp = db[chn]['cusp']['sigma'], db[chn]['cusp']['flat']
        sigma_prod, flat_prod = float(s_cusp.split('*us')[0]), float(f_cusp.split('*us')[0])
        """sigma_cusp[i], flat_cusp[i] = float(s_cusp.split('*us')[0]), float(f_cusp.split('*us')[0])
        wf_cusp = np.zeros(( n_ev, sconv+1 ))
        cusp_func = cusp_filter(fsize, sigma_cusp[i]*us, int(flat_cusp[i]*us), tau*ns)
        cusp_func(wfs[:,:fsize],wf_cusp)
        cusp_ene = np.max(wf_cusp,axis=1)"""
        if sigma_table is not None:
            sigma_cusp[i], flat_cusp[i] = sigma_table[i], 0.5
            wf_cusp = np.zeros(( n_ev, sconv+1 ))
            cusp_func = cusp_filter(fsize, sigma_cusp[i]*us, int(flat_cusp[i]*us), tau*ns)
            cusp_func(wfs[:,:fsize],wf_cusp)
            cusp_ene = np.max(wf_cusp,axis=1)
        else:
            fwhm_rel = 10
            for sigma in [sigma_prod,20,100,500]:
                if sigma == sigma_prod: flat = flat_prod
                else: flat = 0.5
                wf_cusp = np.zeros(( n_ev, sconv+1 ))
                cusp_func = cusp_filter(fsize, sigma*us, int(flat*us), tau*ns)
                cusp_func(wfs[:,:fsize],wf_cusp)
                temp_ene = np.max(wf_cusp,axis=1)
                try:
                    temp_centr, temp_fwhm, temp_fwhm_err, hc, bc = fit_gaussian_peak(temp_ene, relative = 1)
                    print(f'sigma = {sigma} us, flat = {flat}, FWHM = {temp_fwhm:.3f} +/- {temp_fwhm_err:.3f} keV')
                except: continue
                if temp_fwhm < fwhm_rel:
                    fwhm_rel = temp_fwhm
                    cusp_ene = temp_ene
                    sigma_cusp[i] = sigma
                    flat_cusp[i] = flat
        if sigma_cusp[i] != sigma_prod: hit_config = hit_config_prod.split('.json')[0] + '_' + str(int(sigma_cusp[i])) + '.json'
        else: hit_config = hit_config_prod
        print('Using hit config',hit_config)
        with open(hit_config, "r") as file:
            hit = json.load(file)
        m_cal = hit[chn]['operations']['cuspEmax_ctc_cal']['parameters']['a']
        q_cal = hit[chn]['operations']['cuspEmax_ctc_cal']['parameters']['b']
        cusp_ene = m_cal * cusp_ene + q_cal
        try:
            pos[i], fwhm[i], fwhm_err[i], hc, bc = fit_gaussian_peak(cusp_ene)
            print(f'Result: sigma = {sigma_cusp[i]} us, flat = {flat_cusp[i]}, FWHM = {fwhm[i]:.3f} +/- {fwhm_err[i]:.3f} keV')
            axis2.plot(bc,hc,label=f'{det}: {pos[i]:.2f} keV, FWHM={fwhm[i]:.2f} keV')
        except: continue
        test_dict[f'{n_pos}'][chn] = {'n_pulser':n_ev,
                                      'n_cal':n_cal,
                                      'position':pos[i],
                                      'fwhm':fwhm[i],
                                      'fwhm_err':fwhm_err[i],
                                      'sigma_cusp':sigma_cusp[i],
                                      'flat_cusp':flat_cusp[i]}
        # stability plot
        xm = (np.percentile(cusp_ene, 50))
        xlo, xhi, nb = xm - 20, xm + 20, 500
        dt = np.linspace(0,len(cusp_ene),len(cusp_ene))
        tspace = np.linspace(0,len(cusp_ene),100)
        pspace = np.linspace(xm - 2*fwhm[i], xm + 2*fwhm[i],100)
        plt.sca(axx)
        php = Histdd(dt, cusp_ene, bins=(tspace, pspace))
        php.plot(log_scale=True,cmap='viridis',colorbar=True)
        havg = php.average(axis=1)
        hstd = [php[ii:].std() for ii in range(php[:,:].shape[0])]
        avg, std = havg[:].mean(), havg[:].std()/np.sqrt(len(havg[:]))
        plt.plot(tspace[1:],havg[:],color='r',label=f'avg {avg:.2f} $\pm$ {std:.2f} keV')
        axx.set_title(f'{det} - {chn} - pulser')
        axx.legend(title=f'S{string} - {chn} - {det}',loc='upper left')
        axx.set_xlabel('time')
        axx.set_ylabel('energy (keV)')
        print()
        if lh5_file is not None:
            store = lh5.LH5Store()
            try:
                store.gimme_file(lh5_file, "r")
            except:
                print('new lh5 file')
            store.write_object(Array(cusp_ene), name="pulser_energy", lh5_file=lh5_file,
                               wo_mode='overwrite',group=chn)
            store.write_object(Array(timestamp), name="timestamp", lh5_file=lh5_file,
                               wo_mode='overwrite',group=chn)
    axis2.legend(loc='upper left')
    axis2.set_xlabel('energy (keV)')
    if plot_dir is not None:
        fig.savefig(f'{plot_dir}/pos{n_pos}_wfs.png',dpi=300, bbox_inches='tight')
    axis0.errorbar(det_names,fwhm,yerr=fwhm_err,marker='o',ls='',label=f'String {string}')
    axis1.plot(det_names,pos,marker='o',ls='',label=f'String {string}')
    axis0.set_ylabel('FWHM (keV)')
    axis0.set_xticklabels(det_names, rotation = 90, ha="right")
    axis0.legend(loc='upper right')
    axis0.grid()
    axis1.set_ylabel('pulser position (keV)')
    axis1.set_xticklabels(det_names, rotation = 90, ha="right")
    axis1.legend(loc='upper right')
    axis1.grid()
    print('Updating dict file',dict_file)
    with open(dict_file,'w') as f:
        json.dump(test_dict, f, indent=4)
    if plot_dir is not None:
        fig0.savefig(f'{plot_dir}/pos{n_pos}_fwhms.png',dpi=300, bbox_inches='tight')
        fig1.savefig(f'{plot_dir}/pos{n_pos}_positions.png',dpi=300, bbox_inches='tight')
        fig2.savefig(f'{plot_dir}/pos{n_pos}_fwhms_strings.png',dpi=300, bbox_inches='tight')


def fit_gaussian_peak(energies, nb = 500, ran = 20, relative = False):
    xm = (np.percentile(energies, 50))
    xlo, xhi = xm - ran, xm + ran
    hc, bc = np.histogram(energies, bins=np.linspace(xlo,xhi,nb))
    bc = pgh.get_bin_centers(bc)
    #guess, b_ = au.get_gaussian_guess(hc, bc)
    #par, pcov = curve_fit(au.gaussian, bc, hc, p0=guess)
    perr = np.sqrt(np.diag(pcov))
    if relative: fwhm, fwhm_err = par[2]*2.355/par[1]*100, perr[2]*2.355/par[1]*100
    else: fwhm, fwhm_err = par[2]*2.355, perr[2]*2.355
    return par[1], fwhm, fwhm_err, hc, bc
        
        
def process_calibration(chn, file_dir, sigma, flat, tau, n_events = 50000, wcut=2100, sconv=200, ns = 1/16, us = 1000/16, axis = None ):
    print(chn,'Calibration')
    store = lh5.LH5Store()
    count = 0
    for p, d, files in os.walk(file_dir):
        d.sort()    
        for i, f in enumerate(sorted(files)):
            if (f.endswith(".lh5")) & ("raw" in f):
                lh5_file = f"{file_dir}/{f}"
                #print(lh5_file)
                wfs0, n_rows = store.read_object(f"{chn}/raw/waveform/values", lh5_file)
                baseline0, n_rows = store.read_object(f"{chn}/raw/baseline", lh5_file)
                wfs, baseline = wfs0.nda, baseline0.nda
                n_ev, wsize = len(wfs), len(wfs[0])
                nsize, fsize = round(wsize-wcut), round(wsize-wcut-sconv)
                wfs, dts = wfs[:,:nsize], np.linspace(0,nsize/us,nsize)
                wfs = bl_subtract(wfs, baseline)
                wf_cusp = np.zeros(( n_ev, sconv+1 ))
                cusp_func = cusp_filter(fsize, sigma*us, int(flat*us), tau*ns)
                cusp_func(wfs[:,:fsize],wf_cusp)
                if count == 0:
                    n_tot = n_ev
                    cusp_ene = np.max(wf_cusp,axis=1)
                else:
                    n_tot += n_ev
                    cusp_ene = np.append(cusp_ene, np.max(wf_cusp,axis=1))
                count += 1
                print('n.file',count,'tot. events',n_tot)
            if n_tot > n_events: break
    
    #result_map = pcal.calibrate_th228(cusp_ene, pulser = 0, plot=plot)
    # calibration
    xlo = np.percentile(cusp_ene, 5)
    xhi = np.percentile(cusp_ene, 100)
    nb = 3000#int((xhi-xlo)/xpb)
    
    hist, bin_edges = np.histogram(cusp_ene, bins=np.linspace(xlo,xhi,nb))
    bin_centers = pgh.get_bin_centers(bin_edges)
    
    peak_idxs, _ = pcal.find_peaks(hist,prominence=hist.max()/10)
    peak_energies = bin_centers[peak_idxs]
    
    peak_max = peak_energies[1]
    peak_last = peak_energies[-1]
        
    if axis is not None:
        plt.figure(figsize=(12, 6.75), facecolor='white')
        axis.plot(bin_centers, hist, ds='steps', lw=1, c='b')
        for e in peak_energies:
            axis.axvline(e, color="r", lw=1, alpha=0.6)
        axis.axvline(peak_max, color='k', lw=2, label='first peak')
        axis.axvline(peak_last, color='g', lw=2, label='last peak')
        axis.set_xlabel("Energy [uncal]", ha='right', x=1)
        axis.set_ylabel("Filtered Spectrum", ha='right', y=1)
        axis.set_yscale('log')
        axis.legend()
    rough_kev_per_adc = (pcal.cal_peaks_th228[0] - pcal.cal_peaks_th228[-1])/(peak_max-peak_last)
    rough_kev_offset = pcal.cal_peaks_th228[0] - rough_kev_per_adc * peak_max
    return rough_kev_per_adc, rough_kev_offset
    
                
def pulser_dsp_analysis(run, chmap, test_dict_file, n_test = 1, n_acq = 0, nstring = None, plot_dir = None ):
    store = lh5.LH5Store()
    lh5_file = f'r{run:03}/pulser_test/dsp/l200_r{run:03}_pulser_test{n_test}_acq{n_acq}.lh5'
    with open(test_dict_file,'r') as file:
        test_dict = json.load(file)
    fig0, axis0 = plt.subplots(figsize=(20,8), facecolor='white')
    fig1, axis1 = plt.subplots(figsize=(20,8), facecolor='white')
    fig2, axis2 = plt.subplots(nrows=4, ncols=3,figsize=(24,16), facecolor='white')
    chn_off = (26,33,46,47,48,50,84,85,94,112,116,117,120,122)
    ge_off = ['V07298B', 'P00665A', 'V01386A', 'V01403A', 'V01404A', 'B00091D', 'P00537A', 'B00091B', 'P00538B', 'P00661A', 'P00665B', 'P00698B']
    ge_drift = ['V01406A', 'V01415A', 'V01387A', 'P00665C','P00748B', 'P00748A']
    strings = np.array([int(chmap[ch]['location']['string']) for ch in chmap.keys() if chmap[ch]['system']=='geds'])
    if nstring is None: nstring = strings.max()
    for string in range(1,nstring+1):
        print('string',string)
        ax2 = axis2.flat[string-1]
        ge_keys = [ch for ch in chmap.keys() if chmap[ch]['system']=='geds' and chmap[ch]['location']['string']==string]
        if run < 20:
            ge_keys = [ch for n, ch in zip(ge_numbers,det_names) if n not in chn_off ]
        else:
            ge_keys = [ch for ch in ge_keys if ch not in ge_off]
            ge_keys = [ch for ch in ge_keys if ch not in ge_drift]
        ge_numbers = [chmap[ch]['daq']['fcid'] for ch in ge_keys]
        chns = [f'ch{n:03}' for n in ge_numbers]
        ge_rawid = [chmap[ch]['daq']['rawid']  for ch in ge_keys]
        if run < 20: ge_table = chns
        else: ge_table = [f'ch{id}' for id in ge_rawid]
        fwhm, fwhm_err = np.zeros(len(ge_table)), np.zeros(len(ge_table))
        pos, pos_err = np.zeros(len(ge_table)), np.zeros(len(ge_table))
        #fig, axis = plt.subplots(nrows=4, ncols=3,figsize=(24,16), facecolor='white')
        for i, chn in enumerate(ge_table[:]):
            det = ge_keys[i]
            #ax = axis.flat[i]
            ene, n_rows = store.read_object(f'{chn}/pulser_energy', lh5_file)
            timestamp, n_rows = store.read_object(f'{chn}/timestamp', lh5_file)
            ene, timestamp = ene.nda, timestamp.nda
            # stability
            xm = (np.percentile(ene, 50))
            xlo, xhi, nb = xm - 20, xm + 20, 500
            #tspace = np.linspace(timestamp.min(),timestamp.max(),100)
            #dt = [datetime.fromtimestamp(ts) for ts in timestamp]
            #dspace_p = [datetime.fromtimestamp(ts) for ts in tspace]
            """dt = np.linspace(0,len(ene),len(ene))
            tspace = np.linspace(0,len(ene),100)
            pspace = np.linspace(xm - 5, xm + 5,100)
            plt.sca(ax)
            php = Histdd(dt, ene, bins=(tspace, pspace))
            php.plot(log_scale=True,cmap='viridis',colorbar=True)
            havg = php.average(axis=1)
            hstd = [php[ii:].std() for ii in range(php[:,:].shape[0])]
            avg, std = havg[:].mean(), havg[:].std()/np.sqrt(len(havg[:]))
            plt.plot(tspace[1:],havg[:],color='r',label=f'avg {avg:.2f} $\pm$ {std:.2f} keV')
            #plt.plot(tspace[1:],havg[:]-hstd,color='b',ls='-')
            #plt.plot(tspace[1:],havg[:]+hstd,color='b',ls='-')
            ax.set_title(f'{det} - {chn} - pulser')
            ax.legend(title=f'S{string} - {chn} - {det}',loc='upper left')
            ax.set_xlabel('time')
            ax.set_ylabel('energy (keV)')"""
            # energy results
            hc, bc = np.histogram(ene, bins=np.linspace(xlo,xhi,nb))
            bc = pgh.get_bin_centers(bc)
            #guess, b_ = au.get_gaussian_guess(hc, bc)
            #par, pcov = curve_fit(au.gaussian, bc, hc, p0=guess)
            perr = np.sqrt(np.diag(pcov))
            pos[i], fwhm[i], fwhm_err[i] = par[1], par[2]*2.355, perr[2]*2.355
            test_dict[f'test{n_test}'][f'{n_acq}'][chn] = {
                'position':pos[i],
                'fwhm':fwhm[i],
                'fwhm_err':fwhm_err[i],
                #'avg':avg,
                #'std':std
            }
            ax2.plot(bc,hc,label=f'{det}: {pos[i]:.2f} keV, FWHM={fwhm[i]:.1f} keV')
        ax2.legend(title=f'String {string}',loc='upper left')
        ax2.set_xlabel('energy (keV)')
        if string == 1: dets, fwhms, fwhm_errs, poss = ge_keys, fwhm, fwhm_err, pos
        else: dets,fwhms,fwhm_errs,poss =np.append(dets,ge_keys),np.append(fwhms,fwhm),np.append(fwhm_errs,fwhm_err),np.append(poss,pos)
        #if plot_dir is not None and string == 1:
        #    fig.savefig(f'{plot_dir}/test{n_test}_acq{n_acq}_stability_string{string}.png',dpi=300, bbox_inches='tight')
        axis0.errorbar(ge_keys,fwhm,yerr=fwhm_err,marker='o',ls='',label=f'String {string}')
        axis1.plot(ge_keys,pos,marker='o',ls='',label=f'String {string}')
    axis0.set_ylabel('FWHM (keV)')
    axis0.set_xticklabels(dets, rotation = 90, ha="right")
    axis0.legend(loc='upper right')
    axis0.grid()
    axis1.set_ylabel('pulser position (keV)')
    axis1.set_xticklabels(dets, rotation = 90, ha="right")
    axis1.legend(loc='upper right')
    axis1.grid()
    print('Updating dict file',test_dict_file)
    with open(test_dict_file,'w') as f:
        json.dump(test_dict, f, indent=4)
    if plot_dir is not None:
        fig0.savefig(f'{plot_dir}/test{n_test}_acq{n_acq}_fwhms.png',dpi=300, bbox_inches='tight')
        fig1.savefig(f'{plot_dir}/test{n_test}_acq{n_acq}_positions.png',dpi=300, bbox_inches='tight')
        fig2.savefig(f'{plot_dir}/test{n_test}_acq{n_acq}_fwhms_strings.png',dpi=300, bbox_inches='tight')
        

    
def plot_pulser_fft(chns, det_names, f_raw, nwf = 10, apply_mask = False, fc_bl = False, us = 1000/16,
                          ylim=(1e4,1e9), plot_dir = None, title='String 1' ):
    f = h5py.File(f_raw,'r')
    fig, axis = plt.subplots(figsize=(12,6.75), facecolor='white')
    for i, chn in enumerate(chns):
        det = det_names[i]
        wfs = f[f'{chn}/raw/waveform/values'][:]
        timestamp = f[f'{chn}/raw/timestamp'][:]
        baseline = f[f'{chn}/raw/baseline'][:]
        bl_mean, bl_std, bl_slope, bl_intercept = linear_slope_fit(wfs[:,:1000])
        wf_max = wfs[:,:].max(axis=1)
        min_thr = bl_mean.mean()-(wf_max.mean()-bl_mean.mean())
        max_thr = bl_mean.mean()+(wf_max.mean()-bl_mean.mean())/2
        mask = (wfs[:,2000:].max(axis=1)>max_thr) & (wfs[:,:].min(axis=1)>min_thr) & (np.abs(bl_slope)<0.2)
        if apply_mask: wfs, timestamp, baseline = wfs[mask], timestamp[mask], baseline[mask]
        n_ev, wsize = len(wfs), len(wfs[0])
        dts = np.linspace(0,wsize/us,wsize)
        print(chn,'n. pulser events',n_ev,'original size',wsize)
        if fc_bl: wfs = bl_subtract(wfs, baseline)
        #freq, power = au.calculate_fft(wfs[:,:2500], nbls = 1000)
        plt.plot(freq[1:], power[1:],label=f'{chn} - {det}')
    plt.xlabel('frequency (MHz)')
    plt.ylabel('power spectral density')
    plt.ylim(ylim[0],ylim[1])
    plt.xscale('log')
    #plt.yscale('log')
    plt.legend(title=title,loc='upper right')

def create_script_ulite(local_dir, run, n_test, n_acq, daq_conv, plot_dir, log_dir):
    print('Creating scripts for submission in U-LITE')
    job_name = f'pul_{n_test}_{n_acq}'
    curr_script = log_dir + '/script_' + job_name + '.sh'
    fs = open(curr_script,'w')
    if daq_conv: fs.write(f'python {local_dir}/pulser_analysis.py -p -r {run} -nt {n_test} -na {n_acq} -dc > {log_dir}/output_{job_name}.out')
    else: fs.write(f'python {local_dir}/pulser_analysis.py -p -r {run} -nt {n_test} -na {n_acq} > {log_dir}/output_{job_name}.out')
    fs.close()
    scripterr = log_dir + '/file_' + job_name + '.err'
    scriptlog = log_dir + '/file_' + job_name + '.log'
    queue_cmd = 'qsub -q legend -l mem=4000mb -N ' + job_name + ' -V -d ' + local_dir + ' -m abe -M valerio.dandrea@lngs.infn.it -e localhost:' + scripterr + ' -o localhost:' + scriptlog + ' ' + curr_script
    print(queue_cmd)
    os.system(queue_cmd)
    print()

if __name__=="__main__":
    main()
