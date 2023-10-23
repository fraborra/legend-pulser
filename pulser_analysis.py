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
from pygama.pargen.dsp_optimize import run_one_dsp

from dspeed.processors import bl_subtract, linear_slope_fit
from dspeed.processors import zac_filter
from dspeed.processors import cusp_filter
from dspeed.processors import presum

import lgdo.lh5_store as lh5
from lgdo import Array

from legendmeta import LegendMetadata

def main():
    par = argparse.ArgumentParser(description="pulser processing")
    arg, st, sf = par.add_argument, "store_true", "store_false"
    arg("-dc", "--daq_conversion", action=st, help="daq_conversion")
    arg("-r", "--run", nargs=1, action="store", help = "data run")
    arg("-na", "--acq_number", nargs=1, help="number of acquisition")
    args = vars(par.parse_args())
    
    local_dir = os.path.dirname(os.path.realpath(__file__))
    
    if args["run"]:
        period, run = int(args["run"][0]), int(args["run"][1])
        run, period = f'r{run:03}', f'p{period:02}'
    else:
        print('Run and period not provided')
        return
    run_dir = f'{local_dir}/l200-{period}/{run}'
    
    log_dir = local_dir + '/_out'
    try: os.mkdir(log_dir)
    except: FileExistsError: print ("Directory '%s' already exists" % log_dir)
    else: print ("Directory '%s' created" % log_dir)
    
    
    n_test = int(args["test_number"][0])
    n_acq = int(args["acq_number"][0])
    daq_conv = False
    if args['daq_conversion']: daq_conv = True
        
    plot_dir = run_dir + '/plots'
    try: os.mkdir(plot_dir)
    except: FileExistsError: print (f'Directory {plot_dir} already exists')
    else: print(f'Directory {plot_dir} created')
    
    dsp_dir = run_dir + '/dsp'
    try: os.mkdir(dsp_dir)
    except: FileExistsError: print (f'Directory {dsp_dir} already exists')
    else: print(f'Directory {dsp_dir} created')

    raw_dir = f'/lfs/l1/legend/users/dandrea/test_data/raw/tst/{period}-{run}'
    
    # metadata
    version = 'tmp/auto'
    meta_path = f'/lfs/l1/legend/data/public/prodenv/prod-blind/{version}/inputs/'
    lmeta = LegendMetadata(path=meta_path)
    chmap = lmeta.hardware.configuration.channelmaps.on("20230311T235840Z")
    dsp_proc_chain = lmeta.dataprod.config.tier_dsp['L200-p01-r%-T%-ICPC-dsp_proc_chain']
    #par_dsp = lmeta.dataprod.overrides.dsp.cal.p03.r000['L200-p03-r000-cal-T%-par_dsp-overwrite']
    par_dir = f'/lfs/l1/legend/data/public/prodenv/prod-blind/{version}/generated/par'
    par_dsp = f'{par_dir}/dsp/cal/p06/r005/l200-p06-r005-cal-20230717T130655Z-par_dsp.json'
    par_hit = f'{par_dir}/hit/cal/p06/r005/l200-p06-r005-cal-20230717T130655Z-par_hit.json'
    
    test_dict_file = f'l200-{period}/{run}/l200-{period}-{run}-pulser-dict.json'
    with open(test_dict_file,'r') as file:
        test_dict = json.load(file)
    this_dict = test_dict[f'test{n_test}'][f'{n_acq}']
    print('Analysis of pulser test',n_test,'acquistion',n_acq)
    lh5_file = f'{dsp_dir}/l200-{period}-{run}_pulser-test_acq{n_acq}.lh5'
    
    this_dict = test_dict[f'test{n_test}'][f'{n_acq}']
    key_list = [f'l200-p02-r{run:03}-tst-{key}' for key in this_dict['key_list']]
    raw_list = [f'{raw_dir}/{key}_raw.lh5' for key in key_list]
    print('raw_list:',raw_list)
    pulser_waveform(raw_list, chmap, dsp_config, hit_config, test_dict_file, n_test=n_test, n_acq=n_acq,
                    wcut=2100, plot_dir=plot_dir,lh5_file=lh5_file)


def plot_pulser_waveforms(ge_key, det_name, raw_list, puls_chn = 'ch1027201',
                          nwf=10, us=1000/16, xlim=(3060,3090), ylim=(1e4,1e9), plot_dir=None ):
    meta_path = f'/lfs/l1/legend/users/dandrea/legend-metadata'
    lmeta = LegendMetadata(path=meta_path)
    chmap = lmeta.hardware.configuration.channelmaps.on("20230311T235840Z")
    rawid = chmap[ge_key]['daq']['rawid']
    chn = f'ch{rawid}'
    store = lh5.LH5Store()
    daq_puls, n_rows = store.read_object(f"{puls_chn}/raw/daqenergy", raw_list)
    daq_puls = daq_puls.nda
    pulser_tag = daq_puls>1000
    waveforms, n_rows = store.read_object(f"{chn}/raw/waveform/values", raw_list, idx=pulser_tag)
    timestamp, n_rows = store.read_object(f"{chn}/raw/timestamp", raw_list, idx=pulser_tag)
    baseline, n_rows = store.read_object(f"{chn}/raw/baseline", raw_list, idx=pulser_tag)
    n_ev, wsize = waveforms.nda.shape
    dts = np.linspace(0,wsize/us,wsize)
    print(chn,'n. pulser events',n_ev,'size',wsize)
    
    fig, axis = plt.subplots(figsize=(12,6.75), facecolor='white')
    axin = axis.inset_axes([0.55, 0.1, 0.4, 0.55])
    for j in range(nwf):
        wf = waveforms.nda[j,:]
        if j < 10: axis.plot(wf,label=f'{j} {(timestamp.nda[j]-timestamp.nda[0])*1e3:.2f} ms')
        else: axis.plot(wf)
        axin.plot(wf)
    axis.set_xlabel('samples')
    axis.legend(title=f'{ge_key} - {det_name}',loc='upper left')
    axin.set_xlim(xlim[0],xlim[1])
    axin.set_yticklabels('')


def pulser_processing(period, run, ge_key, test_dict_file, raw_dir, puls_chn = 'ch1027201',
                      nwf = 10, us = 1000/16, plot_dir = None ):
    with open(test_dict_file,'r') as file:
        test_dict = json.load(file)
    meta_path = f'/lfs/l1/legend/users/dandrea/legend-metadata'
    lmeta = LegendMetadata(path=meta_path)
    chmap = lmeta.hardware.configuration.channelmaps.on("20230311T235840Z")
    dsp_proc_chain = lmeta.dataprod.config.tier_dsp['L200-p01-r%-T%-ICPC-dsp_proc_chain']
    par_dsp = lmeta.dataprod.overrides.dsp.cal.p03.r000['L200-p03-r000-cal-T%-par_dsp-overwrite']
    rawid = chmap[ge_key]['daq']['rawid']
    chn = f'ch{rawid}'
    store = lh5.LH5Store()
    
    fig, axis = plt.subplots(nrows=2,ncols=4,figsize=(20,8), facecolor='white')
    fig1, axis1 = plt.subplots(nrows=2,ncols=4,figsize=(20,8), facecolor='white')
    string = int(chmap[ge_key]['location']['string'])
    cc4_id = chmap[ge_key]['electronics']['cc4']['id']
    attenuations = np.array(test_dict[cc4_id]['attenuations'])
    out_voltages = np.array([pulser_db_in_voltage(att) for att in attenuations])
    print(out_voltages)
    rough_energies = np.array([pulser_db_in_voltage(att,energy=1) for att in attenuations])
    amplitudes = np.array(test_dict[cc4_id]['amplitudes'])
    keys = test_dict[cc4_id]['keys']
    pulser_pos = np.zeros(len(keys))
    print('Pulser processing for',ge_key,'String',string,'CC4',cc4_id)
    for i, key in enumerate(keys[:]):
        raw_file = f'{raw_dir}/l200-{period}-{run}-tst-{key}-tier_raw.lh5'
        daq_puls, n_rows = store.read_object(f"{puls_chn}/raw/daqenergy", raw_file)
        daq_puls = daq_puls.nda
        pulser_tag = daq_puls>1000
        
        waveforms, n_rows = store.read_object(f"{chn}/raw/waveform", raw_file, idx=pulser_tag)
        timestamp, n_rows = store.read_object(f"{chn}/raw/timestamp", raw_file, idx=pulser_tag)
        baseline, n_rows = store.read_object(f"{chn}/raw/baseline", raw_file, idx=pulser_tag)
        
        n_ev, wsize = waveforms.values.nda.shape
        dts = np.linspace(0,wsize/us,wsize)
        print(key, 'n. pulser events',n_ev,'size',wsize)
        ax = axis.flat[i]
        axin = ax.inset_axes([0.55, 0.1, 0.4, 0.55])
        for j in range(nwf):
            wf = waveforms.values.nda[j]
            ax.plot(dts,wf,label=f'{j} {(timestamp.nda[j]-timestamp.nda[0])*1e3:.2f} ms')
            axin.plot(dts,wf)
        ax.set_xlabel('time ($\mu$s)')
        ax.legend(title=f'{ge_key} - CC4-{cc4_id}\nPulser att. {attenuations[i]} dB',loc='upper left')
        axin.set_xlim(40.4,55.2)
        axin.set_yticklabels('')
        
        # DSP production
        start_t = time.time()
        tb_data = lh5.Table(col_dict={"waveform": waveforms,"timestamp": timestamp,"baseline": baseline})
        dsp_data = run_one_dsp(
            tb_data,
            dsp_proc_chain,
            db_dict=par_dsp
        )
        print(f'Time to produce DSP data {time.time()-start_t:.1f} s')
        
        # energy spectrum
        energies = dsp_data['cuspEmax'].nda
        ene_mean = np.mean(energies)
        ene_space = np.linspace(ene_mean-100,ene_mean+100,500)
        pulser_pos[i] = ene_mean
        ax1 = axis1.flat[i]
        ax1.hist(energies,bins=ene_space,label='data')
        ax1.set_xlabel('uncalibrated energy (ADC)')
        ax1.legend(title=f'{ge_key} - CC4-{cc4_id}\nPulser att. {attenuations[i]}',loc='upper left')
        
        #hit_config = hit_config_prod
        #print('Using hit config',hit_config)
        #with open(hit_config, "r") as file:
        #    hit = json.load(file)
        #m_cal = hit[chn_old]['operations']['cuspEmax_ctc_cal']['parameters']['a']
        #q_cal = hit[chn_old]['operations']['cuspEmax_ctc_cal']['parameters']['b']
        #cusp_ene = m_cal * cusp_ene + q_cal
        
        """
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
        """
    fig2, ax2 = plt.subplots(figsize=(12,6.75), facecolor='white')
    ax2.plot(out_voltages,pulser_pos,label='data')
    ax2.legend(title=f'{ge_key} - CC4-{cc4_id}')
    ax2.set_xlabel('pulser voltages (V)')
    ax2.set_ylabel('pulser position (ADC)')

def pulser_db_in_voltage(attenuation_db, pulser_voltage = 2.5, energy = False):
    ratio = 10.0**(-attenuation_db / 20.0)
    output = ratio * pulser_voltage
    output_MeV = 10.0**(-32 / 20.0) * pulser_voltage
    if energy: return output/output_MeV
    else: return output

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

if __name__=="__main__":
    main()
