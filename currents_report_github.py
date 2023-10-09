#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 18:56:14 2022

@author: oystein and jógvan

Script for making standardized report for current measurement
All input comes from the postprocessing script.

Need to make:
    - rose
    - What method do we use for nanpercentile? The results can differ a lot.
    - You need to manually compile bibtex
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.dates as dt
import matplotlib.ticker as ticker
import os
from natsort import natsorted # pip install natsort
from datetime import timedelta
import utide
import pickle
def nearest_date_ind(items, pivot):
    time_diff = np.abs([date - pivot for date in items])
    return time_diff.argmin(0)
import warnings
warnings.filterwarnings('ignore') # ignore All nan warnings for slices

    
from pylatex import Document, Section, Subsection, Tabular, Math, TikZ, Axis, \
    Plot, Figure, Matrix, Alignat, NoEscape, Command, SubFigure, Table, Label
from pylatex.section import Chapter
import strandalinja_US
import time as counttime
import cmocean
tic_all = counttime.perf_counter()
plt.close('all')
###############
# Settings
savefig = True
plot_map = True 
plotwidth = 15
plotheight = 10

folder = '../../../Data/ADCP/VELF2301/'
instrument = 'AWAC_RAW'
data_folder = folder+'proc_ASCII/'
report_folder = folder+'curr_report/'
titlename = 'VELF2301'
name = 'VELF2301' 
firstbin = 1
lastbin = 5
topbin = 5
midbin = 3
botbin = 1
u10 = 1.65 #Three months
u50 = 1.85 #Three months

# Create output folter
outputfolder = report_folder+'cur_fig/'
if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)

# Read data from read_awac_rawascii.py
mag_all = np.load(data_folder+'mag_corr.npy')
dir_all = np.load(data_folder+'dir_corr.npy')
depth = np.load(data_folder+'d.npy')
depth_av = np.load(data_folder+'d_av.npy')
bindepths = np.load(data_folder+'bindepths.npy')
binheights = np.load(data_folder+'binheights.npy')
time = np.load(data_folder+'time.npy', allow_pickle=True)
dir_all_rad = np.deg2rad(dir_all)
lat, long, mag_dec,instr_height = np.load(data_folder+'vars.npy')
ID, date_first, date_last, instrument_type, serial_number, instrument_frequency, profile_interval, average_interval,\
number_of_measurements, bin_size, number_of_bins, blanking_distance,ping_hz = np.load(data_folder+'string_vars.npy')

###Calculate U and V vectors
vel_u_all = mag_all*np.sin(dir_all_rad)
vel_v_all = mag_all*np.cos(dir_all_rad)
########################
### Average currents ###
dir_all_rad_mat = np.pi/2-dir_all_rad
z = mag_all*np.exp(dir_all_rad_mat*1j)
Z = np.nanmean(z[:,firstbin-1:lastbin-1],axis=1)

# Calculate velocity and angle
mag_av = np.sqrt(np.real(Z)**2+np.imag(Z)**2)
dir_av_rad = np.mod(np.pi/2-np.angle(Z),2*np.pi)
dir_av = np.rad2deg(dir_av_rad)

vel_u_av = mag_av*np.sin(dir_av_rad)
vel_v_av = mag_av*np.cos(dir_av_rad)

def set_legend_and_grid_and_tightlayout_and_show_and_save(figurename, loc):
    fig = plt.gcf() 
    for i in range(0,len(fig.axes)):
        fig.axes[i].grid(True)
        fig.axes[i].legend(loc=loc)
    plt.tight_layout()
    plt.show()
    if savefig == True:
        plt.savefig(outputfolder+figurename+'.png',dpi = 300)
        plt.savefig(outputfolder+figurename+'.pdf',dpi = 600)        
    else:
        print('savefig is set to False')
    
########################
### Timeseries plots ###
### Current speed
fig, axs = plt.subplots(4, sharex=True, figsize = (plotwidth*0.5, 4/3*plotwidth*0.5))
axs[0].plot(time, mag_all[:,topbin-1],'k', label = 'Top bin', linewidth=0.75)
axs[0].set(title = 'Timeseries of current speed')
axs[0].set(ylabel = 'Current speed, m/s')
axs[1].plot(time, mag_all[:,midbin-1],'k', label = 'Middle bin', linewidth=0.75)
axs[1].set(ylabel = 'Current speed, m/s')
axs[2].plot(time, mag_all[:,botbin-1],'k', label = 'Bottom bin', linewidth=0.75)
axs[2].set(ylabel = 'Current speed, m/s')
axs[3].plot(time, mag_av,'k', label = 'Average', linewidth=0.75)
axs[3].set(ylabel = 'Current speed, m/s')
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
# axs[3].xaxis.set_major_locator(dt.MonthLocator())
axs[3].xaxis.set_major_formatter(dt.DateFormatter('%d %b'))
axs[3].xaxis.set_minor_locator(dt.DayLocator())
axs[3].xaxis.set_minor_formatter(ticker.NullFormatter())
set_legend_and_grid_and_tightlayout_and_show_and_save('timeseries','best')

# Plot current speed and direction for each week. 1 weeks per image
weeks = int(np.ceil((time[-1]-time[0]).days/7))
titletext = 'Timeseries of current speed and current direction for week '
for i in range(0,weeks):  
    week_start = time[0]+timedelta(days=i*7)
    week_end = time[0]+timedelta(days=(i+1)*7)
    week_start_idx = nearest_date_ind(time, week_start)
    week_end_idx = nearest_date_ind(time, week_end)
    fig, axs = plt.subplots(2, sharex=True, figsize = (0.6*plotwidth*0.9, 0.8*4/3*plotwidth*0.32))
    axs[0].plot(time[week_start_idx:week_end_idx], mag_all[week_start_idx:week_end_idx,topbin-1],'k', label = 'Top bin', linewidth=0.75)
    axs[0].plot(time[week_start_idx:week_end_idx], mag_all[week_start_idx:week_end_idx,midbin-1],'r', label = 'Middle bin', linewidth=0.75)
    axs[0].plot(time[week_start_idx:week_end_idx], mag_all[week_start_idx:week_end_idx,botbin-1],'g', label = 'Bottom bin', linewidth=0.75)
    axs[0].plot(time[week_start_idx:week_end_idx], mag_av[week_start_idx:week_end_idx],'b', label = 'Average', linewidth=0.75)
    axs[0].grid(True)
    axs[0].legend(loc='upper right')
    axs[0].set(title = titletext + str(i+1)+str('.'))
    axs[0].set(ylabel = 'Current speed, m/s')
    axs[1].plot(time[week_start_idx:week_end_idx], dir_all[week_start_idx:week_end_idx,topbin-1],'k', label = 'Top bin', linewidth=0.75)
    axs[1].plot(time[week_start_idx:week_end_idx], dir_all[week_start_idx:week_end_idx,midbin-1],'r', label = 'Middle bin', linewidth=0.75)
    axs[1].plot(time[week_start_idx:week_end_idx], dir_all[week_start_idx:week_end_idx,botbin-1],'g', label = 'Bottom bin', linewidth=0.75)
    axs[1].plot(time[week_start_idx:week_end_idx], dir_av[week_start_idx:week_end_idx],'b', label = 'Average', linewidth=0.75)
    axs[1].grid(True)
    axs[1].set_ylim([0,360])
    axs[1].set(ylabel = 'Current direction, °')
    axs[1].legend(loc='upper right')    
    axs[1].set_yticks([0,45,90,135,180,225,270,315,360])
    axs[1].xaxis.set_major_formatter(dt.DateFormatter('%d %b'))
    plt.tight_layout()
    if savefig == True:
        plt.savefig(outputfolder+'timeseries_week_'+str(i+1)+'.png',dpi = 300)
        plt.savefig(outputfolder+'timeseries_week_'+str(i+1)+'.pdf',dpi = 600)
    

#############
### Depth
fig = plt.subplots(figsize =(plotwidth*0.7*0.8, 2/3*plotwidth*0.7*0.8))
plt.plot(time, depth, 'k', linewidth=0.75)
plt.axhline(np.mean(depth), color = "k")
plt.text(time[-1],depth_av,'Average = '+str(depth_av)+'m',ha='right',color='k', backgroundcolor='w')
plt.title(titlename + '. Water level')
plt.ylabel('Water level  (m)')
set_legend_and_grid_and_tightlayout_and_show_and_save('water_level','best')

#####################################
### For all bins make scatterplot ###
# Find maximum at the 8 directions.
max_velocity=np.zeros((max((topbin,lastbin)) - min((botbin,firstbin)) +1,8))
max_angle=np.zeros((max((topbin,lastbin)) - min((botbin,firstbin)) +1,8))
max_idx=np.zeros((max((topbin,lastbin)) - min((botbin,firstbin)) +1,8),dtype=int)
max_idx_store=np.zeros((max((topbin,lastbin)) - min((botbin,firstbin)) +1,8),dtype=int)

for n in range(min((botbin,firstbin)), max((topbin,lastbin))+1):   
    if np.isnan(mag_all[:,n-1]).all() == False:        
        idx = np.where((dir_all[:,n-1]>360-22.5) | (dir_all[:,n-1]<22.5))[0]
        max_idx[n-1,0] = int(np.argmax(mag_all[idx,n-1]))
        max_velocity[n-1,0] = mag_all[idx[max_idx[n-1,0]],n-1]
        max_angle[n-1,0] = dir_all_rad[idx[max_idx[n-1,0]], n-1]
        max_idx_store[n-1,0]=idx[max_idx[n-1,0]]
            
        for i in range(1,8):
            # idx = np.where((dir_all>22.5*i) & (dir_all_rad<=22.5*(i+1)) 
            idx = np.where((dir_all[:,n-1]>45*i-22.5) & (dir_all[:,n-1]<45*i+22.5))[0]
            max_idx[n-1,i] = np.argmax(mag_all[idx,n-1])
            max_velocity[n-1,i] = mag_all[idx[max_idx[n-1,i]],n-1]
            max_angle[n-1,i] = dir_all_rad[idx[max_idx[n-1,i]],n-1]
            max_idx_store[n-1,i] = idx[max_idx[n-1,i]]
    else:
        max_idx[n-1,:] = np.zeros((1,8))
        max_velocity[n-1,:] = np.zeros((1,8))
        max_angle[n-1,:] = np.zeros((1,8)) 

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize =(0.65*plotheight, 0.7*plotheight))
    ax.plot(dir_all_rad[:,n-1], mag_all[:,n-1],'.', markersize=2)
    ax.plot(max_angle[n-1,:], max_velocity[n-1,:],'r.', markersize=5)

    ### Correct to geographic direction
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi/2)
    
    n_directions = 16
    angles = [n / float(n_directions) * 2 * np.pi for n in range(n_directions)]
    plt.xticks(angles, color='black', size=10, zorder = 5)
    ax.set_xticklabels(['N', '22.5°', 'NE', '67.5°', 'E', '112.5°', 'SE', '157.5°', 'S', '202.5°', 'SW', '247.5°', 'W', '292.5°', 'NW','337.5°'])
    
    for i in range(0,8):
        ax.text(max_angle[n-1,i], max_velocity[n-1,i]+np.max(max_velocity)*0.05, "%.3f" %max_velocity[n-1,i], ha="center", va="center")
            
    ax.set_rmin(0)
    ax.set_rmax(np.ceil((np.nanmax(mag_all[:,botbin:topbin])/0.2))*0.2+0.1)
    ax.set_rlabel_position(70)  # Move radial labels away from plotted line
    ax.grid(True)
    plt.title(titlename + '. Bin '+str(n)+', depth: %.1f m'%bindepths[n-1])
    plt.tight_layout()
    plt.show()
    if savefig == True:
        plt.savefig(outputfolder+'scatter_bin'+str(n)+'.png',dpi = 300)
        plt.savefig(outputfolder+'scatter_bin'+str(n)+'.pdf',dpi = 600)
max_vel_store = max_velocity
max_ang_store = max_angle


#####################################
### Average scatter plot ###
# Find maximum at the 8 directions.  
max_velocity=np.zeros(8)
max_angle=np.zeros(8)
max_idx=np.zeros(8,dtype=int)
max_idx_store=np.zeros(8,dtype=int)
        
idx = np.where((dir_av>360-22.5) | (dir_av<22.5))[0]
max_idx[0] = int(np.argmax(mag_av[idx]))
max_velocity[0] = mag_av[idx[max_idx[0]]]
max_angle[0] = dir_av_rad[idx[max_idx[0]]]
max_idx_store[0]=idx[max_idx[0]]
        
for i in range(1,8):
        # idx = np.where((dir_all>22.5*i) & (dir_all_rad<=22.5*(i+1)) 
    idx = np.where((dir_av>45*i-22.5) & (dir_av<45*i+22.5))[0]
    max_idx[i] = np.argmax(mag_av[idx])
    max_velocity[i] = mag_av[idx[max_idx[i]]]
    max_angle[i] = dir_av_rad[idx[max_idx[i]]]
    max_idx_store[i] = idx[max_idx[i]]
    

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize =(0.65*plotheight, 0.7*plotheight))
ax.plot(dir_av_rad, mag_av,'.', markersize=2)
ax.plot(max_angle, max_velocity,'r.', markersize=5)

### Correct to geographic direction
ax.set_theta_direction(-1)
ax.set_theta_offset(np.pi/2)
    
n_directions = 16
angles = [n / float(n_directions) * 2 * np.pi for n in range(n_directions)]
plt.xticks(angles, color='black', size=10, zorder = 5)
ax.set_xticklabels(['N', '22.5°', 'NE', '67.5°', 'E', '112.5°', 'SE', '157.5°', 'S', '202.5°', 'SW', '247.5°', 'W', '292.5°', 'NW','337.5°'])

for i in range(0,8):
    ax.text(max_angle[i], max_velocity[i]+np.max(max_velocity)*0.15, "%.3f" %max_velocity[i], ha="center", va="center")       
ax.set_rmin(0)
ax.set_rmax(np.ceil((np.nanmax(mag_all[:,botbin:topbin])/0.2))*0.2+0.1)
ax.set_rlabel_position(70)  # Move radial labels away from plotted line
ax.grid(True)
          
plt.title(titlename + '. Average over depths %.1f - %.1f m'%(bindepths[lastbin-1], bindepths[firstbin-1]))

plt.tight_layout()
plt.show()

if savefig == True:
    plt.savefig(outputfolder+'scatter_average.png',dpi = 300)
    plt.savefig(outputfolder+'scatter_average.pdf',dpi = 600)
   
# # Frequency table
# bins = [0,0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,0.45,0.5,0.6,0.7,0.8,0.9,1.0,1.25,1.5,1.75,2.0]
# f = open(outputfolder+"table_bin.txt", "w")
# f.write('\\addtolength{\\tabcolsep}{-0.3em}\n')
# f.write('\\begin{table}[h!]\n')
# f.write('\centering\n')
# f.write('\small\n')
# f.write('\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|}\n')
# f.write('\hline\n')
# f.write('Bin & Depth & '+str(bins[1:])[1:-1].replace(',',' &')+str('\\\ \hline \n'))
# for i in range(0,np.size(mag_all,axis=1)):
#     hist, b= np.histogram(mag_all[:,i],bins=bins)
#     hist_1000 = 1000/(np.nansum(hist))*hist # in parts per 1000.
#     if np.sum(hist) > 0:
#         f.write(str(i+1)+' & ' +str(np.round(bindepths[i],1)) + ' ' + str(np.round(hist_1000).astype(int))[1:-1].replace('   ','&').replace('  ','&').replace(' ','&').replace('&',' & ')+str('\\\ \n'))
# f.write('\hline\n')
# f.write('\end{tabular}\n')
# f.write('\caption{Frequency of in parts per thousand of speeds equal or exceeding values.}\n')
# f.write('\label{tab:frequency_bins}\n')
# f.write('\end{table}\n')
# f.write('\\addtolength{\\tabcolsep}{0.0em}\n')
# f.close()




i = 0
f = open(outputfolder+"max_speed.txt", "w")
#f.write('\\addtolength{\\tabcolsep}{-0.3em}\n')
f.write('\\begin{table}[h!]\n')
f.write('\centering\n')
# f.write('\small\n')
f.write('\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|}\n')
f.write('\hline\n')
f.write('\\textbf{No.}'+    '&'+ '\\textbf{Octant (°)}'+'&'+ '\multicolumn{4}{|c|}{\\textbf{Maximum current speed (cm/s)}}'+ '&'+ '\multicolumn{4}{|c|}{\\textbf{Direction of maximum speed (°)}}'+ '\\'+'\\\n')
f.write('\hline\n')
f.write(    '&'+   '\multicolumn{1}{|r|}{\\textbf{Bin}}'+    ' &'+ '\\textbf{Top}'+'&'+ '\\textbf{Middle}'+'&'+ '\\textbf{Bottom}'+'&'+ '\\textbf{Average}'+
        '&'+ '\\textbf{Top}'+'&'+ '\\textbf{Middle}'+'&'+ '\\textbf{Bottom}'+'&'+ '\\textbf{Average}'+ '\\'+'\\')
f.write('\hline\n')
f.write(    '&'+   '\multicolumn{1}{|r|}{\\textbf{Depth}}'+    ' &'+ "%.1f" % (bindepths[topbin-1])+'m'+'&'+ "%.1f" % (bindepths[midbin-1])+'m'+'&'+ "%.1f" % (bindepths[botbin-1])+'m'+'&'+ "%.1f" % (bindepths[lastbin-1])+'-'+"%.1f" % (bindepths[firstbin-1])+'m'+ '&'+ "%.1f" % (bindepths[topbin-1])+'m'+'&'+ "%.1f" % (bindepths[midbin-1])+'m'+'&'+ "%.1f" % (bindepths[botbin-1])+'m'+'&'+ "%.1f" % (bindepths[lastbin-1])+'-'+"%.1f" % (bindepths[firstbin-1])+'m'+ '\\'+'\\')
f.write('\hline\n')
f.write('\\textbf{'+str(i+1)+'}'+'&'+ '\\textbf{'+str(360-22.5)+'-'+ str(45*i+22.5)+ '}'+'&'+ "%.2f" % (max_vel_store[topbin-1,i])+ '&'+ "%.2f" % (max_vel_store[midbin-1,i])+ '&'+ "%.2f" % (max_vel_store[botbin-1,i])+ '&'+ "%.2f" % (max_velocity[i])+ '&'+ "%.0f" % (max_ang_store[topbin-1,i]*180/np.pi)+ '&'+ "%.0f" % (max_ang_store[midbin-1,i]*180/np.pi)+ '&'+ "%.0f" % (max_ang_store[botbin-1,i]*180/np.pi)+ '&'+ "%.0f" % (max_angle[i]*180/np.pi)+ '\\'+'\\\n')
f.write('\hline\n')
for i in range(1,8):
   f.write('\\textbf{'+str(i+1)+'}'+ '&'+'\\textbf{'+ str(45*i-22.5)+ '-'+ str(45*i+22.5)+'}'+ '&'+ "%.2f" % (max_vel_store[topbin-1,i])+ '&'+ "%.2f" % (max_vel_store[midbin-1,i])+ '&'+ "%.2f" % (max_vel_store[botbin-1,i])+ '&'+ "%.2f" % (max_velocity[i])+ '&'+ "%.0f" % (max_ang_store[topbin-1,i]*180/np.pi)+ '&'+ "%.0f" % (max_ang_store[midbin-1,i]*180/np.pi)+ '&'+ "%.0f" % (max_ang_store[botbin-1,i]*180/np.pi)+ '&'+ "%.0f" % (max_angle[i]*180/np.pi)+ '\\'+'\\\n')
   f.write('\hline\n')
f.write('\end{tabular}\n')
f.write('\caption{Maximum currents.}\n')
f.write('\label{tab:max_speed}\n')
f.write('\end{table}\n')
f.close()


i = 0
f = open(outputfolder+"NS9415_50y.txt", "w")
f.write('\\begin{table}[h!]\n')
f.write('\centering\n')
# f.write('\small\n')
f.write('\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|}\n')
f.write('\hline\n')
f.write('\\textbf{No.}'+    '&'+ '\\textbf{Octant (°)}'+'&'+ '\multicolumn{8}{|c|}{\\textbf{Dimensioning current speed (cm/s)}}'+ '\\'+'\\')
f.write('\hline')
f.write(' '+    '&'+ ' '+'&'+ '\multicolumn{4}{|c|}{\\textbf{10 year return period}}'+ '&'+ '\multicolumn{4}{|c|}{\\textbf{50 year return period}}'+ '\\'+'\\\n')
f.write('\hline')
f.write(    '&'+   '\multicolumn{1}{|r|}{\\textbf{Bin}}'+    ' &'+ '\\textbf{Top}'+'&'+ '\\textbf{Middle}'+'&'+ '\\textbf{Bottom}'+'&'+ '\\textbf{Average}'+
        '&'+ '\\textbf{Top}'+'&'+ '\\textbf{Middle}'+'&'+ '\\textbf{Bottom}'+'&'+ '\\textbf{Average}'+ '\\'+'\\')
f.write('\hline\n')
f.write(    '&'+   '\multicolumn{1}{|r|}{\\textbf{Depth}}'+    ' &'+ "%.1f" % (bindepths[topbin-1])+'m'+'&'+ "%.1f" % (bindepths[midbin-1])+'m'+'&'+ "%.1f" % (bindepths[botbin-1])+'m'+'&'+ "%.1f" % (bindepths[lastbin-1])+'-'+"%.1f" % (bindepths[firstbin-1])+'m'+ '&'+ "%.1f" % (bindepths[topbin-1])+'m'+'&'+ "%.1f" % (bindepths[midbin-1])+'m'+'&'+ "%.1f" % (bindepths[botbin-1])+'m'+'&'+ "%.1f" % (bindepths[lastbin-1])+'-'+"%.1f" % (bindepths[firstbin-1])+'m'+ '\\'+'\\')
f.write('\hline\n')
f.write('\\textbf{'+str(i+1) + str('}')+ '&'+'\\textbf{'+str(360-22.5)+ '-'+ str(45*i+22.5)+'}'+ '&'+ "%.2f" % (max_vel_store[topbin-1,i]*u10)+ '&'+ "%.2f" % (max_vel_store[midbin-1,i]*u10)+ '&'+ "%.2f" % (max_vel_store[botbin-1,i]*u10)+ '&'+ "%.2f" % (max_velocity[i]*u10)+ '&'+ "%.2f" % (max_vel_store[topbin-1,i]*u50)+ '&'+ "%.2f" % (max_vel_store[midbin-1,i]*u50) + '&'+ "%.2f" % (max_vel_store[botbin-1,i]*u50)+ '&'+ "%.2f" % (max_velocity[i]*u50)+ '\\'+'\\')
f.write('\hline\n')
for i in range(1,8):
    f.write('\\textbf{'+str(i+1)+'}'+ '&'+'\\textbf{'+ str(45*i-22.5)+ '-'+ str(45*i+22.5)+'}'+ '&'+ "%.2f" % (max_vel_store[topbin-1,i]*u10)+ '&'+ "%.2f" % (max_vel_store[midbin-1,i]*u10)+ '&'+ "%.2f" % (max_vel_store[botbin-1,i]*u10)+ '&'+ "%.2f" % (max_velocity[i]*u10)+  '&'+ "%.2f" % (max_vel_store[topbin-1,i]*u50)+ '&'+ "%.2f" % (max_vel_store[midbin-1,i]*u50)+ '&'+ "%.2f" % (max_vel_store[botbin-1,i]*u50)+ '&'+ "%.2f" % (max_velocity[i]*u50)+'\\'+'\\')
    f.write('\hline\n')
f.write('\end{tabular}\n')
f.write('\caption{10- and 50-year dimensioning currents according to NS9415:2021.}\n')
f.write('\label{tab:NS9415_50y}\n')
f.write('\end{table}\n')
f.close()

#####################
### Vertical plot ###    
fig, ax = plt.subplots(figsize =(0.7*plotheight, plotheight))
plt.plot(np.nanmax(mag_all,0), -bindepths, 'r.--', linewidth=1.5, label = 'Max speed for all bins')
plt.plot((np.nanmax(mag_av),np.nanmax(mag_av)), (-bindepths[firstbin-1],-bindepths[lastbin-1]), 'r', linewidth=2, label = 'Max speed for bins '+str(firstbin)+'-'+str(lastbin))
plt.plot(np.nanmean(mag_all,0), -bindepths, 'g.--', linewidth=1.5, label = 'Mean speed for all bins')
plt.plot((np.nanmean(mag_av),np.nanmean(mag_av)), (-bindepths[firstbin-1],-bindepths[lastbin-1]), 'g', linewidth=2, label = 'Mean speed for bins '+str(firstbin)+'-'+str(lastbin))
plt.plot(np.nanpercentile(mag_all, 10, axis=0, out=None), -bindepths, 'k.--', linewidth=0.75, label = 'Percentiles: 10,50,90,95,99,99.9')
plt.plot(np.nanpercentile(mag_all, 50, axis=0, out=None), -bindepths, 'k.--', linewidth=0.75)
plt.plot(np.nanpercentile(mag_all, 90, axis=0, out=None), -bindepths, 'k.--', linewidth=0.75)
plt.plot(np.nanpercentile(mag_all, 95, axis=0, out=None), -bindepths, 'k.--', linewidth=0.75)
plt.plot(np.nanpercentile(mag_all, 99, axis=0, out=None), -bindepths, 'k.--', linewidth=0.75)
plt.plot(np.nanpercentile(mag_all, 99.9, axis=0, out=None), -bindepths, 'k.--', linewidth=0.75)
for i in range(botbin,topbin+1):
    plt.text(np.nanmax(mag_all,0)[i-1], -bindepths[i-1],str(i), fontsize = 'large')
ax.set_ylim([-np.mean(depth),0])
ax.set_xlim([0,np.ceil(np.nanmax(mag_all[:,np.where((bindepths>0))])*10)/10])
plt.grid(True)
plt.title(titlename + '. Current speeds against depth')
plt.ylabel('Depth (m)')
plt.xlabel('Current speed (m/s)')
set_legend_and_grid_and_tightlayout_and_show_and_save('profile','best')    


# #####################
#%% ### UTIDE ### 
# Create Latex document for appendix
app2 = Document(documentclass='memoir')
with app2.create(Chapter('Tidal predictions')):

    for i in range(0,4):
        if i == 0:
            temp_u = vel_u_av; temp_v = vel_v_av; temp_mag = mag_av; temp_dir = dir_av; filename = 'averagebin'
        if i == 1:
            temp_u = vel_u_all[:,topbin-1]; temp_v = vel_v_all[:,topbin-1]; temp_mag = mag_all[:,topbin-1]; temp_dir = dir_all[:,topbin-1]; filename = 'topbin'
        if i == 2:
            temp_u = vel_u_all[:,midbin-1]; temp_v = vel_v_all[:,midbin-1]; temp_mag = mag_all[:,midbin-1]; temp_dir = dir_all[:,midbin-1]; filename = 'midbin'
        if i == 3:
            temp_u = vel_u_all[:,botbin-1]; temp_v = vel_v_all[:,botbin-1]; temp_mag = mag_all[:,botbin-1]; temp_dir = dir_all[:,botbin-1]; filename = 'botbin'
        print('Staring Utide')
        coef = utide.solve(
            np.array(time), temp_u, temp_v,
            lat=lat,
            verbose=False,
            # nodal=True,
            trend = False,
            # trend=True,
            # method='ols',
            # conf_int='linear',
            # Rayleigh_min=1.0,
          
        )
        tide = utide.reconstruct(time, coef)
        
        print('\nVariance calulated as for t_tide (Matlab)')
        tide_temp_mag = np.sqrt(tide['u']**2+tide['v']**2)
        tide_temp_mag_angle = np.angle(-tide['v']-tide['u']*1j)+np.pi
        tide_temp_mag_angle_deg = np.rad2deg(tide_temp_mag_angle)
        
        cov_tide_u = np.cov(tide['u'])
        cov_tide_v = np.cov(tide['v'])
        cov_tide_total = cov_tide_u+cov_tide_v
        cov_temp_u = np.ma.cov(np.ma.masked_invalid(temp_u))
        cov_temp_v = np.ma.cov(np.ma.masked_invalid(temp_v))
        cov_vel_total = cov_temp_u+cov_temp_v
        print('var(u) = '+str(np.round(cov_temp_u,5))+'\t'+'var(up) = '+str(np.round(cov_tide_u,5)))
        print('var(up)/var(u) = '+str(round(cov_tide_u/cov_temp_u*100,1))+' %')
        print('')
        print('var(v) = '+str(np.round(cov_temp_v,5))+'\t'+'var(vp) = '+str(np.round(cov_tide_v,5)))
        print('var(vp)/var(v) = '+str(round(cov_tide_v/cov_temp_v*100,1))+' %')
        print('')
        print('total var = '+str(np.round(cov_vel_total,5))+'\t'+'total var pred = '+str(np.round(cov_tide_total,5)))
        print('total var/total var pred = '+str(round(cov_tide_total/cov_vel_total*100,1))+' %')
        print('')
        print('Sum of 6 largest amplitudes: '+str(np.round(sum(coef.Lsmaj[0:6]),2)))
        if i == 0:
            if cov_tide_total/cov_vel_total > 0.5 and sum(coef.Lsmaj[0:6])>0.15:
                print('Total variance from tidal prediction> 50% and sum of six largest majors > 0.15 m/s > Tidal driven current')
                tidal_driven = True
            else:
                print('The current is not tidal driven')                                              
                tidal_driven = False
        fig, ax = plt.subplots(nrows=4, sharex=True, figsize =(plotwidth*1/2, 2/3*plotwidth))
        ax[0].plot(time, temp_mag, label='Observations',color='Tab:blue',linewidth=0.50)
        ax[0].plot(time, np.sqrt(tide['u']**2+tide['v']**2), label='Prediction', color='Tab:green',linewidth=0.50)
        ax[0].plot(time,  temp_mag-np.sqrt(tide['u']**2+tide['v']**2), label='Difference',color='Tab:red',linewidth=0.50)
        ax[0].set_ylabel('Magnitude, m/s')
        
        ax[1].plot(time, temp_dir, label='Observations', color='Tab:blue',linewidth=0.50)
        ax[1].plot(time, tide_temp_mag_angle_deg, label='Prediction',  color='Tab:green',linewidth=0.50)
        ax[1].set_ylabel('Direction, °')
        
        ax[2].plot(time, temp_u, label='Observations', color='Tab:blue',linewidth=0.50)
        ax[2].plot(time, tide['u'], label='Prediction', color='Tab:green',linewidth=0.50)
        ax[2].plot(time, temp_u-tide['u'], label='Difference', color='Tab:red',linewidth=0.50)
        ax[2].set_ylabel('Easting, m/s')
        
        ax[3].plot(time, temp_v, label='Observations',  color='Tab:blue',linewidth=0.50)
        ax[3].plot(time, tide['v'], label='Prediction', color='Tab:green',linewidth=0.50)
        ax[3].plot(time, temp_v-tide['v'], label='Difference',  color='Tab:red',linewidth=0.50)
        ax[3].set_ylabel('Northing, m/s')
        
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        set_legend_and_grid_and_tightlayout_and_show_and_save('tidal_prediction_'+filename,'upper right')
        
        # Plot only two weeks
        time_idx = np.arange(nearest_date_ind(time,time[int(len(time)/2)]-timedelta(days=7)),
                             nearest_date_ind(time,time[int(len(time)/2)]+timedelta(days=7)))
        
        ax[0].set_xlim([time[nearest_date_ind(time,time[int(len(time)/2)]-timedelta(days=7))],
                          time[nearest_date_ind(time,time[int(len(time)/2)]+timedelta(days=7))]])
        
        set_legend_and_grid_and_tightlayout_and_show_and_save('tidal_prediction_2weeks_'+filename,'best')
        print(np.sqrt(tide['u']**2+tide['v']**2))
        if i == 0:

            tide_measured_umax = np.max(np.sqrt(tide['u']**2+tide['v']**2))
            print('Largest tidal predicted velocity in the measurement period ' + str(round(tide_measured_umax,2)))           
            # tide = utide.reconstruct(time, coef)
            # date_list = [base - datetime.timedelta(days=x) for x in range(numdays)]
            date_list = np.array([time[0] + timedelta(minutes=x*30) for x in range(int(365*60*24*60/30))]) # Every half hour # Every 12th minute to save RAM.
            tide_60 = utide.reconstruct(date_list, coef)#, constit=['M2'])
            tide_60_umax = np.max(np.sqrt(tide_60['u']**2+tide_60['v']**2))
            print('Largest tidal predicted velocity in a 60-year period ' + str(round(tide_60_umax,2)))
            # plt.figure()
            # plt.figure
            # plt.plot(np.sqrt(tide_60['u']**2+tide_60['v']**2))#[0:1000000])
            with open(outputfolder+'/../'+titlename+'.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump([coef, long, lat, ID, date_first, date_last, instrument_type, serial_number, instrument_frequency, profile_interval, average_interval,\
                number_of_measurements, bin_size, number_of_bins, blanking_distance,ping_hz, tide_60_umax], f)

            
        if i == 0:
            with app2.create(Figure(position='h!')) as image:
                image.add_image('cur_fig/tidal_prediction_'+filename+'.pdf', width=NoEscape(r'1.0\textwidth'), placement=NoEscape(r'\centering'))
                image.add_caption('Vector averaged velocity tidal prediction of bin '+str(firstbin) + ' to bin ' +str(lastbin) + '.')                  
            with app2.create(Figure(position='h!')) as image:
                image.add_image('cur_fig/tidal_prediction_2weeks_'+filename+'.pdf', width=NoEscape(r'1.0\textwidth'), placement=NoEscape(r'\centering'))
                image.add_caption('Two-week close-up of the vector averaged velocity tidal prediction of bin '+str(firstbin) + ' to bin ' +str(lastbin) + '.')    
        else:
            with app2.create(Figure(position='h!')) as image:
                image.add_image('cur_fig/tidal_prediction_'+filename+'.pdf', width=NoEscape(r'1.0\textwidth'), placement=NoEscape(r'\centering'))
                image.add_caption('Tidal prediction for the '+filename[0:3]+' '+filename[3:6]+'.')             
            with app2.create(Figure(position='h!')) as image:
                image.add_image('cur_fig/tidal_prediction_2weeks_'+filename+'.pdf', width=NoEscape(r'1.0\textwidth'), placement=NoEscape(r'\centering'))
                image.add_caption('Two-week close-up of the '+filename[0:3]+' '+filename[3:6]+' tidal prediction.')    
                     
        app2.append(NoEscape('\clearpage'))
        table1 = Table(position="h")
        tabular1 = Tabular('|c|c|c|c|c|')
        tabular1.add_hline()
        tabular1.add_row((NoEscape(r'\textbf{Const.}'),NoEscape(r'\textbf{Major}'), NoEscape(r'\textbf{Minor}'), NoEscape(r'\textbf{G, °}'), NoEscape(r'\textbf{Inc, °}')))
        for j in range(0,len(coef.Lsmaj)):
            tabular1.add_hline()
            tabular1.add_row((str(coef.name[j]),str("%.5f" % coef.Lsmaj[j]),str("%.5f" % coef.Lsmin[j]),str(np.round(coef.g[j],1)), str(np.round(coef.theta[j],1))))
        tabular1.add_hline()
        table1.append(NoEscape(r'\centering'))
        table1.append(tabular1)
        if i == 0:
            table1.add_caption(NoEscape('Tidal constituents for the vector averaged velocity of bin '+str(firstbin) + ' to bin ' +str(lastbin)+
                               ', Umean = ' + "%.5f" % coef.umean + ' m/s. Vmean = ' + "%.5f" % coef.vmean + " m/s. "+
                               r"""$\frac{var(total)}{var(prediction)} = $ """+ "%.2f" % (cov_tide_total/cov_vel_total*100) +'\%.'+
                               ' The sum of the 6 largest major constituents is '+"%.2f" % sum(coef.Lsmaj[0:6])+' m/s.'))
        else:
            table1.add_caption(NoEscape('Tidal constituents for the '+filename[0:3]+' '+filename[3:6]+'. Umean = ' + "%.5f" % coef.umean + 
                               ' m/s, Vmean = ' + "%.5f" % coef.vmean + r""" m/s. $\frac{var(total)}{var(psrediction)} = $"""+"%.2f" % (cov_tide_total/cov_vel_total*100) +' \%.'+
                               ' The sum of the 6 largest major constituents is '+"%.2f" % sum(coef.Lsmaj[0:6]) +' m/s.'))
        app2.append(table1)
        app2.append(NoEscape('\clearpage'))   

app2.generate_tex(outputfolder+'/../appendix2')    
# Remove begin document
with open(outputfolder+'/../appendix2.tex', 'r') as fin:
    data = fin.read().splitlines(True)
with open(outputfolder+'/../appendix2.tex', 'w') as fout:
    fout.writelines(data[12:-2])


###############
## Contour plot
def contour_plot_settings(h,ax,title,limit):
    ax.axis('square')
    ax.set_xlim([-limit,limit])
    ax.set_ylim([-limit,limit])
    ax.axhline(0,color='k',alpha=0.25)
    ax.axvline(0,color='k',alpha=0.25)
    for i in range(1,int(max_val/0.25)+2):
        circle = plt.Circle((0, 0), i*0.25,fill=False,alpha=0.5)
        ax.add_patch(circle)
    fig.colorbar(h[3], ax=ax)
    ax.set(ylabel = 'N, m/s')
    ax.set(title = title)
    # plt.clim(0,cbar_max)
    
max_val=np.nanmax((np.abs(vel_v_av),np.abs(vel_u_av)))
max_val=max_val-(max_val % 0.25)+0.25 # round to nearest 0.2
# Get max for colorbar, create dummy figure
fig, ax0 = plt.subplots(nrows=1,ncols=1, figsize =(plotwidth*1/2, 2/3*plotwidth))
h1=ax0.hist2d(vel_u_all[:,topbin-1], vel_v_all[:,topbin-1], bins=100, density=False, range = [[-max_val,max_val],[-max_val,max_val]])
h2=ax0.hist2d(vel_u_all[:,midbin-1], vel_v_all[:,midbin-1], bins=100, density=False, range = [[-max_val,max_val],[-max_val,max_val]])
h3=ax0.hist2d(vel_u_all[:,botbin-1], vel_v_all[:,botbin-1], bins=100, density=False, range = [[-max_val,max_val],[-max_val,max_val]])
plt.close(fig=plt.gcf().number)

cbar_max = np.max((h1[0],h2[0],h3[0]))
fig, ((ax0, ax1), (ax2, ax3), (ax4, ax5)) = plt.subplots(nrows=3,ncols=2, figsize =(plotwidth*1/2, 2/3*plotwidth))
h0=ax0.hist2d(vel_u_all[:,topbin-1], vel_v_all[:,topbin-1], bins=100, density=False, range = [[-max_val,max_val],[-max_val,max_val]],vmin = 0, vmax=cbar_max,cmin = 0.5,cmap='jet')#,cmap='GnBu')
h2=ax2.hist2d(vel_u_all[:,midbin-1], vel_v_all[:,midbin-1], bins=100, density=False, range = [[-max_val,max_val],[-max_val,max_val]],vmin = 0, vmax=cbar_max,cmin = 0.5,cmap='jet')#,cmap='GnBu')
h4=ax4.hist2d(vel_u_all[:,botbin-1], vel_v_all[:,botbin-1], bins=100, density=False, range = [[-max_val,max_val],[-max_val,max_val]],vmin = 0, vmax=cbar_max,cmin = 0.5,cmap='jet')#,cmap='GnBu')
ax4.set(xlabel = 'E, m/s')

h1=ax1.hist2d(vel_u_all[:,topbin-1], vel_v_all[:,topbin-1], bins=100, density=False, range = [[-max_val,max_val],[-max_val,max_val]],vmin = 0, vmax=cbar_max,cmin = 0.5,cmap='jet')#,cmap='GnBu')
h3=ax3.hist2d(vel_u_all[:,midbin-1], vel_v_all[:,midbin-1], bins=100, density=False, range = [[-max_val,max_val],[-max_val,max_val]],vmin = 0, vmax=cbar_max,cmin = 0.5,cmap='jet')#,cmap='GnBu')
h5=ax5.hist2d(vel_u_all[:,botbin-1], vel_v_all[:,botbin-1], bins=100, density=False, range = [[-max_val,max_val],[-max_val,max_val]],vmin = 0, vmax=cbar_max,cmin = 0.5,cmap='jet')#,cmap='GnBu')
ax5.set(xlabel = 'E, m/s')
# max_val=max_val/2
contour_plot_settings(h0,ax0,'Top bin.',max_val)#,cbar_max)
contour_plot_settings(h2,ax2,'Middle bin.',max_val)
contour_plot_settings(h4,ax4,'Bottom bin.',max_val)
contour_plot_settings(h1,ax1,'Top bin.',max_val/2)#,cbar_max)
contour_plot_settings(h3,ax3,'Middle bin.',max_val/2)
contour_plot_settings(h5,ax5,'Bottom bin.',max_val/2)

plt.suptitle('Occurances of measured speeds')
plt.tight_layout()

plt.show()
if savefig == True:
    plt.savefig(outputfolder+'contour.png',dpi = 300)
    plt.savefig(outputfolder+'contour.pdf',dpi = 600)
                    

####################
#%% Histogram table
bins = [0,0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,0.45,0.5,0.6,0.7,0.8,0.9,1.0,1.25,1.5,1.75,2.0]
f = open(outputfolder+"table_bin.txt", "w")
f.write('\\addtolength{\\tabcolsep}{-0.3em}\n')
f.write('\\begin{table}[h!]\n')
f.write('\centering\n')
f.write(r'\tiny'+str('\n'))
f.write('\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|}\n')
f.write('\hline\n')
f.write(r'''\textbf{Bin} & \textbf{Depth (m)} & '''+ r'''\multicolumn{19}{c|}{\textbf{Current speed (cm/s)}}'''+str('\\\ \hline \n'))
f.write('~ & ~ & '+r'\textbf{'+str(bins[0:-1])[1:-1].replace(',','}&').replace(' ',r'\textbf{')+str('} \\\ \hline \n'))
f.write(NoEscape('~ & ~ & '+str(np.array(bins[1:])*0)[1:-1].replace('0.',r'\textbf{-} &')[:-1]+str('\\\ \hline \n')))
f.write(NoEscape('~ & ~ & '+r'\textbf{'+str(bins[1:])[1:-1].replace(',','}&').replace(' ',r'\textbf{')+str('} \\\ \hline \n')))
for i in range(0,np.size(mag_all,axis=1)):
    # i=1
    hist, b= np.histogram(mag_all[:,i],bins=bins)
    hist_1000 = 1000/(np.nansum(hist))*hist # in parts per 1000.
    if np.sum(hist) > 0:
        #generate string
        tempstr = str(i+1)+' & ' +str(np.round(bindepths[i],1)) + ' & '
        for j in range(0,len(hist_1000)):
            if hist_1000[j]>=1:
                tempstr = tempstr + str(round(hist_1000[j])) + ' & '
            if hist_1000[j]<1:
                if hist_1000[j]==0:
                    tempstr = tempstr + str('0') + ' & '
                else:
                    j = 16
                    tempstr = tempstr + str(round(hist_1000[j],3)) + ' & '
                    
        tempstr = tempstr[:-3]
        tempstr = tempstr + str('\\\ \n')
        f.write(tempstr)
        f.write('\hline\n')
f.write('\end{tabular}\n')
f.write('\caption{Velocities in selected intervals in parts per thousand of speeds. Velocities occuring less than once shown with one decimal, unless the frequency is zero.}\n')
f.write('\label{tab:interval_bins}\n')
f.write('\end{table}\n')
f.write('\\addtolength{\\tabcolsep}{0.0em}\n')
f.close()

#######################
#%% Frequency table for mean values
fig, ax1 = plt.subplots(figsize =(plotwidth, 2/3*plotwidth))
ax2 = ax1.twinx()
ax1.hist(mag_av,np.arange(0,2,0.05),edgecolor='black', weights=np.ones_like(mag_av) / len(mag_av)*100)
# ax2.hist(mag_av,np.arange(0,2,0.001),edgecolor='C1', weights=np.ones_like(mag_av) / len(mag_av)*100, cumulative=1,histtype='step')
n,bins,patches = ax2.hist(mag_av,np.arange(0,2,0.001),edgecolor='C1', density =1, cumulative=1,histtype='step')
patches[0].set_xy(patches[0].get_xy()[:-1]) # Manually delete last point as it is always zero
ax1.set_xlabel('Current speed, m/s')
ax1.set_ylabel('Frequency, %', color='C0')
ax2.set_ylabel('Cumulative, %', color='C1')
plt.suptitle('Depth averaged current distribution')
plt.tight_layout()
plt.show()
if savefig == True:
    plt.savefig(outputfolder+'distribution.png',dpi = 300)
    plt.savefig(outputfolder+'distribution.pdf',dpi = 600)
    

# # consecutive time of each current speed
# event_hours = []
# vel_range = np.arange(0,2,0.1)
# for i in vel_range:
#    # j = i/10#
#     idx=np.where((mag_av>i) & (mag_av<i+0.1))[0]
#     #loop and count
#     count = []
#     dtime = [0] #dummy
#     temp = 1
#     time_start=time[0]
#     for k in range(0,len(idx)-1):      
#         if idx[k]+1 == idx[k+1]:
#             temp = temp + 1
#         else:
#             if temp>1:
#                 count.append(temp)
#                 temp = 1
#                 time_end=time[k]
#                 dtime.append((time_end-time_start).seconds)
#                 time_start=time[k]
#     event_hours.append(max(dtime)/3600)
# fig, ax = plt.subplots(figsize =(plotwidth*1/2, 2/3*plotwidth))
# ax.grid(True)
# ax.set_xlabel('Velocity magnitude, m/s')
# ax.set_ylabel('Maximum duration, hours')
# ax.set(title='')

##############################################
# consecutive time of each current speed range
event_hours = []
event_hours_50 = []
event_hours_90 = []
event_hours_95 = []
bin_size_interval = 0.002
if titlename == 'SKUA2201':
    vel_range = np.arange(bin_size_interval,3.0,bin_size_interval)  
else:
    vel_range = np.arange(bin_size_interval,2.0,bin_size_interval)  
    
for i in vel_range:
    idx=np.where(mag_av<i)[0]
    time_start=time[0]
    dtime = [0]
    if idx.size > 1: 
        for k in range(0,len(idx)-1):
            if idx[k] != idx[k+1]-1:
                time_end=time[k]
                dtime.append((time_end-time_start).total_seconds())
                time_start=time[k+1]
    else:
        k=0
    time_end=time[k]
    dtime.append((time_end-time_start).total_seconds())
    time_start=time[k+1]
    event_hours.append(max(dtime)/3600)  
    event_hours_50.append(np.nanpercentile(np.array(dtime)/3600,50))   
    event_hours_90.append(np.nanpercentile(np.array(dtime)/3600,90))    
    event_hours_95.append(np.nanpercentile(np.array(dtime)/3600,95))          
fig, ax = plt.subplots(ncols=2,figsize =(0.7*plotwidth*0.8, 0.7*1/2*plotwidth*0.65))
ax[0].grid(True)
ax[0].set_xlabel('Velocity magnitude, m/s')
ax[0].set_ylabel('Maximum duration, hours')
ax[0].set(title='Entire measurement')
ax[0].plot(vel_range,event_hours, c='C0',label='Maximum')#,width=bin_size)
ax[0].plot(vel_range,event_hours_50,c='C1',label='50 Percentile')#,width=bin_size)
ax[0].plot(vel_range,event_hours_90,c='C2',label='90 Percentile')#,width=bin_size)
ax[0].plot(vel_range,event_hours_95,c='C3',label='95 Percentile')#,width=bin_size)
ax[0].legend()
ax[1].grid(True)
ax[1].set_xlabel('Velocity magnitude, m/s')
ax[1].set_ylabel('Maximum duration, hours')
ax[1].set(title='12-hour close up')
ax[1].plot(vel_range,event_hours, c='C0',label='Maximum')#,width=bin_size)
ax[1].plot(vel_range,event_hours_50,c='C1',label='50 Percentile')#,width=bin_size)
ax[1].plot(vel_range,event_hours_90,c='C2',label='90 Percentile')#,width=bin_size)
ax[1].plot(vel_range,event_hours_95,c='C3',label='95 Percentile')#,width=bin_size)ze
ax[1].set_ylim([0,12])
ax[1].legend()
ax[1].set_xlim([0,vel_range[np.argmin((np.abs(np.array(event_hours)-12)))]*1.05])
# fig.suptitle('Concurrent depth averaged velocity below threshold')
plt.tight_layout()
if savefig == True:
    plt.savefig(outputfolder+'concurrent_velocity_below.png',dpi = 300)
    plt.savefig(outputfolder+'concurrent_velocity_below.pdf',dpi = 600)
    
###############################################
# consecutive time of each current speed range
event_hours = []
event_hours_50 = []
event_hours_90 = []
event_hours_95 = []
# bin_size_interval = 0.002
# vel_range = np.arange(bin_size_interval,2.0,bin_size_interval)
for i in vel_range:
    idx=np.where(mag_av>i)[0]
    time_start=time[0]
    dtime = [0]
    if idx.size > 1: 
        for k in range(0,len(idx)-1):
            if idx[k] != idx[k+1]-1:
                time_end=time[k]
                dtime.append((time_end-time_start).total_seconds())
                time_start=time[k+1]
    else:
        k=0
    time_end=time[k]
    dtime.append((time_end-time_start).total_seconds())
    time_start=time[k+1]
    event_hours.append(max(dtime)/3600)
    event_hours_50.append(np.nanpercentile(np.array(dtime)/3600,50))   
    event_hours_90.append(np.nanpercentile(np.array(dtime)/3600,90))    
    event_hours_95.append(np.nanpercentile(np.array(dtime)/3600,95))                 
fig, ax = plt.subplots(ncols=2,figsize =(0.7*plotwidth*0.8, 0.7*1/2*plotwidth*0.65))
ax[0].grid(True)
ax[0].set_xlabel('Velocity magnitude, m/s')
ax[0].set_ylabel('Maximum duration, hours')
ax[0].set(title='Entire measurement')
ax[0].plot(vel_range,event_hours, c='C0',label='Maximum')#,width=bin_size)
ax[0].plot(vel_range,event_hours_50,c='C1',label='50 Percentile')#,width=bin_size)
ax[0].plot(vel_range,event_hours_90,c='C2',label='90 Percentile')#,width=bin_size)
ax[0].plot(vel_range,event_hours_95,c='C3',label='95 Percentile')#,width=bin_size)
ax[0].legend()
ax[1].grid(True)
ax[1].set_xlabel('Velocity magnitude, m/s')
ax[1].set_ylabel('Maximum duration, hours')
ax[1].set(title='Close up')
ax[1].plot(vel_range,event_hours, c='C0',label='Maximum')#,width=bin_size)
ax[1].plot(vel_range,event_hours_50,c='C1',label='50 Percentile')#,width=bin_size)
ax[1].plot(vel_range,event_hours_90,c='C2',label='90 Percentile')#,width=bin_size)
ax[1].plot(vel_range,event_hours_95,c='C3',label='95 Percentile')#,width=bin_size)ze
ax[1].set_ylim([0,3])
try:
    ax[1].set_xlim([0,vel_range[np.min(np.where((np.array(event_hours) == 0.0)))]*1.05])
except:
    ax[1].set_xlim([0,2.0])
    
ax[1].legend(loc='upper left')
# ax[1].set_xlim([0,0vel_range[np.argmin((np.abs(np.array(event_hours)-12)))]*1.05])
# fig.suptitle('Concurrent depth averaged velocity above threshold.')
plt.tight_layout()
if savefig == True:
    plt.savefig(outputfolder+'concurrent_velocity_above.png',dpi = 300)
    plt.savefig(outputfolder+'concurrent_velocity_above.pdf',dpi = 600)

############
# Plot PVD
# If value is nan, the previous datapoint is used.
fig, ax = plt.subplots(ncols=1,figsize =(plotwidth*0.5, plotwidth*0.5))
leg = ['Bottom bin','Middle bin','Top bin']
aa=0
temp_NS = 0 # use to replace nan
temp_EW = 0 # use to replace nan
for a in [botbin-1,midbin-1,topbin-1]:
    pvd_NS = [0]
    pvd_EW = [0]
    invalid = []
    fromaverage = []
    for i in range(np.size(vel_u_all,0)-1):
        if np.isnan(vel_u_all[i,a]) == True:
            if np.isnan(vel_u_av[i]) == False: 
                pvd_NS.append(pvd_NS[-1]+vel_u_av[i]*(time[i+1]-time[i]).total_seconds())
                fromaverage.append(i)
            else:
                pvd_NS.append(pvd_NS[-1]+temp_NS)
                invalid.append(i)
        else:
            pvd_NS.append(pvd_NS[-1]+vel_u_all[i,a]*(time[i+1]-time[i]).total_seconds())
            temp_NS = vel_u_all[i,a]*(time[i+1]-time[i]).total_seconds()
        
        if np.isnan(vel_v_all[i,a]) == True:
            if np.isnan(vel_u_av[i]) == False: 
                pvd_EW.append(pvd_EW[-1]+vel_v_av[i]*(time[i+1]-time[i]).total_seconds())
                fromaverage.append(i)
            else:
                pvd_EW.append(pvd_EW[-1]+temp_EW)
                invalid.append(i)        
            #pvd_EW.append(pvd_EW[-1]+temp_EW)
        else:
            pvd_EW.append(pvd_EW[-1]+vel_v_all[i,a]*(time[i+1]-time[i]).total_seconds())
            temp_EW = vel_v_all[i,a]*(time[i+1]-time[i]).total_seconds()
            
    ax.plot(np.array(pvd_EW)/1000,np.array(pvd_NS)/1000,label=leg[aa])        

    if len(fromaverage)>0:
        ax.plot(np.array(pvd_EW)[fromaverage]/1000,np.array(pvd_NS)[fromaverage]/1000,'.k',markersize=1,label='__nolabel__')
    if len(invalid)>0:
        ax.plot(np.array(pvd_EW)[invalid]/1000,np.array(pvd_NS)[invalid]/1000,'xr',markersize=6,label='__nolabel__')
    aa = aa+1

pvd_NS = [0]
pvd_EW = [0]
temp_NS = 0 # use to replace nan
temp_EW = 0 # use to replace nan
for i in range(np.size(vel_u_av,0)-1):   
    invalid = []
    if np.isnan(vel_u_av[i]) == True:
        pvd_NS.append(pvd_NS[-1]+temp_NS)
        invalid.append(i)
    else:
        pvd_NS.append(pvd_NS[-1]+vel_u_av[i]*(time[i+1]-time[i]).total_seconds())
        temp_NS = vel_u_av[i]*(time[i+1]-time[i]).total_seconds()
    if np.isnan(vel_v_av[i]) == True:
        pvd_EW.append(pvd_EW[-1]+temp_EW)
        invalid.append(i)
    else:
        pvd_EW.append(pvd_EW[-1]+vel_v_av[i]*(time[i+1]-time[i]).total_seconds())
        temp_EW = vel_v_av[i]*(time[i+1]-time[i]).total_seconds()
ax.plot(np.array(pvd_EW)/1000,np.array(pvd_NS)/1000,label='Average')
if len(invalid)>0:
    ax.plot(np.array(pvd_EW)[invalid]/1000,np.array(pvd_NS)[invalid]/1000,'xr',markersize=6,label='invalid')
ax.legend()
ax.axis('equal')
ax.set(xlabel = 'Distance east/west (Km)')
ax.set(ylabel = 'Distance north/south (Km)')   
plt.tight_layout()
if savefig == True:
    plt.savefig(outputfolder+'PVD.png',dpi = 300)
    plt.savefig(outputfolder+'PVD.pdf',dpi = 600)


##############
# Generate map 
if plot_map == True:
    tic = counttime.perf_counter()
    temp = np.load('../rfb_interpolation/scatter_bathymetry.npy')
    def sort_bathymetry(temp):
        temp = np.delete(temp,np.where(temp[:,0]<long-0.3),axis = 0)
        temp = np.delete(temp,np.where(temp[:,0]>long+0.3),axis = 0)
        temp = np.delete(temp,np.where(temp[:,1]<lat-0.15),axis = 0)
        temp = np.delete(temp,np.where(temp[:,1]>lat+0.15),axis = 0)
        return temp
    botnur = sort_bathymetry(temp)
    fig, ax = strandalinja_US.strandalinja_US_plot_all_simple()    
    scatter = ax.scatter(botnur[:,0],botnur[:,1],s=0.1,c=botnur[:,2],cmap=cmocean.cm.deep,marker='.')
    ax.set_xlim([long-0.3,long+0.3])
    ax.set_ylim([lat-0.15,lat+0.15])
    ax.set_xlabel('Longitude, °')
    ax.set_ylabel('Latitude, °')
    plt.subplots_adjust(bottom=0.02)
    plt.subplots_adjust(top=0.98)
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    plt.colorbar(scatter, cax=cax) # Similar to fig.colorbar(im, cax = cax) 
    ax.plot(long,lat,'or',markersize=8)
    ax.text(long+0.006,lat,titlename,color ='r')   
    if savefig == True:
        plt.savefig(outputfolder+'map.png',dpi = 300)
        plt.savefig(outputfolder+'map.pdf',dpi = 600)
    toc = counttime.perf_counter()   
    print('Time ' + str(toc-tic))

#%%
##################
#Create latex pdf
# Copy template tex files if they do not exist
import os
if not os.path.exists(report_folder+'stylefiles'):
    import shutil, errno
    shutil.copytree('latex_template',report_folder+'stylefiles' )
    shutil.copy('metadata.tex',report_folder )
    shutil.copy('introduction.tex',report_folder )
    shutil.copy('abstract.tex',report_folder )
    shutil.copy('summary.tex',report_folder )
    shutil.copy('discussion.tex',report_folder )
    shutil.copy('report.bbl',report_folder )
    shutil.copy('report.blg',report_folder )
#geometry_options = {"documentclass":'article',"tmargin": "3cm", "lmargin": "1cm","rmargin": "1cm"}
doc = Document(documentclass='memoir',document_options =['12pt','a4paper','fleqn','twosided,article'])
doc.preamble.append(NoEscape(r'\usepackage{subcaption}'))
doc.preamble.append(NoEscape(r'\input{stylefiles/dokumentstilur}'))
doc.preamble.append(NoEscape(r'\input{stylefiles/kommandoir}'))
doc.preamble.append(NoEscape(r'\input{metadata}'))
doc.preamble.append(NoEscape(r'\input{stylefiles/titlepage}'))
doc.preamble.append(NoEscape(r'\input{stylefiles/opna}'))
doc.append(NoEscape(r'\frontmatter'))
doc.append(NoEscape(r'\pagestyle{empty}'))
doc.append(NoEscape(r'\titlepage'))
doc.append(NoEscape(r'\opn'))
doc.append(NoEscape(r'\mainmatter'))
doc.append(NoEscape(r'\pagestyle{mergedstyle}'))
doc.append(NoEscape(r'\openany'))
# mdata=pd.read_csv(metadataname, sep='\t').to_numpy()
#f = open(outputfolder+"metadata.txt", "w")

# def find_string(string, number):
#     # Find substring in string. "number" is the time the substring occurs.
#     count = 0
#     for i in range(len(mdata)):
#         x = mdata[i][0].find(string)
#         if x > -1:
#             count = count + 1
#             if count == number:
#                 break
#     return i

        
doc.append(NoEscape(r'\input{introduction}'))
with doc.create(Chapter('Deployment data')):
    doc.append(NoEscape(r'''Table \ref{tab:deployment} summarizes the deployment and instrument information. 
Figure \ref{fig:map} shows the measurement location and the available bathymetry data.'''))


# lat, long, mag_dec,instr_height = np.load(data_folder+'vars.npy')
# ID, date_first, date_last, instrument_type, serial_number, instrument_frequency, profile_interval, average_interval,\
# number_of_measurements, bin_size, number_of_bins, blanking_distance = np.load(data_folder+'string_vars.npy')

    #doc.append(NoEscape('{'))
    # doc.append(Command('centering'))
    table1 = Table(position="h")
    tabular1 = Tabular('|c|c|',pos='center')
    tabular1.add_hline()
    tabular1.add_row((NoEscape(r'\textbf{Deployment}'), NoEscape(r'\textbf{Instrument}')))
    tabular1.add_hline()
    tabular1.add_row(('ID: '+str(titlename), 'Type: '+instrument_type))
    tabular1.add_hline()
    tabular1.add_row(('Time of first measurement: ' + date_first,'Serial number: '+serial_number))
    tabular1.add_hline()
    tabular1.add_row(('Time of last measurement: ' + date_last,'Instrument frequency: ' + instrument_frequency))
    tabular1.add_hline()
    tabular1.add_row(('Longitude: ' +str(np.round(long,4))+str("°"),'Profile interval: '+profile_interval))
    tabular1.add_hline()
    tabular1.add_row(('Latitude: '+str(np.round(lat,4))+str("°"),'Average interval: '+average_interval))
    tabular1.add_hline()
    tabular1.add_row(('Longitude: '+str(int(long)) + '°' + str(np.round((int(long)-long)*60,4)) + str("'") ,'Number of measurements: '+number_of_measurements))
    tabular1.add_hline()
    tabular1.add_row(('Latitude: '+str(int(lat)) + '°' + str(np.round((-int(lat)+lat)*60,4)) + str("'"),'Bin size: '+ bin_size))
    tabular1.add_hline()
    tabular1.add_row(('','Number of bins: ' + number_of_bins))  
    tabular1.add_hline()
    tabular1.add_row(('','Blanking distance: ' +blanking_distance))  
    tabular1.add_hline()
    tabular1.add_row(('','Instrument height: ' + str(instr_height) + ' m'))  
    tabular1.add_hline()
    tabular1.add_row(('','Ping frequency: ' + ping_hz))  
    tabular1.add_hline()    
    tabular1.add_row(('','Pings per averaging interval: ' + str(int(float(average_interval[0:3])*float(ping_hz[0:3])))))  
    tabular1.add_hline()  
    table1.append(NoEscape(r'\centering'))
    table1.append(tabular1)
    table1.add_caption('Deployment data and instrument settings.')
    table1.append(Label(NoEscape(r'tab:deployment')))
    doc.append(table1)
    with doc.create(Figure(position='h!')) as image:
        image.add_image('cur_fig/map.png', width=NoEscape(r'1.0\textwidth'), placement=NoEscape(r'\centering'))
        image.add_caption('Overview of the measurement location and available bathymetry data. The bathymetry is shown using raw point data. The data from Andrias Reinert, Írland, Magnus Heinason, Landsverk, RAO and Tjaldur. Possible multibeam datasets are not included.')
    image.append(Label(NoEscape(r'fig:map')))
    doc.append(NoEscape('\clearpage'))
with doc.create(Chapter('Processing')):    
    doc.append(NoEscape(r'''Table \ref{tab:binsfromhead} shows the bin distance stored in the instrument log file, excluding the instrument height above bottom, and
the bin centre time averaged depth based on the pressure sensor, including the instrument height above bottom. The bins near and above the water surface are invalid.
 For individual bin analysis the heighest bin with good data quality, the bottom bin and a bin between the two bins are selected.
The bins "averaging start" to "averaging end" are used for vertical vector averaging current velocities.'''))
    if os.path.exists(report_folder+'processingextra.tex'):
        doc.append(NoEscape(r'\input{processingextra}'))  
  
    table1 = Table(position="h")
    tabular1 = Tabular('|c|c|c|c|')
    #generate "comment"
    comments = [''  for x in range(len(bindepths))]
    comments[botbin-1] = 'Bottom bin'
    comments[midbin-1] = 'Middle bin'
    comments[topbin-1] = 'Top bin'
    if comments[firstbin-1] != '':
        comments[firstbin-1] = comments[firstbin-1] + ', averaging start'   
    else:
        comments[firstbin-1] = 'Averaging start'   
    if comments[lastbin-1] != '':
        comments[lastbin-1] = comments[lastbin-1] + ', averaging end'   
    else:
        comments[lastbin-1] = 'Averaging end'   
        
    tabular1.add_hline()
    tabular1.add_row((NoEscape(r'\textbf{Bin}'),NoEscape(r'\textbf{Distance from head, m}'), NoEscape(r'\textbf{Bin centre depth (m)}'), NoEscape(r'\textbf{Comment}')))

    for i in range(0,len(bindepths)):       
        tabular1.add_hline()
        tabular1.add_row((str(i+1),"%.1f" % (binheights[i]-instr_height),"%.1f" % bindepths[i],comments[i]))

    tabular1.add_hline()
    table1.append(NoEscape(r'\centering'))
    table1.append(tabular1)
    table1.add_caption('Current profile cell center distance from head (m), excluding instrument height from bottom.')
    table1.append(Label(NoEscape(r'tab:binsfromhead')))
    doc.append(table1)
    
    # table1 = Table(position="h")
    # tabular1 = Tabular('|c|c|c|',pos='center')
    # # with doc.create(Tabular('|c|c|c|',pos='center')) as tabular1:
    # tabular1.add_hline()
    # tabular1.add_row((NoEscape(r'\textbf{Bin name}'), NoEscape(r'\textbf{Bin number}'), NoEscape(r'\textbf{Bin centre depth (m)}')))
    # tabular1.add_hline()
    # tabular1.add_row(('Bottom', str(botbin),"%.1f" % bindepths[botbin-1]))
    # tabular1.add_hline()
    # tabular1.add_row(('Middle',str(midbin),"%.1f" % bindepths[midbin-1]))
    # tabular1.add_hline()
    # tabular1.add_row(('Top',str(topbin),"%.1f" % bindepths[topbin-1])) 
    # tabular1.add_hline()
    # tabular1.add_row(('First',str(firstbin),"%.1f" % bindepths[firstbin-1])) 
    # tabular1.add_hline()
    # tabular1.add_row(('Last',str(lastbin),"%.1f" % bindepths[lastbin-1]))  
    # tabular1.add_hline()
        
    # table1.append(NoEscape(r'\centering'))
    # table1.append(tabular1)

    # table1.add_caption('Bin selection and bin centre depths.') 
    # table1.append(Label(NoEscape(r'tab:bins')))
    
    # doc.append(table1)                  
 
    doc.append(NoEscape('\clearpage'))
with doc.create(Chapter('Results')):
    doc.append(NoEscape(r'''Figure \ref{fig:waterlevel} shows the water level from the pressure sensor at the instrument location. 
Figure \ref{fig:timeseries} shows the velocity magnitude and direction for the duration of the measurement.
Figure \ref{fig:timeseriesweek1} to Figure \ref{fig:timeseriesweek'''+str(weeks)+r'''} in appendix \ref{chap:Timeseriesperweek} shows the time series for each week.'''+str('\n\n')))
    doc.append(NoEscape(r'Figure \ref{fig:histogram} an overview of the distribution of current velocities.'+str('\n\n')))
    

    doc.append(NoEscape(r'''If the current measurement is used for dimensioning, the maximum instantaneous current velocity must be extracted in the 
cardinal and intercardinal directions, as shown in Figure \ref{fig:scatteraverage}. The vertical vector average current velocity is used to reduce the impact of potential outliers.
Figure \ref{fig:scatter1} to Figure \ref{fig:scatter''' + str(int(np.ceil((max((topbin,lastbin))+1)/4)))+r'''} in Appendix \ref{chap:Scatterplots} shows the speed and direction for each bin.'''+r''' Table 
\ref{tab:max_speed} shows the maximum current for each bin and Table \ref{tab:NS9415_50y} shows the dimensioning currents according to NS9415:2021.'''+str('\n\n')))

    doc.append(NoEscape(r'''Figure \ref{fig:profile} shows the current speed against depth. The vertical bold lines show how the mean and maximum velocity represent the current velocity profile.'''+str('\n\n')))

    doc.append(NoEscape(r'''Figure \ref{fig:concurrentbelow} and Figure \ref{fig:concurrentabove} show the consecutive current speed below and above
 a given threshold to indicate if the sustained current speeds are suitable for fish farming. Table \ref{tab:interval_bins} shows the current velocities in parts per thousand for selected intervals. If the current speed is too low for a prolonged period,
 the site might experience a lack of water exchange leading to oxygen issues. If the consecutive current speed is too high, the fish might not be able
 to swim against the current, resulting in fatigue. Using data from \citet{hva2021} for 705 g salmon and the method from \citet{gui2014} and \citet{liu2023}, the average 705 g salmon
 can swim against tidal driven currents up to 1.08 m/s.'''+str('\n\n')))


    if tidal_driven == True:
        doc.append(NoEscape(r'''Appendix \ref{chap:Tidalpredictions} shows the tidal prediction. \citet{lar2019} defines tidal driven currents as the
                            sum of the six largest tidal constituents should be larger than 0.15 m/s and that tidal prediction variance accounts for at least 50\% of the total measurement variance. 
                            By this definition, the measurement shows that the current is tidal driven. The tidal prediction results in a maximum current speed of ''' + "%.2f" % tide_measured_umax +
                            ''' m/s in the measurement period, while the 60-year maximum predicted tidal current speed is ''' + "%.2f" % tide_60_umax + ''' m/s. 
                            The measurement is performed at ''' + "%.2f" % (tide_measured_umax/tide_60_umax) + '''\% of the strongest tidal current. Assuming similar weather conditions during the strongest 
                            tidal current as during the measurement period, the 60-year maximum current speed would thus be ''' + "%.2f" % (tide_60_umax/tide_measured_umax*max(max_velocity)) + ''' m/s''' + str('.\n\n')))   
        

    else:
        doc.append(NoEscape(r'''Appendix \ref{chap:Tidalpredictions} shows the tidal prediction. \citet{lar2019} defines tidal driven currents by that the
                            sum of the six largest tidal constituents should be larger than 0.15 m/s and that tidal prediction variance accounts for at least 50\% of the total measurement variance. 
                            By this definition the measurement shows that the current is not tidal driven.'''+str('\n\n'))) 
    
                            
    doc.append(NoEscape(r'''Appendix \ref{chap:ProgressiveVectorDiagram} shows the Progressive Vector Diagram (PVD) for the top, bottom, middle, and vertical vector average bin.'''+str('\n\n')))

                            
    with doc.create(Figure(position='h!')) as image:
        image.add_image('cur_fig/water_level.pdf', width=NoEscape(r'0.8\textwidth'), placement=NoEscape(r'\centering'))
        image.add_caption('Time series of measured water level including instrument height from bottom.')
    image.append(Label(NoEscape(r'fig:waterlevel')))
    with doc.create(Figure(position='h!')) as image:
        image.add_image('cur_fig/timeseries.pdf', width=NoEscape(r'1.0\textwidth'), placement=NoEscape(r'\centering'))
        image.add_caption('Time series of measured velocity magnitude. Top bin = bin '+str(topbin) +'. Middle bin = bin '
                          + str(midbin) + '. Bottom bin = bin '+str(botbin)+
                          '. Average is the vector average of bin '+str(firstbin) + ' to bin ' +str(lastbin)+'.')
    image.append(Label(NoEscape(r'fig:timeseries')))

    with doc.create(Figure(position='h!')) as image:
        image.add_image('cur_fig/contour.pdf', width=NoEscape(r'1.0\textwidth'), placement=NoEscape(r'\centering'))
        image.add_caption('2D histogram with 100 evenly distributed bins showing the distribution of occurances of good observations.')
    image.append(Label(NoEscape(r'fig:histogram')))
    with doc.create(Figure(position='h!')) as image:
        image.add_image('cur_fig/scatter_average.pdf', width=NoEscape(r'1.0\textwidth'), placement=NoEscape(r'\centering'))
        image.add_caption('Vector average velocity magnitude and direction. The eight red dots mark the maximum velocity in the four cardinal and four intercardinal directions.')
    image.append(Label(NoEscape(r'fig:scatteraverage')))
    

    with doc.create(Figure(position='h!')) as image:
        image.add_image('cur_fig/profile.pdf', width=NoEscape(r'0.8\textwidth'),placement=NoEscape(r'\centering'))
        image.add_caption('Velocity profile through the depth of the measurement.')
    image.append(Label(NoEscape(r'fig:profile')))
    
    doc.append(NoEscape('\clearpage'))      
    with doc.create(Figure(position='h!')) as image:
        image.add_image('cur_fig/concurrent_velocity_below.pdf', width=NoEscape(r'1.0\textwidth'), placement=NoEscape(r'\centering'))
        image.add_caption('Concurrent measured velocity below treshold.')
    image.append(Label(NoEscape(r'fig:concurrentbelow')))    
    with doc.create(Figure(position='h!')) as image:
        image.add_image('cur_fig/concurrent_velocity_above.pdf', width=NoEscape(r'1.0\textwidth'), placement=NoEscape(r'\centering'))
        image.add_caption('Concurrent measured velocity above treshold.')
    image.append(Label(NoEscape(r'fig:concurrentabove')))
    latex_document = outputfolder+'table_bin.txt'
    with open(latex_document) as file:
        tex= NoEscape(file.read())
    doc.append(tex)
    doc.append(NoEscape('\clearpage'))


with doc.create(Section('Dimensioning Currents')):    

    latex_document = outputfolder+'max_speed.txt'
    with open(latex_document) as file:
        tex= NoEscape(file.read())
    doc.append(tex)
    
    latex_document = outputfolder+'NS9415_50y.txt'
    with open(latex_document) as file:
        tex= NoEscape(file.read())
    doc.append(tex)
doc.append(NoEscape('\clearpage'))
doc.append(NoEscape(r'\input{discussion}'))
doc.append(NoEscape('\clearpage'))
doc.append(NoEscape(r'\input{summary}')) 
doc.append(NoEscape(r'\include{Bibliography}')) 
doc.append(NoEscape(r'\small')) 
doc.append(NoEscape(r'\bibliography{D:/Dropbox/literature/ref/ref.bib}')) 

# with open('appendix.tex') as file:
#     tex= NoEscape(file.read())
# doc.append(tex)  

############################## 
# Generate appendix.
app1 = Document(documentclass='memoir')
offset = 0
plot_avg = 0
n_offset = 0
with app1.create(Chapter('Scatter plots')): 
    for k in range(0, int(np.ceil((max((topbin,lastbin))+1)/4))):
        with app1.create(Figure(position='h!')) as main_figure:
            #for n in range(min((botbin,firstbin))+offset, max((topbin,lastbin))+3): 
            for n in range(min((botbin,firstbin))+offset, min(min((botbin,firstbin))+4+offset,max((topbin,lastbin))+2)):     
                if plot_avg==0:
                    with app1.create(SubFigure(width=NoEscape(r'0.475\linewidth'))) as image:
                        image.add_image('cur_fig/scatter_average.pdf', width=NoEscape(r'\linewidth'))#, placement=NoEscape(r'\centering'))
                        image.add_caption('')
                        plot_avg = 1
                        n_offset = 1
                else:    
                    with app1.create(SubFigure(width=NoEscape(r'0.475\linewidth'))) as image:
                        image.add_image('cur_fig/scatter_bin'+str(n-n_offset)+'.pdf', width=NoEscape(r'\linewidth'))#, placement=NoEscape(r'\centering'))
                        image.add_caption('')
                    # image.add_label('Scatter'+str(n))
                if (n % 2) == 1:
                    #doc.append(NoEscape(r'%% Leave a black line to create line break')  )  
                    app1.append(NoEscape(r'\medskip')) 
                else:
                    app1.append(NoEscape(r'\hfill')) 
                offset = offset + 1
                # if offset >3:
                #     break
            main_figure.add_caption('Current velocity and direction. The eight red dots mark the maximum velocity in the four cardinal and four intercardinal directions.')
        main_figure.append(Label(NoEscape(r'fig:scatter'+str(k+1))))
        
app1.generate_tex(outputfolder+'/../appendix1')    
# Remove begin document
with open(outputfolder+'/../appendix1.tex', 'r') as fin:
    data = fin.read().splitlines(True)
with open(outputfolder+'/../appendix1.tex', 'w') as fout:
    fout.writelines(data[13:-2])
    
# Appendix 2 is generated where the tidal prediction is generated

# Appendix for time series per week
app3 = Document(documentclass='memoir')
with app3.create(Chapter('Time series per week')):        
    for i in range(0,weeks):
        with app3.create(Figure(position='h!')) as image:
            image.add_image('cur_fig/timeseries_week_'+str(i+1)+'.pdf', width=NoEscape(r'1.0\textwidth'), placement=NoEscape(r'\centering'))
            image.add_caption('Time series of measured velocity magnitude for week '+str(i+1)+' of measurements.')
        image.append(Label('fig:timeseriesweek'+str(i+1)))  
app3.generate_tex(outputfolder+'/../appendix3')    
# Remove begin document
with open(outputfolder+'/../appendix3.tex', 'r') as fin:
    data = fin.read().splitlines(True)
with open(outputfolder+'/../appendix3.tex', 'w') as fout:
    fout.writelines(data[11:-2])
    
# Appendix for quality control 
app4 = Document(documentclass='memoir')
files =[]
os.listdir(outputfolder+'/../quality_check')
for file in os.listdir(outputfolder+'/../quality_check'):
    if file.endswith('.pdf'):    # check only text files
        files.append(file)
files = natsorted(files)       
files = files[1:] #Remove water depth plot 

# dfind max bin 
with app4.create(Chapter('Quality checks')):     
    app4.append(NoEscape(r'''The measurements must undergo a series of quality checks as the raw data will have errors because of possible poor signal strength and objects interfering with the beam.
The pitch, roll and tilt shown in Figure \ref{fig:tilt} can show the instrument stability and indicate if the instrument is displaced during the measurement period.

Figure \ref{fig:volt} shows the battery voltage. Figure \ref{fig:temp} shows the temperature sensor results. 

Figure \ref{fig:A} shows the bin signal strength over the depth and the thresholds used for filtering measurements with invalid signal strength. The signal must pass four checks to be valid:
\begin{enumerate}
\item The signal strength must be lower than a specified upper threshold.
\item The signal strength must be larger than a specified lower threshold.
\item The signal strength must be lower than the mean signal strengh times a supplier specified factor. 
\item The bin location must be below the measurement sidelobe height,
$\text{sidelobe} = \text{cos}(\text{beamangle})/(\text{180}\cdot\pi) \cdot \text{(instumentdepth} - \text{instrument height)}$.
\end{enumerate}
To further quality check the data, each bin is plotted on top of the previous bin as shown in Figure \ref{fig:uv_bin1} to Figure \ref{fig:uv_bin''' + str(number_of_bins) + r'''} to check 
for potential outliers between each bin. 
The final check is a plot of the standard deviation of the vertical velocity as shown in Figure \ref{fig:std_w}, where it is seen that the corrected data has a much lower vertical velocity standard deviation than the original data.  
 ''' +str('\n\n')))    

    for j in files:
        with app4.create(Figure(position='ht')) as image:
            image.add_image('quality_check/'+j, width=NoEscape(r'1.0\textwidth'), placement=NoEscape(r'\centering'))
            image.add_caption('')
            image.append(NoEscape(r'''\label{fig:'''+r"{}".format(j[:-4])+'}'))
         #image.add_caption('Time series of measured velocity magnitude for week '+str(i+1)+' of measurements.')
    # image.append(Label('fig:timeseriesweek'+str(i+1)))  
app4.generate_tex(outputfolder+'/../appendix4')    
# Remove begin document
with open(outputfolder+'/../appendix4.tex', 'r') as fin:
    data = fin.read().splitlines(True)
with open(outputfolder+'/../appendix4.tex', 'w') as fout:
    fout.writelines(data[12:-2])

app5 = Document(documentclass='memoir')
with app5.create(Chapter('Progressive Vector Diagram')):        
    # for i in range(0,weeks):
    with app5.create(Figure(position='h!')) as image:
        image.add_image('cur_fig/PVD.pdf', width=NoEscape(r'1.0\textwidth'), placement=NoEscape(r'\centering'))
        image.add_caption(NoEscape('''Progressive Vector Diagram (PVD) of the top, middle, bottom bin and vertical vector averaged bin. 
                                   The vertical vector average value is used when a bin point value is invalid. Invalid points are marked with black dots. 
                                   If the vertical vector average point also is invalid, the previous value is repeated and marked with red crosses.'''))
    image.append(Label('fig:PVD'))  
app5.generate_tex(outputfolder+'/../appendix5')    
# Remove begin document
with open(outputfolder+'/../appendix5.tex', 'r') as fin:
    data = fin.read().splitlines(True)
with open(outputfolder+'/../appendix5.tex', 'w') as fout:
    fout.writelines(data[12:-2])    



doc.append(NoEscape(r'\appendix'))
doc.append(NoEscape(r'\include{appendix3}'))    
doc.append(NoEscape(r'\clearpage'))    
doc.append(NoEscape(r'\include{appendix1}')) 
doc.append(NoEscape(r'\clearpage'))         
doc.append(NoEscape(r'\include{appendix2}'))    
doc.append(NoEscape(r'\clearpage'))         
doc.append(NoEscape(r'\include{appendix5}'))  
doc.append(NoEscape(r'\clearpage'))         
doc.append(NoEscape(r'\include{appendix4}')) 
   
doc.generate_pdf(outputfolder+'/../report', compiler='pdflatex',clean_tex=False,clean=False)
doc.generate_pdf(outputfolder+'/../report', compiler='pdflatex',clean_tex=False,clean=False)
toc_all = counttime.perf_counter()
print('Total time to generate report: ' + "%.2f" % (toc_all-tic_all) + ' s')
#1311
