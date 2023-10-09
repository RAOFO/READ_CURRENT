#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 23:38:19 2022

@author: oystein

Purpose: Convert Storm current data to AquaSim current data.
Notes:  *.v1 is data in east/west.
        *.v2 is data in north/south.
        header file is *.hdr.
        data is in compass north, meaning it should be transformed by 5°.
        
        Need to do:
            -remove current measurements with high vertical velocities
            -ASCII export
            -Check height of first bin
            -Check how depth is calculated for RDI 
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pandas as pd
def filter_non_digits(string: str) -> str:
    result = ''
    for char in string:
        if char in '1234567890.':
            result += char
    return result 
#from windrose import WindroseAxes
plt.close('all')


folder = '../../../Data/ADCP/VELF2301/'
filename = 'VELF2301'
name = 'VELF2301' # Name to use for titles etc.
instrument = 'AWAC_RAW' #Data type. Can be 'AWAC', 'AWAC_RAW', 'AquaPro' or 'RDI
data_folder = folder+'RawASCII/'
report_folder = folder+'curr_report/'
amp_tresh_low = 45   # 45 for AWAC 600 kHz, 25 for Aquadopp profiler 1200kHz
amp_tresh_high = 170 # 165 for AWAC 600 kHz, 200 for Aquadopp profiler 1200kHz
ampmult = 2.5     # 2 for AWAC 600 kHz, 2.5 for Aquadopp profiler 1200kHz
ens_start = 100
ens_end = 7915 #15023
instr_height = 0.3
mag_dec = -4.02
orientation = 1 # 1 = upwards, -1 = downwards
waterdepth = 22 # Estimated water depth when downwards looking
lat = 61.976510 #20231003, From deployment
long = -6.850194 #20231003, From deployment

##########################
### Standard variables ###
# First, check the noise level of the instrument. See the readings in the software, under the Current tab.
# Pinging in air should produce signal strengths (amplitudes) of 15-30 counts
if instrument == 'AWAC_RAW' or instrument == 'Aquapro' or instrument == 'AWAC':
    beam_angle = 25
    depth_col = 14
    head_col = 11
    pitch_col = 12
    roll_col = 13
    temp_col = 15
    volt_col = 9
    if instrument == 'AWAC': 
        instrument_type = 'AWAC'
    if instrument == 'AWAC_RAW': 
        instrument_type = 'AWAC'
    if instrument == 'Aquapro': 
        instrument_type = 'Aquapro'    
if instrument == 'RDI':
    # amp_tresh_low = 37   # 45 for AWAC 600 kHz, 25 for Aquadopp profiler 1200kHz
    # amp_tresh_high = 165 # 165 for AWAC 600 kHz, 200 for Aquadopp profiler 1200kHz
    ampmult = 2      # 2 for AWAC 600 kHz, 2.5 for Aquadopp profiler 1200kHz
    beam_angle = 25
    depth_col = 14
    head_col = 12
    pitch_col = 10
    roll_col = 11
    temp_col = 13 #NA
    volt_col = 17 #NA
    instrument_type = 'RDI Workhorse Sentinel'#'RDI'
    
if instrument == 'VECTOR':
    instrument_type = 'Vector'
    
    beam_angle = 0
    depth_col = 0
    head_col = 11
    pitch_col = 12
    roll_col = 13
    temp_col = 14
    volt_col = 9
    
plotwidth = 10
savefig = True   #True or False
nan_num = -99
##########################

if not os.path.exists(report_folder):
        os.makedirs(report_folder)

if not os.path.exists(report_folder+'quality_check'):
        os.makedirs(report_folder+'quality_check')
        
if not os.path.exists(folder+'proc_ASCII'):
        os.makedirs(folder+'proc_ASCII')
        
        
#%%%%%%%%%%%%%%%%%%%%
## Read for export metadata

if instrument == 'RDI':
    mdata=pd.read_csv(data_folder+'u.txt',nrows = 12, sep = '      ',encoding = "ISO-8859-1").to_numpy()
    
else:
    mdata=pd.read_csv(data_folder+filename+'.hdr', sep='\t',encoding = "ISO-8859-1").to_numpy()
def find_string(string, number):
    # Find substring in string. "number" is the time the substring occurs.
    count = 0
    for i in range(len(mdata)):
        x = mdata[i][0].find(string)
        if x > -1:
            count = count + 1
            if count == number:
                break
    return i
mdata_var = []


# Strings to export
ID = name
if instrument_type == 'AWAC' or instrument_type == 'Aquapro':
    bin_length = int(float(filter_non_digits(mdata[find_string('Cell size',1)][0])))/100
    # date_first = str(mdata[3]).split(' ')[-3].split('/')[1] + '/' + str(mdata[3]).split(' ')[-3].split('/')[0] + '/'+ str(mdata[3]).split(' ')[-3].split('/')[2]
    # date_last = str(mdata[4]).split(' ')[-3].split('/')[1] + '/' + str(mdata[4]).split(' ')[-3].split('/')[0] + '/'+ str(mdata[4]).split(' ')[-3].split('/')[2]
    bin_length_first = bin_length+float(filter_non_digits(mdata[find_string('Blanking distance',1)][0]))
    serial_number = str(mdata[find_string('Serial number',2)]).split(' ')[-2]+' '+str(mdata[find_string('Serial number',2)]).split(' ')[-1][:-2]
    number_of_bins =        filter_non_digits(mdata[find_string('Number of cells',1)][0])
    measurement_load =  filter_non_digits(mdata[find_string('Measurement load',1)][0]) #+ ' kHz'
    profile_interval=       filter_non_digits(mdata[find_string('Profile interval',1)][0]) + ' s'
    average_interval=       filter_non_digits(mdata[find_string('Average interval',1)][0])+' s'
    # profile_interval=       filter_non_digits(mdata[find_string('Time/Ping',1)][0]) + ' s'
    # average_interval=       filter_non_digits(mdata[find_string('Pings/Ens',1)][0])+' s'
    number_of_measurements= filter_non_digits(mdata[find_string('Number of measurements',1)][0])
    blanking_distance=      filter_non_digits(mdata[find_string('Blanking distance',1)][0]) + ' m'
if instrument == 'RDI':
    bin_length = int(float(filter_non_digits(mdata[find_string('Bin Size',1)][0])))
    # dates read from SEN
    #date_first = str(mdata[4]).split(' ')[-1][:-3]+ str(' ')+str(mdata[5]).split(' ')[-1][:-6] 
    #date_last = str(mdata[4]).split(' ')[-2] + str(' ') + str(mdata[4]).split(' ')[-1][:-2]
    bin_length_first = bin_length+float(filter_non_digits(mdata[find_string('1st Bin Range',1)][0][2:]))
    serial_number = ''
    number_of_bins = mdata[9][0].split('\t')[-1]
    measurement_load =  '100'
    profile_interval=       str(float(filter_non_digits(mdata[find_string('Ensemble Interval (s)',1)][0]))) + ' s'
    average_interval=       str(float(filter_non_digits(mdata[find_string('Pings/Ens',1)][0]))) + ' s'
    #number_of_measurements= filter_non_digits(mdata[find_string('Number of measurements',1)][0]) Done in sen file
    blanking_distance=      str(float(filter_non_digits(mdata[find_string('1st Bin Range',1)][0][2:]))) + ' m'
if instrument == 'VECTOR':
    serial_number = str(mdata[find_string('Serial number',2)]).split(' ')[-2]+' '+str(mdata[find_string('Serial number',2)]).split(' ')[-1][:-2]

    average_interval =  str(mdata[find_string('Burst interval',1)]).split(' ')[-1][:-2]
    profile_interval =  str(mdata[find_string('Sampling rate',1)]).split(' ')[-2] + ' ' + str(mdata[find_string('Sampling rate',1)]).split(' ')[-1][:-2]
    number_of_bins = 1 
    measurement_load =  filter_non_digits(mdata[find_string('Measurement load',1)][0]) #+ ' kHz'
    blanking_distance=      '0.15 m'
instrument_frequency =  filter_non_digits(mdata[find_string('Head frequency',1)][0]) + ' kHz'
bin_size =          str(bin_length) + ' m'


if instrument == 'RDI':
    instrument_frequency =  filter_non_digits(mdata[find_string('Broadband',1)][0]) + ' kHz'
    ping_hz = str(round(1/float(filter_non_digits(mdata[find_string('Time/Ping',1)][0])),3))
if instrument == 'AWAC' or instrument == 'AWAC_RAW':
    if instrument_frequency == '1000 kHz':
        ping_hz = '7 Hz'
    if instrument_frequency == '600 kHz':
        ping_hz = '4 Hz'
    if instrument_frequency == '400 kHz':
        ping_hz = '2 Hz'
    ping_hz = "%.2f" % (int(float(ping_hz[:-3]))*int(float(measurement_load))/100) + str(' hz (')+ ping_hz[:-2] + 'hz @ ' + str(measurement_load) + ' %)'    
if instrument == 'Aquapro'  :
    if instrument_frequency == '2000 kHz':
        ping_hz = '23 Hz'  
    if instrument_frequency == '1000 kHz':
        ping_hz = '7 Hz'
    if instrument_frequency == '600 kHz':
        ping_hz = '4 Hz'      
    if instrument_frequency == '400 kHz':
        ping_hz = '3 Hz'  
    ping_hz = "%.2f" % (int(float(ping_hz[:-3]))*int(float(measurement_load))/100) + str(' hz (')+ ping_hz[:-2] + 'hz @ ' + str(measurement_load) + ' %)'    


if instrument == 'VECTOR'  : 
    instrument_frequency =  filter_non_digits(mdata[find_string('Head frequency ',1)][0]) + ' kHz'
    ping_hz = '32 Hz' #str(round(1/float(filter_non_digits(mdata[find_string('Time/Ping',1)][0])),3))
    
    ping_hz = "%.2f" % (int(float(ping_hz[:-3]))*int(float(measurement_load))/100) + str(' hz (')+ ping_hz[:-2] + 'hz @ ' + str(measurement_load) + ' %)'    


#%%%%%%%%%%%%%%%%%%%%
## Read files.
def readdate(path):#,idx):
    data = np.genfromtxt(path, delimiter = '')   
#    data = np.delete(data,idx, axis= 0)
    date = data[ens_start-1:ens_end-1,0:5]
    return date


if instrument == 'AWAC_RAW' or instrument == 'Aquapro':
    def readdata(path):
        data = np.genfromtxt(path, delimiter = '')   
        return data[ens_start-1:ens_end-1,:]

    data_v1 = readdata(data_folder+filename+'.v1')
    data_v2 = readdata(data_folder+filename+'.v2')
    data_v3 = readdata(data_folder+filename+'.v3')
    data_sen = readdata(data_folder+filename+'.sen')
    data_a1 = readdata(data_folder+filename+'.a1')
    data_a2 = readdata(data_folder+filename+'.a2')
    data_a3 = readdata(data_folder+filename+'.a3')

if instrument == 'AWAC':
    def readdata(path):
        data = np.genfromtxt(path, delimiter = '')   
        return data[ens_start-1:ens_end-1,:]

    data_v1 = readdata(data_folder+filename+'.v1')
    data_v2 = readdata(data_folder+filename+'.v2')
    data_v3 = readdata(data_folder+filename+'.v3')
    data_sen = readdata(data_folder+filename+'.sen')
    data_a1 = readdata(data_folder+filename+'.a1')
    data_a2 = readdata(data_folder+filename+'.a2')
    data_a3 = readdata(data_folder+filename+'.a3')    
# if instrument == 'AWAC':
#     def readdata(path):
#         data = np.genfromtxt(path, delimiter = '')   
#         return data[ens_start-1:ens_end-1,:]

#     data_v1 = readdata(data_folder+filename+'.v1')
#     data_v2 = readdata(data_folder+filename+'.v2')
#     data_v3 = readdata(data_folder+filename+'.v3')
#     data_sen = readdata(data_folder+filename+'.sen')
#     data_a1 = readdata(data_folder+filename+'.a1')
#     data_a2 = readdata(data_folder+filename+'.a2')
#     data_a3 = readdata(data_folder+filename+'.a3')
    
if instrument == 'RDI':
    def readdata(path):
        data = np.genfromtxt(path, delimiter = '\t',skip_header=16)   
        return data[ens_start-1:ens_end-1,:]

    # data_v1 = readdata(data_folder+filename+'.u')/1000
    # data_v2 = readdata(data_folder+filename+'.v')/1000
    # data_v3 = readdata(data_folder+filename+'.w')/1000
    # data_v1 = np.delete(data_v1, slice(0,9), axis = 1)
    # data_v2 = np.delete(data_v2, slice(0,9), axis = 1)
    # data_v3 = np.delete(data_v3, slice(0,9), axis = 1)
    # data_sen = readdata(data_folder+'anc.txt')
    data_v1 = readdata(data_folder+'u.txt')/1000
    data_v2 = readdata(data_folder+'v.txt')/1000
    data_v3 = readdata(data_folder+'w.txt')/1000
    if os.path.exists(data_folder+'e.txt'):
        data_verror = readdata(data_folder+'e.txt')/1000
    if os.path.exists(data_folder+'err.txt'):
        data_verror = readdata(data_folder+'err.txt')/1000
    data_v1 = np.delete(data_v1, slice(0,9), axis = 1)
    data_v2 = np.delete(data_v2, slice(0,9), axis = 1)
    data_v3 = np.delete(data_v3, slice(0,9), axis = 1)
    data_verror = np.delete(data_verror, slice(0,9), axis = 1)
    # data_sen=pd.read_csv(data_folder+'anc.txt', skiprows=16, delimiter ='\t',decimal=',').to_numpy()[ens_start-1:ens_end-1,:].astype(float)
    data_sen = readdata(data_folder+'anc.txt')

if instrument == 'VECTOR':
    def readdata(path,usecols):
        data = np.genfromtxt(path,usecols=usecols)   
        return data[ens_start-1:ens_end-1]
    
    data_v1 = readdata(data_folder+filename+'.dat',2).T
    data_v2 = readdata(data_folder+filename+'.dat',3).T
    data_v3 = readdata(data_folder+filename+'.dat',4).T
    data_sen = readdata(data_folder+filename+'.sen',range(0,14))
    

if instrument == 'VECTOR':    
    data_v1_corr = np.empty((np.size(data_v1,0)))
    data_v2_corr = np.empty((np.size(data_v1,0)))
    data_v3_corr = np.empty((np.size(data_v1,0)))
else:
    data_v1_corr = np.empty((np.size(data_v1,0),np.size(data_v1,1)))
    data_v2_corr = np.empty((np.size(data_v1,0),np.size(data_v1,1)))
    data_v3_corr = np.empty((np.size(data_v1,0),np.size(data_v1,1)))
    
# Measurement was upside down
if filename == 'TEST2302': 
    data_v1 = np.flip(data_v1, axis = 1)
    data_v2 = np.flip(data_v2, axis = 1)
    data_v3 = np.flip(data_v3, axis = 1)
    data_verror = np.flip(data_verror, axis = 1)

# Measurement was upside down
# if filename == 'SIBC2303': 
#     data_v1 = np.flip(data_v1, axis = 1)
#     data_v2 = np.flip(data_v2, axis = 1)
#     data_v3 = np.flip(data_v3, axis = 1)
#     data_verror = np.flip(data_verror, axis = 1)

#data_v1_corr[:] = np.NaN
#data_v2_corr[:] = np.NaN
#data_v3_corr[:] = np.NaN

#str2date = lambda x: datetime.strptime(x, '%d.%m.%Y %H:%M')
#
#timevec = np.transpose(np.array([data_sen[:,2],data_sen[:,1],data_sen[:,0],data_sen[:,3],data_sen[:,4],data_sen[:,5]]))
#
#time = np.zeros(len(data_sen[:,0]))   #datetime(np.transpose(np.array([data_sen[:,2],data_sen[:,1],data_sen[:,0],data_sen[:,3],data_sen[:,4],data_sen[:,5]])))
#for i in range(len(time)):
#    time[i] = datetime(int(timevec[i,0]),int(timevec[i,2]),int(timevec[i,1]),int(timevec[i,3]),int(timevec[i,4]),int(timevec[i,5]))


#dates = ['01/01/2016 04:50','01/01/2016 05:50','01/01/2016 06:50','01/01/2016 07:50']
#DATE = [datetime.strptime(x,'%m/%d/%Y %H:%M') for x in dates]

testtime = [] #np.empty(len(data_sen[:,0])) #np.zeros(len(data_sen[:,0]))   #datetime(np.transpose(np.array([data_sen[:,2],data_sen[:,1],data_sen[:,0],data_sen[:,3],data_sen[:,4],data_sen[:,5]])))
if ((instrument == 'AWAC') or (instrument=='AWAC_RAW') or (instrument=='Aquapro') or (instrument=='VECTOR')):
    for i in range(len(data_sen[:,0])):
        testtime.append(str(int(data_sen[i,2]))+'.'+str(int(data_sen[i,0]))+'.'+str(int(data_sen[i,1]))+' '+str(int(data_sen[i,3]))+':'+str(int(data_sen[i,4]))+':'+str(int(data_sen[i,5])))
if instrument == 'RDI':
    for i in range(len(data_sen[:,0])):
        testtime.append(str(int(data_sen[i,1]+2000))+'.'+str(int(data_sen[i,2]))+'.'+str(int(data_sen[i,3]))+' '+str(int(data_sen[i,4]))+':'+str(int(data_sen[i,5]))+':'+str(int(data_sen[i,6])))
    
time = [datetime.strptime(x,'%Y.%m.%d %H:%M:%S') for x in testtime]
# if instrument_type == 'RDI':
date_first = str(time[0])
date_last = str(time[-1])
number_of_measurements= str(len(data_sen[:,0]))
###############
### sensors ###
if ((instrument == 'AWAC') or (instrument == 'AWAC_RAW') or (instrument=='Aquapro')):
    depth = data_sen[:,depth_col-1]/1.025+instr_height      # Water level
if instrument == 'RDI':
    depth = data_sen[:,depth_col-1]/1.025+instr_height      # Water level
if instrument == 'VECTOR':
    depth = instr_height*np.ones((np.size(data_v1,0)))
head = np.mod(data_sen[:,head_col-1] + mag_dec,360)     # heading corrected for magnetic declination
pitch = data_sen[:,pitch_col-1]                         # Pitch
roll = data_sen[:,roll_col-1]                           # Roll
temp = data_sen[:,temp_col-1]                           # Temperature
volt = data_sen[:,volt_col-1]                           # Voltage


if orientation == 1:
    depth_av = np.round(np.nanmean(depth), decimals=2) 
else:
    depth_av = waterdepth 

nums = np.arange(0,np.size((data_sen),axis=0))+1
fig = plt.subplots(figsize =(plotwidth, 2/3*plotwidth))
plt.plot(nums, depth, linewidth=0.75)
plt.plot([nums[0],nums[-1]],[depth_av,depth_av], linewidth=2.0)
plt.text(nums[-1],depth_av,depth_av)
plt.grid(True)
plt.xlabel('Ensemble')
plt.title(str(name) + ', water level')
plt.ylabel('Water level  (m)')
plt.tight_layout()
plt.show()
if savefig == True:
    plt.savefig(report_folder+'quality_check/1d.png',dpi = 300)
    plt.savefig(report_folder+'quality_check/1d.pdf',dpi = 600)


# fig = plt.subplots(figsize =(plotwidth, 2/3*plotwidth))
# plt.plot(nums, head, linewidth=0.75)
# #plt.plot([nums[0],nums[-1]],[depth_av,depth_av])
# #plt.text(nums[-1],depth_av,depth_av)
# plt.grid(True)
# plt.xlabel('Ensemble')
# plt.title(str(name) + ', heading')
# plt.ylabel('Heading (dgr, true north)')
# plt.tight_layout()
# plt.show()
# if savefig == True:
#     plt.savefig(report_folder+'quality_check/head.png',dpi = 300)
#     plt.savefig(report_folder+'quality_check/head.pdf',dpi = 600)
    
fig, ax = plt.subplots(figsize =(plotwidth, 2/3*plotwidth))
ax2 = ax.twinx()
p1=ax.plot(nums, pitch, linewidth=0.75,label='Pitch')
p2=ax.plot(nums, roll, linewidth=0.75,label = 'Roll')
p3=ax2.plot(nums, head, linewidth=0.75,color='C2', label = 'Heading')
ax.grid(True)
ax.set_xlabel('Ensemble')
ax.set_title(str(name) + ', tilt')
ax.set_ylabel('Tilt, °')
ax2.set_ylabel('Heading (°, true north)')
plt.tight_layout()
# ax.legend(['Pitch','Roll'])#
# ax2.legend('Heading')
lns = p1+p2+p3
labs = [l.get_label() for l in lns]
ax2.legend(lns, labs, loc=1)
plt.show()
if savefig == True:
    plt.savefig(report_folder+'quality_check/tilt.png',dpi = 300)
    plt.savefig(report_folder+'quality_check/tilt.pdf',dpi = 600)

fig = plt.subplots(figsize =(plotwidth, 2/3*plotwidth))
plt.plot(nums, temp, linewidth=0.75)
plt.grid(True)
plt.xlabel('Ensemble')
plt.title(str(name) + ', temperature')
plt.ylabel('Temperature  (dgr C)')
plt.tight_layout()
#plt.legend('Pitch','Roll')
plt.show()
if savefig == True:
    plt.savefig(report_folder+'quality_check/temp.png',dpi = 300)
    plt.savefig(report_folder+'quality_check/temp.pdf',dpi = 600)

fig = plt.subplots(figsize =(plotwidth, 2/3*plotwidth))
plt.plot(nums, volt, linewidth=0.75)
plt.grid(True)
plt.xlabel('Ensemble')
if instrument == 'RDI':
    plt.title(str(filename) + ', voltage')
    plt.ylabel('Battery (??)')
    plt.tight_layout()
    plt.show()
    if savefig == True:
        plt.savefig(report_folder+'quality_check/battery.png',dpi = 300)
        plt.savefig(report_folder+'quality_check/battery.pdf',dpi = 600)
else:
    plt.title(str(filename) + ', voltage')
    plt.ylabel('Voltage  (V)')
    plt.tight_layout()
    plt.show()
    if savefig == True:
        plt.savefig(report_folder+'quality_check/volt.png',dpi = 300)
        plt.savefig(report_folder+'quality_check/volt.pdf',dpi = 600)

############
### Bins ###
if instrument == 'VECTOR':
    bins = 1
    binheights = depth[0]# np.array([depth[0],depth[0]])
    bindepths = np.array([depth[0],depth[0]])
else:
    bins = np.arange(0,np.size(data_v1,1))+1
    binheights = np.arange(0,np.size(data_v1,1))*bin_length+bin_length_first+instr_height
    bindepths = orientation*(orientation*depth_av - binheights)
    if orientation == -1:
        bindepths = binheights    



#################
### Amplitude  ###
### check error ##

#################
### Flag data ###
if orientation == 1:
    sidelobeheight = np.cos(beam_angle/180*np.pi)*(np.min(depth) - instr_height)
    goodbins = np.array(np.where(binheights < sidelobeheight))
else:
    sidelobeheight = np.cos(beam_angle/180*np.pi)*(depth_av - instr_height)
    goodbins = np.array(np.where(binheights < sidelobeheight))
    #goodbins = bins
        
print('Correcting data')        
if instrument == 'AWAC_RAW' or instrument =='Aquapro':
    a1av = np.zeros(np.size(data_a1,1))
    a2av = np.zeros(np.size(data_a1,1))
    a3av = np.zeros(np.size(data_a1,1))
    a1max = np.zeros(np.size(data_a1,1))
    a2max = np.zeros(np.size(data_a1,1))
    a3max = np.zeros(np.size(data_a1,1))
    a1min = np.zeros(np.size(data_a1,1))
    a2min = np.zeros(np.size(data_a1,1))
    a3min = np.zeros(np.size(data_a1,1))
    for m in range(1,np.size(data_a1,1)+1):
        a1av[m-1] = np.nanmean(data_a1[:,m-1])
        a2av[m-1] = np.nanmean(data_a2[:,m-1])
        a3av[m-1] = np.nanmean(data_a3[:,m-1])
        a1max[m-1] = np.nanmax(data_a1[:,m-1])
        a2max[m-1] = np.nanmax(data_a2[:,m-1])
        a3max[m-1] = np.nanmax(data_a3[:,m-1])  
        a1min[m-1] = np.nanmin(data_a1[:,m-1])
        a2min[m-1] = np.nanmin(data_a2[:,m-1])
        a3min[m-1] = np.nanmin(data_a3[:,m-1])
    

    for i in range(0,np.size(data_v1,0)):
    #    for k in range(0,np.size(goodbins)): #np.where(binheights < sidelobeheight):
    #        j = goodbins[0,k]
        for j in range(0,np.size(data_v1,1)):
            if np.min((data_a1[i,j], data_a2[i,j], data_a3[i,j])) < amp_tresh_low:
                data_v1_corr[i,j] = np.NaN
                data_v2_corr[i,j] = np.NaN
                data_v3_corr[i,j] = np.NaN
            elif np.max((data_a1[i,j], data_a2[i,j], data_a3[i,j])) > amp_tresh_high:
                data_v1_corr[i,j] = np.NaN
                data_v2_corr[i,j] = np.NaN
                data_v3_corr[i,j] = np.NaN
            elif np.mean((data_a1[i,j], data_a2[i,j], data_a3[i,j])) > ampmult*np.mean((a1av[j], a2av[j], a3av[j])):
                data_v1_corr[i,j] = np.NaN
                data_v2_corr[i,j] = np.NaN
                data_v3_corr[i,j] = np.NaN 
            elif np.any(goodbins == j) == False: #j in goodbins == False:
                data_v1_corr[i,j] = np.NaN
                data_v2_corr[i,j] = np.NaN
                data_v3_corr[i,j] = np.NaN 
            else:
                data_v1_corr[i,j] = data_v1[i,j]
                data_v2_corr[i,j] = data_v2[i,j]
                data_v3_corr[i,j] = data_v3[i,j]


 
            # data_v3_std = 5*np.nanstd(data_v3[:,j])+np.nanmean(data_v3[:,j])
            # data_v3_std_minus = -5*np.nanstd(data_v3[:,j])+np.nanmean(data_v3[:,j])
            # # for i in range(0,np.size(data_v1,0)):                                
            # if data_v3[i,j] > data_v3_std or data_v3[i,j] < data_v3_std_minus:
            #         data_v1_corr[i,j] = np.NaN
            #         data_v2_corr[i,j] = np.NaN
            #         data_v3_corr[i,j] = np.NaN            
            #         # data_err_corr[i,j] = np.NaN
        # print('Corrected bin ' +str(j))
    if 'nanbins' in locals():
        for i in nanbins:
            data_v1_corr[:,i-1] = np.NaN
            data_v2_corr[:,i-1] = np.NaN
            data_v3_corr[:,i-1] = np.NaN 

print('Correcting data')        
if instrument == 'AWAC':
    a1av = np.zeros(np.size(data_a1,1))
    a2av = np.zeros(np.size(data_a1,1))
    a3av = np.zeros(np.size(data_a1,1))
    a1max = np.zeros(np.size(data_a1,1))
    a2max = np.zeros(np.size(data_a1,1))
    a3max = np.zeros(np.size(data_a1,1))
    a1min = np.zeros(np.size(data_a1,1))
    a2min = np.zeros(np.size(data_a1,1))
    a3min = np.zeros(np.size(data_a1,1))
    for m in range(1,np.size(data_a1,1)+1):
        a1av[m-1] = np.nanmean(data_a1[:,m-1])
        a2av[m-1] = np.nanmean(data_a2[:,m-1])
        a3av[m-1] = np.nanmean(data_a3[:,m-1])
        a1max[m-1] = np.nanmax(data_a1[:,m-1])
        a2max[m-1] = np.nanmax(data_a2[:,m-1])
        a3max[m-1] = np.nanmax(data_a3[:,m-1])  
        a1min[m-1] = np.nanmin(data_a1[:,m-1])
        a2min[m-1] = np.nanmin(data_a2[:,m-1])
        a3min[m-1] = np.nanmin(data_a3[:,m-1])    
    data_v1_corr = np.copy(data_v1)
    data_v2_corr = np.copy(data_v2)
    data_v3_corr = np.copy(data_v3)
# myVar exists.


# Error Velocity: A key quality control parameter that derives from the four
# beam geometry of an ADCP. Each pair of opposing beams provides one
# measurement of the vertical velocity and one component of the horizontal
# velocity, so there are actually two independent measurements of vertical
# velocity that can be compared. If the flow field is homogeneous, the difference 
# between these vertical velocities will average to zero. To put the error
# velocity on a more intuitive footing, it is scaled to be comparable to the
# variance in the horizontal velocity. In a nutshell, the error velocity can be
# treated as an indication of the standard deviation of the horizontal velocity
# measurements.
if instrument == 'RDI':
    # data_v3_std = 2*np.nanstd(data_v3[:,:])+np.nanmean(data_v3[:,:])
    # data_v3_std_minus = -2*np.nanstd(data_v3[:,:])+np.nanmean(data_v3[:,:])
    data_v1_corr = np.copy(data_v1)
    data_v2_corr = np.copy(data_v2)
    data_v3_corr = np.copy(data_v3)
    data_err_corr = np.copy(data_verror)

    for j in range(0,np.size(data_v1,1)):
        # data_err_corr_std = np.nanstd(data_err_corr[:,j])+np.nanmean(data_err_corr[:,j])
        # data_err_corr_std_minus = -np.nanstd(data_err_corr[:,j])+np.nanmean(data_err_corr[:,j])
        data_err_corr_std = 7*np.nanstd(data_err_corr[:,j])+np.nanmean(data_err_corr[:,j])
        data_err_corr_std_minus = -7*np.nanstd(data_err_corr[:,j])+np.nanmean(data_err_corr[:,j])
        
        data_v3_std = 5*np.nanstd(data_v3[:,j])+np.nanmean(data_v3[:,j])
        data_v3_std_minus = -5*np.nanstd(data_v3[:,j])+np.nanmean(data_v3[:,j])
        
        print(data_err_corr_std)
        print(data_err_corr_std_minus)
        # data_v3_std = 2*np.nanstd(data_v3[:,j])+np.nanmean(data_v3[:,j])
        # data_v3_std_minus = -2*np.nanstd(data_v3[:,j])+np.nanmean(data_v3[:,j])
        for i in range(0,np.size(data_v1,0)):
       # if np.min((data_a1[i,j], data_a2[i,j], data_a3[i,j])) < amp_tresh_low:        
            if data_err_corr[i,j] > data_err_corr_std or data_err_corr[i,j] < data_err_corr_std_minus:
                    data_v1_corr[i,j] = np.NaN
                    data_v2_corr[i,j] = np.NaN
                    data_v3_corr[i,j] = np.NaN
                    data_err_corr[i,j] = np.NaN
                                  
            if data_v3[i,j] > data_v3_std or data_v3[i,j] < data_v3_std_minus:
                    data_v1_corr[i,j] = np.NaN
                    data_v2_corr[i,j] = np.NaN
                    data_v3_corr[i,j] = np.NaN            
                    data_err_corr[i,j] = np.NaN
            # # if (abs(data_verror[i,j])> np.sqrt(data_v1[i,j]**2+data_v2[i,j]**2)*0.25) and (data_verror[i,j]<data_err_corr_std) and (data_verror[i,j]>data_err_corr_std_minus):
            # if (data_verror[i,j]>data_err_corr_std) or (data_verror[i,j]<data_err_corr_std_minus):    
            #     if (abs(data_verror[i,j])> np.sqrt(data_v1[i,j]**2+data_v2[i,j]**2)*0.25):
            #         data_err_corr[i,j] = np.NaN
            #         data_v1_corr[i,j] = np.NaN
            #         data_v2_corr[i,j] = np.NaN
            #         data_v3_corr[i,j] = np.NaN
            
if instrument == 'VECTOR':
    data_v1_corr = np.copy(data_v1)
    data_v2_corr = np.copy(data_v2)
    data_v3_corr = np.copy(data_v3)
    # for j in range(0,np.size(data_v1,1)):
    data_v3_std = 5*np.nanstd(data_v3)+np.nanmean(data_v3)
    data_v3_std_minus = -5*np.nanstd(data_v3)+np.nanmean(data_v3)
    data_v1_std = 4*np.nanstd(data_v1)+np.nanmean(data_v1)
    data_v1_std_minus = -4*np.nanstd(data_v1)+np.nanmean(data_v1)
    data_v2_std = 4*np.nanstd(data_v2)+np.nanmean(data_v2)
    data_v2_std_minus = -4*np.nanstd(data_v2)+np.nanmean(data_v2)
    for i in range(0,np.size(data_v1,0)):

        if data_v1[i] > data_v1_std or data_v1[i] < data_v1_std_minus:
                data_v1_corr[i] = np.NaN
                data_v2_corr[i] = np.NaN
                data_v3_corr[i] = np.NaN    
        if data_v2[i] > data_v2_std or data_v2[i] < data_v2_std_minus:
                data_v1_corr[i] = np.NaN
                data_v2_corr[i] = np.NaN
                data_v3_corr[i] = np.NaN    
                # data_err_corr[i] = np.NaN
        if data_v3[i] > data_v3_std or data_v3[i] < data_v3_std_minus:
                data_v1_corr[i] = np.NaN
                data_v2_corr[i] = np.NaN
                data_v3_corr[i] = np.NaN    
#################################
### East-west and north-south ###
nums = np.arange(0,np.size((data_v1),axis=0))+1

if instrument == 'VECTOR':
    jrange = range(1,2)
    data_v1.shape = (len(data_v1),1)
    data_v2.shape = (len(data_v2),1)
    data_v3.shape = (len(data_v3),1)
    data_v1_corr.shape = (len(data_v1_corr),1)
    data_v2_corr.shape = (len(data_v2_corr),1)
    data_v3_corr.shape = (len(data_v3_corr),1)
else:
    jrange = range(1,np.size(data_v1,1)+1)

print('Plot UV')        
# for j in range(1,np.size(data_v1,1)+1):   
for j in jrange:   
    fig, axs = plt.subplots(6, sharex=True, figsize = (plotwidth, 4/3*plotwidth))
    axs[0].plot(nums, data_v1[:,j-1],'r', label = 'Raw data', linewidth=0.75)
    axs[0].plot(nums, data_v1_corr[:,j-1],'g', label = 'Corrected', linewidth=0.75)
    axs[0].grid(True)
    axs[0].set(title = str(name) + ', data for bin '+str(j)+' at depth {:0.2f}m.'.format(bindepths[j-1]))
    axs[0].set(ylabel = 'u')
    axs[1].plot(nums, data_v1[:,j-1],'r' , label = 'Bin del.'+str(j), linewidth=0.75)
    axs[1].plot(nums, data_v1_corr[:,j-1],'g' , label = 'Bin '+str(j)+' corr.', linewidth=0.75)
    if j > 1:
        # axs[1].plot(nums, data_v1[:,j-1]-data_v1[:,j-2],color = 'r', label = 'Bin '+str(j-1), linewidth=0.75)
        # axs[1].plot(nums, data_v1_corr[:,j-1]-data_v1[:,j-2],color = 'g', label = 'Bin '+str(j-1), linewidth=0.75)
        axs[1].plot(nums, data_v1[:,j-2],color = (0.5,0.5,0.5), label = 'Bin '+str(j-1), linewidth=0.75)
        axs[1].legend(loc='upper right')
    else:
        axs[1].legend(loc='upper right')
    axs[1].grid(True)
    axs[1].set(ylabel = 'u')
    
    axs[2].plot(nums, data_v2[:,j-1],'r' , label = 'Raw data', linewidth=0.75)
    axs[2].plot(nums, data_v2_corr[:,j-1],'g' , label = 'Corrected', linewidth=0.75)
    axs[2].grid(True)
    axs[2].set(ylabel = 'v')
    
    axs[3].plot(nums, data_v2[:,j-1], 'r', label = 'Bin '+str(j), linewidth=0.75)
    axs[3].plot(nums, data_v2_corr[:,j-1], 'g', label = 'Bin '+str(j)+' corr.', linewidth=0.75)
    if j > 1:
        axs[3].plot(nums, data_v2[:,j-2],color = (0.5,0.5,0.5), label = 'Bin '+str(j-1), linewidth=0.75)
        #axs[3].legend(loc='upper right')
    axs[3].grid(True)
    axs[3].set(ylabel = 'v')

    # Including signal strength
    if instrument == 'AWAC_RAW' or instrument == 'Aquapro':
        axs[4].plot(nums, data_a1[:,j-1],'r', label='Beam 1', linewidth=0.75)
        axs[4].plot(nums, data_a2[:,j-1],'g', label='Beam 2', linewidth=0.75)
        axs[4].plot(nums, data_a3[:,j-1],'b', label='Beam 3', linewidth=0.75)
        axs[4].plot([nums[0], nums[-1]], [amp_tresh_high, amp_tresh_high],'k--', linewidth=1.5)
        axs[4].plot([nums[0], nums[-1]], [ampmult*np.mean((a1av[j-1], a2av[j-1], a3av[j-1])), ampmult*np.mean((a1av[j-1], a2av[j-1], a3av[j-1]))],'k.-', linewidth=1.5)
        axs[4].plot([nums[0], nums[-1]], [amp_tresh_low, amp_tresh_low],'k:', linewidth=1.5)
        axs[4].grid(True)
        axs[4].set(ylabel = 'A')
        axs[4].set(xlabel = 'Ensemble')
        axs[4].legend(loc='upper right')
    if instrument == 'AWAC':
        axs[4].plot(nums, data_a1[:,j-1],'r', label='Beam 1', linewidth=0.75)
        axs[4].plot(nums, data_a2[:,j-1],'g', label='Beam 2', linewidth=0.75)
        axs[4].plot(nums, data_a3[:,j-1],'b', label='Beam 3', linewidth=0.75)
        axs[4].plot([nums[0], nums[-1]], [amp_tresh_high, amp_tresh_high],'k--', linewidth=1.5)
        axs[4].plot([nums[0], nums[-1]], [ampmult*np.mean((a1av[j-1], a2av[j-1], a3av[j-1])), ampmult*np.mean((a1av[j-1], a2av[j-1], a3av[j-1]))],'k.-', linewidth=1.5)
        axs[4].plot([nums[0], nums[-1]], [amp_tresh_low, amp_tresh_low],'k:', linewidth=1.5)
        axs[4].grid(True)
        axs[4].set(ylabel = 'A')
        axs[4].set(xlabel = 'Ensemble')
        axs[4].legend(loc='upper right')
    if instrument == 'RDI':
        axs[4].plot(nums, data_verror[:,j-1],'r', label='Raw data', linewidth=0.75)        
        axs[4].plot(nums, data_err_corr[:,j-1],'g', label='Corrected', linewidth=0.75)   
        axs[4].grid(True)
        axs[4].set(ylabel = 'Error, m/s')
        axs[4].set(xlabel = 'Ensemble')
        axs[4].legend(loc='upper right')
    # Including Vertical velocity
    axs[5].plot(nums, data_v3[:,j-1],'r', label='Vert. vel.', linewidth=0.75)
    axs[5].plot(nums, data_v3_corr[:,j-1],'b', label='Vert. vel. corrected', linewidth=0.75)
    axs[5].grid(True)
    axs[5].set(ylabel = 'w')
    axs[5].set(xlabel = 'Ensemble')
    #axs[5].legend(loc='upper right')
    
    #fig.suptitle(str(name) + ', data for bin '+str(j)+' at depth {:0.2f}m.'.format(bindepths[j-1]))   
    plt.tight_layout()
    plt.show()
    
    if savefig == True:
        plt.savefig(report_folder+'quality_check/uv_bin'+str(j)+'.png',dpi = 300)
        plt.savefig(report_folder+'quality_check/uv_bin'+str(j)+'.pdf',dpi = 600)
    


################
### Vertical ###
print('Plot Vertical')     
if instrument != 'VECTOR':
    verstd = np.zeros(np.size(data_v1,1))
    verstd_corr = np.zeros(np.size(data_v1,1))
    for k in range(0,np.size(data_v1,1)):
        verstd[k] = np.nanstd(data_v3[:,k])
        verstd_corr[k] = np.nanstd(data_v3_corr[:,k])
        
    fig = plt.subplots(figsize =(plotwidth/2, 3/4*plotwidth))
    plt.plot(verstd, binheights,'.-', linewidth=0.75,label='Std deviation')
    plt.plot(verstd_corr, binheights,'.-', linewidth=0.75,label='Corr. std deviation')
    for k in range(0,np.size(data_v1,1)):
        plt.text(verstd[k], binheights[k],k+1)
    plt.plot((np.min(verstd), np.max(verstd)),(depth_av, depth_av),'k', linewidth=1.5)
    
    if orientation == 1:
        plt.text(np.max(verstd)*0.8, depth_av, 'Waterlevel') # = '+str(depth_av)+'m')
        plt.plot((np.min(verstd), np.max(verstd)),(np.max(depth), np.max(depth)),':k', linewidth=1.5)
        plt.plot((np.min(verstd), np.max(verstd)),(np.min(depth), np.min(depth)),':k', linewidth=1.5)
        plt.plot((np.min(verstd), np.max(verstd)),(np.cos(beam_angle/180*np.pi)*np.min(depth), np.cos(beam_angle/180*np.pi)*np.min(depth)),'k--', linewidth=1.5)
        plt.text(np.max(verstd)*0.9, np.max(depth), 'Max')
        plt.text(np.max(verstd)*0.9, np.min(depth), 'Min')
        plt.text(np.max(verstd)*0.8,  np.cos(beam_angle/180*np.pi)*np.min(depth), 'Sidelobe')
    else:
        plt.text(np.max(verstd)*0.7, depth_av, 'Estimated depth') # = '+str(depth_av)+'m')
        plt.plot((np.min(verstd), np.max(verstd)),(np.cos(beam_angle/180*np.pi)*depth_av, np.cos(beam_angle/180*np.pi)*depth_av),'k--', linewidth=1.5)
        plt.text(np.max(verstd)*0.8,  np.cos(beam_angle/180*np.pi)*depth_av, 'Sidelobe')
        
    plt.title('Std of vertical velocity')
    plt.ylabel('Depth (m)')
    plt.xlabel('Std(w), (m/s)')
    plt.grid(True)
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.show()
    if savefig == True:
            plt.savefig(report_folder+'quality_check/std_w.png',dpi = 300)
            plt.savefig(report_folder+'quality_check/std_w.pdf',dpi = 600)



   
    

######################
### Amplitude plot ###
print('Plotting amplitide')
if ((instrument == 'AWAC_RAW') or (instrument=='AWAC') or (instrument=='Aquapro')):   
    fig = plt.subplots(figsize =(plotwidth/2, 3/4*plotwidth))
    plt.plot(a1av, binheights,'.-r', linewidth=1.0)
    plt.plot(a2av, binheights,'.-g', linewidth=1.0)
    plt.plot(a3av, binheights,'.-b', linewidth=1.0)
    plt.plot(a1max, binheights,'--r', linewidth=1.0)
    plt.plot(a2max, binheights,'--g', linewidth=1.0)
    plt.plot(a3max, binheights,'--b', linewidth=1.0)
    plt.plot(a1min, binheights,':r', linewidth=1.0)
    plt.plot(a2min, binheights,':g', linewidth=1.0)
    plt.plot(a3min, binheights,':b', linewidth=1.0)
    for k in range(0,np.size(data_a1,1)):
        plt.text(np.max(np.array([[a1av[k], a2av[k], a3av[k]]])), binheights[k],k+1)
    a_max = np.max(np.array([[a1max, a2max, a3max]]))
    a_min = np.min(np.array([[a1min, a2min, a3min]]))
    plt.plot((a_min, np.max(ampmult*np.mean(np.vstack((a1av, a2av, a3av)), axis=0))),(depth_av, depth_av),'b', linewidth=1.5)
    
    if orientation == 1:
        plt.text(np.max(ampmult*np.mean(np.vstack((a1av, a2av, a3av)), axis=0))*0.75, depth_av, 'Waterlevel') # = '+str(depth_av)+'m')
        plt.plot((a_min, np.max(ampmult*np.mean(np.vstack((a1av, a2av, a3av)), axis=0))),(np.max(depth), np.max(depth)),':b', linewidth=1.5)
        plt.plot((a_min, np.max(ampmult*np.mean(np.vstack((a1av, a2av, a3av)), axis=0))),(np.min(depth), np.min(depth)),':b', linewidth=1.5)
        plt.plot((a_min, np.max(ampmult*np.mean(np.vstack((a1av, a2av, a3av)), axis=0))),(np.cos(beam_angle/180*np.pi)*np.min(depth), np.cos(beam_angle/180*np.pi)*np.min(depth)),'k--', linewidth=1.5)
        plt.text(np.max(ampmult*np.mean(np.vstack((a1av, a2av, a3av)), axis=0))*0.9, np.max(depth), 'Max')
        plt.text(np.max(ampmult*np.mean(np.vstack((a1av, a2av, a3av)), axis=0))*0.9, np.min(depth), 'Min')
        plt.text(np.max(ampmult*np.mean(np.vstack((a1av, a2av, a3av)), axis=0))*0.8,  np.cos(beam_angle/180*np.pi)*np.min(depth), 'Sidelobe')
    else:
        plt.text(np.max(ampmult*np.mean(np.vstack((a1av, a2av, a3av)), axis=0))*0.5, depth_av, 'Estimated depth') # = '+str(depth_av)+'m')
        plt.plot((a_min, np.max(ampmult*np.mean(np.vstack((a1av, a2av, a3av)), axis=0))),(np.cos(beam_angle/180*np.pi)*depth_av, np.cos(beam_angle/180*np.pi)*depth_av),'k--', linewidth=1.5)
        plt.text(np.max(ampmult*np.mean(np.vstack((a1av, a2av, a3av)), axis=0))*0.8,  np.cos(beam_angle/180*np.pi)*depth_av, 'Sidelobe')
        
    # Cutoff
    plt.plot((amp_tresh_low, amp_tresh_low), (binheights[0], binheights[-1]), 'k:', linewidth=1.5)
    plt.plot((amp_tresh_high, amp_tresh_high), (binheights[0], binheights[-1]), 'k--', linewidth=1.5)
    plt.plot(ampmult*np.mean(np.vstack((a1av, a2av, a3av)), axis=0), binheights, 'k--', linewidth=1.5)
    
    # plt.figure()
    # plt.scatter(data_a1)
    
    plt.legend(('Average 1','Average 2','Average 3','Max 1','Max 2','Max 3','Min 1','Min 2','Min 3'), loc='upper right')
    
    plt.title('Bins related to signal strength')
    plt.ylabel('Depth (m)')
    plt.xlabel('A (counts)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    if savefig == True:
        plt.savefig(report_folder+'quality_check/A.png',dpi = 300)
        plt.savefig(report_folder+'quality_check/A.pdf',dpi = 600)
     
    
# Calculate velocity and angle
print('Calculating velocity and angle')
data_velocity = np.sqrt(data_v1**2+data_v2**2)
data_velocity_3D = np.sqrt(data_v1**2+data_v2**2+data_v3**2)
data_angle = np.angle(-data_v2-data_v1*1j)+np.pi
data_angle_deg = np.rad2deg(data_angle)+mag_dec
data_angle_deg = np.mod(data_angle_deg,360)
data_velocity_corr = np.sqrt(data_v1_corr**2+data_v2_corr**2)
data_velocity_corr_3D = np.sqrt(data_v1_corr**2+data_v2_corr**2+data_v3_corr**2)

data_angle_corr = np.angle(-data_v2_corr-data_v1_corr*1j)+np.pi
data_angle_deg_corr = np.rad2deg(data_angle_corr)+mag_dec
data_angle_deg_corr = np.mod(data_angle_deg_corr,360)


print('Storing raw data')


np.save(folder+'proc_ASCII/'+'pitch',pitch)
np.save(folder+'proc_ASCII/'+'roll',roll)
np.save(folder+'proc_ASCII/'+'head',head)
np.save(folder+'proc_ASCII/'+'u_corr',data_v1_corr)
np.save(folder+'proc_ASCII/'+'v_corr',data_v2_corr)
np.save(folder+'proc_ASCII/'+'w_corr',data_v3_corr)
np.save(folder+'proc_ASCII/'+'mag',data_velocity)
np.save(folder+'proc_ASCII/'+'mag_3D',data_velocity_3D)
np.save(folder+'proc_ASCII/'+'dir',data_angle_deg)
np.save(folder+'proc_ASCII/'+'mag_corr',data_velocity_corr)
np.save(folder+'proc_ASCII/'+'mag_corr_3D',data_velocity_corr_3D)
np.save(folder+'proc_ASCII/'+'dir_corr',data_angle_deg_corr)
np.save(folder+'proc_ASCII/'+'d',depth)
np.save(folder+'proc_ASCII/'+'d_av',depth_av)
np.save(folder+'proc_ASCII/'+'binheights',binheights)
np.save(folder+'proc_ASCII/'+'bindepths',bindepths)
np.save(folder+'proc_ASCII/'+'time',time)
np.save(folder+'proc_ASCII/'+'vars',[lat,long,mag_dec,instr_height])#,date_first])
np.save(folder+'proc_ASCII/'+'string_vars',[ID, date_first, date_last, instrument_type, serial_number, instrument_frequency, profile_interval, average_interval,
number_of_measurements, bin_size, number_of_bins, blanking_distance,ping_hz])#,date_first])
np.save(folder+'proc_ASCII/'+'mdata',mdata)
# np.save(folder+'proc_ASCII/'+'metadata',ght])
print(binheights)
print(bindepths)

