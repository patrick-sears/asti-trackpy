#!/usr/bin/env python3

import sys
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('figure', figsize=(10,6))
mpl.rc('image', cmap='gray')

import math

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

import pims
import trackpy as tp



if len(sys.argv) < 2:
  print( "Need config file as argument." )
  exit( 1 )

fname_config = sys.argv[1]

f_config = open( fname_config, 'r' )


############################################ **config
for l in f_config:
  if not l.startswith('!'):  continue
  l = l.strip()
  ll = l.split(' ')
  key = ll[0]
  if key == '!imdir':
    if len(ll) == 2:    imdir = ll[1]
    else:               imdir = f_config.readline().strip()
  elif key == '!um_per_pix':
    um_per_pix = float(ll[1])
  elif key == '!ms_per_frame':
    ms_per_frame = float(ll[1])
  elif key == '!fname_out1':
    fname_out1 = ll[1]
  elif key == '!fname_out2':
    fname_out2 = ll[1]
  elif key == '!track_ims_dir':
    track_ims_dir = ll[1]
  elif key == '!track_ims_vidname':
    track_ims_vidname = ll[1]
  #
  elif key == '!imfile_basename':
    imfile_basename = ll[1]
  elif key == '!imfile_suffix':
    imfile_suffix = ll[1]
  elif key == '!imfile_i_first':
    imfile_i_first = int(ll[1])
  elif key == '!imfile_i_last':
    imfile_i_last = int(ll[1])
  elif key == '!imfile_digs':
    imfile_digs = int(ll[1])
  #
  elif key == '!im_w':
    im_w = int(ll[1])
  elif key == '!im_h':
    im_h = int(ll[1])
  #
  elif key == '!min_frames_in_a_track':
    min_frames_in_a_track = int(ll[1])
  #
  elif key == '!particle_size':
    particle_size = float(ll[1])
  elif key == '!search_range':
    search_range = float(ll[1])
  elif key == '!tpl_minmass':
    tpl_minmass = int(ll[1])
  elif key == '!min_track_len_px':
    min_track_len_px = float(ll[1])
  #
  else:
    print("Unrecognized key.")
    print("  key = ", key)
    # exit(1)
    # Not exciting because the same config file
    # will be used for "sup" programs.
############################################ **config





im_n = imfile_i_last - imfile_i_first + 1


# frames = pims.ImageSequence('../dataset01/xp-0708v10a-v016-part/*.jpg', as_grey=True)
# imfiles = imdir + "/*.jpg"
imfiles = []
for i in range(im_n):
  j = imfile_i_first + i
  locname = imdir+'/'+imfile_basename
  locname += '{0:04d}'.format(j)
  locname += imfile_suffix
  imfiles.append( locname )
  #
  if not os.path.isfile( imfiles[i] ):
    print("Error:  Missing file.")
    print("  : ", imfiles[i] )
    exit(1)


# frames = pims.ImageSequence(imfiles, as_grey=True)
frames = pims.ImageSequence(imfiles)



print("Finding particles in all images.")
print("  Applying trackpy batch....")
par1 = tp.batch(frames[:], particle_size, minmass=tpl_minmass )
print("Done finding particles in all images.")
print()


############################################
# The dataframe f now contains all particles found in every image.
# Each item in f knows its x y coordinates.
#
# For more info on tracky batch, see
#  http://soft-matter.github.io/trackpy/v0.3.0/generated/trackpy.batch.html
# Return value:  DataFrame
#  y
#  x
#  mass:  total integrated brightness of the blob
#  size:  radius of gyration of its Gaussian-like profile
#  ecc:   eccentricity (0 is circular)
#  signal
#  raw_mass
#  ep
#  frame:  The frame #?
############################################

print("--------------------")
print("Dataframe f columns:")
# print(df.columns.tolist())
# print(f.dtypes)
print(par1.dtypes)
print("--------------------")



############################################
print("Remove particles that are too close to the edge of the image.")
print("  Number initially:  ", len(par1))

particle_x_min = 2 * particle_size
particle_y_min = 2 * particle_size

particle_x_max = im_w - 2 * particle_size
particle_y_max = im_h - 2 * particle_size

par1 = par1[(par1.x > particle_x_min) & (par1.x < particle_x_max) & (par1.y > particle_y_min) & (par1.y < particle_y_max) ]
print("  Number after:      ", len(par1))
############################################




#################################
print("Link particles to find tracks...")
tra1 = tp.link_df(par1, search_range, memory=0)
tra1_nunique = tra1['particle'].nunique()
print("Done linking.")
print("  Number of tracks found:  ", tra1_nunique)
#################################


print()



#################################
print("Apply min_frames_in_track.")
tra2 = tp.filter_stubs(tra1, min_frames_in_a_track)

tra2_nunique = tra2['particle'].nunique()
print('  Before:', tra1['particle'].nunique())
print('  After:', tra2_nunique)
print("Done")
print()
#################################


### print("Saving dataframe to csv.")
### tra2.to_csv('z1-tracked_particles.csv', index=False)



### print( "--------------------------------" )
### print( "tra1.head:" )
### print( tra1.head() )
### 
### print( "--------------------------------" )
### print( "tra2.head:" )
### print( tra2.head() )
### 
### print( "--------------------------------" )











#################################
# Get the data from the track DataFrame into 
# lists.  These are the tp-filtered tracks.
# They aren't p-filtered yet.

tra2_i_frame = []
tra2_x_pos = []
tra2_y_pos = []
tra2_i_track = []



print( "--------------------------------" )
############################################
# Convert dataframe rows to lists.

# pit_:  The column index for that value in the dataframe.
pit_frame = 0
pit_x = 2
pit_y = 1
pit_particle = 10

# The dataframe itertuples() functions iterates
# through each row, each one formated as a tuple.
i=0
for urow in tra2.itertuples():
  tra2_i_frame.append( urow[pit_frame] )
  tra2_x_pos.append( urow[pit_x] )
  tra2_y_pos.append( urow[pit_y] )
  tra2_i_track.append( urow[pit_particle] )
  i += 1

n_tra2 = i

### f90 = open('z2-tra2_itertuples.csv', 'w')
### line = 'frame,x_pos,y_pos,track'
### f90.write(line+'\n')
### for i in range(n_tra2):
###   line = str(tra2_i_frame[i])
###   line += ','+str(tra2_x_pos[i])
###   line += ','+str(tra2_y_pos[i])
###   line += ','+str(tra2_i_track[i])
###   f90.write(line+'\n')
############################################
print( "--------------------------------" )



###  #################################
###  # This saves all the tracks before p-filtering.
###  ofile1 = open(fname_out1, 'w')
###  for i in range( n_tra2 ):
###    oline = str(tra2_i_track[i]) + '\t'
###    oline += str(tra2_i_frame[i]) + '\t'
###    oline += str(tra2_x_pos[i]) + '\t'
###    oline += str(tra2_y_pos[i]) + '\n'
###    ofile1.write( oline )
###  
###  ofile1.close()





print()


# -------------------------------------------------------
# -------------------------------------------------------
# -------------------------------------------------------
# -------------------------------------------------------
# -------------------------------------------------------
# -------------------------------------------------------

############################################
############################################
############################################
# c_track:
################################
#  t_frame   frame index
#  x         pixels
#  y         pixels invy
################################
class c_track():
  def __init__(self):
    self.track_id = -1
    self.t_frame = []
    self.x = []
    self.y = []
    self.n = 0
  def track_len2_1(self):
    if self.n < 2:  return 0.0
    return (self.x[0]-self.x[self.n-1])**2 + (self.y[0]-self.y[self.n-1])**2
  def speed1_um_per_s(self):
    if self.n < 2: return 0.0
    sp = math.sqrt(self.track_len2_1())
    sp *= um_per_pix / (self.n-1) / ms_per_frame * 1000.0
    return sp
############################################
############################################
############################################






############################################
# Convert lists of data into list of c_track objects
# New system for creating l_tracks, 2018-08-26.

l_track = []
for j in range(tra2_nunique):
  l_track.append( c_track() )

n_found = 0
for i in range( n_tra2 ):
  i_track = tra2_i_track[i]
  i_frame = tra2_i_frame[i]
  x_pos = tra2_x_pos[i]
  y_pos = tra2_y_pos[i]
  #
  found = False
  for j in range(tra2_nunique):
    if l_track[j].track_id == i_track:
      found = True
      break
  if not found:
    j = n_found
    n_found += 1
    # print("Not found.  j = ", j)
  # else:
    # print("Found.      j = ", j)
  #
  l_track[j].track_id = i_track
  l_track[j].t_frame.append( i_frame )
  l_track[j].x.append( x_pos )
  l_track[j].y.append( y_pos )
  l_track[j].n += 1
############################################




print("Number of l_track objects:  ", len(l_track))







############################################
# Remove tracks that have not moved more than 10 px
min_track_len_px2 = min_track_len_px**2

l_track2 = []

tracks_to_remove = []

n_tracks_initial = len(l_track)
n_final = 0
mi = 0
print("Remove tracks that are too short.")
######################
for m in l_track:
  if( m.track_len2_1() >= min_track_len_px2 ):
    l_track2.append( m )
    n_final += 1
  else:
    tracks_to_remove.append( int(m.track_id) )
  mi += 1
######################
print('  Intial tracks:  ', n_tracks_initial)
print('  Final tracks:   ', n_final)



### print( "--------------------------------" )
### print( "Before gone:" )
### print( "tra2.head:" )
### print( tra2.head() )


############################################
# The actual removal of bad tracks is done here.
for gone in tracks_to_remove:
  #print( "Remove:  ", gone )
  tra2 = tra2[ tra2.particle != gone]
############################################



### print( "--------------------------------" )
### print( "After gone:" )
### print( "tra2.head:" )
### print( tra2.head() )
### 
### print( "--------------------------------" )



#################################
# This saves all the tracks after p-filtering.
ofile1 = open(fname_out1, 'w')
oline = "n_track "+str(n_final)
ofile1.write(oline+'\n')
oline = "track_sizes"
for i in range( n_final ):
  oline += ' '+str(l_track2[i].n)
ofile1.write(oline+'\n')


for i in range( n_final ):
  ofile1.write( '\n' )
  oline = '! track '+str(i)
  ofile1.write(oline+'\n')
  oline = 'track_size '+str(l_track2[i].n)
  ofile1.write(oline+'\n')
  oline  = 'track_id '+str( int(l_track2[i].track_id) )
  ofile1.write(oline+'\n')
  #
  ofile1.write('---\n')
  ofile1.write('i_frame x y\n')
  #
  for j in range( l_track2[i].n ):
    oline = str( l_track2[i].t_frame[j] ) + '\t'
    oline += str( l_track2[i].x[j] ) + '\t'
    oline += str( l_track2[i].y[j] ) + '\n'
    ofile1.write( oline )

ofile1.write( '\n' )
ofile1.close()






### ################################# ---)
### fou2 = open(fname_out2, 'w')
### fou2.write( "track_id\tspeed(um/s)\n" )
### 
### for m in l_track2:
###   print( "->", m.track_id )
###   sp = m.speed1_um_per_s()
###   print( "   speed =", sp, "um/s" )
###   fou2.write( str(int(m.track_id))+"\t"+str(sp)+"\n" )
### 
### fou2.close()
### ################################# ---)





############################################
# The plt.figure() call only works if the X server 
# is running.  Not via ssh.
if 'DISPLAY' not in os.environ:
  print("Exciting because no display server found.")
  exit(1)


plt.figure()
ax1 = plt.gca()  # gca = get current axes

tp.plot_traj(tra2, ax=ax1)

plt.xlim( 0, im_w )
plt.ylim( im_h, 0 )

# ax1.axis('equal')  # didn't work with setting xlim and ylim.
plt.gca().set_aspect('equal')



# plt.show()
oname = track_ims_dir + "/" + track_ims_vidname + ".png"
print( "oname = [" + oname + "]" )

plt.savefig( oname, bbox_inches='tight' )


print( "end" )





