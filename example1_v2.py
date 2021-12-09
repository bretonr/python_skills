## v2 (Jedi version)
## This version is very efficient in terms of execution speed.

## This example creates a list of sources and then picks a subset to be part of
## catalogue 1 and another subset to be part of catalogue 2. The location of
## the sources is given a small random offset in each catalogue to mimic the
## effect of images of a same field giving different results for two different
## observations (due to wavelength, weather, etc). The two catalogues are then
## cross-matched.


## Module import
import numpy as np
import os
import matplotlib.pyplot as plt
from icecream import ic
ic.enable()

## Profiler
import yappi
RUN_YAPPI = True

## Setting random number generator
rng = np.random.default_rng(12345)



#------------------------------------------------------------------------------
## Create catalogue of mock sources

## Multiplier factor so scale processing
multiplier = 1

## Number of pixels in image
nx = 1000 * multiplier
ny = 1000 * multiplier

## Number of actual sources
nsources = 16 * multiplier
## Number of sources in catalogue 1
nsources1 = 10 * multiplier
## Number of sources in catalogue 2
nsources2 = 7 * multiplier

## Generating random positions for the actual sources
x = rng.uniform(0, nx, size=nsources-1)
y = rng.uniform(0, ny, size=nsources-1)
x = np.r_[x, x[3]-3]
y = np.r_[y, y[3]+5]

## Making a random selection (i.e. fetching indices) of the actual sources to be in catalogue 1
inds1 = rng.choice(nsources, size=nsources1, replace=False)
## Fetching coordinates and adding small offsets
x1 = x[inds1] + rng.normal(loc=0., scale=2., size=nsources1)
y1 = y[inds1] + rng.normal(loc=0., scale=2., size=nsources1)

## Making a random selection (i.e. fetching indices) of the actual sources to be in catalogue 2
inds2 = rng.choice(nsources, size=nsources2, replace=False)
## Fetching coordinates and adding small offsets
x2 = x[inds2] + rng.normal(loc=0., scale=2., size=nsources2)
y2 = y[inds2] + rng.normal(loc=0., scale=2., size=nsources2)

## Retrieve the list of matching sources
match_actual = np.intersect1d(inds1, inds2)



#------------------------------------------------------------------------------
## Perform the cross matching by calculating Euclidian distance

## Profiler
if RUN_YAPPI:
    yappi.set_clock_type("cpu") # Use set_clock_type("wall") for wall time
    yappi.start()


## Calcualte the mutual distances
dist = np.sqrt( (x1[:,None]-x2[None,:])**2 + (y1[:,None]-y2[None,:])**2 )

## Best match one-way
oneway1 = dist.argmin(axis=1)
oneway2 = dist.argmin(axis=0)

## Restrict matches to be bidirectional
match_arr = np.zeros_like(dist, dtype=int)
match_arr[np.arange(nsources1), oneway1] += 1
match_arr[oneway2, np.arange(nsources2)] += 1

## NOTE: only v1 yields the correct results
(match1_v1, match2_v1) = np.nonzero( (match_arr == 2) * (dist < 20) )
(match1_v2, match2_v2) = np.nonzero( (match_arr == 2) )
(match1_v3, match2_v3) = np.nonzero( (dist < 20) )


## Profiler
if RUN_YAPPI:
    yappi.get_func_stats().print_all()
    yappi.get_thread_stats().print_all()



#------------------------------------------------------------------------------
## Plot the results
fig1, ax1 = plt.subplots()
s = 100
ax1.scatter(x, y, s=s, c='k', marker='.')
ax1.scatter(x1, y1, s=s, c='k', marker='+')
ax1.scatter(x2, y2, s=s, c='k', marker='x')
ax1.scatter(x[match_actual], y[match_actual], s=s*2, ec='b', fc='none', marker='o')
ax1.scatter(x1[match1_v1], y1[match1_v1], s=s*1.5, ec='r', marker='+')
ax1.scatter(x2[match2_v1], y2[match2_v1], s=s*1.5, ec='r', marker='x')
ax1.set_title('Two-way nearest + Distance threshold')

fig2, ax2 = plt.subplots()
s = 100
ax2.scatter(x, y, s=s, c='k', marker='.')
ax2.scatter(x1, y1, s=s, c='k', marker='+')
ax2.scatter(x2, y2, s=s, c='k', marker='x')
ax2.scatter(x[match_actual], y[match_actual], s=s*2, ec='b', fc='none', marker='o')
ax2.scatter(x1[match1_v2], y1[match1_v2], s=s*1.5, ec='r', marker='+')
ax2.scatter(x2[match2_v2], y2[match2_v2], s=s*1.5, ec='r', marker='x')
ax2.set_title('Two-way nearest')

fig3, ax3 = plt.subplots()
s = 100
ax3.scatter(x, y, s=s, c='k', marker='.')
ax3.scatter(x1, y1, s=s, c='k', marker='+')
ax3.scatter(x2, y2, s=s, c='k', marker='x')
ax3.scatter(x[match_actual], y[match_actual], s=s*2, ec='b', fc='none', marker='o')
ax3.scatter(x1[match1_v3], y1[match1_v3], s=s*1.5, ec='r', marker='+')
ax3.scatter(x2[match2_v3], y2[match2_v3], s=s*1.5, ec='r', marker='x')
ax3.set_title('Distance threshold')


