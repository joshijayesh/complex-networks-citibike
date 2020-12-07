# Just testing if we can plot the map from python... This aint straightforward :(

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from pylab import rcParams
rcParams['figure.figsize'] = 50,50


y0, x0, y1, x1 = 40.67095, -74.045751, 40.86541, -73.85990
cx, cy = (x0 + x1) / 2, (y0 + y1) / 2


'''
m = Basemap(
    projection='mill',
    resolution='h',
    ellps='WGS84',
    llcrnrlon=x0, llcrnrlat=y0,
    urcrnrlon=x1, urcrnrlat=y1,
    #lat_0=cx, lon_0 =cy,
    area_thresh=0.1,
    epsg=2263
)

# m.drawcoastlines()
#m.fillcontinents(color='green')
#m.drawstates()
#m.drawcounties(linewidth=3)
#m.drawrivers(linewidth=3)
#m.drawmapboundary(fill_color='blue')
m.arcgisimage(service='ESRI_StreetMap_World_2D', xpixels=2000, verbose=True)
plt.show()
'''

request = cimgt.OSM()
ax = plt.axes(projection=request.crs)
ax.set_extent([x0, x1, y0, y1])

ax.add_image(request, 8) # interpolation='bilinear')

plt.show()

