# code for computing measures of spatial autocorrelation of random generated points
# for different grids and sampling rates, specifically, do othe following

# For i in I iterations (e.g. 1000)
#  Create a random distribution of points
#  Create spatial weights matrix
#  For $\rho$ in `[0, 0.2, 0.4, 0.8, 1]`
#   Use spatial autoregressive model $Y = (I - \rho W)^{-1} \epsilon$ to generate autocorrelated variable for each point
#   Loop over each sample size `["3%","5%","10%","20%","50%","100%"]`
#    and then over each grid size `["6x6","8x8","10x10","12x12","15x15"]`
#    Aggregating the sampled points to each grid cell.
#    Computing global spatial autocorrelation stats
#    outputting data to a list
#  save to a csv file 


import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
import shapely
import pysal
import csv
import time

# constat data gen variables
total_points = 10000
knn_for_W = 30
iterations = 1

# load in grid data (was generated in QGIS by hand)
grid_6 = gpd.read_file("grids/grid_6x6.geojson")
grid_8 = gpd.read_file("grids/grid_8x8.geojson")
grid_10 = gpd.read_file("grids/grid_10x10.geojson")
grid_12 = gpd.read_file("grids/grid_12x12.geojson")
grid_15 = gpd.read_file("grids/grid_15x15.geojson")

# setup lists for looping samples and grids
grids = [grid_6, grid_8, grid_10, grid_12, grid_15]
#grids = [grid_6]
samples = ["sample_03","sample_05","sample_10","sample_20","sample_50","sample_100"]
#samples = ["sample_100"]
sample_names = ["3%","5%","10%","20%","50%","100%"]
#sample_names = ["100%"]
grid_names = ["6x6","8x8","10x10","12x12","15x15"]
#grid_names = ["6x6"]

# timing
start_time = time.time()

# start iterating over iterations
i = 0
while i < iterations:
    
    try:
        
        # storing outputs in a simple python list
        outputs = []

        print(i, start_time - time.time())

        # generate random location of points
        xy = 1.2 * np.random.rand(total_points,2)
        xydf = pd.DataFrame(data=xy)
        xydf.columns = ['x', 'y']

        # standard normal errors for each point
        errors_ind = np.random.normal(0, 1, total_points)
        xydf["errors"] = errors_ind

        # an identiy matrix needed for generating simulated values at each point
        I = np.identity(total_points)

        # weights matrix for k nearest
        kd = pysal.lib.cg.kdtree.KDTree(xy)
        W = pysal.lib.weights.KNN(kd, knn_for_W)
        W.transform = 'r' # row normalizing

        # extract the sparse weights matrix as a full np array for matrix multiplication
        W = (W.sparse)
        W = (W.toarray())

        for rho in [0, 0.2, 0.4, 0.8, 0.999]:

            # spatial autoregression
            # $Y = (I - \rho W)^{-1} \epsilon$
            Y = np.matrix(I - rho * W).I.dot(errors_ind)

            # append these Y values to the point data frame
            xydf["sim"] = np.transpose(Y)

            # use binomial distribution to select whether point is sampled
            xydf["sample_03"] = np.random.binomial(1, 0.03, size=total_points)
            xydf["sample_05"] = np.random.binomial(1, 0.05, size=total_points)
            xydf["sample_10"] = np.random.binomial(1, 0.10, size=total_points)
            xydf["sample_20"] = np.random.binomial(1, 0.20, size=total_points)
            xydf["sample_50"] = np.random.binomial(1, 0.50, size=total_points)
            xydf["sample_100"] = np.random.binomial(1, 1, size=total_points) 

            s = 0
            for sample in samples:

                # subset data by each sample
                xydf_s = xydf.loc[(xydf[sample] == 1)]

                # set up a geodataframe for this, to allow for future spatial join
                geometry = [shapely.geometry.Point(xy) for xy in zip(xydf_s.x, xydf_s.y)]
                gdf = gpd.GeoDataFrame(xydf_s,  geometry=geometry)

                g = 0
                for grid in grids:

                    print(i, rho, sample_names[s], grid_names[g], start_time - time.time())

                    # spatial join the grid IDs to the point data
                    xy_with_grid = gpd.sjoin(gdf, grid, how="inner", op='intersects')

                    # generate means and proportions in each cell of the grid
                    grid_desc = xy_with_grid.groupby(['id']).agg({'errors': "count",'sim': "mean"})

                    # update some of the column names
                    grid_desc["mean"] = grid_desc["sim"]
                    del grid_desc['sim'], grid_desc['errors']

                    # join back to grid boundaries
                    grid_join = grid.merge(grid_desc, on='id')

                    # compute spatial weights matrix
                    Wg = pysal.lib.weights.Queen.from_dataframe(grid_join)
                    Wg.transform = 'r' # row normalizing

                    mi = pysal.explore.esda.Moran(np.array(grid_join["mean"]), Wg, two_tailed=False)

                    #YVar='mean'
                    #XVars=['id']
                    grid_join["var"] = np.random.normal(0, 1, len(grid_join))
                    Ym=grid_join['mean'].values.reshape((len(grid_join),1))
                    Xm=grid_join[['var']].values
                    mlag = pysal.model.spreg.ml_lag.ML_Lag(Ym,Xm,w=Wg,name_y='mean', name_x=['var'] )

                    # output the values
                    output = [i, rho, sample_names[s], grid_names[g], round(mi.I, 3), round(mi.p_norm, 3), round(mlag.rho,3),  round(mlag.z_stat[2][1],3)]
                    outputs.append(output)

                    g += 1

                s += 1

        with open("outputs/" + str(i) + ".csv", "w") as csvfile:
            writer = csv.writer(csvfile)
            for row in outputs:
                writer.writerow(row)
    
        i += 1
    
    except:
        
        None