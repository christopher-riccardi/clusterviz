"""
Code by:
    [+] Christopher Riccardi, PhD Student at University of Florence (Italy) https://www.bio.unifi.it/vp-175-our-research.html
        Currently Guest Researcher at Sun Lab, University of Southern California, Los Angeles (USA) https://dornsife.usc.edu/profile/fengzhu-sun/
        PhD Project Title: Computational modelling of omics data from condition-dependent datasets.
        Advisor: Marco Fondi (U Florence)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN

def make_grid(grid_size, spacing):
    ## Create a square, fine mesh of points. This way we can operate
    ## on the grid and update values instead of adding random points
    ## in a not clearly defined space.
    # using numpy's built-in methods to create an ordered grid
    x = np.linspace(0, spacing, grid_size)
    y = np.linspace(0, spacing, grid_size)
    xx, yy = np.meshgrid(x, y)
    # collapse into one dimension
    x_coords = xx.flatten()
    y_coords = yy.flatten()
    return x_coords, y_coords

def add_Gaussian_noise(mean, sd, signal):
    ## P(x) = 1 / (σ√(2π)) * e^(-(x-μ)^2 / (2σ^2))
    noise = np.random.normal(mean, sd, size=signal.shape)
    return signal + noise

def get_closest_point(grid, x, y):
    ## Look for the closest point in a grid where the 'z' coordinate is 0
    ## 'z' does not actually add a third dimension here, it's a switch for
    ## informing the matrix on which values are writable and which are not
    distances = list(np.sqrt((grid[(grid['z']==0)]['x'] - x)**2 + (grid[(grid['z']==0)]['y'] - y)**2))
    i = distances.index(min(distances))
    return grid[grid['z']==0].iloc[i]

def add_padding(grid, xs, ys):
    ## a simple and clever way to avoid collisions, simply add some padding 
    ## around the shapes in every direction
    padding = 25
    for i in range(len(xs)): ## xs, or ys it's the same. They have the same length
        grid.loc[ (grid['x']<xs[i]+padding) & (grid['x']>xs[i]-padding) & (grid['y']<ys[i]+padding) & (grid['y']>ys[i]-padding), 'z'] = 1
    return grid

if __name__=='__main__':
    ## we're hard-coding the filenames for the purpose of publication
    Pseudoalteromonas = pd.read_csv('Pseudoalteromonas_dataframe_antismash.txt', sep='\t', index_col=[0])
    Pseudomonas = pd.read_csv('Pseudomonas_dataframe_antismash_updated.txt', sep='\t', index_col=[0])
    Psychrobacter = pd.read_csv('Psychrobacter_dataframe_antismash.txt', sep='\t', index_col=[0])
    Shewanella = pd.read_csv('Shewanella_dataframe_antismash.txt', sep='\t', index_col=[0])
    
    ## add taxonomy, here we group by genera
    Pseudoalteromonas['Genus']='Pseudoalteromonas'
    Pseudomonas['Genus']='Pseudomonas'
    Psychrobacter['Genus']='Psychrobacter'
    Shewanella['Genus']='Shewanella'
    
    our_Pseudoalteromonas = np.loadtxt('Novel_Pseudoalteromonas', dtype=str)
    our_Pseudomonas = np.loadtxt('Novel_Pseudomonas', dtype=str)
    our_Psychrobacter = np.loadtxt('Novel_Psychrobacter', dtype=str)
    our_Shewanella = np.loadtxt('Novel_Shewanella', dtype=str)
    
    ## concatenate data, and perform some sanity check operations
    df = pd.concat([Pseudoalteromonas, Pseudomonas, Psychrobacter, Shewanella])
    df = df.fillna(0.0)
    
    ## keep track of taxonomy but leave the matrix numerical before feeding it to the algo
    genera = df['Genus']
    del df['Genus']

    ## I'm trying DBSCAN on simple presence/absence matrix
    ## essentially DBSCAN identifies dense regions in a dataset and assigns each data point to a cluster 
    ## or labels it as noise, based on the density of points around it.
    dbscan = DBSCAN(eps=1.8, min_samples=2)
    cluster_labels = dbscan.fit_predict(df)
    clusters = np.unique(cluster_labels) # unique clusters
    df['Genus'] = genera
    df['Label'] = cluster_labels
    df['Ours'] = 0
    
    ## Ours are the genomes we sampled and sequenced
    df.loc[our_Pseudoalteromonas, 'Ours'] = 1
    df.loc[our_Pseudomonas, 'Ours'] = 1
    df.loc[our_Psychrobacter, 'Ours'] = 1
    df.loc[our_Shewanella, 'Ours'] = 1
    df = df.sort_values(by=['Label'])
    
    ## initialize x and y values
    df['x'] = np.nan
    df['y'] = np.nan
    
    ## initialize the grid
    grid_params=[60, 300, 20]
    x_y = make_grid(grid_params[0], grid_params[1])
    grid = pd.DataFrame()
    grid['x'] = x_y[0]
    grid['y'] = x_y[1]
    grid['z'] = 0

    ## -===== Ready to start =====- ##
    for _, c in enumerate(df['Label'].unique()):
        print(f'Main cluster: {c}')
        sub = df[df['Label']==c]
        pos = grid[(grid['z']==0)].sample(1)
        
        ## each cluster has an initial centroid c_x, c_y
        c_x, c_y = float(pos['x']), float(pos['y'])
        print(f'Pos for cluster {c}: {c_x},{c_y}')
        c_xs, c_ys = [], []
        gs = sub['Genus'].value_counts() # collect genera in the cluster
        for i, g in enumerate(gs):
            xs, ys = [], []
            x, y = c_x, c_y
            ## around the centroid arise mini centroids
            print(f'Subcluster: {g}')
            for subcluster in range(g):
                pos = get_closest_point(grid, x, y)
                x, y = float(pos['x']), float(pos['y'])
                print(f'New x, y: {x},{y}')
                grid.loc[(grid['x']==x) & (grid['y']==y), 'z'] = 1 ## Always update grid
                xs.append(x)
                ys.append(y)
                x, y = np.mean(xs), np.mean(ys) # recursively update the centroids and nearest point
            
            ## done with subcluster, add to dataframe
            df.loc[(df['Label']==c) & (df['Genus']==gs.index[i]), 'x'] = xs
            df.loc[(df['Label']==c) & (df['Genus']==gs.index[i]), 'y'] = ys
            
            ## add noise to break grid-like structure
            df.loc[(df['Label']==c) & (df['Genus']==gs.index[i]), 'x'] = add_Gaussian_noise(0, 4, np.array(xs))
            df.loc[(df['Label']==c) & (df['Genus']==gs.index[i]), 'y'] = add_Gaussian_noise(0, 4, np.array(ys))
            
            ## once we're done with these subclusters, after having added the x,y to the dataframe
            ## fill the cluster's general xs and ys. These are needed to add padding before moving on
            ## to the next cluster
            c_xs += xs
            c_ys += ys
        ## Done with this cluster. Provide some padding
        add_padding(grid, c_xs, c_ys)

    ## save results to plot and save data frame
    plt.axis('equal')
    scat = sns.scatterplot(x='x', y='y', data=df, style='Ours', hue='Genus')
    fig = scat.get_figure()
    fig.savefig('cluster_clouds.svg')
    df.to_csv('clusters_dataframe.tsv', sep='\t')
