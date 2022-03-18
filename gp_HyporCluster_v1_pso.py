# gp_HyporheicCluster.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance
from sklearn.cluster import KMeans


class GetDataSet:
    """
    GetDataSet
        method 1: from folder import xlsx file
        method 2: control FEFLOW -> start simulation, then use ifm plug-ins get data
    """
    def __init__(self, filepath="C:\JunXiang\VSCode\Excel_py"):
        self.filepath = filepath
        self.riverNodes = pd.read_excel(self.filepath + "\\nearRiverNodes_aq1_v2.xlsx")

    def getDataLoc(self, nodes):
        loc = self.riverNodes
        loc = loc[['Node'] + ['X'] + ['Y'] + ['Z']]
        loc = pd.merge(loc,
                       nodes,
                       how='inner',
                       on=['Node'])

        return loc

    def importData(self):
        saturation = pd.read_excel(self.filepath + "\\new_gp_Saturation.xlsx")
        darcyZ = pd.read_excel(self.filepath + "\\new_gp_Darcy_z.xlsx")
        pressure = pd.read_excel(self.filepath + "\\new_gp_Pressure.xlsx")

        filter_saturation = pd.merge(self.riverNodes['Node'],
                                     saturation,
                                     how='inner',
                                     on=['Node'])

        filter_darcyZ = pd.merge(self.riverNodes['Node'],
                                 darcyZ,
                                 how='inner',
                                 on=['Node'])

        filter_pressure = pd.merge(self.riverNodes['Node'],
                                   pressure,
                                   how='inner',
                                   on=['Node'])

        data = pd.concat([self.riverNodes['Node'],
                          pd.DataFrame(filter_saturation, columns=['S']),
                          pd.DataFrame(filter_darcyZ, columns=['VZ']),
                          pd.DataFrame(filter_pressure, columns=['P'])],
                          axis=1)
        return data

    def ifmGetData(self):
        import sys
        import ifm
        sys.path.append("C:\\Program Files\\DHI\\2020\\FEFLOW 7.3\\bin64")
        doc = ifm.loadDocument("D:\\FEM_FILE\Simulation\\new_gp_WithoutPump.fem")

        saturation, darcyZ, pressure, darcyX, darcyY = [], [], [], [], []

        doc.startSimulator()
        for node in self.riverNodes['Node']:
            saturation.append(doc.getResultsFlowSaturationValue(int(node) - 1))
            darcyZ.append(doc.getResultsZVelocityValue(int(node) - 1))
            pressure.append(doc.getResultsFlowPressureValue(int(node) - 1))
            darcyX.append(doc.getResultsXVelocityValue(int(node) - 1))
            darcyY.append(doc.getResultsYVelocityValue(int(node) - 1))


        doc.stopSimulator()

        data = pd.concat([self.riverNodes['Node'],
                          pd.DataFrame(darcyZ, columns=['VZ']),
                          pd.DataFrame(saturation, columns=['S']),
                          pd.DataFrame(pressure, columns=['P']),
                          pd.DataFrame(darcyX, columns=['VX']),
                          pd.DataFrame(darcyY, columns=['VY'])],
                          axis=1)

        data.to_excel("D:\\VSCode\Excel_py\\HyporFeatures.xlsx")
        return data

class PreProcess:
    """
    1. Use dataframe type to preprocess
    2. Then convert df to ndarray for training
    """
    def __init__(self):
        pass

    def removeNoise(self, init_data):
        """ Remove saturation == 0 """
        data = init_data[init_data['S'] > 0.1]

        return data

    def convert_NDimensionData(self, data):
        """ Convert dataframe to ndarray"""
        data = np.array(data)

        return data

    def absolute(self, data):
        return abs(data)

    def norm(self, data):
        scaler = MinMaxScaler(feature_range=(0, 1))

        return scaler.fit_transform(data.values.reshape(-1, 1))

def dataset2Dplot():
    import seaborn as sns
    sns.set(style = "darkgrid")
    sns.despine(top = True, right = True)
    sns.set_theme()

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(init_data['P'],
                init_data['S'],
                label="node",
                color='blue',
                alpha=0.2)

    ax1.legend(loc="upper right")

    ax1.set(xlabel="pressure",
            ylabel="saturation",
            title="Pressure vs Saturation")
    plt.clf()

    ax2 = fig.add_subplot(111)

    ax2.scatter(init_data['P'],
                init_data['VZ'],
                label="node",
                color="blue",
                alpha=0.2)

    ax2.legend(loc="upper right")

    ax2.set(xlabel="pressure",
            ylabel="darcy flux(z-vector)",
            title="Pressure vs Darcy flux")
    plt.clf()

    ax3 = fig.add_subplot(111)

    ax3.scatter(init_data['S'],
                init_data['VZ'],
                label="node",
                color="blue",
                alpha=0.2)

    ax3.legend(loc="upper right")

    ax3.set(xlabel="saturation",
            ylabel="darcy flux(z-vector)",
            title="Saturation vs Darcy flux")

    fig.tight_layout()

def dataset3Dplot():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.scatter(process_data[:, 4],
               process_data[:, 5],
               process_data[:, 6],
               alpha=0.3)

    ax.set(xlabel=data_label[0],
           ylabel=data_label[1],
           zlabel=data_label[2])

    plt.title('Hyporheic Dataset')

def theoretical():
    # Theoretically
    df = pd.DataFrame(init_data, columns=["Node", "VZ", "P", "S"])
    # hyporheic data
    df = df[abs(df["VZ"]) >= 0.0002]
    df = df[df["S"] >= 0.6]
    df = df[df["P"] <= 600]
    df["Hypor"] = 1

    # non-hyporheic data
    dd = pd.DataFrame(init_data, columns=["Node", "VZ", "P", "S"])
    dd = pd.DataFrame(init_data, columns=["Node", "VZ", "P", "S"]).append(df)
    dd = dd.drop_duplicates(subset=['Node'], keep=False)
    dd["Hypor"] = 0

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(df["VZ"],
               df["P"],
               df["S"],
               c='red',
               alpha=0.3,
               label="group1")

    ax.scatter(dd["VZ"],
               dd["P"],
               dd["S"],
               c='blue',
               alpha=0.3,
               label="group2")

    plt.legend()
    ax.set_xlabel(data_label[0])
    ax.set_ylabel(data_label[1])
    ax.set_zlabel(data_label[2])
    ax.view_init(elev=200,azim=-45)

    # export
    output = pd.concat([dd, df], axis=0)
    output.to_excel("D:\VSCode\Excel_py\\hypor_theoretic.xlsx")

def hyporheicLocationPlot(y):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.scatter(loc['X'],
               loc['Y'],
               loc['Z'],
               c=y,
               alpha=0.1)

    ax.set(xlabel='x',
           ylabel='y',
           zlabel='z')

    plt.show()

def featuresClusterResult(y):
    color_label = ['red', 'blue']
    y = [color_label[y_i] for y_i in y]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.scatter(data['S'],
               data['P'],
               data['VZ'],
               c=y,
               alpha=0.3)

    # ax.set(xlabel=data_label[3],
    #        ylabel=data_label[4],
    #        zlabel=data_label[5],
    #        )
    # ax.set_xlabel('Saturation', fontsize=12)
    # ax.set_ylabel('Pressure', fontsize=12)
    # ax.set_zlabel('Darcy flux', fontsize=12)

    plt.title('Hyporheic Clustering')
    plt.show()

class Clustering:
    def __init__(self, process_data, data_label):
        self.process_data = process_data.copy()
        self.data_label = data_label

    def kmeans(self, weighted=None, max_iter=500):
        if weighted == None:
            weighted = [1, 10, 10, 15, 0.1, 8, 25]

        P = PreProcess()
        # Normolized datasets
        for label in self.data_label:
            self.process_data[label] = P.norm(self.process_data[label])

        self.process_data *= weighted
        train_x = P.convert_NDimensionData(self.process_data).copy()[:, 1:7]
        y = KMeans(n_clusters=2, max_iter=max_iter).fit(train_x).predict(train_x)

        return y






particle_dim = 7

class Particle:
    particle = np.array([1] * particle_dim)
    pb = np.array([1] * particle_dim)
    pb_loss = np.math.inf
    loss = np.math.inf
    v = np.array([0] * particle_dim)

    def get_fit(self):
        # conditioned weighting of features which are negative value
        self.particle = np.array([0.1 if self.particle[n] < 0.1 else round(self.particle[n], 1) for n in range(len(self.particle))])

        # conditioned weighting of Node's feature
        self.particle[0] = 1.

        result_label = C.kmeans(list(self.particle))
        df = pd.concat([process_data, pd.DataFrame(result_label, columns=['label'])], axis=1)

        df['VZ'] = P.norm(df['VZ'])
        var_VZc1 = np.var(df[df['label']==1]['VZ'])
        var_VZc2 = np.var(df[df['label']==0]['VZ'])

        df['P'] = P.norm(df['P'])
        var_Pc1 = np.var(df[df['label']==1]['P'])
        var_Pc2 = np.var(df[df['label']==0]['P'])

        self.loss = 8*min(var_VZc1, var_VZc2) + 3*min(var_Pc1, var_Pc2)

        if self.loss < self.pb_loss:
            self.pb_loss = self.loss
            self.pb = self.particle

class PSO:
    def __init__(self, w=0.5, c1=3, c2=5):
        self.w, self.c1, self.c2 = w, c1, c2
        self.particles = [Particle() for _ in range(particle_num)]
        for i in range(particle_num):
            self.particles[i].particle = np.random.rand(particle_dim)

        self.gb = np.copy(self.particles[0].particle)
        self.gb_loss = np.math.inf

    def get_all_fit(self):
        for i in range(particle_num):
            self.particles[i].get_fit()
            if self.particles[i].pb_loss < self.gb_loss:
                self.gb_loss = self.particles[i].pb_loss
                self.gb = np.copy(self.particles[i].particle)
                print(self.gb, self.gb_loss)

    def updater(self):
        r1, r2 = np.random.rand(), np.random.rand()

        for i in range(particle_num):
            self.particles[i].v = self.w * self.particles[i].v \
                                      + self.c1 * r1 * (self.particles[i].pb - self.particles[i].particle) \
                                      + self.c2 * r2 * (self.gb - self.particles[i].particle)
            self.particles[i].particle += self.particles[i].v




if __name__ == "__main__":
    G = GetDataSet()
    # init_data = G.ifmGetData()
    init_data = pd.read_excel('C:\JunXiang\VSCode\Excel_py\\HyporFeatures.xlsx')

    P = PreProcess()
    data = P.removeNoise(init_data)
    nodes = data['Node']
    loc = G.getDataLoc(nodes)
    data = pd.merge(data, loc, how='inner', on='Node')

    data_label = ['X', 'Y', 'Z', 'S', 'P', 'VZ']

    data = data[['Node'] + data_label]

    process_data = data.copy()
    process_data['VZ'] = P.absolute(process_data['VZ'])

    # Normolized datasets
    for label in data_label:
        process_data[label] = P.norm(process_data[label])

    particle_dim = len(data.columns)
    particle_num = 10
    iter_num = 50

    C = Clustering(process_data, data_label)
    PS = PSO()

    for iteration in range(50):
        PS.get_all_fit()
        PS.updater()


    # process_data *= weighted

    # Set up features weighted (learning target parameter) -> higher weighted means important (easily) to distinct hyporheic and groundwater
    weighted = [1.0, 0.1, 0.1, 0.1, 1.1, 0.9, 0.9]
    weighted = [1.0, 2, 2, 3, 1, 3, 8]

    process_data *= weighted

    train_x = P.convert_NDimensionData(process_data).copy()[:, 1:7]

    # dataset3Dplot()

    init_centroids = "k-means++"
    y = KMeans(n_clusters=2, init=init_centroids, max_iter=500).fit(train_x).predict(train_x)

    featuresClusterResult(y)

    hyporheicLocationPlot(y)
    df = pd.DataFrame([nodes.values, y, data['VZ'], abs(data['VZ'])]).T
    df.columns = ['Nodes', 'HYP', 'VZ', 'absVZ']
    df.to_excel("C:\\JunXiang\VSCode\Excel_py\\hypor.xlsx", index=False)

