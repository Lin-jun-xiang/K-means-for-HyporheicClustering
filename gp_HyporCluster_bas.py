# gp_HyporheicCluster.py
from matplotlib import markers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import skew
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
    color_label = ['red', 'blue']
    y = [color_label[y_i] for y_i in y]

    plt.ion()
    for i in range(1, 10):
        plt.clf()
        plt.scatter(data[data['Slice']==i]['X'],
                    data[data['Slice']==i]['Y'],
                    c=y[len(data[data['Slice']==i]['X'])*(i-1):len(data[data['Slice']==i]['X'])*i],
                    alpha=0.2)
        plt.show()
        plt.pause(0.5)

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
        self.gb_loss = np.math.inf

    def kmeans(self, weighted=None, max_iter=400):
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

    def sign(self, a):
        if a > 0:
            return 1
        elif a < 0:
            return -1
        else:
            return 0

    def lossFunc(self, weighted):
        """
        Evolution the clustering results.

        Here has two index to judge the clustering results is good/bad.

            1. Distance between each groups: The higher distance is good.
            2. Distance between each data of each groups: The lower distance is good.

        Combine the two index, then:
            -> loss = data_dist / groups_dist (Davies bouldin score)
            -> fittness = groups_dist / data_dist
            * choose one to be the optimizing function.
        """
        result_label = self.kmeans(weighted=weighted)
        df = pd.concat([self.process_data, pd.DataFrame(result_label, columns=['label'])], axis=1)
        df['VZ'] = P.norm(df['VZ'])

        var_VZc1 = np.var(df[df['label']==1]['VZ'])
        var_VZc2 = np.var(df[df['label']==0]['VZ'])

        df['P'] = P.norm(df['P'])
        var_Pc1 = np.var(df[df['label']==1]['P'])
        var_Pc2 = np.var(df[df['label']==0]['P'])

        skewness_Pc1 = skew(df[df['label']==0]['P'])
        skewness_Pc2 = skew(df[df['label']==1]['P'])

        # from sklearn import metrics
        # loss = metrics.silhouette_score(self.process_data, result_label, metric='euclidean')
        loss = min(var_VZc1, var_VZc2) + 0.05*(1 / (1 + max(skewness_Pc1, skewness_Pc2)))

        if loss < self.gb_loss:
            self.gb_loss = loss
            # print(skew(df[df['label']==0]['P']), skew(df[df['label']==1]['P']))
            print('VZ var=', round(min(var_VZc1, var_VZc2),4), \
                  'skewnss=', round(0.1*(1 / (1 + max(skewness_Pc1, skewness_Pc2))),4), \
                  'loss=', loss, 'w=', weighted)

        return loss

    def bas(self,
            target_parameters=None,
            eta=0.95,
            iter=900,
            step=0.5,
            d0=1):

        """Beetle Antennae Search Algorithm

        -> optimizing the target parameters(feature weight), the first feature (Node) is not our target!

        -> funtion parameters as following:
           1. target_parameters: the features weighted (learning target parameter) -> higher weighted means important (easily) to distinct hyporheic and groundwater
           2. dir: random all target parameters direction
        """
        if target_parameters == None:
            target_parameters = [.1 for _ in range(len(self.process_data.columns))]

        for i in range(iter):
            dir = np.random.rand(len(target_parameters))

            # conditioned weighting of features which are negative value
            target_parameters = [0.1 if target_parameters[n] < 0.1 else round(target_parameters[n], 5) for n in range(len(target_parameters))]

            # conditioned weighting of Node's feature
            target_parameters[0] = 1.

            # Normalize dir
            norm = np.sqrt(sum(d**2 for d in dir))
            dir /= norm

            xl = list(np.round(target_parameters + d0*dir/2, 1))
            xr = [0.1 if target_parameters[n] - d0*dir[n]/2 < 0.1 else round(target_parameters[n] - d0*dir[n]/2, 5) for n in range(len(target_parameters))]

            xl[0], xr[0] = 1., 1.

            fl = self.lossFunc(xl)
            fr = self.lossFunc(xr)

            target_parameters = target_parameters - step*dir*self.sign(fl-fr)
            step = eta * (abs(fl) + abs(fr)) * 5

            # print('epoch=', i+1, 'step=', round(step, 5))
        print('BAS finished!')
        return target_parameters

def plotly_():
    import plotly.graph_objects as go

    color_label = ['red', 'blue']
    ycolor = [color_label[y_i] for y_i in y]

    fig = go.Figure(data=[go.Scatter3d(x=data['S'],
                                       y=data['P'],
                                       z=data['VZ'],
                                       mode='markers',
                                       marker=dict(
                                           size=2,
                                           color=ycolor,
                                           opacity=0.3
                                       ))])
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    fig.show()

if __name__ == "__main__":
    G = GetDataSet()
    # init_data = G.ifmGetData()
    init_data = pd.read_excel('C:\JunXiang\VSCode\Excel_py\\HyporFeatures_Sensity_LowerK_v2.xlsx')

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

    C = Clustering(process_data, data_label)
    weighted = C.bas()


    # Set up features weighted (learning target parameter) -> higher weighted means important (easily) to distinct hyporheic and groundwater
    weighted = [1.0, 0.7, 0.3, 1.1, 0.5, 0.7, 1.0] # Higher K
    weighted = [1.0, 0.2, 0.2, 0.4, 0.6, 0.3, 0.9] # Lower K
    weighted = [1.0, 0.1, 0.1, 0.43, 0.49524, 0.23401, 0.8] # Lower K v2
    weighted = [1.0, 0.4, 0.2, 0.6, 0.3, 0.4, 0.9] # Higher HBC
    weighted = [1.0, 0.31944, 0.15177, 0.9354, 0.17141, 0.62148, 2.27677] # Lower HBC
    weighted = [1, 1, 1, 1, 1, 5, 1]
    weighted = [1, 1, 1, 1, 1, 1, 5]
    weighted = [1.0, 0.12, 0.1, 0.53, 0.1, 0.31, 0.98] # Basic case

    process_data *= weighted

    train_x = P.convert_NDimensionData(process_data).copy()[:, 1:7]

    # dataset3Dplot()
    init_centroids = "k-means++"
    y = KMeans(n_clusters=2, init=init_centroids, max_iter=400).fit(train_x).predict(train_x)

    featuresClusterResult(y)

    # hyporheicLocationPlot(y)
    df = pd.DataFrame([nodes.values, y, data['VZ'], abs(data['VZ']), data['P']]).T
    df.columns = ['Nodes', 'HYP', 'VZ', 'absVZ', 'Pressure']
    df.to_excel("C:\\JunXiang\VSCode\Excel_py\\hypor_Sensitivity_LowerK_v2.xlsx", index=False)






    # show the feature range of HZ
    d = pd.concat([data, pd.DataFrame(y, columns=['label'])], axis=1)
    import seaborn as sns
    max(abs(d[d['label']==0]['VZ']))
    np.mean(abs(d[d['label']==0]['VZ']))
    sns.distplot(abs(d[d['label']==0]['VZ']), color='dodgerblue', label='HZ')
    sns.distplot(abs(d[d['label']==1]['VZ']), color='coral', label='Groundwater')
    plt.legend(loc='upper right')

    max(abs(d[d['label']==0]['P']))
    min(abs(d[d['label']==0]['P']))
    sns.distplot(abs(d[d['label']==0]['P']), color='dodgerblue', label='HZ')
    sns.distplot(abs(d[d['label']==1]['P']), color='coral', label='Groundwater')
    plt.legend(loc='upper right')

    # show the skweness
    df = pd.concat([process_data, pd.DataFrame(y, columns=['label'])], axis=1)
    df['VZ'] = P.norm(df['VZ'])

    var_VZc1 = np.var(df[df['label']==1]['VZ'])
    var_VZc2 = np.var(df[df['label']==0]['VZ'])
    print(var_VZc1, var_VZc2)

    df['P'] = P.norm(df['P'])
    var_Pc1 = np.var(df[df['label']==1]['P'])
    var_Pc2 = np.var(df[df['label']==0]['P'])
    print(var_Pc1, var_Pc2)

    skewness_Pc1 = skew(df[df['label']==0]['P'])
    skewness_Pc2 = skew(df[df['label']==1]['P'])
    print(skewness_Pc1, skewness_Pc2)
    plt.scatter(df[df['label']==0]['P'], df[df['label']==0]['VZ'], alpha=0.1, color='red')
    plt.scatter(df[df['label']==1]['P'], df[df['label']==1]['VZ'], alpha=0.1)

    # import seaborn as sns
    # sns.distplot(df[df['label']==0]['P'], color='blue', label='Groundwater', hist=None)
    # sns.distplot(df[df['label']==1]['P'], color='red', label='HZ', hist=None)
    # plt.xlabel('Pressure')
    # plt.legend(loc='upper right')
