import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def etudeStatYtrain():
    dataYtrain=pd.read_csv('Ytrain.csv')
    l,r=dataYtrain.shape
    dt={}
    for id in range(l):
        classe=int(dataYtrain['prdtypecode'][id])
        if classe in dt.keys():
            dt[classe].add(id)
        else:
            dt[classe]={id}
    # nombre de classe pour le doc

    nbClass=len(dt.keys()) # 27 pour le cas étudié

    # tracé du graphe de nb de produit en fonction des classes 
    def traceClasse():
        X=[]
        Y=[]
        for classe in sorted(dt.keys()):
            X.append(classe)
            Y.append(len(dt[classe]))
        plt.style.use('dark_background')
        plt.figure(figsize=(15, 12))
        sns.barplot(x=X,y=Y)
        plt.show()
    print('nombre de classe : ',nbClass,'\nnombre de produit : ',l)
    traceClasse()




