import numpy as np

def get_ave_results(filename):
    results = np.loadtxt(filename)
    results = results[:,:4]
    samples = results.shape[0]
    results[np.isnan(results)]=0
    ave_results = np.sum(results,axis=0)/samples
    ACC = ave_results[0]
    NMI = ave_results[1]
    F1 = ave_results[2]
    ARI = ave_results[3]
    return ACC,NMI,F1,ARI

facebook_file = "facebook_evaluate0.45.txt"
ACC, NMI, F1, ARI = get_ave_results(facebook_file)
print("ACC: {}  NMI: {} F1: {} ARI: {}".format(round(ACC, 4), round(NMI, 4),round(F1, 4), round(ARI,4)))




