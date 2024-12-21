import kNearestNeighbours as knn
import time
import MultiLayeredPerceptron as mlp
import SupportVectorMachine as svm
import nearestClassCentroid as ncc

start_time=time.time()
print(u'\u2500' * 50)
ncc.NCC()
print(u'\u2500' * 50)
knn.KNN(1)
print(u'\u2500' * 50)
knn.KNN(2)
print(u'\u2500' * 50)
mlp.MLP()
print(u'\u2500' * 50)
kernels=['linear','rbf','poly','sigmoid']
svm.SVM(Kernel=kernels[1], verbal=True)
print(u'\u2500' * 50)
total_time=time.time() - start_time
minutes= total_time//60
seconds= total_time - minutes*60
print(f"Program complete in {int(minutes)} minutes, {seconds:.10f} seconds")