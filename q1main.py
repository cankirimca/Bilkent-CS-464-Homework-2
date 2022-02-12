import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_pcs(dataset, components):  
  cov = np.cov(centered , rowvar = False)
  eig_val , eig_vec = np.linalg.eigh(cov)
  #get sorted eigenvalue indexes
  indexes_asc= np.argsort(eig_val)
  indexes_desc = np.flip(indexes_asc)
  eig_val = eig_val[indexes_desc]
  eig_vec = eig_vec[:,indexes_desc]
  component_list = eig_vec[:,0:components]
  return component_list, eig_val

def apply_pca(centered, component_list, no_of_components):  
  return np.dot(centered, component_list[:,0:no_of_components])

def reconstruct(applied, pc_list, no_of_components):
  return (applied @ pc_list[:, 0:no_of_components].T) + (np.sum(images, axis=0)/len(images))

df = pd.read_csv('images.csv')
images = df.to_numpy()

centered = images - np.mean(images , axis = 0)

#obtain the first 500 principal components
#to decrease the runtime, pcs are only generated once
pc_list, pve_list = get_pcs(centered, 500)

total_var = np.sum(pve_list)
for i in range(10):
  print("PVE for component", i + 1, ":", pve_list[i]/total_var)

for i in range(10):
  pca_matrix = np.reshape(pc_list[:,i], (48,48))
  #plt.imshow(pca_matrix, cmap = 'gray')  
  #plt.title("Principal Component "+ str(i+1))
  #plt.colorbar()
  #plt.show()  

pves = []
k_values = [1, 10, 50, 100, 500]
for i in k_values:
  pves.append(np.sum(pve_list[0:i]/total_var) * 100)
for i in range(5):
  print("Total PVE for the first", k_values[i], "components:", pves[i], "%")  
#plt.plot(k_values, pves)
#plt.xlabel('Number of Components')
#plt.ylabel('PVE (%)')
#plt.title('Total PVE vs Number of Components')
#plt.show()

for i in [1,10,50,100,500]:
  applied = apply_pca(centered, pc_list, i)
  reconstructed = reconstruct(applied, pc_list, i)
  reconst_matrix = np.reshape(reconstructed[0,:], (48,48))
  #plt.imshow(reconst_matrix, cmap = 'gray')  
  #plt.title("Reconstructed with "+ str(i) + " components")
  #plt.colorbar()
  #plt.show()