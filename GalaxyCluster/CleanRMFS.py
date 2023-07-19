""" 
This script checks if any RMFs that were created are blank. If any RMFs are, then they will be removed. Finally, a clean RMF pickle file will be created.

This should not be used for creating synthetic data! This should only be used if the rmf pickle file is necessary. I really only use this as a verifcation step.
"""
import numpy as np
import pickle
import os
# Path to rmfs (A)
def clean_rmfs(filename):
    responses_data = pickle.load(open(filename+'.pkl', 'rb'))
    responses = [response for response in responses_data]
    # Check for good rmfs
    not_indices = []  # List of indices corresponding to all-zero rmfs
    for spec_ct, spec_obs in enumerate(responses):  # Step through all the rmfs
        if not np.any(spec_obs):  # The rmfs is only zeros
            not_indices.append(spec_ct)  # Add to list of indices
    print('Number of all-zero rmfs: %i'%len(not_indices))
    print('Total number of rmfs: %i'%(len(responses)))

    # Remove bad RMFS
    #good_responses = list(np.delete(responses, not_indices))
    #print('Number of good RMFs after cleaning: %i'%(len(good_responses)))
    #clean_filename = filename + '_clean'
    #pickle.dump(good_responses, open(clean_filename+'.pkl', 'wb'))

if __name__ == "__main__":
    RMFS_dir = '/export/home/carterrhea/Documents/ChandraData/MachineLearningPaperI/RMFs'
    for file in os.listdir(RMFS_dir):
        if file.endswith(".pkl") and 'not' not in file:
            filename = os.path.join(RMFS_dir, file)
            clean_rmfs(filename=filename)
