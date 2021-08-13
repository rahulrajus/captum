from google_cca import get_cca_similarity
import numpy as np

def svcca(acts1, acts2):
    # print("Results using SVCCA keeping 20 dims")

    # # Mean subtract activations
    cacts1 = acts1 - np.mean(acts1, axis=1, keepdims=True)
    cacts2 = acts2 - np.mean(acts2, axis=1, keepdims=True)

    # Perform SVD
    U1, s1, V1 = np.linalg.svd(cacts1, full_matrices=False)
    U2, s2, V2 = np.linalg.svd(cacts2, full_matrices=False)

    svacts1 = np.dot(s1[:4]*np.eye(4), V1[:4])
    # can also compute as svacts1 = np.dot(U1.T[:20], cacts1)
    svacts2 = np.dot(s2[:4]*np.eye(4), V2[:4])

    svcca_results = get_cca_similarity(svacts1, svacts2, epsilon=1e-10, verbose=False)
    print(svcca_results)

np.random.seed(1337)
act = np.random.randn(10, 20).astype(np.float32)
actss1 = act[:5]
actss2 = act[5:]

# print(repr(actss1))

svcca(actss1, actss2)
