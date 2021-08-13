from io import BytesIO


def reorder_matrix(mat, showdendrogram=False):
    import scipy.cluster.hierarchy as sch

    Y = sch.linkage(mat, method="centroid")
    Z = sch.dendrogram(Y, orientation="left", no_plot=not showdendrogram)

    index = Z["leaves"]
    mat = mat[index, :]
    mat = mat[:, index]
    return mat, index


def save_matrix(mat, title, io):
    import pylab

    fig = pylab.figure(figsize=(7, 7), dpi=100)
    axmatrix = fig.add_axes([0, 0, 0.9, 0.9], label="axes1")
    axmatrix.matshow(mat, aspect="auto", origin="lower")
    fig.savefig(io, format="PNG")
