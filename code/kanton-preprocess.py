
import paths, numpy as np, pandas as pd, requests, os, json
from utils.preprocess import featureSelection
from scipy.io import mmread
from scipy import sparse


# Boolean. Whether to save the preprocessed data or not.
save_data = True

print('===')
print("=== Preprocessing {v} data".format(v=paths.kanton_name))
print("===")

def download_file(url, destination):
    response = requests.get(url)
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            f.write(response.content)
        print(f"File downloaded successfully from {url}.")
    else:
        print(f"Failed to download file from {url}. Status code: {response.status_code}")


def preprocess(metafile, countfile, line, n=1000, decay=1.5, n_components=50):
    meta = pd.read_csv(metafile, sep="\t")

    counts = mmread(str(countfile))
    counts = sparse.csc_matrix(counts).T

    ind = meta["in_FullLineage"].values
    if line is not None:
        ind = ind & (meta["Line"].values == line)

    seqDepths = np.array(counts[ind, :].sum(axis=1))
    stage = meta["Stage"].values[ind].astype("str")


    print('Number of cells:', counts.shape[0])
    print('Number of clusters:', np.unique(stage).size)
    print('Number of genes:', counts.shape[1])
    print(f'Fraction of zeros in the data matrix: {counts.size / np.prod(counts.shape):.2f}')

    meanExpr, zeroRate = meanExpr_zeroRate_kanton(counts[ind, :], threshold=0, atleast=10)

    impGenes = featureSelection(meanLogExpression=meanExpr, nearZeroRate=zeroRate, n=n, decay=decay)

    # Transformations
    logcounts = np.log2(counts[:, impGenes][ind, :].toarray() / seqDepths * np.median(seqDepths) + 1)
    X = logcounts - logcounts.mean(axis=0)
    U, s, V = np.linalg.svd(X, full_matrices=False)
    X = np.dot(U, np.diag(s))
    X = X[:, np.argsort(s)[::-1]][:, :n_components]

    return X, logcounts, stage


def meanExpr_zeroRate_kanton(data, threshold=0, atleast=10):
    zeroRate = 1 - np.squeeze(np.array((data > threshold).mean(axis=0)))
    A = data.multiply(data > threshold)
    A.data = np.log2(A.data)
    meanExpr = np.zeros_like(zeroRate) * np.nan
    detected = zeroRate < 1
    meanExpr[detected] = np.squeeze(np.array(A[:, detected].mean(axis=0))) / (
        1 - zeroRate[detected]
    )

    lowDetection = np.array(np.sum(data > threshold, axis=0)).squeeze() < atleast
    zeroRate[lowDetection] = np.nan
    meanExpr[lowDetection] = np.nan
    return meanExpr, zeroRate


def load_kanton(root_path=paths.kanton_data):
    root_path = os.path.join(root_path, "human-409b2")
    if not os.path.exists(root_path):
        os.mkdir(root_path)

    try:
        x = np.load(os.path.join(root_path, "preprocessed-data.npy"))
        logcounts = np.load(os.path.join(root_path, "gene-selected-data.npy"))
        y = np.load(os.path.join(root_path, "labels.npy"))
        with open(os.path.join(root_path, "metadata.json"), "r") as f:
            d = json.load(f)

    except FileNotFoundError:

        metafile = os.path.join(root_path, "metadata_human_cells.tsv")
        countfile = os.path.join(root_path, "human_cell_counts_consensus.mtx")

        urls = []
        if not os.path.exists(metafile):
            urls.append("https://www.ebi.ac.uk/biostudies/files/E-MTAB-7552/metadata_human_cells.tsv")
        if not os.path.exists(countfile):
            urls.append("https://www.ebi.ac.uk/biostudies/files/E-MTAB-7552/human_cell_counts_consensus.mtx")

        if len(urls) > 0:
            print("Downloading data")
        for url in urls:
            filename = os.path.join(root_path, url.split("/")[-1])
            download_file(url, filename)

        print("Preprocessing data")
        line = "409b2"
        x, logcounts, y = preprocess(metafile, countfile, line)

        print('Shape of the data after gene selection:', logcounts.shape, '\n')
        print('Shape of the data after PCA:', x.shape, '\n')

        # meta data
        d = {"label_colors": {
            "iPSCs": "navy",
            "EB": "royalblue",
            "Neuroectoderm": "skyblue",
            "Neuroepithelium": "lightgreen",
            "Organoid-1M": "gold",
            "Organoid-2M": "tomato",
            "Organoid-3M": "firebrick",
            "Organoid-4M": "maroon",
        }, "time_colors": {
            "  0 days": "navy",
            "  4 days": "royalblue",
            "10 days": "skyblue",
            "15 days": "lightgreen",
            "  1 month": "gold",
            "  2 months": "tomato",
            "  3 months": "firebrick",
            "  4 months": "maroon",
        }}

        # cluster assignments
        meta = pd.read_csv(metafile, sep="\t")
        mask = (meta["Line"] == "409b2") * meta["in_FullLineage"]

        d["clusters"] = list(meta[mask]["cl_FullLineage"])

        d["color_to_time"] = {v: k for k, v in d["time_colors"].items()}

        if save_data:
            np.save(os.path.join(root_path, "preprocessed-data"), x)
            np.save(os.path.join(root_path, "gene-selected-data"), logcounts)
            np.save(os.path.join(root_path, "labels"), y)

            with open(os.path.join(root_path, "metadata.json"), "w") as f:
                json.dump(d, f)

    return x, logcounts, y, d

if __name__ == "__main__":
    x, logcounts, y, d = load_kanton()
    print("pca shape: ", x.shape)
    print("gene selected shape: ", logcounts.shape)
    print("label shape: ", y.shape)
    print("meta data: ", d.keys())