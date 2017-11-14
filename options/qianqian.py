import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from gensim import corpora, models
import gensim
from sklearn.decomposition import NMF
from sklearn.manifold import SpectralEmbedding
pd.options.display.max_rows = 1000
%run scripts/start.py

#https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/32156
# feat number 1: "manager_id grouping building_id tfidf to nmf",
# feat number 2: "manager_id grouping building_id tfidf deepwalkembedding",
# feat number 3: "manager_id grouping street_address tfidf to nmf",
# feat number 4: "manager_id grouping display_address tfidf to nmf",
# feat number 5: "building_id grouping manager_id tfidf to nmf",
# feat number 6: "building_id grouping street_address tfidf to nmf",
# feat number 7: "manager_id grouping price_t1 mean",
# feat number 8: "manager_id grouping room_sum mean",
# feat number 9: "manager_id grouping created difference mean",
# feat number 10: "manager_id grouping created count mean",
# feat number 11: "manager_id grouping created 24-hour mean(how often manager post during each hour)",
# feat number 12: "manager_id grouping building_id zero count",
# feat number 13: "manager_id grouping building_id zero ratio",
# feat number 14: "manager_id grouping latitude median",
# feat number 15: "manager_id grouping longitude median",
# feat number 16: "latitude,longitude grouping distance to clustering center",
# feat number 17: "manager_id latitude,longitude median grouping distance to clustering center",
# feat number 18: "manager_id grouping label encoding"

def generate_gpd_feature(df, group_by_feat, grouped_by_feat, reduce_num, fname, method = "nmf"):
    gpd = df.groupby(group_by_feat)
    corpus = gpd.apply(lambda df: df[grouped_by_feat].values.tolist())
    indices = corpus.index # used later to join with main dataset
    corpus = list(corpus)
    dictionary = corpora.Dictionary(corpus)
    corpus = [dictionary.doc2bow(text) for text in corpus]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    corpus = None # free up some RAM
    corpus_tfidf = [doc for doc in corpus_tfidf]
    corpus_tfidf = gensim.matutils.corpus2csc(corpus_tfidf).T
    if method == "nmf":
        nmf = NMF(n_components=reduce_num)
        res = nmf.fit_transform(corpus_tfidf)
    elif method == "se":
        se = SpectralEmbedding(n_components=reduce_num)
        res = se.fit_transform(corpus_tfidf.toarray())
    assert res.shape[0] == df[group_by_feat].unique().shape[0]
    res = pd.DataFrame(res, index=indices)
    feat_final = pd.merge(left = df[[group_by_feat]], right = res, left_on=group_by_feat, right_index=True)
    feat_final = feat_final.iloc[:, 1:]
    root = "features/qq"
    fpath = os.path.join(root, fname)
    feat_final.to_csv(fpath)
    print("%s complete." % fname)
    return res

df_fu = IO_full()
df_fu.created = pd.to_datetime(df_fu.created)
cols = [ col for col in df_fu.columns if col not in ["manager_id", "display_address"]]
cols = ["manager_id", "display_address"] + cols
df_fu = df_fu[cols]

# ****************
# features 1 - 6 *
# ****************

ft1 = generate_gpd_feature(df_fu, "manager_id", "building_id", 10, "full_man_bui_nmf.csv")
ft2 = generate_gpd_feature(df_fu, "manager_id", "building_id", 10, "full_man_bui_se.csv", method = "se")
ft3 = generate_gpd_feature(df_fu, "manager_id", "street_address", 10, "full_man_strAddr_nmf.csv")
ft4 = generate_gpd_feature(df_fu, "manager_id", "display_address", 10, "full_man_disAddre_nmf.csv")
ft5 = generate_gpd_feature(df_fu, "building_id", "manager_id", 10, "full_bui_man_nmf.csv")
ft6 = generate_gpd_feature(df_fu, "building_id", "street_address", 10, "full_bui_strAddr_nmf.csv")

# ****************
# features 7 - 15 *
# ****************

gpd = df_fu.groupby("manager_id")
manager_indices = gpd.size().index # used later to merge with main df. I use size() but could be any agg func

# FEAT 7
df_fu["room_sum"] = df_fu["bathrooms"]+df_fu["bedrooms"]
df_fu["price_t1"] = df_fu["price"] / df_fu["room_sum"]
df_fu.loc[(df_fu["room_sum"] == 0), "price_t1"] = df_fu["price"]/0.5
ft7 = gpd.mean()["price_t1"]


# FEAT 8
ft8 = gpd.mean()["room_sum"]

# FEAT 9
#"manager_id grouping created difference mean"
def get_diff(df):
    df = df.sort_values(by = ["created"])
    return df["created"].diff().mean()
ft9 = gpd.apply(get_diff)
ft9 = ft9.dt.seconds.fillna(-1)

# FEAT 10
# "manager_id grouping created count mean",
ft10 = gpd.size()

# FEAT 11
# "manager_id grouping created 24-hour mean(how often manager post during each hour)"
def average_per_hour(df):
    post_per_hour = df.groupby("unique_hour").size()
    mean_per_manager = post_per_hour.mean()
    return mean_per_manager
df_fu["unique_hour"] = df_fu.created.astype("str").str[:13]
ft11 = gpd.apply(average_per_hour)

# FEAT 12
# "manager_id grouping building_id zero count",
df_fu["building_is_zero"] = (df_fu["building_id"] == "0")*1
gpd = df_fu.groupby("manager_id")
ft12 = gpd.sum()["building_is_zero"]

# FEAT 13
def ratio_building_zero(df):
    return len(df[df["building_is_zero"] == 1])/len(df)
ft13 = gpd.apply(ratio_building_zero)

# FEATS 14 AND 15
ft14 = gpd.median()["latitude"]
ft15 = gpd.median()["longitude"]

for i, feat in enumerate([ft7, ft8, ft9, ft10, ft11, ft12, ft13, ft14, ft15]):
    feat = pd.DataFrame(feat, index = manager_indices)
    feat_final = pd.merge(left = df_fu[["manager_id"]], right = feat, left_on="manager_id", right_index=True, how="left")
    feat_final = feat_final.iloc[:, 1:]
    fpath = "features/qq/full_manager_gpd_feat_%s.csv" % (7+i)
    feat_final.to_csv(fpath, index = None)
    print("Feat number %s complete." % (7+i))

# *************
# features 16 *
# *************

def cluster_latlon(n_clusters, data, method = "birch"):
    from sklearn.cluster import Birch
    from sklearn.cluster import KMeans
    def fit_cluster(coords, method):
        if method == "birch":
            model = Birch(branching_factor=100, n_clusters=n_clusters, threshold=0.01,compute_labels=True)
            model.fit(coords)
            clusters=model.predict(coords)
        elif method == "kmeans":
            model = KMeans(n_clusters=n_clusters)
            model.fit(coords)
            clusters=model.predict(coords)
        return clusters, model
    # data must be df_full
    #split the data between "around NYC" and "other locations" basically our first two clusters 
    lat_lon_cols = ['latitude', "longitude"]
    mask = (data.longitude>-74.05)&(data.longitude<-73.75)&(data.latitude>40.4)&(data.latitude<40.9)
    data_c=data.loc[mask, lat_lon_cols].copy(deep=True)
    data_e=data.loc[~mask, lat_lon_cols].copy(deep=True)
    #put it in matrix form
    coords=data_c.as_matrix(columns=lat_lon_cols)
    # fit cluster algo
    clusters, model = fit_cluster(coords, method)
    data_c["cluster_"+str(n_clusters)]=clusters
    data_e["cluster_"+str(n_clusters)]=-1 #assign cluster label -1 for the non NYC listings 
    assert data_c.shape[0] + data_e.shape[0] == data.shape[0]
    data = pd.merge(data, data_c[["cluster_"+str(n_clusters)]], left_index = True, right_index = True, how = 'left')
    data = pd.merge(data, data_e[["cluster_"+str(n_clusters)]], left_index = True, right_index = True, how = 'left')
    data["cluster_"+str(n_clusters)] = data["cluster_"+str(n_clusters)+"_x"].fillna(0)+data["cluster_"+str(n_clusters)+"_y"].fillna(0)
    assert data[data["cluster_"+str(n_clusters)].isnull()].shape[0] == 0
    return data, model

n_clusters = 35
df_fu, kmeans = cluster_latlon(n_clusters, df_fu, method = "kmeans")
print("created clusters.")


non_nyc_centroid = df_fu.loc[df_fu["cluster_35"] == -1, ["latitude", "longitude"]]
non_nyc_centroid = non_nyc_centroid.mean(axis = 0).values
centroids = { val[0] : list(val[1]) for val in zip(range(n_clusters + 1), kmeans.cluster_centers_) } # centroids are in same order as labels, i.e. from 0 to n_cluster-1
centroids[-1] = non_nyc_centroid
def compute_distance(r):
    def euclidean(x, y):
        return np.sqrt(np.sum((x-y)**2))
    x = r[["latitude", "longitude"]].values
    y = centroids[r["cluster_35"]]
    return euclidean(x, y)

df_fu["centroid_dist"] = df_fu.apply(compute_distance, axis = 1)
df_fu[["centroid_dist"]].to_csv("features/qq/full_distance_to_centroids.csv")


# *****************
# Simple features *
# *****************

%run "scripts/start.py"
df_fu = make_simple_features(df_fu)

simple_feats = [
"bedrooms",
"bathrooms",
"price",
"longitude",
"latitude",    
"created_month",
"created_day",
"created_hour",
"num_photos",
"num_features",
"num_description_words",
"p_per_bathroom",
"p_per_bedroom",
"log_price",
"room_sum",
"room_diff",
"price_t1"
            ]

df_fu[simple_feats].to_csv("features/qq/full_simple_feats.csv")

# ***************
# Magig feature *
# ***************

import pandas as pd
import numpy as np
%run scripts/start.py
df_fu = IO_full()
magic_all_ids = pd.read_csv("downloads/listing_image_time.zip") # all 124k rows loaded
magic_feat = df_fu[["listing_id"]] # Base to merge with 
magic_feat = pd.merge(magic_feat, magic_all_ids, how = "left", left_on = "listing_id", right_on = "Listing_Id")
magic_feat = magic_feat[["time_stamp"]]
magic_feat.to_csv("features/mk/full_magic.csv")






