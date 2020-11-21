#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 11:36:52 2020

@authors: Jackie Zhang, Alicia Chen

@citations: 
    https://www.kaggle.com/jaykrishna/topic-modeling-enron-email-dataset
    https://www.kaggle.com/cguzman09/extracting-emails-from-enron-data-set
    https://blog.exploratory.io/visualizing-k-means-clustering-results-to-understand-the-characteristics-of-clusters-better-b0226fb3dd10
    https://realpython.com/k-means-clustering-python/
"""

"""
|------------|
|  IMPORTS   |
|------------|
"""
import os, sys, email,re, math, time, itertools, operator, string, nltk, gensim
import numpy as np 
import pandas as pd
import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
from mlxtend.plotting import plot_linear_regression
import networkx as nx
import nxviz as nv
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.cluster import KMeans, MiniBatchKMeans
from kneed import DataGenerator, KneeLocator
from sentiment_module import sentiment

"""
|------------------------|
|  1. DATA PREPARATION   |
|------------------------|
"""
start_1 = time.time()
os.chdir("/Users/zhanyina/Documents/MSA/AA502 Analytics Methods and Applications I/Text Mining/Enron/Data and Results")

''' Parsing out two columns into 18 columns for easier readability'''
# Read the data into a DataFrame
emails_df_raw = pd.read_csv('emails.csv')
# Examine the data
print(emails_df_raw.shape) #(517401, 2)
emails_df_raw.head()

# Helper functions
def get_text_from_email(msg):
    '''To get the content from email objects'''
    parts = []
    for part in msg.walk():
        if part.get_content_type() == 'text/plain':
            parts.append( part.get_payload() )
    return ''.join(parts)

def split_email_addresses(line):
    '''To separate multiple email addresses'''
    if line:
        addrs = line.split(',')
        addrs = frozenset(map(lambda x: x.strip(), addrs))
    else:
        addrs = None
    return addrs
            
# Parse the emails into a list email objects
messages = list(map(email.message_from_string, emails_df_raw['message']))
emails_df = emails_df_raw
emails_df_raw.drop('message', axis=1, inplace=True)
# Get fields from parsed email objects
keys = messages[0].keys()
for key in keys:
    emails_df[key] = [doc[key] for doc in messages]
# Parse content from emails
emails_df['content'] = list(map(get_text_from_email, messages))
# Split multiple email addresses
emails_df['From'] = emails_df['From'].map(split_email_addresses)
emails_df['To'] = emails_df['To'].map(split_email_addresses)

# Extract the root of 'file' as 'user'
emails_df['user'] = emails_df['file'].map(lambda x:x.split('/')[0])
del messages

# check cleaned and reconstructed dataframe
print(emails_df.shape) #(517401, 18)
pd.set_option('display.max_columns', None)
print(emails_df.head())
print(emails_df['content'][1])
# get all 18 column names
list(emails_df.columns) 
len(list(emails_df.columns))

# export to csv
emails_df.to_csv('EnronData_Processed.csv')

end_1 = time.time()
print("Part 1 ran for ", end_1 - start_1, " seconds!")


"""
|---------------------------------------------------------------------|
| 2. Finding bidirectional relationships of 100+ email correspondences|
|---------------------------------------------------------------------|
"""
start_2 = time.time()

# read in pre-processed Enron data
df = pd.read_csv('EnronData_Processed.csv')
df = df.rename(columns={"Unnamed: 0":"ID", "X-From":"XFrom",
                        "X-To":"XTo", "X-Origin":"XOrigin", }, errors="raise")
del df['file']
del df['Message-ID']
del df['Mime-Version']
del df['Content-Type']
del df['Content-Transfer-Encoding']
del df['X-cc']
del df['X-bcc']
del df['X-Folder']
del df['X-FileName']
df_s = df[["From", "To"]]
df_s.head()
df_s= df_s.dropna()
len(df_s)

df_s.To = [sub.replace("frozenset({", "").replace("})","") for sub in df_s.To]
df_s.From = [sub.replace("frozenset({", "").replace("})","") for sub in df_s.From]

relationship = {}
all_pairs = []
for ind in df_s.index: 
    cur_from = df_s['From'][ind]
    cur_to = df_s['To'][ind].split(",")
    
    for email in cur_to:
        pair = [cur_from]
        pair.append(email.strip())
        pair = [sub.replace("'", "") for sub in pair]
        # print(pair)
        all_pairs.append(sorted(pair))

for p in all_pairs:
    key = '_'.join(p)
    if key not in relationship:
        relationship[key] = 0
    else:
        relationship[key] = relationship[key] + 1
    

BD_rel = {key:value for (key, value) in relationship.items() if value >= 100}
BD_rel = dict(sorted(BD_rel.items(), key=operator.itemgetter(1),reverse=True))

end_2 = time.time()
print("Part 2 ran for ", end_2 - start_2, " seconds!")

"""
|---------------------------------------------------------------|
| 3. Filtering all emails to only those with 100+ relationships |
|---------------------------------------------------------------|
"""
start_3 = time.time()

df_f = df[["ID","From", "To"]]
df_f= df_f.dropna()
df_f.To = [sub.replace("frozenset({", "").replace("})","") for sub in df_f.To]
df_f.From = [sub.replace("frozenset({", "").replace("})","") for sub in df_f.From]

qualifying_rel = list(BD_rel.keys())
qualifying_rel = [sub.split("_") for sub in qualifying_rel]

email_IDs = set()
for ind in df_f.index: 
    email_id = df_f['ID'][ind]
    cur_from = df_f['From'][ind]
    cur_to = df_f['To'][ind].split(",")
    
    for email in cur_to:
        pair = [cur_from]
        pair.append(email.strip())
        pair = [sub.replace("'", "") for sub in pair]
        if pair in qualifying_rel:
            email_IDs.add(email_id)

len(email_IDs) / len(df) # --> 25%!! [130433/517401]

good_df = df[df['ID'].isin(email_IDs)]
good_df.To = [sub.replace("frozenset({", "").replace("})","") for sub in good_df.To]
good_df.From = [sub.replace("frozenset({", "").replace("})","") for sub in good_df.From]

end_3 = time.time()
print("This chunk 2 ran for ", end_3 - start_3, " seconds!") #331 sec

"""
|------------------------------------------------------------|
| 4. Further reducing email sample size by centrality metrics|
|------------------------------------------------------------|
"""
start_4 = time.time()

good_df['recipient1'] = good_df.apply (lambda row: row["To"].split(",")[0], axis=1)
end = time.time()
good_df.recipient1[4685]

G = nx.from_pandas_edgelist(good_df, 'To', 'recipient1', edge_attr=['Date', 'Subject'])

"""filtering data by top 100 most popular nodes/emails"""
# Degree Centrality --> popularity
cent = nx.degree_centrality(G)
name = []
centrality = []

for key, value in cent.items():
    name.append(key)
    centrality.append(value)

cent = pd.DataFrame()    
cent['name'] = name
cent['centrality'] = centrality
cent = cent.sort_values(by='centrality', ascending=False)

pop_emails = cent[:100].name.values.tolist()
pop_emails = [sub.replace("'", "") for sub in pop_emails]
len(pop_emails)

email_IDs_2 = set()
for ind in good_df.index: 
    email_id = good_df['ID'][ind]
    cur_from = good_df['From'][ind].strip("'")
    if cur_from in pop_emails:
        email_IDs_2.add(email_id)
    else:
        cur_to = good_df['To'][ind].split(",")
        
        for email in cur_to:
            if email.strip("'") in pop_emails:
                email_IDs_2.add(email_id)

final_df = good_df[good_df['ID'].isin(email_IDs_2)]

''' randomly sampling 10% of thus far filtered dataset '''
final_df = final_df.sample(frac=0.1, random_state=1)
final_df.to_csv('Sampled_Enron.csv')

end_4 = time.time()
print("Part 4 ran for ", end_4 - start_4, " seconds!")

"""
|--------------------------------------|
| 5. Clearning email content with regex|
|--------------------------------------|
"""
start_5 = time.time()

# # start here if you are starting directly
# sample = pd.read_excel("Sampled_Enron.csv")

# else start here
sample = final_df

content_list = []
for x in sample['content']: 
    x = x.replace("\n", "%%%")
    abc = re.match('''(.*?)((%%%\w+\s\w+@\w+%%%\d{2}\/\d{2}\/\d{4} \d{2}:\d{2} \w+)|(At \d{2}:\d{2} \w\w \d{1,2}\/\d{2}\/\d{4} -\d{4}, you wrote:)|(From: )|(To: )|(-----Original Message-----)|(Original Message:)|((-)+ Forwarded by)|(%%%\w+\s\w+%%%\d{2}\/\d{2}\/\d{4} \d{2}:\d{2} \w+%%%To:))(.*)''',x)
    if abc:
        defg = abc.group(1).replace("%%%", "\n")
        content_list.append(defg)
    else: 
        content_list.append(x)
    
contentlist2 =[]
for x in content_list:
    contentlist2.append(x.replace("%%%", "\n"))

contentlist2[0:10]

sample['newcontent']=""
sample['newcontent']=contentlist2
        
sample.to_excel('final_cleaned_data.xlsx')

end_5 = time.time()
print("Part 5 ran for ", end_5 - start_5, " seconds!")

"""
|-----------------------------|
| 6.* Social Network Analysis |
|-----------------------------|
"""
start_6 = time.time()

# CircosPlot
plot = nv.CircosPlot(G)
plot.draw()
plt.show()

# drawing the network
plt.figure(figsize=(20,20))
pos = nx.spring_layout(G, k=.1)
nx.draw_networkx(G, pos, node_size=25, node_color='red', with_labels=True, edge_color='blue')
plt.show()

# Degree Centrality --> popularity
plt.figure(figsize=(10, 25))
_ = sns.barplot(x='centrality', y='name', data=cent[:10], orient='h')
_ = plt.xlabel('Degree Centrality')
_ = plt.ylabel('Correspondent')
_ = plt.title('Top 15 Degree Centrality Scores in Enron Email Network')
plt.show()

G = nx.from_pandas_edgelist(sample, 'To', 'recipient1', edge_attr=['Date', 'Subject'])

# Betweenness Centrality --> Bridge
between = nx.betweenness_centrality(G)
name = []
betweenness = []

for key, value in between.items():
    name.append(key)
    betweenness.append(value)

bet = pd.DataFrame()
bet['name'] = name
bet['betweenness'] = betweenness
bet = bet.sort_values(by='betweenness', ascending=False)


plt.figure(figsize=(10, 25))
_ = sns.barplot(x='betweenness', y='name', data=bet[:15], orient='h')
_ = plt.xlabel('Degree Betweenness Centrality')
_ = plt.ylabel('Correspondent')
_ = plt.title('Top 15 Betweenness Centrality Scores in Hillary Clinton Email Network')
plt.show()

# Eigenvector Centrality --> Influence/Prestige
eigenvector_cent = nx.adjacency_matrix(G).todense()

# Closeness Centrality --> Centralness
closeness_cent = nx.closeness_centrality(G)

end_6 = time.time()
print("Part 6 ran for ", end_6 - start_6, " seconds!")

"""
|---------------------|
| 7. Topic Clustering |
|---------------------|
"""
start_7 = time.time()

""" LDA on top of TF-IDF"""
# use this when starting here and want to start with cleaned filtered dataset
really_good_df = pd.read_excel("final_cleaned_data_v2.xlsx")

# # use this if you are continuing from part 6 and above
# really_good_df = sample

temp = really_good_df.newcontent.dropna()
emails = list(temp)

# Remove punctuation
email = [ ]

punc = string.punctuation.replace( '-', '' )
for i in range( 0, len( emails ) ):
    email.append( re.sub( '[' + punc + ']+', '', emails[ i ] ) )
        

# Porter stem
porter_lda = nltk.stem.porter.PorterStemmer()
stems_lda = { }

for i in range( 0, len( email ) ):
    tok = email[ i ].split()
    for j in range( 0, len( tok ) ):
        if tok[ j ] not in stems_lda:
            stems_lda[ tok[ j ] ] = porter_lda.stem( tok[ j ] )
        tok[ j ] = stems_lda[ tok[ j ] ]

    email[ i ] = ' '.join( tok )

# Remove empty sentences after stop word removal
i = 0
while i < len( email ):
    if len( email[ i ] ) == 0:
        del email[ i ]
    else:
        i += 1

# Count raw term frequencies
# for LDA you only uses Count Vectorizer, NEVER TFIDF.. sklearn already does TFIDF inside LDA
count = CountVectorizer( stop_words='english' )
term_vec_lda = count.fit_transform( email )

n_topic = 10

# Build a string list of [ 'Topic 1', 'Topic 2', ..., 'Topic n' ]
col_nm = [ ]
for i in range( 1, n_topic + 1 ):
    col_nm += [ f'Topic {i}' ]

# Fit an LDA model to the term vectors, get cosine similarities
lda_model = LDA( n_components=n_topic )
concept = lda_model.fit_transform( term_vec_lda )
X = cosine_similarity( concept )

# Print top 12 terms for each topic
feat = count.get_feature_names()
topic_list = [ ]
for i,topic in enumerate( lda_model.components_ ):
    top_n = [ feat[ i ] for i in topic.argsort()[ -12: ] ]
    top_feat = ' '.join( top_n )
    topic_list.append( f"topic_{'_'.join(top_n[ :3 ] ) }" )

    print( f'Topic {i}: {top_feat}' )

        
""" k-means clustering! """

# elbow method to select best k
start_k = time.time()
kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}
sse = []
for k in range(1, 20):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

end_k = time.time()
print("Finding optimal k ran for ", end_k - start_k, " seconds!") # 503.4

# plotting the elbow graph
plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

# asking computer to find the best k 
kl = KneeLocator(
    range(1, 20), sse, curve="convex", direction="decreasing"
)

kl.elbow  #4


# Fit LDA vectors to clusters
clust = KMeans( n_clusters=4, random_state=101 ).fit( X )

km_df_lda = really_good_df[["From", "To", "newcontent"]]
km_df_lda["Cluster"] = km_df_lda.labels_

#obtain cluster labels for each cluster
km_df_lda.to_csv("sample_with_cluster_labels.csv")

''' examine each clusters '''
brainstorm_lda = pd.DataFrame()
for i in range(0,8):
    brainstorm_lda = brainstorm_lda.append(km_df_lda[km_df_lda.Cluster == i].head(10))
len(brainstorm_lda)
brainstorm_lda.to_excel('brainstorm_lda_v3.xlsx')

""" visualizing topic clustering!!! """

#read in initial dataframe
sample = pd.read_csv("sample_with_cluster_labels.csv")

for x in range(0,4):
    full_sent = list(sample['newcontent'][sample['cluster_no'] == x])

    # Remove empty sentences
    i = 0
    while i < len( full_sent ):
        if len( full_sent[ i ] ) == 0:
            del full_sent[ i ]
        else:
            i += 1

    # Remove punctuation
    sent = [ ]

    punc = string.punctuation.replace( '-', '' )
    for i in range( 0, len( full_sent ) ):
        sent.append( re.sub( '[' + punc + ']+', '', full_sent[ i ] ) )

    # Porter stem
    porter = nltk.stem.porter.PorterStemmer()
    stems = { }

    for i in range( 0, len( sent ) ):
        tok = sent[ i ].split()
        for j in range( 0, len( tok ) ):
            if tok[ j ] not in stems:
                stems[ tok[ j ] ] = porter.stem( tok[ j ] )
            tok[ j ] = stems[ tok[ j ] ]

        sent[ i ] = ' '.join( tok )

    # Remove empty sentences after stop word removal
    i = 0
    while i < len( sent ):
        if len( sent[ i ] ) == 0:
            del sent[ i ]
        else:
            i += 1

    txt = []

    for x in sent: 
        txt.extend(x.split(" "))
    txt

    def remove_stop_word( txt ):
        """Remove all stop words from text blo
        Args:
          txt (list of string): Text block as list of individual terms

        Returns:
          (list of string): Text block with stop words removed
        """

        term_list = [ ]
        stop_word = nltk.corpus.stopwords.words( 'english' )

        for term in txt:
            term_list += ( [ ] if term in stop_word else [ term ] )

        return term_list

    no_stopwords = remove_stop_word(txt)

    # Count occurrences of unique terms in each document

    term_dict = {}
    for x in no_stopwords:
        if x in term_dict.keys():
            term_dict[x] += 1
        else:
            term_dict[x] = 1

    sorted_term_dict = {k: v for k, v in sorted(term_dict.items(), key=lambda item: item[1], reverse=True)}

    no_stopwords = remove_stop_word(txt)

    # Count occurrences of unique terms in each document
    term_dict = {}
    for x in no_stopwords:
        if x in term_dict.keys():
            term_dict[x] += 1
        else:
            term_dict[x] = 1

    sorted_term_dict = {k: v for k, v in sorted(term_dict.items(), key=lambda item: item[1], reverse=True)}

    clustfreq = pd.DataFrame(sorted_term_dict.items(), columns=['word', 'count'])

    clustfreq.to_csv('clust{}freq.csv'.format(x))

#get lists of stopwords
def cluster_stopwords(x):
    full_sent = list(sample['newcontent'][sample['cluster_no'] == x])

    # Remove empty sentences
    i = 0
    while i < len( full_sent ):
        if len( full_sent[ i ] ) == 0:
            del full_sent[ i ]
        else:
            i += 1

    # Remove punctuation
    sent = [ ]

    punc = string.punctuation.replace( '-', '' )
    for i in range( 0, len( full_sent ) ):
        sent.append( re.sub( '[' + punc + ']+', '', full_sent[ i ] ) )

    # Porter stem
    porter = nltk.stem.porter.PorterStemmer()
    stems = { }

    for i in range( 0, len( sent ) ):
        tok = sent[ i ].split()
        for j in range( 0, len( tok ) ):
            if tok[ j ] not in stems:
                stems[ tok[ j ] ] = porter.stem( tok[ j ] )
            tok[ j ] = stems[ tok[ j ] ]

        sent[ i ] = ' '.join( tok )

    # Remove empty sentences after stop word removal
    i = 0
    while i < len( sent ):
        if len( sent[ i ] ) == 0:
            del sent[ i ]
        else:
            i += 1

    txt = []

    for x in sent: 
        txt.extend(x.split(" "))

    def remove_stop_word( txt ):
        """Remove all stop words from text blo
        Args:
          txt (list of string): Text block as list of individual terms

        Returns:
          (list of string): Text block with stop words removed
        """
        term_list = [ ]
        stop_word = nltk.corpus.stopwords.words( 'english' )

        for term in txt:
            term_list += ( [ ] if term in stop_word else [ term ] )

        return term_list
    return remove_stop_word(txt)

#create lists of stopwords
stop1 = cluster_stopwords(0)
stop2 = cluster_stopwords(1)
stop3 = cluster_stopwords(2)
stop4 = cluster_stopwords(3)

#write text files with stopwords for later word cloud visualization
with open('stop1.txt', 'w') as f:
    for item in stop1:
        f.write("%s\n" % item)

with open('stop2.txt', 'w') as f:
    for item in stop2:
        f.write("%s\n" % item)
        
with open('stop3.txt', 'w') as f:
    for item in stop3:
        f.write("%s\n" % item)

with open('stop4.txt', 'w') as f:
    for item in stop4:
        f.write("%s\n" % item)

# clusters over time

#create new columns for month, year, etc. and fill them from extracts from the emails
sample['Month'] = ""
sample['Year'] = ""
sample['Year-Month'] = ""
sample['total_sentiment'] = ""

monthlist = [dt.datetime.strptime(y, "%b").month for y in [x.split(" ")[2] for x in sample['Date']]]
yearlist = [x.split(" ")[3] for x in sample['Date']]

sample['Month'] = monthlist
sample['Year'] = yearlist

sample['Year-Month'] = sample['Year'].astype(str) + "-" + sample['Month'].astype(str)

#define function that recodes months into quarters 
def quarters(series):
    if series in [1,2,3]:
        return 1
    elif series in [4,5,6]:
        return 2
    elif series in [7,8,9]:
        return 3
    elif series in [10,11,12]:
        return 4

#create column for quarters 
sample['Quarter'] = ""
sample['Quarter'] = sample['Month'].apply(quarters)

sample['Year-Quarter'] = ""
sample['Year-Quarter'] = sample['Year'].astype(str) + "-" + sample['Quarter'].astype(str)


#create table with proportion in each cluster per quarter
pivot_sample = sample[['cluster_no', 'Year-Quarter']]

cluster_counts = pivot_sample.pivot_table(index='Year-Quarter', columns='cluster_no', values='cluster_no', aggfunc=len, fill_value=0)

total_counts = sample[['newcontent', 'Year-Quarter']].groupby(['Year-Quarter']).count()

#cluster_counts_all = total_counts.join(cluster_counts, on=['Year-Month'], how='left')

cluster_counts['all'] = cluster_counts[0] + cluster_counts[1] + cluster_counts[2] + cluster_counts[3]

cluster_counts['prop0'] = cluster_counts[0] / cluster_counts['all']
cluster_counts['prop1'] = cluster_counts[1] / cluster_counts['all']
cluster_counts['prop2'] = cluster_counts[2] / cluster_counts['all']
cluster_counts['prop3'] = cluster_counts[3] / cluster_counts['all']
cluster_counts = cluster_counts[1:]

cluster_counts.head()

cluster_counts.to_csv("cluster_counts.csv")

# histograms

#create histogram of word counts per email
#word counts were created by taking the cluster-labeled emails and adding counts to them 
hist0 = pd.read_csv('cluster_0.csv')
hist1 = pd.read_csv('cluster_1.csv')
hist2 = pd.read_csv('cluster_2.csv')
hist3 = pd.read_csv('cluster_3.csv')

#plot cluster 1 word counts 
font = {'size'   : 22}

plt.rc('font', **font)

fontP = FontProperties()
fontP.set_size('large')

plt.style.use('seaborn-whitegrid')
plt.figure(num=None, figsize=(9, 6), dpi=80, facecolor='w', edgecolor='k')

plt.xlim(right=1000)

plt.hist(hist0['count'], density=False, bins=100)  
plt.ylabel('Number of emails')
plt.xlabel('Word count');
plt.title('Cluster 1 ("Internal business") word counts')


# plot cluster 2 word counts
fontP = FontProperties()
fontP.set_size('large')

plt.figure(num=None, figsize=(9, 6), dpi=80, facecolor='w', edgecolor='k')

plt.hist(hist1['count'], density=False, bins=100)  # `density=False` would make counts
plt.ylabel('Number of emails')
plt.xlabel('Word count');
plt.title('Cluster 2 ("External business") word counts')
plt.xlim(left=0, right=1000)

# plot cluster 3 word counts
fontP = FontProperties()
fontP.set_size('large')

plt.style.use('seaborn-whitegrid')
plt.figure(num=None, figsize=(9, 6), dpi=80, facecolor='w', edgecolor='k')


plt.hist(hist2['count'], density=False, bins=50)  # `density=False` would make counts
plt.ylabel('Number of emails')
plt.xlabel('Word count');
plt.title('Cluster 3 ("Internal logistics") word counts')
plt.xlim(left=0, right=1000)

# plot cluster 4 word counts 
fontP = FontProperties()
fontP.set_size('large')

plt.style.use('seaborn-whitegrid')
plt.figure(num=None, figsize=(9, 6), dpi=80, facecolor='w', edgecolor='k')

plt.hist(hist3['count'], density=False, bins=50)  # `density=False` would make counts
plt.ylabel('Number of emails')
plt.xlabel('Word count');
plt.title('Cluster 4 ("Information exchange") word counts')
plt.xlim(left=0, right=1000)

#get cumulative percentages per each cluster so that we can plot a stacked bar chart of proportion of cluster per quarter
qclu = pd.read_csv("cluster_counts_simple.csv")
qclu.head()

#plot stacked bar chart of proportion of emails in each cluster per quarter
plt.style.use('seaborn-whitegrid')
fontP.set_size('medium')
plt.figure(num=None, figsize=(18, 7), dpi=80, facecolor='w', edgecolor='k')

N = 15
ind = np.arange(N)    # the x locations for the groups
width = 0.35
p1 = plt.bar(ind, qclu['cum1'], width, zorder=4, label='Internal business', color='#0B5351')
p2 = plt.bar(ind, qclu['cum2'], width, zorder=3, label='External business', color="#4E8098"
             #bottom=qclu['Cluster 1']
            )
p3 = plt.bar(ind, qclu['cum3'], width, zorder=2, label='Internal logistics', color="#00A9A5" #bottom=qclu['Cluster 2']
            )
p4 = plt.bar(ind, qclu['cum4'], width,zorder=1, label='Information exchange', color="#90C2E7" #bottom=qclu['Cluster 3']
            )

plt.legend(handles=[p1, p2, p3, p4], title='Cluster names', bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)
plt.ylabel('Percentage')
plt.title('Proportion of emails in each cluster per quarter')
plt.xticks(ind, (qclu['Year-Quarter']), rotation=45)
plt.yticks(np.arange(0, 1, .10))
plt.ylim(top=1)

plt.show()

end_7 = time.time()
print("Part 7 ran for ", end_7 - start_7, " seconds!")

"""
|-----------------------|
| 8. Sentiment Analysis |
|-----------------------|
"""
start_8 = time.time()

%matplotlib inline
plt.style.use('seaborn-whitegrid')

enron = pd.read_excel("final_cleaned_data_v2.xlsx")

""" SENTIMENTS OVERALL """
orig_email = enron.newcontent.dropna()
# getting rid of http addresses and newline newtab symbols
orig_email = orig_email.apply(lambda x : re.sub(r'http\S+', '', x))
orig_email = orig_email.apply(lambda x : re.sub('\n', '', x))
orig_email = orig_email.apply(lambda x : re.sub('\t', '', x))

sep_email=orig_email.str.split(".").to_frame().values.tolist()

# separating emails into sentences
sentences = []
i = 0
for email in sep_email:
    email = email[0]
    j = 0
    while j < len( email ):
        if len( email[ j ] ) == 0:
            del email[ j ]
        else:
            j += 1
    for sentence in email:
        sentences.append(sentence)
    i += 1

#  Calulate average valence and arousal for each sentence using Dr. Healey's package 
#  which uses the ANEW and Happiness sentiment dictionaries	
sentiments = [sentiment.sentiment(sentence.split(" ")) for sentence in sentences]

''' visualizing '''
valences = [sent["valence"] for sent in sentiments] 
valences = [i for i in valences if i != 0]
arousals = [sent["arousal"] for sent in sentiments] 
arousals = [i for i in arousals if i != 0]

sentiments_overall = list(zip(valences,arousals))

# high arousal, high valence --> orange #fdae61: 0
# low arousal, high valence --> green #abdda4: 1
# low arousal, low valence --> blue #2b83ba: 2
# high arousal, low valence --> red #d7191c: 3 
color = []
for sent in sentiments_overall:
    if sent[0]>=5: # high valence
        if sent[1]>=5: # high arousal 
             color.append(0)
        else: # low arousal
            color.append(1)
    else: # low valence
        if sent[1]>=5: # high arousal
             color.append(3)
        else: # low arousal
            color.append(2)
        
colormap = np.array(['#fdae61', '#abdda4', '#2b83ba', '#d7191c'])

fontP = FontProperties()
fontP.set_size('x-small')

plt.scatter(valences, arousals, c=colormap[color], alpha = 0.3)

orange = mpatches.Patch(color='#fdae61', label='excited, happy')
green = mpatches.Patch(color='#abdda4', label='contented, relaxed')
blue = mpatches.Patch(color='#2b83ba', label='bored, unhappy')
red = mpatches.Patch(color='#d7191c', label='tense, upset')

plt.title('Sentiment Per Email Sentence Overall')
plt.xlabel('Valence')
plt.ylabel('Arousal')
plt.xlim(1, 9)
plt.ylim(1, 9)
plt.savefig('sentiments_overall.png', dpi=300)
plt.show()

plt.legend(handles=[orange,green, blue, red], 
            loc='upper left', prop=fontP)
plt.savefig('sentiments_overall_legend.png', dpi=300)


""" SENTIMENTS OVER TIME """
#create new columns for month, year, etc. and fill them from extracts from the emails
enron['Month'] = ""
enron['Year'] = ""
enron['Month-Year'] = ""
enron['total_sentiment'] = ""

monthlist = [dt.datetime.strptime(y, "%b").month for y in [x.split(" ")[2] for x in enron['Date']]]
yearlist = [x.split(" ")[3] for x in enron['Date']]

enron['Month'] = monthlist
enron['Year'] = yearlist
enron['Month-Year'] = enron['Month'].astype(str) + enron['Year'].astype(str)

email_time = enron[['newcontent', 'Month-Year', 'Month', 'Year']]
# getting rid of http addresses and newline newtab symbols
email_time.newcontent = email_time.newcontent.apply(lambda x : re.sub(r'http\S+', '', x))
email_time.newcontent = email_time.newcontent.apply(lambda x : re.sub('\n', '', x))
email_time.newcontent = email_time.newcontent.apply(lambda x : re.sub('\t', '', x))

email_time['sep_email']=email_time.newcontent.str.split(".").to_frame().values.tolist()

valence_list = []
arousal_list = []
for index, row in email_time.iterrows():
    per_email = row['sep_email'][0]
    # separating emails into sentences
    sentences_per_email = []
    if len(per_email) != 0:
        for sentence in per_email:
            if (len(sentence) != 0):
                sentences_per_email.append(sentence)
        
    email_length = len(sentences_per_email)
    
    # !!!!!! must .lower() all strings else sentiment_module does not work!!!
    sentences_per_email = [x.lower() for x in sentences_per_email]
    
    #  Calulate average valence and arousal for each sentence using Dr. Healey's package 
    #  which uses the ANEW and Happiness sentiment dictionaries	
    sentiments_per_email = [sentiment.sentiment(sentence.split(" ")) for sentence 
                            in sentences_per_email]
    
    # aggregate and average each email's valence and arousal
    total_valence = 0
    total_arousal = 0
    for sent_pair in sentiments_per_email:
        total_valence += sent_pair['valence']
        total_arousal += sent_pair['arousal']
    
    valence_per_email = 0 if email_length==0 else total_valence/email_length
    arousal_per_email = 0 if email_length==0 else total_arousal/email_length
    
    valence_list.append(valence_per_email)
    arousal_list.append(arousal_per_email)


enron['valence'] = valence_list
enron['arousal'] = arousal_list

#create a groupby dataframe with total sentiment and total word count over the months
mean_sentiment = enron[['valence', 'arousal', 'Month-Year']].groupby(['Month-Year']).mean()

''' visualizing '''
mean_sentiment = mean_sentiment.reset_index().rename(columns={"Month-Year": "month_year"})
mean_sentiment.month_year = mean_sentiment.month_year.apply(lambda x :  
                                dt.datetime.strptime(x, "%m%Y").date())
mean_sentiment = mean_sentiment.sort_values(by='month_year')

emotions_list = []
emotions_colors = []

# coloring my points with 4 colors depending on their high/low valence/arousal
for index, row in mean_sentiment.iterrows():
    cur_valence = row.valence
    cur_arousal = row.arousal
    
    if cur_valence >=5:
        if cur_arousal >= 5:
            emotions_colors.append(0)
        else: 
            # high valence, low arousal
            emotions_colors.append(1)
        
    else:
        if cur_arousal >= 5:
            emotions_colors.append(3)
        else: 
             # low valence, low arousal
            emotions_colors.append(2)

mean_sentiment["colors"] = emotions_colors

# Sentiment Over Time (monthly)
x = mean_sentiment[1:].drop([12]).month_year     
y = mean_sentiment[1:].drop([12]).arousal
y_v = mean_sentiment[1:].drop([12]).valence
c = mean_sentiment[1:].drop([12]).colors

colormap = np.array(['#fdae61', '#abdda4', '#2b83ba', '#d7191c'])
plt.scatter(x, y_v, s=100, c=colormap[c])

plt.title('Sentiment Over Time')
plt.xlabel('Year-Month')
plt.ylabel('Valence')
plt.xticks(rotation=30)

# constructing an int array corresponding to my months so I can plot the linear line
x_try = np.array([ 1,  3,  5,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
       35, 36, 37, 38, 39, 40, 41, 42, 43, 44])

plt.plot(x, -0.018735*x_try + 4.4557, 'm')

plt.savefig('sentiments_over_time.png', dpi=300, bbox_inches="tight")
plt.show()

""" SENTIMENTS OVER CLUSTERS """
enron = pd.read_csv("sample_with_cluster_labels.csv")

enron0 = enron[enron['cluster_no'] == 0]
enron1 = enron[enron['cluster_no'] == 1]
enron2 = enron[enron['cluster_no'] == 2]
enron3 = enron[enron['cluster_no'] == 3]

def get_average_sentiment(datafr): 

    """ this function gets the average sentiment for each dataframe """
    orig_email = datafr.newcontent.dropna()
    # getting rid of http addresses and newline newtab symbols
    orig_email = orig_email.apply(lambda x : re.sub(r'http\S+', '', x))
    orig_email = orig_email.apply(lambda x : re.sub('\n', '', x))
    orig_email = orig_email.apply(lambda x : re.sub('\t', '', x))

    sep_email=orig_email.str.split(".").to_frame().values.tolist()

    # separating emails into sentences
    sentences = []
    i = 0
    for email in sep_email:
        email = email[0]
        j = 0
        while j < len( email ):
            if len( email[ j ] ) == 0:
                del email[ j ]
            else:
                j += 1
        for sentence in email:
            sentences.append(sentence)
        i += 1

    #  Calulate average valence and arousal for each sentence using Dr. Healey's package 
    #  which uses the ANEW and Happiness sentiment dictionaries	
    sentiments = [sentiment.sentiment(sentence.split(" ")) for sentence in sentences]

    clust0valence = 0
    clust0arousal = 0
    for x in sentiments:
        clust0valence += x['valence']
        clust0arousal += x['arousal']

    #cluster 0 sentiments 
    avgval0 = clust0valence/len(sentiments)
    avgarousal0 = clust0arousal/len(sentiments)
    return [avgval0, avgarousal0]

#create empty dataframe to store average valence and arousal scores
cluster_sentiments = pd.DataFrame(columns=['valence', 'arousal'])

#append stuff from the function into the dataframe 
cluster_sentiments = cluster_sentiments.append(pd.Series(get_average_sentiment(enron0), 
                                                         index = cluster_sentiments.columns), ignore_index=True)
cluster_sentiments = cluster_sentiments.append(pd.Series(get_average_sentiment(enron1), 
                                                         index = cluster_sentiments.columns), ignore_index=True)
cluster_sentiments = cluster_sentiments.append(pd.Series(get_average_sentiment(enron2), 
                                                         index = cluster_sentiments.columns), ignore_index=True)
cluster_sentiments = cluster_sentiments.append(pd.Series(get_average_sentiment(enron3), 
                                                         index = cluster_sentiments.columns), ignore_index=True)

cluster_colors=[]
for index, row in cluster_sentiments.iterrows():
    cur_valence = row.valence
    cur_arousal = row.arousal
    
    if cur_valence >=5:
        if cur_arousal >= 5:
            cluster_colors.append(0)
        else: 
            # high valence, low arousal
            cluster_colors.append(1)
        
    else:
        if cur_arousal >= 5:
            cluster_colors.append(3)
        else: 
             # low valence, low arousal
            cluster_colors.append(2)
            
colormap = np.array(['#fdae61', '#abdda4', '#2b83ba', '#d7191c'])
cluster_colormap = ["#0B5351", "#4E8098", "#00A9A5", "#90C2E7"]

x= list(cluster_sentiments.valence)
y= list(cluster_sentiments.arousal)
n = ["Internal business",
     "External business",
     "Internal logistics",
     "Information exchange"]

plt.scatter(x, y, s=100, c=colormap[cluster_colors])
for i, txt in enumerate(n):
    plt.annotate(txt, (x[i]+.02, y[i]+.02), 
                  fontsize=12, fontweight='bold',
                  bbox={'facecolor': cluster_colormap[i], 'alpha': 0.4, 'pad': 1})
plt.title('Sentiment Over Clusters')
plt.xlabel('Valence')
plt.ylabel('Arousal')
plt.savefig('sentiments_over_clusters.png', bbox_inches="tight", dpi=300)
plt.show()

end_8 = time.time()
print("Part 8 ran for ", end_8 - start_8, " seconds!")
