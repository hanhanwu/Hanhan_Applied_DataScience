# These are the data exploration functions I often use
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

# How to convert str features to "category" while keeping their missing values
le = preprocessing.LabelEncoder()
isnull_mask = raw_df[cat_cols].isnull()

le_cat_df = raw_df[cat_cols]
for le_col in encoding_cols:
    le_cat_df[le_col] = le.fit_transform(le_cat_df[le_col].astype('str'))
    
le_cat_df = le_cat_df.where(~isnull_mask, raw_df[cat_cols])
le_cat_df = le_cat_df.astype('category')
print(le_cat_df.isnull().sum())


# plot 1 line with annotations
def plot_performance_lst(performance_lst, y_label, title):
    plt.figure(figsize=(15,7))
    ax = plt.gca()
    ax.set_ylim([0, 1]) # set y-axis range
    
    x = [i+1 for i in range(len(performance_lst))]
    y = performance_lst
    
    ax.plot(x, y, color='g')
    
    # anchor text to show text in the plot
    anchored_text = AnchoredText(f'Average {y_label} is {round(np.mean(performance_lst), 4)}', loc=3, prop={'size': 12})  # the location code: https://matplotlib.org/3.1.0/api/offsetbox_api.html
    ax.add_artist(anchored_text)  
    
    # annotate y_value along the line
    for i,j in zip(x,y):
        ax.annotate(str(round(j, 4)),xy=(i,j))  
    
    plt.xlabel('epoch #')
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
    
    
# plot multiple dot-line with annotation
def plot_performance_per_threshold(threshold_performance_dct, color_dct, title, fig_size=[15, 7]):
    plt.figure(figsize=(fig_size[0], fig_size[1]))
    ax = plt.gca()
    ax.set_ylim([0, 1]) # set y-axis range
    
    x = threshold_performance_dct['threshold_lst']
    
    for performance_name, performance_lst in threshold_performance_dct.items():
        if (performance_name=='threshold_lst'):
            continue
        ax.plot(x, performance_lst, color=color_dct[performance_name], label=performance_name, marker='*')
        
        # annotate y_value along the line
        for i,j in zip(x, performance_lst):
            ax.annotate(str(j),xy=(i,j)) 
    
    plt.xlabel('Positive Label Threshold')
    plt.ylabel('Performance')
    plt.legend(loc='best')
    plt.title(title)
    plt.show()



# plot numerical features distribution (histogram)
# Usage: https://github.com/hanhanwu/Hanhan_My_Garden/blob/main/code/secret_guest/data_distributions.ipynb
def plot_num_feature_distribution(feature_df, n_rows, n_cols, 
                                  exclude_cols=[], fsize=[40, 20], 
                                  color='g', font_size=20):    
    plt.rcParams.update({'font.size': font_size})

    features = feature_df.columns

    i = 0
    fig=plt.figure(figsize=(fsize[0], fsize[1]))
    for feature in features:
        if feature in exclude_cols:
            continue
        i += 1
        ax=fig.add_subplot(n_rows,n_cols,i) 
        bins = np.linspace(np.nanmin(feature_df[feature]), np.nanmax(feature_df[feature]), 100)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        plt.hist(feature_df[feature], bins, alpha=0.75, 
                 label='median = ' + str(round(np.nanmedian(feature_df[feature]), 3)), 
                 color = color, edgecolor = 'k', range=(bins.min(),bins.max()),
                 weights=np.zeros_like(feature_df[feature]) + 1. / feature_df[feature].shape[0])  # weights here covert count into percentage for y-axis
        plt.legend(loc='best')
        plt.title('Feature: ' + feature)
        plt.xlabel('Feature Values')
        plt.ylabel('Density')
    fig.tight_layout()
    plt.show()


# plot numerical feature distribution in each group
# Usage: https://github.com/hanhanwu/Hanhan_My_Garden/blob/main/code/secret_guest/data_distributions.ipynb
def plot_num_feature_distribution_per_group(grouped_df, group_col, n_rows, n_cols, exclude_col=[], 
                                            figsize=[20, 10], font_scale=1, 
                                            bins=100, palette=["red", "blue", "green"]):
    sns.set(font_scale=font_scale)
    f, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1]))
    
    features = [col for col in grouped_df.columns if col not in exclude_col]
    for ax, feature in zip(axes.flat, features):
        sns.histplot(grouped_df, x=feature,
                     hue=group_col, ax=ax,
                     stat='density', palette=palette, bins=bins)


# plot 1 numerical value dist per group
df.pivot(columns=seg_col, values=value_col)\
            .plot.hist(alpha=0.4, legend=True, bins=100, density=True, color=['green', 'red', 'blue'], figsize=(10,5))


# plot numerical features distribution (KDE)
n_rows = 3
n_cols = 3

i = 0
fig=plt.figure(figsize=(20,10))

for feature in score_df.columns:
    if feature in ['rid', 'prediction_prob', 'prediction', 'score']:
        continue
    i += 1
    ax=fig.add_subplot(n_rows,n_cols,i) 
    bins = np.linspace(0, 1, 100)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    sns.kdeplot(score_df.loc[(score_df['prediction']==1) & (score_df['score'] > 0)][feature].values,
                 color='green', label='pred trustworthy', alpha=0.5)
    sns.kdeplot(score_df.loc[(score_df['prediction']==0) & (score_df['score'] > 0)][feature].values,
                 color='red', label='pred non-trustworthy', alpha=0.5)

    plt.legend(loc='best', fontsize=10)
    plt.title(feature + ' (score > 0)', fontsize=15)
    plt.xlabel('Feature Value', fontsize=15)
    plt.ylabel('Density', fontsize=15)
fig.tight_layout()
plt.show()
    
    
    
# Plot categorical features distribution
# Usage: https://github.com/hanhanwu/Hanhan_My_Garden/blob/main/code/secret_guest/data_distributions.ipynb
def plot_cat_feature_distribution(df, n_rows, n_cols, exclude_cols=[], fsize=[40, 20], font_size=20):
    plt.rcParams.update({'font.size': font_size})
    
    feature_df = df.copy()
    feature_df = feature_df.astype('str').fillna('NA')
    features = feature_df.columns

    i = 0
    fig=plt.figure(figsize=(fsize[0], fsize[1]))
    for feature in features:
        if feature in exclude_cols:
            continue
        i += 1
        ax=fig.add_subplot(n_rows,n_cols,i) 
        axes = plt.gca()
        x_values = feature_df[feature].value_counts().index.values
        x_pos = np.arange(len(x_values))
        y_values = feature_df[feature].value_counts().values
        plt.bar(x_pos, y_values, align='center', alpha=0.6, color='b')
        plt.xticks(x_pos, x_values, rotation=45, ha="right")
        plt.xlabel('Distinct Categorical Value', fontsize=font_size)
        plt.ylabel('Value Count', fontsize=font_size)
        plt.title(feature, fontsize=font_size)

        rects = axes.patches
        total_ct = sum(y_values)

        for v, count in zip(rects, y_values):
            height = v.get_height()
            axes.text(v.get_x() + v.get_width() / 2, height/2, str(round(count*100.0/total_ct, 2))+'%',
                    ha='center', va='bottom')
    fig.tight_layout()
    plt.show()


# Plot categorical features distribution in each group
# Usage: https://github.com/hanhanwu/Hanhan_My_Garden/blob/main/code/secret_guest/data_distributions.ipynb
def plot_cat_feature_distribution_per_group(grouped_df, group_col, n_rows, n_cols,
                                            exclude_cols=[], fsize=[40, 20], 
                                            font_size=20, palette=['green', 'red', 'orange']):
    plt.rcParams.update({'font.size': font_size})
    
    feature_df = grouped_df.copy()
    feature_df = feature_df.astype('str').fillna('NA')
    features = feature_df.columns

    i = 0
    fig=plt.figure(figsize=(fsize[0], fsize[1]))
    for feature in features:
        if feature in exclude_cols:
            continue
        i += 1
        ax=fig.add_subplot(n_rows, n_cols, i) 
        axes = plt.gca()
        
        class_ct_df = feature_df[[feature, group_col]]\
          .groupby([feature, group_col], as_index=False)[group_col]\
          .agg(['count']).reset_index()
        class_ct_df['perct'] = round(class_ct_df['count']*100/len(feature_df), 2)
        
        ax=sns.barplot(x=feature, hue=group_col, y='perct', data=class_ct_df, palette=palette)
        for container in ax.containers:
            ax.bar_label(container)
    fig.tight_layout()
    plt.show()
    


# output more percentile -> easier to see outliers than pandas `describe()` function
def check_percentile(target_df):
    dct = {}
    idx = 0

    for col in target_df.columns:
        if target_df[col].dtypes != 'O' and col != 'label' and col != 'id':
            idx += 1
            dct[idx] = {'feature': col,
                                'min': np.nanpercentile(target_df[col], 0), 'perct1': np.nanpercentile(target_df[col], 1),
                                'perct5': np.nanpercentile(target_df[col], 5), 'perct25': np.nanpercentile(target_df[col], 25),
                                'perct50': np.nanpercentile(target_df[col], 50), 'perct75': np.nanpercentile(target_df[col], 75),
                                'perct90': np.nanpercentile(target_df[col], 90), 'perct99': np.nanpercentile(target_df[col], 99),
                               'perct99.9': np.nanpercentile(target_df[col], 99.9), 'max': np.nanpercentile(target_df[col], 100)}
    dist_df = pd.DataFrame(dct).T
    dist_df = dist_df[['feature', 'min', 'perct1', 'perct5', 'perct25', 'perct50', 'perct75',
                                 'perct90', 'perct99', 'perct99.9', 'max']]
    return dist_df

# Compare the boundary for each label
def boundary_compare(df1, df2, b_name1, b_name2):
    dct1, dct2 = {}, {}
    idx = 0

    for col in df1.columns:
        if col != 'is_trustworthy':
            idx += 1
            dct1[idx] = {'feature': col, b_name1: np.nanpercentile(df1[col], 100)}
            dct2[idx] = {'feature': col, b_name2: np.nanpercentile(df2[col], 100)}
            
    dist_df1 = pd.DataFrame(dct1).T
    dist_df1 = dist_df1[['feature', b_name1]]
    dist_df2 = pd.DataFrame(dct2).T
    dist_df2 = dist_df2[['feature', b_name2]]
    
    boundary_comapre_df = dist_df1.merge(dist_df2, on='feature')
    boundary_comapre_df['smaller_boundary'] = boundary_comapre_df[[b_name1,b_name2]].min(axis=1)
    boundary_comapre_df['boundary_diff'] = abs(boundary_comapre_df[b_name1] - boundary_comapre_df[b_name2])
    boundary_comapre_df['boundary_diff_ratio'] = boundary_comapre_df['boundary_diff']/(boundary_comapre_df['smaller_boundary']+0.0001)
    return boundary_comapre_df


# remove outliers of specific numerical features
def remove_outliers(target_df, low, high, exclude_cols):
    """
    Remove outliers smaller than than the lower percentile value or those larger than the higher percentile value.
    For those features in exclude_cols, not remove outliers.
    
    param: target_df: num_df
    param: low: lower percentile
    param: high: higher percentile
    param: exclude_cols: columns that no need to remove outliers
    
    return: processed num_df
    """
    processed_df = target_df.copy()
    quant_df = target_df.quantile([low, high])
    cols = [col for col in target_df.columns if col not in exclude_cols and col != 'id' and col != 'label']
    quant_df = quant_df[cols]
    quant_df.index = [low, high]

    for col in target_df:
        if col != 'id' and col != 'label':
            continue
        if col not in exclude_cols:
            processed_df.loc[processed_df[col] > quant_df[col].values[1], col] = quant_df[col].values[1]  # giant outliers convert to higher bound value
            processed_df.loc[processed_df[col] < quant_df[col].values[0], col] = quant_df[col].values[0]  # low outliers convert to lower bound value

    return processed_df

    
# replace null with median or mode
def replace_na(feature_df, agg):
    processed_df = feature_df.copy()
    features = feature_df.columns
    
    for feature in features:
        if agg == 'median':
            processed_df[feature] = processed_df[feature].fillna(np.nanmedian(feature_df[feature]))
        elif agg == 'mode':
            processed_df[feature] = processed_df[feature].fillna(processed_df[feature].mode().iloc[0])
    return processed_df


# scatter plot to show linearity relationship
def multi_scatter_plot(n_rows, n_cols, sample_data, y):
    i = 0
    area = np.pi*3
    
    fig=plt.figure(figsize=(40, 15))
    for feature in sample_data.columns:
        i += 1
        ax=fig.add_subplot(n_rows,n_cols,i)

        plt.scatter(sample_data[feature], sample_data[y], s=area, c='g', alpha=0.5)
        plt.title('Feature & Label Relationships', fontsize=30)
        plt.xlabel(feature, fontsize=30)
        plt.ylabel(y, fontsize=30)
    fig.tight_layout()
    plt.show()
    
    
# residual plot of all features vs the label, to find linearity relationship
ridge = Ridge()
visualizer = ResidualsPlot(ridge)

X_train, X_test, y_train, y_test = train_test_split(processed_labeled_num_df.iloc[:, 1:-1], 
                                                    processed_labeled_num_df.iloc[:, 0],
                                                    train_size=0.75, test_size=0.25)
visualizer.fit(X_train, y_train)  # Fit the training data to the model
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.poof()


# Check close to constant features
def get_constant_cols(df, exclude_cols=[], major_rate=0.9999):
    constant_cols = []
    
    for col in df.columns:
        if col in exclude_cols:
            continue
        if (df[col].dropna().nunique() == 1) or (df[col].value_counts().iloc[0]/len(df) >= major_rate):
            constant_cols.append(col)
    return constant_cols


# normalize into 0..1 range
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(df)
norm_df = scaler.transform(df)


# Get cat cols' correlation
def get_cat_correlation(df, method='chi2', p_threshold=0.05):
    corr_lst = []
    cat_cols = df.columns
    
    start = timeit.default_timer()
    for i in range(len(cat_cols)-1):
        for j in range(i+1, len(cat_cols)):
            contingency_table = pd.crosstab(df[cat_cols[i]].fillna('NA'), df[cat_cols[j]].fillna('NA'))  # Fill NA with 'NA' so that cramerv won't throw error when a feature's missing rate is high
            
            if method == 'chi2':
                test_static, p, dof, expected_feq = chi2_contingency(contingency_table)
                corr_result = 'correlated' if p <= p_threshold else 'not correlated'
            elif method == 'cramer':
                corr_result = contingency.association(contingency_table, method='cramer')
            else:
                print('"method" has to be "chi2" or "cramer"')
                return None
                
            corr_lst.append([cat_cols[i], cat_cols[j], corr_result])
            
    corr_df = pd.DataFrame(corr_lst, columns=['var1', 'var2', method])
    print(round(timeit.default_timer() - start, 4), "seconds")
    return corr_df

# Get 2D highly correlated num features 
## NOTE: Please normalize the feature before doing this, otherwise features with higher values tend to show higher correlation
def get_num_correlated_features(df, corr_method='pearson', threshold=0.9, exclude_cols=[]):
    """
    Find correlated feature pairs. Higher threshold, higher correlation.
    :param data: features input, pandas dataframe
    :param corr_method: ‘pearson’, ‘kendall’, ‘spearman’
    :param threshold: the correlation threshold decides which feature pairs are highly correlated, abs value between 0..1 range
    """
    corr_dct = {}
    
    corr_matrix = df.corr(method=corr_method).abs()  # create correlation matrix
    upper_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))  # upper triangle
    
    for col in df.columns:
        if col in exclude_cols:
            continue
        corr_lst = list(upper_matrix[col].where(upper_matrix[col] >= threshold).dropna().index)
        if len(corr_lst) > 0:
            corr_dct.setdefault(col, [])
            corr_dct[col].extend(corr_lst)
    
    drop_lst = [column for column in upper_matrix.columns if any(upper_matrix[column] > threshold)]

    return corr_dct, drop_lst

# Get 3D+ highly correlated num features, to deal with multicollinearity issue
## Normally when VIF is between 5 and 10, there could be multicollineary issue of the feature. 
## When VIF > 10, it's too high and the feature should be removed.
## NOTE: Please normalize the feature before doing this, otherwise features with higher values tend to show higher correlation
## NOTE: deal with nan before using this method, otherwise SVD won't converge
from statsmodels.stats.outliers_influence import variance_inflation_factor
def get_multicollineary_features(df, vif_threshold=10, exclude_cols=[]):
    cols = [col for col in df.columns if col not in exclude_cols]
    feature_df = df[cols].fillna(-999)
    
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(feature_df.values, i) for i in range(feature_df.shape[1])]
    vif["features"] = feature_df.columns  # This will get VIF for each feature. To drop individual feature, start from the one with highest VIF
    drop_lst = vif.loc[vif['VIF Factor']>vif_threshold]['features'].values
    return vif, list(drop_lst)

# Correlation: categorical vs categorical (chi2); numerical vs categorical (f_classif, namely ANOVA)
def dependency_chosen_features(feature_df, label_col, pvalue_threshold, feature_type):
    if feature_type == 'num':
        _, pvalue_lst = f_classif(feature_df, feature_df[label_col])
    else:
        _, pvalue_lst = chi2(feature_df, feature_df[label_col])
    
    features = feature_df.columns
    
    high_dependency_features = []
    
    for i in range(len(features)):
        if features[i] != label_col and pvalue_lst[i] <= pvalue_threshold:
            high_dependency_features.append(features[i])
    return high_dependency_features

# Show kernel density distribution, calculate K-L score to show difference between the 2 probability distributions
## Lower K-L score, the more similarity between 2 probability distributions
## Check what is probability distribution ("probability" is the probability of each category): https://machinelearningmastery.com/divergence-between-probability-distributions/
## wasserstein_distance measures the distance between 2 distributions
import numpy as np
from scipy.stats import gaussian_kde
from scipy.stats import wasserstein_distance
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

def calc_kl_score(x1, x2):
    """
    Fits a gaussian distribution to x1 and x2 and calculates the K-L score
    between x1 and x2.
     
    :param x1: list. Contains float / integers representing a feature.
    :param x2: list. Contains float / integers representing a different feature.
    :return float
    """
    positions = np.linspace(0,1,1000) # (Optional) If plotting, you can increase this number to generate a smoother KDE plot
    kernel1 = gaussian_kde(x1)
    values1 = kernel1(positions)
    kernel2 = gaussian_kde(x2)
    values2 = kernel2(positions)
    return entropy(values1,values2)

# wasserstein_distance works better than K-L, especially when the support of 2 distributions are different
## such as one distribution has much fatter tail than the other
def plot_dist_diff(df, df1, df2, n_rows, n_cols, exclude_cols, label1, label2):
    dist_diff_dct = {}
    
    features = df.columns
    print('Number of features: ' + str(len(features)))
    
    i = 0
    fig=plt.figure(figsize=(30,40))
    
    for feature in features:
        if feature in exclude_cols:
            continue
        i += 1
        ax=fig.add_subplot(n_rows,n_cols,i) 
        bins = np.linspace(min(df[feature]), max(df[feature]), 100)
        
        v1 = df1.loc[~df1[feature].isnull()][feature]
        v2 = df2.loc[~df2[feature].isnull()][feature]
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        sns.distplot(v1, color='green', label=label1)
        sns.distplot(v2, color='purple', label=label2)
        
        kl_score = calc_kl_score(v1, v2)
        w_dist = wasserstein_distance(v1, v2) # wasserstein_distance works better than K-L, especially when the support of 2 distributions are different
        dist_diff_dct[feature] = {'w_dist':w_dist, 'kl_score':kl_score}
        
        plt.legend(loc='best', fontsize=20)
        plt.title('Feature: ' + feature + ', Divergence:' + str(round(w_dist, 8)), fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('Feature Values', fontsize=18)
        plt.ylabel('Percentage', fontsize=18)
        
    fig.tight_layout()
    plt.show()
    
    return dist_diff_dct
