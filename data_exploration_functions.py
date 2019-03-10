# These are the data exploration functions I often use

# plot numerical features distribution
def show_num_feature_distribution(feature_df, n_rows, n_cols):
    plt.rcParams.update({'font.size': 20})

    features = feature_df.columns

    i = 0
    fig=plt.figure(figsize=(40, 15))
    for feature in features:
        if feature == 'id' or feature == 'label':
            continue
        i += 1
        ax=fig.add_subplot(n_rows,n_cols,i) 
        bins = np.linspace(np.nanmin(feature_df[feature]), np.nanmax(feature_df[feature]), 100)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        plt.hist(feature_df[feature], bins, alpha=0.75, label='median = ' + str(round(np.median(feature_df[feature]), 3)), 
                 color = 'g', edgecolor = 'k', range=(bins.min(),bins.max()),
                 weights=np.zeros_like(feature_df[feature]) + 1. / feature_df[feature].shape[0])
        plt.legend(loc='best')
        plt.title('Feature: ' + feature)
        plt.xlabel('Feature Values')
        plt.ylabel('Percentage')
    fig.tight_layout()
    plt.show()
    
    
# Plot categorical features distribution
def plot_cat_feature_distribution(feature_df, n_rows, n_cols):
    plt.rcParams.update({'font.size': 20})

    features = feature_df.columns

    i = 0
    fig=plt.figure(figsize=(40, 20))
    for feature in features:
        if feature == 'rid' or feature == 'isfraud':
            continue
        i += 1
        ax=fig.add_subplot(n_rows,n_cols,i) 
        axes = plt.gca()

        x_values = feature_df[feature].value_counts().index.values
        x_pos = np.arange(len(x_values))
        y_values = feature_df[feature].value_counts().values
        plt.bar(x_pos, y_values, align='center', alpha=0.6)
        plt.xticks(x_pos, x_values)
        plt.xlabel('Distinct Categorical Value')
        plt.ylabel('Percentage')
        plt.title(feature)

        rects = axes.patches
        total_ct = sum(y_values)

        for v, count in zip(rects, y_values):
            height = v.get_height()
            axes.text(v.get_x() + v.get_width() / 2, height/2, str(round(count*100.0/total_ct, 2))+'%',
                    ha='center', va='bottom')
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


# plot numerical feature distribution for each class
def show_num_distribution_has_label(labeled_feature_df, label_col, n_rows, n_cols):
    plt.rcParams.update({'font.size': 20})

    features = [col for col in labeled_feature_df.columns if col != label_col]
    fraud_df = labeled_feature_df.loc[labeled_feature_df[label_col]==1]
    nonfraud_df = labeled_feature_df.loc[labeled_feature_df[label_col]==0]

    i = 0
    fig=plt.figure(figsize=(40, 15))
    for feature in features:
        if feature == 'rid' or feature == 'isfraud':
            continue
        i += 1
        ax=fig.add_subplot(n_rows,n_cols,i) 
        bins = np.linspace(np.nanmin(labeled_feature_df[feature]), np.nanmax(labeled_feature_df[feature]), 100)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        plt.hist(fraud_df[feature], bins, alpha=0.75, label='fraud', 
                 color = 'r', edgecolor = 'k', range=(bins.min(),bins.max()),
                 weights=np.zeros_like(fraud_df[feature]) + 1. / fraud_df[feature].shape[0])
        plt.hist(nonfraud_df[feature], bins, alpha=0.5, label='nonfraud', 
                 color = 'b', edgecolor = 'k', range=(bins.min(),bins.max()),
                 weights=np.zeros_like(nonfraud_df[feature]) + 1. / nonfraud_df[feature].shape[0])
        plt.legend(loc='best')
        plt.title('Feature: ' + feature)
        plt.xlabel('Feature Values')
        plt.ylabel('Percentage')
    fig.tight_layout()
    plt.show()

    
  # plot categorical feature distribution for each class
  def plot_cat_feature_distribution_with_label(labeled_feature_df, label_col, n_rows, n_cols):
    plt.rcParams.update({'font.size': 20})

    features = labeled_feature_df.columns
    fraud_df = labeled_feature_df.loc[labeled_feature_df[label_col]==1]
    nonfraud_df = labeled_feature_df.loc[labeled_feature_df[label_col]==0]

    i = 0
    fig=plt.figure(figsize=(40, 20))
    for feature in features:
        if feature == 'rid' or feature == 'isfraud':
            continue
        i += 1
        ax=fig.add_subplot(n_rows,n_cols,i) 
        axes = plt.gca()
        width = 0.2

        fraud_x_values = fraud_df[feature].value_counts().index.values
        fraud_x_pos = np.arange(len(fraud_x_values))
        fraud_y_values = fraud_df[feature].value_counts().values
        
        nonfraud_x_values = nonfraud_df[feature].value_counts().index.values
        nonfraud_x_pos = np.arange(len(nonfraud_x_values))
        nonfraud_y_values = nonfraud_df[feature].value_counts().values
        
        plt.bar(nonfraud_x_pos, nonfraud_y_values, width, align='center', alpha=0.6, color='green', label='nonfraud')
        plt.bar(fraud_x_pos+width, fraud_y_values, width, align='center', alpha=0.6, color='red', label='fraud')
        plt.xticks(nonfraud_x_pos+width/2, nonfraud_x_values)
        plt.xlabel('Distinct Categorical Value')
        plt.ylabel('Percentage')
        plt.title(feature)

        rects = axes.patches
        nonfraud_total_ct = sum(nonfraud_y_values)
        fraud_total_ct = sum(fraud_y_values)

        for v, count in zip(rects, nonfraud_y_values):
            height = v.get_height()
            axes.text(v.get_x() + v.get_width() / 2, height/2, str(round(count*100.0/nonfraud_total_ct, 2))+'%',
                    ha='center', va='bottom')
        
        for v, count in zip(rects, fraud_y_values):
            height = v.get_height()
            axes.text(v.get_x() + v.get_width()*1.5, height/(nonfraud_total_ct/fraud_total_ct), str(round(count*100.0/fraud_total_ct, 2))+'%',
                    ha='center', va='bottom')
            
        ax.legend()
    fig.tight_layout()
    plt.show()
