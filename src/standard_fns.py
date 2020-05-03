# Function to get all the prefixes from an array of cols, given suffixes
# eg, in this case we have all columns ending in _mean, _worst or _se
# This function removes those and returns array with the actual features.
# eg [radius, mean, concatvity...] etc
# This is helpful in data exploration phase.
def get_prefixes(cols,suffixes):
    result = []
    for col in cols:
        for suffix in suffixes:
            if suffix in col:
                result.append(col.replace(suffix,''))
    
    return(pd.unique(result))

# Similar to above, given a prefix (eg radius), returns an array with all the columns
# that contain this prefix (eg [radius_mean, radius_worst, radius_se])
# Also useful in exploratory phase.
def get_like_cols(prefix,all_cols):
    result = []
    for col in all_cols:
        if prefix in col:
            result.append(col)
    return(result)

# Generic function to create a boxplot for specified cols.
# Useful when used in conjunction with one of the 2 fns above.
def create_boxplot(df,cols):
    #for col in cols:
    df.boxplot(column = cols)

# Generic function to create a boxplot for all columns with like prefix 
# (eg all radius columns)
def get_boxplot_for_like_cols(prefix,df):
    all_cols = df.columns
    cols = get_like_cols(prefix,all_cols)
    create_boxplot(df,cols)

# Function to calculate VIF for 2 columns in a dataframe.
# If r_value = 1, set VIF to be 1
def get_VIF_simple(df,cols):
    proj0 = np.asarray(df[cols[0]])
    proj1 = np.asarray(df[cols[1]])
    slope, intercept, r_value, p_value, std_err = stats.linregress(proj0,proj1)
    if(r_value==1):
        VIF = 1
    else:
        VIF = 1/(1-r_value)
    return(VIF)

# Function to calcaulate VIF for 2 arrays
def get_VIF_simple_from_arrays(array1,array2):
    slope, intercept, r_value, p_value, std_err = stats.linregress(array1,array2)
    VIF = 1/(1-r_value)
    return(VIF)

# Function to create VIF matric for a given dataframe (all cols vs all other cols)
def get_VIF(df):
    cols = df.columns
    size = cols.size
    mat = np.zeros((size,size))

    for i in range(size):
        for j in range(size):
            #print([cols[i],cols[j]])
            mat[i,j] = get_VIF_simple(df,np.asarray([cols[i],cols[j]]))
    return mat

# Function to remove the column which generates max VIF (VIF must also be above
# cutoff threshold)
def remove_max_VIF_col(VIF,df,thresh):
    maxval = np.amax(VIF)
    if(maxval >= thresh):
        loc = np.where(VIF == maxval)[0][1]
        newdf = df.drop([df.columns.tolist()[loc]],axis=1)
    else:
        newdf = df
    return(newdf)

# Wrapper function which allows us to drop multicollinear features by running
# previous function in a loop until no columns have VIF > thresh.
# At each stage, we recalculate the VIF before removing columns.
def drop_multicollinear_features(thresh,df):
    dropped = True    
    while dropped:
        dropped = False
        VIFs = get_VIF(df)
        maxval = np.amax(VIFs)
        if maxval > thresh:
            df = remove_max_VIF_col(VIFs,df,thresh)
            dropped = True
    return(df)

# Function which calculates the point biserial correlation 
# (between a catergorical and continous variable), to see
# the correlation between target variable and other (continues)
# variables. Thresh input allows it to filter out cols which
# have correlation below the specified threshold.
def get_PBS_corr_from_cols(df,target_col,cont_cols,thresh = 0 ):
    res = dict()
    for col in cont_cols:
        correlation, pval = pointbiserialr(df[target_col],df[col])
        res[col] = correlation
    inter = pd.Series(res, name='corr').reset_index()
    inter['abs_corr'] = pd.DataFrame.abs(inter['corr'])
    inter = inter[inter['abs_corr'] > thresh ]
    fin_res = inter.sort_values('corr',ascending=False)
    fin_res = fin_res.drop(columns = ['abs_corr'])
    return(fin_res)