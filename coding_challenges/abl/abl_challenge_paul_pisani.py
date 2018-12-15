
# ##################################################
# ABL Data Science Challenge
# Paul Pisani
# December 2018
#
# This script installs some core python libraries for data handling / viz, 
# creates some helper functions to assist with both data pipelining and analysis, 
# then runs through a working example with the groups / scores / students files
# ##################################################




# install / import required packages

# !pip3 install pandas
# !pip3 install numpy
# !pip3 install matplotlib
# !pip3 install seaborn

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import seaborn as sns


# [DATA EXPLORATION / ANALYSIS] define helper function to create flexible stacked bar chart
def stacked_bar_chart_by_category(df, x, y, pcnt_of_total=False):
    
    df = df.groupby([x,y]).size().unstack()
   
    if pcnt_of_total == True: 
        df = df.divide(df.sum(axis=1),axis=0)
    
    fig, ax = plt.subplots()
    rows = df.index.values
    bars = {}
    label_values = []
    label_names = []
    
    for i in range(len(df.columns.values)):
        bar_bottom = sum(bars.values())
        bars[df.columns.values[i]] = df.iloc[:,i].fillna(0)
        grouptype_i = plt.bar(rows.astype(str), bars[df.columns.values[i]], bottom=bar_bottom)
        label_values.append(grouptype_i[0])
        label_names.append(df.iloc[:,i].fillna(0).name)
        if len(label_values) < 20:
            plt.legend(label_values, label_names, loc=(1.04,0))
            
    plt.xlabel(x)
    #ax.set_xticks(ax.get_xticks()[::2])
    plt.xticks(rotation=60)
    plt.ylabel('Distribution of ' + y)
    plt.title('Distribution of ' + y + ' by ' + x)
    plt.show()
    plt.clf()


# [DATA EXPLORATION / ANALYSIS] define helper function to create flexible histogram
def histogram_by_category_with_filter(df, filter_var, filter_value, split_var, hist_var, split_value=None,
                                     min=-40, max=40, splits=41, separate_plots=False):

    plt.subplots(1)
    bins = np.linspace(min, max, num=splits) ## 80/5 = 16+1 = 17
    df_filter = df[df[filter_var] == filter_value]
    split_values = df[split_var].unique()
    if split_value:
        split_values = split_value

    for i in df[split_var].unique():
        df_temp = df_filter[df_filter[split_var] == i][hist_var]
        plt.hist(df_temp, bins, alpha=0.40, label=str(i), density=True, rwidth = 10, edgecolor="black")
        plt.legend(loc=(1.04,0))
        if separate_plots == True:
            #plt.xlabel(hist_var)
            plt.xticks(rotation=60)
            plt.ylabel('Frequency / Density')
            plt.title(hist_var + ' for ' + split_var + ' = ' + str(i) 
                      + ' (' + filter_var + '=' + str(filter_value) + ')')
            plt.subplots()
            plt.clf()

    if separate_plots == False:
        #plt.xlabel(hist_var)
        #ax.set_xticks(ax.get_xticks()[::2])
        plt.xticks(rotation=60)
        plt.ylabel('Frequency / Density')
        plt.title(hist_var + ' for ' + split_var + ' = ' + str(i) 
                      + ' (' + filter_var + '=' + str(filter_value) + ')')
        plt.subplots()
        plt.clf()


# [DATA LOADING / CLEANING] define helper function to check contents of new dataframes
def check_and_explore(df):
    
    print('\n\n' + df.name + ': First 5 Rows')
    print(df.head())
    
    print('\n' + df.name + ': Dataframe Info (Entries, Nulls, Data Types)')
    print(df.info())
    
    print('\n' + df.name + ': Dataframe Description By Column')
    print(df.describe(include='all'))
    
    for i in range(len(df.columns)): 
        print('\n' + df.name + ': Value Counts For', df.columns[i], "Column")
        print(df.groupby(df.columns[i]).size())
        
    for col1 in df.columns:
        for col2 in df.columns:
            if all([col1 != col2, df[col1].nunique() < 15, df[col2].nunique() < 15]): 
                stacked_bar_chart_by_category(df, col1, col2)


# [DATA EXPLORATION / ANALYSIS] define helper function to create flexible ridge chart
def ridge_plot_by_category(df, x, y, split=False, split_val=False):

    df_temp = df
    
    if split and split_val:
        df_temp = df[df[split] == split_val]
        
    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(df_temp, row=y, hue=y, aspect=12, height=1, palette=pal)

    g.map(sns.kdeplot, x, clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
    g.map(sns.kdeplot, x, clip_on=False, color="w", lw=2, bw=.2)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)


    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)

    g.map(label, x)

    g.fig.subplots_adjust(hspace=-.15)

    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)


# [DATA LOADING / CLEANING] define helper function to check for null values via assert statements
def find_null_values(df): 
    
    try:
        assert df.isnull().values.sum() == 0
    except AssertionError:
        print('\nDataframe: ' + df.name)
        print(df[df.isnull().any(axis=1)])


# [DATA LOADING / CLEANING] define helper function #5 - fill in missing rows for students / dates
def fill_missing_dates(df, entity_field, date_field, fill_field, fill_value):
    
    new_rows = []
    
    for i in range(1, df[entity_field].nunique()+1): 
        for j in range(1, df[date_field].nunique()+1): 
            if len(scores_raw[(df[entity_field] == i) & (df[date_field] == j)]) == 0:
                new_row = {entity_field: [i], 
                           date_field: [j],
                           fill_field: [fill_value]}
                new_row_df = pd.DataFrame(data=new_row)
                new_rows.append(new_row_df)
    
    if len(new_rows) > 0: 
        str = df.name
        df = df.append(pd.concat(new_rows), ignore_index=False, sort=True)
        df.sort_values([entity_field, date_field], inplace=True)
        df.reset_index(drop=True)
        df.name = str
        return df
    
    else:
        print('No values to add!')
        return df


# [DATA LOADING / CLEANING] define helper function to type cols (numeric and categorical / string values)
def type_fields(df, unique_value_cutoff=15, range_cutoff=40):
    
    for col in df.columns:

        unique_values = df[col].nunique()

        try: 
            df[col] = df[col].astype('int64')
            print('Successfully cast column ' + col + ' as int64!')
            value_range = max(df[col].values)-min(df[col].values)
            sequential_values_yn = pd.Series(df[col].unique()).sort_values().diff().dropna().eq(1).all()
            if value_range > range_cutoff and unique_values > unique_value_cutoff and sequential_values_yn == False:
                print('-> Consider creating bucketed version of ' + col +
                      ' - many unique / non-sequential vals and large range!')     

        except ValueError:
            try: 
                df[col] = df[col].astype('float16')
                value_range = max(df[col].values)-min(df[col].values)
                sequential_values_yn = pd.Series(df[col].unique()).sort_values().diff().dropna().eq(1).all()
                print('Successfully cast column ' + col + ' as float64!')
                if max(df[col].values)-min(df[col].values) > range_cutoff and df[col].nunique() > unique_value_cutoff and pd.Series(df[col].unique()).diff().dropna().eq(1).all() == False:
                    print('-> Consider creating bucketed version of ' + col + ' - many unique / non-sequential values and large range!')     

            except ValueError:
                try:
                    if unique_values <= 15:
                        df[col] = df[col].astype('category')
                        print('Successfully cast column ' + col + ' as category!')

                    else:
                        df[col] = df[col].astype('string')
                        print('Successfully cast column ' + col + ' as string!')

                except ValueError:
                    'Not able to successfully type col ' + col + ' - should take closer look!'
    
    return df


 # read in raw data files as dataframes
groups_raw = pd.read_csv('groups.csv')
groups_raw.name = 'groups_raw'

students_raw = pd.read_csv('students.csv')
students_raw.name = 'students_raw'

scores_raw = pd.read_csv('scores.csv')
scores_raw.name = 'scores_raw'


# explore new dataframes (cols, size, type, relationships) -- WARNING: long output
check_and_explore(groups_raw)
check_and_explore(students_raw)
check_and_explore(scores_raw)


# find and address null values where necessary
find_null_values(groups_raw)
find_null_values(students_raw)
find_null_values(scores_raw)


# fix unexpected nulls / potentially erroneous values 
students_raw.loc[students_raw['Gender'].isnull(), 'Gender'] = 'M' # all other Martians are M
students_raw.loc[students_raw['Gender'] == 'Fe', 'Gender'] = 'F' # all other Venutians are F, likely typo / miscoded


# fill missing dates for included students (test 8 for one teacher / two classrooms of students)
scores_raw = fill_missing_dates(scores_raw, 'StudentId', 'QuizNumber', 'Score', np.nan)


# make sure all fields can be successfully typed
groups_typed = type_fields(groups_raw)
groups_typed.name = 'groups_typed'

students_typed = type_fields(students_raw)
students_typed.name = 'students_typed'

scores_typed = type_fields(scores_raw)
scores_typed.name = 'scores_typed'


# create bucketed category version of Score field
bins = np.linspace(20, 100, num=9)
scores_typed['ScoreBucket'] = pd.cut(scores_raw["Score"], bins)
scores_typed['ScoreBucket'] = scores_typed['ScoreBucket'].astype('category')


# create score diff columns derived from Score field
scores_typed.sort_values(['StudentId', 'QuizNumber'], inplace=True)

for i in range(1, max(scores_typed['QuizNumber'])):
    field_name = 'ScoreDiff_' + str(i)
    scores_typed[field_name] = scores_typed['Score'].diff(i)
    for j in range(i+1):
        scores_typed.loc[scores_typed['QuizNumber'] == j, field_name] = np.nan
        scores_typed[field_name].astype('float16')


# before joining, rerun script to explore typed dataframes
check_and_explore(groups_raw)
check_and_explore(students_raw)
check_and_explore(scores_raw)


# join tables together - first join scores to students on student id and then join to groups on quiz and class
df1 = pd.merge(left=scores_typed, right=students_typed, left_on='StudentId', right_on='StudentId')
combined_view = pd.merge(left=df1, right=groups_typed, left_on=['ClassId', 'QuizNumber'], right_on=['ClassId','QuizNumber'])
combined_view.name = 'combined_view'
combined_view.sort_values(['StudentId', 'QuizNumber'], inplace=True)
combined_view.reset_index(drop=True)


# create score rank from Score and Class field (top third = High, middle third = Mid, bottom third = Low)
combined_view['ScoreRank'] = combined_view.groupby(['QuizNumber','ClassId'])['Score'].rank(ascending=False)
combined_view['ScoreRank'] = combined_view['ScoreRank'].astype('float16')

bins = np.linspace(0, 30, num=4)
combined_view['ScoreRank_Bucket'] = pd.cut(combined_view['ScoreRank'], bins)
combined_view['ScoreRank_Bucket'] = combined_view['ScoreRank_Bucket'].astype('category')

combined_view.sort_values(['ClassId', 'QuizNumber', 'Score'], inplace=True)


# check and explore new combined view as we did previously 
combined_view = combined_view[combined_view['Score'] > 0]
combined_view.name = 'combined_view'
check_and_explore(combined_view)


# create histograms to chart change in score (t-1) or raw score by different vars (and split by group type)
# sns.set()
# for i in combined_view['QuizNumber'].unique():
#     histogram_by_category_with_filter(combined_view, 'QuizNumber', i, 'GroupType', 'ScoreDiff_1')
# for i in combined_view['ClassId'].unique():
#     histogram_by_category_with_filter(combined_view, 'ClassId', i, 'GroupType', 'ScoreDiff_1')
# for i in combined_view['Teacher'].unique():
#     histogram_by_category_with_filter(combined_view, 'Teacher', i, 'GroupType', 'ScoreDiff_1')      
# for i in combined_view['Race'].unique():
#     histogram_by_category_with_filter(combined_view, 'Race', i, 'GroupType', 'ScoreDiff_1')    
# for i in combined_view['Gender'].unique():
#     histogram_by_category_with_filter(combined_view, 'Gender', i, 'GroupType', 'ScoreDiff_1')   
# for i in combined_view['ScoreRank_Bucket'].unique():
#     histogram_by_category_with_filter(combined_view, 'ScoreRank_Bucket', i, 'GroupType', 'ScoreDiff_1')
# for i in combined_view['QuizNumber'].unique():
#     histogram_by_category_with_filter(combined_view, 'QuizNumber', i, 'Race', 'Score', 
#                                     min=20, max = 100, splits=41)
# for i in combined_view['ClassId'].unique():
#     histogram_by_category_with_filter(combined_view, 'ClassId', i, 'Race', 'Score',
#                                     min=20, max = 100, splits=41)
# for i in combined_view['QuizNumber'].unique():
#     histogram_by_category_with_filter(combined_view, 'QuizNumber', i, 'Gender', 'Score',
#                                     min=20, max = 100, splits=41)
# for i in combined_view['QuizNumber'].unique():
#     histogram_by_category_with_filter(combined_view, 'QuizNumber', i, 'Gender', 'Score',
#                                     min=20, max = 100, splits=41)
# for i in combined_view['ClassId'].unique():
#     histogram_by_category_with_filter(combined_view, 'ClassId', i, 'Teacher', 'Score',
#                                     min=20, max = 100, splits=41)
# for i in combined_view['QuizNumber'].unique():
#     histogram_by_category_with_filter(combined_view, 'QuizNumber', i, 'Teacher', 'Score',
#                                     min=20, max = 100, splits=41)


## create summary boxplots for score diff 1  across key splits

field_list = ['Teacher', 'Gender', 'Race', 'ScoreRank_Bucket']
for i in field_list: 
    f, ax = plt.subplots(figsize=(7, 6))
    sns.boxplot(x="ScoreDiff_1", y=i, data=combined_view, palette="vlag")
    ax.set_title('ScoreDiff_1 by ' + i)
    ax.xaxis.grid(True)

## create summary boxplots for score diff 1-9
field_list = ['ScoreDiff_' + str(i) for i in range(1,10)] 
for i in field_list: 
    f, ax = plt.subplots(figsize=(7, 6))
    sns.boxplot(x=i, y='GroupType', data=combined_view, palette="vlag")
    ax.set_title(i + ' by GroupType')
    ax.xaxis.grid(True)


# create line plots for raw score across key segmentations
sns.lmplot(x="QuizNumber", y="Score", data=combined_view, height=4, aspect=2)
field_list = ['Race', 'Gender', 'Teacher', 'ClassId']
for i in field_list:
    sns.lmplot(x="QuizNumber", y="Score", data=combined_view, 
               hue=i, height=4, aspect=2)


# create additional boxplots for score diff split by group type + additional var
field_list = ['QuizNumber', 'ClassId', 'Teacher', 'Gender', 'Race', 'ScoreRank_Bucket']
for i in field_list: 
    sns.catplot(x=i, y='ScoreDiff_1', col='GroupType', 
                    data=combined_view, palette=sns.color_palette("muted"), 
                    col_wrap = 4, kind="box", height=4, aspect=0.8)


# create ridge plot to see relative distributions across group types
ridge_plot_by_category(combined_view, 'ScoreDiff_1', 'GroupType')


# create heatmap tables for mean score by quiz number and class 
for i in combined_view['ClassId'].unique(): 
    combined_view_temp = combined_view[combined_view['ClassId'] == i]
    pivot_table = combined_view_temp[['ClassId', 'QuizNumber', 'GroupType', 'Score']].groupby(
        ['ClassId', 'QuizNumber', 'GroupType']).agg([np.mean]) # np.std, np.min, np.max
    f, ax = plt.subplots(figsize=(1, 4))
    ax.set_title('Mean Score By QuizNumber And \nGroupType for ClassId = ' + str(i))
    ax.set_ylabel('Mean Score')
    sns.heatmap(pivot_table, annot=True, linewidths=1, ax=ax, cmap='RdYlGn')


# create line plot for avg score by race-gender combos
for i in combined_view['Race'].unique(): 

    df_split = combined_view[combined_view['Race'] == i]

    pivot_table = df_split[['QuizNumber', 'Gender', 'Score']].groupby(
        ['QuizNumber', 'Gender'], as_index=False).agg([np.mean]) # np.std, np.min, np.max

    for j in pivot_table.index.get_level_values('Gender').unique():
        df_temp = pivot_table.unstack(level=1).xs(j, level='Gender', axis=1)
        fig = plt.plot(df_temp, marker = 'o', label=i+'-'+j)

plt.ylabel('Average Score')
#plt.ylim(0,100)
plt.title('Average Score By Quiz Week And Gender')
plt.legend(loc=(1.04,0))
fig = plt.gcf()
fig.set_size_inches(18.5, 6.5)


# create line plot for avg score by teacher
temp = combined_view
temp['Score'] = temp['Score'].astype('float64')
pivot_table = combined_view[['QuizNumber', 'Teacher', 'Score']].groupby(
    ['QuizNumber', 'Teacher'], as_index=False).agg([np.mean]) # np.std, np.min, np.max

for i in pivot_table.index.get_level_values('Teacher').unique():
    df_temp = pivot_table.unstack(level=1).xs(i, level='Teacher', axis=1)
    plt.plot(df_temp, marker = 'o', label=i)

plt.ylabel('Average Score')
#plt.ylim(0,100)
plt.title('Average Score By Quiz Week And Teacher')
plt.legend(loc=(1.04,0))
fig = plt.gcf()
fig.set_size_inches(18.5, 6.5)



















