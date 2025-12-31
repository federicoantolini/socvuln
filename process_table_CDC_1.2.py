import pandas as pd
import numpy as np
import os, time

'''
codes associated to different modes of construction and transformation
of a generic index. Not all modes are relevant for CDC SVI
'''
ModelStructure = {'Deductive':1, 'Hierarchical':2, 'Inductive':3}
Variables = {'Base':4, 'Evacuation':5, 'Resources':6, 'Alternative':7}
Undercount = {'Regular':8, 'Undercount':9}
Scale = {'Blockgroup':10, 'Tract':11}
Transforming = {'None':12, 'Area':13, 'Population':14, 'Custom':15}
Normalizing = {"None" : 16,
               "MinMax0to1" : 17,
               "MinMax1to2" : 18,
               "Zscore" : 19,
               "MaxValue" : 20}
PCASampleSize = {'Normal':21, 'Expanded':22}
PCARotation = {'Varimax':23, 'Unrotated':24}
PCAFactorRetention = {'Kaiser':25, 'Parallel':26}
Weighting = {'Equal':27, 'Expert':28, 'Random':29, 'DEA':30}
Aggregation = {'Additive':31, 'Geometric':32}
DEM = {'Level1':33, 'Level2':34}
Inventory = {'Level1':35, 'Level2':36}
FloodGrid = {'Leve11':37, 'Level2':38}
DepthDamage = {'Level1':39, 'Level2':40}


'''
Themes, or pillars
Variables contribute to different themes, as defined in themes dictionary
Acronyms are used for simplicity
'''
themes_list = ['socioeconomic','household','minorLang','housingTransp']
themes = {'socioeconomic':['POVERTY','UNEMPLOYED','PERCAP','noHSDIPL'],
          'household':['65+','17-','DISABLE','SINGLPAR'],
          'minorLang':['MINORITY','ENGLTW'],
          'housingTransp':['10+UNITS','MOBILEHOMES','HOUSCROWD','noVEHICLE','GROUPQRTRS']
          }
themesAcr = {'socioeconomic':'SE',
          'household':'HH',
          'minorLang':'ML',
          'housingTransp':'HT'
          }



##var_codes = [['POVERTY', ['B17001_002']],
##             ['UNEMPLOYED', ['B23001_008','B23001_015','B23001_022','B23001_029','B23001_036','B23001_043','B23001_050',
##                             'B23001_057','B23001_064','B23001_071','B23001_076','B23001_081','B23001_086',
##                             'B23001_094','B23001_101','B23001_108','B23001_115','B23001_122','B23001_129','B23001_136',
##                             'B23001_143','B23001_150','B23001_157','B23001_162','B23001_167','B23001_172']],
##             #['aggINCOME', ['B19313_001']],
##             ['noHSDIPL', ['B15002_003','B15002_004','B15002_005','B15002_006','B15002_007','B15002_008','B15002_009','B15002_010',
##                           'B15002_020','B15002_021','B15002_022','B15002_023','B15002_024','B15002_025','B15002_026','B15002_027']],
##             ['65+', ['B01001_020', 'B01001_021', 'B01001_022', 'B01001_023', 'B01001_024', 'B01001_025',
##                       'B01001_044', 'B01001_045', 'B01001_046', 'B01001_047', 'B01001_048', 'B01001_049']],
##             ['17-', ['B01001_003', 'B01001_004', 'B01001_005', 'B01001_006',
##                      'B01001_027', 'B01001_028', 'B01001_029', 'B01001_030']],
##             ['DISABLE', ['B18101_004','B18101_007','B18101_010','B18101_013','B18101_016','B18101_019',
##                          'B18101_023','B18101_026','B18101_029','B18101_032','B18101_035','B18101_038']],
##             ['SINGLPAR',['B11003_010','B11003_016']],
##             ['WHITE_noHISP', ['B03002_003']],
##             ['ENGLTW', ['B16004_007','B16004_008','B16004_012','B16004_013','B16004_017','B16004_018','B16004_022','B16004_023',
##                          'B16004_029','B16004_030','B16004_034','B16004_035','B16004_039','B16004_040','B16004_044','B16004_045',
##                          'B16004_051','B16004_052','B16004_056','B16004_057','B16004_061','B16004_062','B16004_066','B16004_067']],
##             ['10+UNITS', ['B25024_007','B25024_008','B25024_009']],
##             ['MOBILEHOMES', ['B25024_010']],
##             ['HOUSCROWD', ['B25014_005','B25014_006','B25014_007','B25014_011','B25014_012','B25014_013']],
##             ['noVEHICLE', ['B25044_003','B25044_010']],
##             ['GROUPQRTRS',['B09019_038']]
##             ]

##var_codes = [['POVERTY', ['S0601',['HC01_EST_VC72']]],
##             ['UNEMPLOYED', ['DP03',['HC03_VC13']]],
##             ['PERCAP', ['B19301',['HD01_VD01']]],
##             ['noHSDIPL', ['S0601',['HC01_EST_VC49']]],
##             ['65+', ['S0101',['HC01_EST_VC31']]],
##             ['17-_abs', ['B09001',['HD01_VD01']]],
##             ['DISABLE', ['DP02',['HC03_VC104']]],
##             ['SINGLPAR_abs', ['DP02', ['HC01_VC10','HC01_VC12']]],
##             ['WHITE_noHISP_abs', ['B01001H',['HD01_VD01']]],
##             ['ENGLTW_abs', ['B16005',['HD01_VD07','HD01_VD08','HD01_VD12','HD01_VD13',
##                                   'HD01_VD17','HD01_VD18','HD01_VD22','HD01_VD23',
##                                   'HD01_VD29','HD01_VD30','HD01_VD34','HD01_VD35',
##                                   'HD01_VD39','HD01_VD40','HD01_VD44','HD01_VD45']]],
##             ['10+UNITS_abs', ['DP04',['HC01_VC18','HC01_VC19']]],
##             ['MOBILEHOMES', ['DP04',['HC03_VC20']]],
##             ['HOUSCROWD_abs', ['DP04',['HC01_VC111','HC01_VC112']]],
##             ['noVEHICLE', ['DP04',['HC03_VC82',]]],
##             ['GROUPQRTRS_abs', ['B26001',['HD01_VD01']]]
##             ]

'''
Variable names in var_list

'''

var_list = ['TOTPOP','5+POP','TOTHH','TOTHU','OCCHU',
            'POVERTY','UNEMPLOYED','PERCAP','noHSDIPL','65+','17-_abs','DISABLE','SINGLPAR_abs',
            'WHITE_noHISP_abs','ENGLTW_abs','10+UNITS_abs','MOBILEHOMES','HOUSCROWD_abs','noVEHICLE','GROUPQRTRS_abs']

'''
var_codes is a dictionary of key-value pairs relating variables with tables and columns.
The key is the variable name.
The pair is a list.
The first element of the list is the code identifying the table
The second element of the list is a list of the column(s) with the value
'''
var_codes = {'TOTPOP' : ['S0601',['HC01_EST_VC01']],
             '5+POP' : ['B16005',['HD01_VD01']],
             'TOTHH' : ['DP02', ['HC01_VC03']],
             'TOTHU' : ['DP04', ['HC01_VC03']],
             'OCCHU' : ['DP04', ['HC01_VC109']],
             'POVERTY' : ['S0601',['HC01_EST_VC72']],
             'UNEMPLOYED' : ['DP03',['HC03_VC13']],
             'PERCAP' : ['B19301',['HD01_VD01']],
             'noHSDIPL' : ['S0601',['HC01_EST_VC49']],
             '65+' : ['S0101',['HC01_EST_VC31']],
             '17-_abs' : ['B09001',['HD01_VD01']],
             'DISABLE' : ['DP02',['HC03_VC104']],
             'SINGLPAR_abs' : ['DP02', ['HC01_VC10','HC01_VC12']],
             'WHITE_noHISP_abs' : ['B01001H',['HD01_VD01']],
             'ENGLTW_abs' : ['B16005',['HD01_VD07','HD01_VD08','HD01_VD12','HD01_VD13',
                                   'HD01_VD17','HD01_VD18','HD01_VD22','HD01_VD23',
                                   'HD01_VD29','HD01_VD30','HD01_VD34','HD01_VD35',
                                   'HD01_VD39','HD01_VD40','HD01_VD44','HD01_VD45']],
             '10+UNITS_abs' : ['DP04',['HC01_VC18','HC01_VC19']],
             'MOBILEHOMES' : ['DP04',['HC03_VC20']],
             'HOUSCROWD_abs' : ['DP04',['HC01_VC111','HC01_VC112']],
             'noVEHICLE' : ['DP04',['HC03_VC82',]],
             'GROUPQRTRS_abs' : ['B26001',['HD01_VD01']]
             }


'''
var_details is a dictionary of key-value pairs that summarizes how each indicators must be treated.
The key identifies an indicator name, a variable name or a transformation type.
The value is a list of variables or codes.
The n-th value of each list goes with the n-th values of the other lists.

For example, the indicator MINORITY uses the value from MINORITY_abs at the numerator 
normalized by TOTPOP, a non-standard transformation will be operated (see TransformColumn function),
and take_inv is True, meaning that the higher the indicator, the lower must be the index/vulnerability
'''
var_details = {'name':['POVERTY','UNEMPLOYED','PERCAP','noHSDIPL','65+','17-','DISABLE','SINGLPAR',
                       'MINORITY','ENGLTW','10+UNITS','MOBILEHOMES','HOUSCROWD','noVEHICLE','GROUPQRTRS'],
               'num':['POVERTY','UNEMPLOYED','PERCAP','noHSDIPL','65+','17-_abs','DISABLE','SINGLPAR_abs',
                       'MINORITY_abs','ENGLTW_abs','10+UNITS_abs','MOBILEHOMES','HOUSCROWD_abs','noVEHICLE','GROUPQRTRS_abs'],
               'denom':['POVPOP','16+POP','TOTPOP','25+POP','TOTPOP','TOTPOP','noINSTPOP','TOTHH',
                        'TOTPOP','5+POP','TOTHU','TOTHU','OCCHU','OCCHU','TOTPOP'],
               'howTransform':[12,12,12,12,12,15,12,15,
                               15,15,15,12,15,12,15],
               'take_inv':[True, True, False, True, True, True, True, True,
                           True, True, True, True, True, True, True]
                           }


'''
var_df is a pandas dataframe that include all of the above information
in one table-like data structure
'''
var_df = pd.DataFrame(var_details,
                      columns=['num', 'denom', 'howTransform', 'take_inv'],
                      index=var_details['name'])




def buildIndicator(df, var, dataACS_path):
    '''
    creates the indicator from corresponding column(s) from ACS .csv file
    Some indicators, e.g., single parent households, are the sum of multiple columns
    '''
    
    filecode = var_codes[var][0]
    varcodes = var_codes[var][1]

    var_df = pd.read_csv(os.path.join(dataACS_path, 'ACS_12_5YR_'+filecode+'_with_ann.csv'),
                         header=0,
                         skiprows=[1],
                         index_col = 0,
                         usecols  = ['GEO.id']+varcodes)
    var_df[varcodes[0]] = pd.to_numeric(var_df[varcodes[0]], errors='coerce')
    var_df['sumvars'] = var_df[varcodes[0]]
    for i in range(1,len(varcodes)):
        var_df[varcodes[i]] = pd.to_numeric(var_df[varcodes[i]], errors='coerce')
        var_df['sumvars'] += var_df[varcodes[i]]
    df[var] = var_df['sumvars']


def NormalizeColumn(df, colName, norm_colName, howNormalize, takeInverse = False):
    '''
    Normalize column indicator, using different methods    
    '''
    col = df[colName]
    #print(col.head())
    maxVal = np.max(col)
    #print(maxVal)
    minVal = np.min(col)
    #print(minVal)
    mean = np.mean(col)
    #print(mean)
    stDev = np.std(col)
    #print(stDev)

    if howNormalize == Normalizing["None"]:
        normalizedVar = col
        
    elif howNormalize == Normalizing["MinMax0to1"]:       #Stretch values between 0 and 1
        if takeInverse == True:
            normalizedVar = (maxVal - col) / (maxVal - minVal)
        else:
            normalizedVar = (col - minVal) / (maxVal - minVal)

    elif howNormalize == Normalizing["MinMax1to2"]:       #Stretch values between 1 and 2
        if takeInverse == True:
            normalizedVar = ((maxVal - col) / (maxVal - minVal)) + 1
        else:
            normalizedVar = ((col - minVal) / (maxVal - minVal)) + 1

    elif howNormalize == Normalizing["Zscore"]:           #Perform a z-score transformation
        normalizedVar = (col - mean) / stDev
        if takeInverse == True:
            normalizedVar = normalizedVar * -1

    elif howNormalize == Normalizing["MaxValue"]:         #Divide by the maximum value
        if takeInverse == True:
            normalizedVar = (1 - (col / maxVal)) + (minVal / maxVal)
        else:
            normalizedVar = col / maxVal
        
    df[norm_colName] = normalizedVar




def TransformColumn(df, colName, transf_colName, howTransform, denColName='TOTPOP'):
    col = df[colName]
    '''
    Transform the indicator, from absolute to relative
    Divide indicator absolute number by unit area, unit total population, or other
    '''
    if howTransform == Transforming["None"]:
        denominator = 1
    elif howTransform == Transforming["Area"]:
        denominator = df['ALAND']
    elif howTransform == Transforming["Population"]:
        denominator = df['TOTPOP'] #######insert total population column name here!!
    elif howTransform == Transforming["Custom"]:
        denominator = df[denColName]/100
        
    df[transf_colName] = col/denominator



def RankColumn(df, colName, rankColName, ascendingBool = False):
    '''
    Assign rank by indicator colName in descending or ascending order
    Ranks in same direction of vulnerability, e.g. high poverty -> high rank -> high vulnerability   
    '''
    df[rankColName] = df[colName].rank(method='min', ascending=ascendingBool)


def PercRankColumn(df, colName, rankColName, ascendingBool = False):
    '''
    Perc and pct stand for percentile
    First rank units/rows by indicator colName (same as RankColumn).
    Ranks in same direction of vulnerability, e.g. high poverty -> high rank -> high vulnerability
    Then rank normalized by number of units
    PercRank values between 0 and 1 (NOT 0 and 100, it is not!)
    '''
    df[rankColName] = df[colName].rank(method='min', ascending=ascendingBool, pct=True)
    # if max PercRank is intended to be 1 and min PercRank is intended to be 0, uncomment the following:
    max_val = df[rankColName].max()
    df.loc[df[rankColName]==max_val, rankColName] = 1
    min_val = df[rankColName].min()
    df.loc[df[rankColName]==min_val, rankColName] = 0



def buildIndex(df, columnList, indicator_name, howWeight, howAggregate):
    '''
    build index from list of indicators
    '''
    
    numOfTerms = len(columnList)
    weights = [1 for i in range(numOfTerms)]
    
    if howWeight == Weighting['Expert']:
        
        #insert weight functions here
        pass
    
    indicator = 0
    if howAggregate == Aggregation["Additive"]:
        indicator = 0
        for i in range(numOfTerms):
            indicator += weights[i] * df[columnList[i]]   
            
    if howAggregate == Aggregation["Geometric"]:
        indicator = 1
        for i in range(numOfTerms):
            indicator *= df[columnList[i]]**weights[i]
    
    df[indicator_name] = indicator

   
             

def deductive():
    '''
    not yet implemented
    '''
    pass

##def get_scaled(data, how_scaling):
##    pass

def get_weighted(data, how_weight):
    '''
    not yet implemented
    '''
    pass




if __name__ == "__main__":
    start = time.time()

    current_folder = os.getcwd()
    dataCDC_path = os.path.join(current_folder, "CDC_data")
    dataACS_path = os.path.join(current_folder, "ACS_original")
    years_of_tracts = ['2010','2016']
    for year in years_of_tracts:
        index_col = 4
        if year == '2010':
            index_col = 1
        tracts_df = pd.read_csv(os.path.join(dataCDC_path, "tracts_"+year+"_coastCounties.csv"),
                                header=0,
                                index_col = index_col)        
        speak_df = pd.read_csv(os.path.join(dataACS_path,"ACS_12_5YR_B16005_with_ann.csv"),
                         header=0,
                         skiprows=[1],
                         index_col=0)
        ##                 keep_default_na=False)
       
        '''
        take unit name column from census table
        drop all columns that are not needed from shapefile attribute table
        add unit name column at the beginning of shapefile attribute table 
        '''
        tracts_df['Geography'] = speak_df['GEO.display-label']
        col = tracts_df['Geography']
        try:
            tracts_df.drop([col.name,'FID','STATEFP','COUNTYFP','NAME','LSAD','ALAND','AWATER'],
                           inplace=True, axis=1)
        except:
            tracts_df.drop([col.name,'FID','STATE','COUNTY','NAME','LSAD','CENSUSAREA'],
                           inplace=True, axis=1)
        tracts_df.insert(0, col.name, col)
        # print (tracts_df.head())


        '''
        create indicators from variables
        calculate/transform/modify some of the indicators
        '''
        for var in var_list :
            buildIndicator(tracts_df, var, dataACS_path)
        
        tracts_df = tracts_df[tracts_df['TOTPOP']>0]                                        # not considering units without population
        tracts_df['MINORITY_abs'] = tracts_df['TOTPOP'] - tracts_df['WHITE_noHISP_abs']     # calculate MINORITY by difference
        tracts_df.drop('WHITE_noHISP_abs', inplace=True, axis=1)

        indicators_list = var_details['name']
        for i in range(len(var_details['name'])):
            try:
                TransformColumn(tracts_df, var_details['num'][i], var_details['name'][i],
                            var_details['howTransform'][i], var_details['denom'][i])
            except:
                print (var_details['name'][i])
        tracts_df.fillna(value=0,inplace=True)                                              # verify this is the intended behavior (there should be no/few NA!)

        '''
        rank tracts by indicator value
        create as many rank columns as indicators
        '''
        ranked_indicators_list = ['r_' + i for i in indicators_list]
        for i in range(len(indicators_list)):
            PercRankColumn(tracts_df, indicators_list[i], ranked_indicators_list[i], var_details['take_inv'][i])

        '''
        calculate theme values
        sum PercRanks of indicators in each theme
        '''
        for theme in themes_list:
            tracts_df["sc_"+themesAcr[theme]] = tracts_df['r_'+themes[theme][0]]
            for i in range(1,len(themes[theme])):
                tracts_df["sc_"+themesAcr[theme]] += tracts_df['r_'+themes[theme][i]]

        '''
        rank tracts by theme value
        create as many rank columns as themes
        '''
        ranked_theme_columnList = []
        for theme in themes_list:
            ranked_theme_columnList.append('r_'+themesAcr[theme])
            PercRankColumn(tracts_df, "sc_"+themesAcr[theme], 'r_'+themesAcr[theme], True)

        '''
        calculate CDC SVI score
        sum four theme values
        '''
        tracts_df['CDC_score'] = tracts_df["sc_"+themesAcr[themes_list[0]]]
        for i in range(1, len(themes_list)):
            tracts_df['CDC_score'] += tracts_df["sc_"+themesAcr[themes_list[i]]]

##        tracts_df['CDC_score'] = tracts_df[ranked_theme_columnList[0]]
##        for i in range(1,len(ranked_theme_columnList)):
##            tracts_df['CDC_score'] += tracts_df[ranked_theme_columnList[i]]
##        NormalizeColumn(tracts_df, "RINDEX", "nZ_RINDEX", 19, True)

        '''
        rank tracts by CDC SVI score
        '''
        PercRankColumn(tracts_df, 'CDC_score', 'CDC_percrank', True)

        '''
        export dataframe to csv, and txt
        these can be joined to shapefile to visualize CDC SVI'''
        tracts_df.to_csv(os.path.join(dataCDC_path,"CDC_"+year+"_coastCounties.csv"))
        tracts_df.to_csv(os.path.join(dataCDC_path,"CDC_"+year+"_coastCounties.txt"))

    print (str(time.time()-start) + " seconds")
