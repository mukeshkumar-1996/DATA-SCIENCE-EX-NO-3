## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Encoding for the feature in the data set.

STEP 4:Apply Feature Transformation for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.

2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.

3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.

4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation

• Reciprocal Transformation

• Square Root Transformation

• Square Transformation

  # 2. POWER TRANSFORMATION
• Boxcox method

• Yeojohnson method

# CODING AND OUTPUT:

      #importing all the neccessary packages
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
    import seaborn as sns
    !pip install category_encoders # if category_encoders is not installed
    from category_encoders import BinaryEncoder
    data=pd.read_csv("data.csv")
    df=pd.DataFrame(data)
    df

<img width="751" height="366" alt="530260657-cff7e59a-ab22-4232-94ad-744a39913f2c" src="https://github.com/user-attachments/assets/7f50a5ab-772a-4e40-87fd-12b500a2ec4e" />

    #Ordinal encoder without specifying order
    oe=OrdinalEncoder()
    df["OE_1]=oe.fit_transform(df[["Ord_1"]])
    df

<img width="817" height="363" alt="530260839-2a89f08d-f480-4225-ba13-364b894fe554" src="https://github.com/user-attachments/assets/e61580ea-bc76-4ef2-bc21-4b1ce3b2568a" />

    #OrdinalEncoder with order specified
    oe=OrdinalEncoder(categories=[["Hot","Warm","Very Hot","Cold"]])
    df["OE1(1)"]=oe.fit_transform(df[["Ord_1"]])
    df

<img width="832" height="361" alt="530260955-f23c8876-e3a9-47cf-a659-8a468b5f76f9" src="https://github.com/user-attachments/assets/8e0a9218-9619-4d86-b1e1-ba6bceac3ac5" />

    #LABEL ENCODER
    le=LabelEncoder()
    df["LE2"]=le.fit_transform(df["Ord_2"])
    df

<img width="902" height="356" alt="530261023-3d89fccc-6203-4b45-9b5e-10f0a1f45390" src="https://github.com/user-attachments/assets/00377def-376d-473a-b9ea-401b91538b87" />
    
    #ONE HOT ENCODER
    ohe=OneHotEncoder(sparse_output=False)
    enc=pd.DataFrame(ohe.fit_transform(df[["bin_1"]]))
    df=pd.concat([df,enc],axis=1)
    df

<img width="956" height="371" alt="530261085-23f02629-8887-419e-93ee-b9211b4d113f" src="https://github.com/user-attachments/assets/c246f68a-1fa6-4d80-9f1e-614037c9cd5e" />

    get_dummies(df,columns=["bin_1"])

<img width="983" height="364" alt="530261180-55648463-4bbf-4877-b701-1b0e9fdc6a7c" src="https://github.com/user-attachments/assets/eae84b3f-b1b2-4914-b7fc-61b25e2c1710" />

     # BINARY ENCODER
     be=BinaryEncoder()
     nd=be.fit_transform(df["bin_2"])
     df=pd.concat([df,nd],axis=1)
     df

<img width="1055" height="356" alt="530261253-bae5422f-8f6f-40a5-95e5-22ef0657c9a4" src="https://github.com/user-attachments/assets/18775225-a326-48dd-a4cc-f0c88250a6d1" />

    #importing all neccessary packasges
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from sklearn.preprocessing import PowerTransformer
    data=pd.read_csv("Data_to_Transform.csv")
    df=pd.DataFrame(data)
    df

<img width="1085" height="416" alt="530261472-f704a0f7-881e-4a94-91ca-1f8782cbbb42" src="https://github.com/user-attachments/assets/a93aad75-1de7-480f-9668-716940092e0c" />

    df.skew()  

<img width="695" height="123" alt="530261539-abf5554d-d4df-4164-b5b8-32a118b7e4db" src="https://github.com/user-attachments/assets/b1973b10-ecd9-47c3-81a5-182dcc7b6631" />

    #log transformation
    np.log(df["Highly Positive Skew"])

<img width="771" height="268" alt="530261609-46d8aebd-01c8-49a5-b9bb-4123e5be579d" src="https://github.com/user-attachments/assets/051c0238-f716-43b3-bbff-612d4452525b" />

    #reciprocal transformation
    np.reciprocal(df["Highly Positive Skew"])
    
<img width="844" height="265" alt="530261652-783586b3-63db-4866-aa45-d6ab23d1bd1e" src="https://github.com/user-attachments/assets/f8c41bf7-e408-46d8-b289-4983ac2a40c5" />

    #square root transformation 
    np.sqrt(df["Highly Positive Skew"])

<img width="818" height="268" alt="530261691-ba22936c-eb47-4c52-b99d-8f971f0547c5" src="https://github.com/user-attachments/assets/17cc728e-dc76-4a63-a45b-f96cfcb2c5b7" />

    #Yeo-Johnson Transformation
    pt_yj=PowerTransformer(method='yeo-johnson')
    df["YJ_skew"]=pt_yj.fit_transform(df[["Highly Negative Skew"]])
    df
 
<img width="1123" height="455" alt="530261828-1700efd9-f2e6-45a4-bf18-b502c8d6d705" src="https://github.com/user-attachments/assets/23a45b37-60f9-4dfb-a293-546d37004a6c" />

    #Box-cox Transformation
    pt_boxcox=PowerTransformer(method="box-cox")
    df["BC_skew"]=pt_boxcox.fit_transform(df[["Moderate Positive Skew"]])
    df

<img width="1068" height="446" alt="530261920-8ff39dde-f97a-4dd5-91a6-e4cbe0aa648b" src="https://github.com/user-attachments/assets/1607159f-a620-4aab-88ee-b999dc30f5ff" />

    #to save the data to a new file
    df.to_csv("Transformed_data.csv",index=False)

# RESULT:
       
              Thus the progarm of read the given data and perform Feature Encoding and Transformation process and save the data to a file is written and executed successfully.
       
