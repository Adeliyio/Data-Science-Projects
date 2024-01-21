<div align="center">
  
[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]

</div>

# <div align="center">Build, train and deploy Machine Learning model using AWS Sagemaker</div>
<div align="center"><img src="https://github.com/Pradnya1208/Create-train-and-deploy-ML-model-using-AWS-Sagemaker/blob/main/images/intro.gif?raw=true" width="60%"></div>

## Overview:
In this project, we will learn how to use Amazon SageMaker to build, train, and deploy a machine learning (ML) model using the XGBoost ML algorithm. Amazon SageMaker is a fully managed service that provides every developer and data scientist with the ability to build, train, and deploy machine learning (ML) models quickly.

In this project, we'll learn how to:

- Create a SageMaker notebook instance
- Prepare the data
- Train the model to learn from the data
- Deploy the model
- Evaluate your ML model's performance

## AWS S3 bucket:
Amazon Simple Storage Service (Amazon S3) is an object storage service offering industry-leading scalability, data availability, security, and performance. Customers of all sizes and industries can store and protect any amount of data for virtually any use case, such as data lakes, cloud-native applications, and mobile apps. With cost-effective storage classes and easy-to-use management features, you can optimize costs, organize data, and configure fine-tuned access controls to meet specific business, organizational, and compliance requirements.
<img src="https://github.com/Pradnya1208/Create-train-and-deploy-ML-model-using-AWS-Sagemaker/blob/main/images/s3.PNG?raw=true">

## Amazon Sagemaker Studio:
Amazon SageMaker Studio provides a single, web-based visual interface where you can perform all ML development steps, improving data science team productivity by up to 10x. SageMaker Studio gives you complete access, control, and visibility into each step required to build, train, and deploy models. You can quickly upload data, create new notebooks, train and tune models, move back and forth between steps to adjust experiments, compare results, and deploy models to production all in one place, making you much more productive. All ML development activities including notebooks, experiment management, automatic model creation, debugging, and model and data drift detection can be performed within SageMaker Studio.

## Getting started
- login to `AWS Management Console`
- Search for `Amazon Sagemaker`
- Go to `Notebook instance` and `Create notebook instance` use `any S3 Bucket`
- Once the status shows `InService` open a `Jupyter Notebook`


## 1) Importing important Libraries:
```
import sagemaker
import boto3 #for accessing s3 bucket
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.session import s3_input, Session
```

## 2) Create S3 bucket:
```
bucketname = "<bucket name>"
my_region = boto3.session.Session().region_name # set the region of the instance
```
```
s3 = boto3.resource('s3')
try:
    if my_region=="ap-south-1":
        s3.create_bucket(Bucket = bucketname)
    print("s3 bucket created successfully")
except Exception as e:
    print("S3 error: ", e)
```
## 3) Set the Output path for saving the model:
```
prefix = "xgboost-as-a-built-in-algo"
output_path = 's3://{}/{}/output'.format(bucketname,prefix)

output_path2 = 's3://{}/{}/output'.format("testbucketforassignone",prefix)
```

## 4) Saving the train and test data in s3 bucket:

Upload the dataset in S3 bucket where the notebook instance's created.

```
import pandas as pd
import urllib
try:
    urllib.request.urlretrieve ("<URL")
    print('Success')
except Exception as e:
    print('Data load error: ',e)

try:
    model_data = pd.read_csv('<FILENAME>',index_col=0)
    print('Success: Data loaded into dataframe.')
except Exception as e:
    print('Data load error: ',e)
```
#### Save the train and test data:
<br>

```
boto3.Session().resource('s3').Bucket("testbucketforassignone").Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
s3_input_train = sagemaker.TrainingInput(s3_data='s3://{}/{}/train'.format("testbucketforassignone", prefix), content_type='csv')
```
Same way we can store the test data.

## 5) Implement the model as shown in [Notebook](https://github.com/Pradnya1208/Build-train-and-deploy-ML-model-using-AWS-Sagemaker/blob/main/Bank%20application%20using%20AWS%20Sagemaker.ipynb)

Following code automatically looks for the XGBoost image URI and builds and XGBoost container

```
from sagemaker import image_uris 
container = sagemaker.image_uris.retrieve("xgboost", boto3.Session().region_name, "1.2-1")
```
## 6) Construct a SageMaker estimator that calls the XGBoost-container:

```
estimator = sagemaker.estimator.Estimator(image_uri=container, 
                                          hyperparameters=hyperparameters,
                                          role=sagemaker.get_execution_role(),
                                          instance_count=1, 
                                          instance_type='ml.m5.2xlarge', 
                                          volume_size=5, # 5 GB 
                                          output_path=output_path2,
                                          use_spot_instances=True,
                                          max_run=300,
                                          max_wait=600)
```
## 7) Training the data:
```
estimator.fit ({'train': s3_input_train,'validation': s3_input_test})
```

## 8) Deploy Machine Learning model as endpoints:
```
xgb_predictor = estimator.deploy(initial_instance_count=1,instance_type='ml.m4.xlarge')
```

## 9) Predicting using Test data:
```
from sagemaker.predictor import csv_serializer
test_data_array = test_data.drop(['y_no', 'y_yes'], axis=1).values #load the data into an array
xgb_predictor.serializer = csv_serializer # set the serializer type
predictions = xgb_predictor.predict(test_data_array).decode('utf-8') # predict!
predictions_array = np.fromstring(predictions[1:], sep=',') # and turn the prediction into an array
```
### Results:
```
Overall Classification Rate: 89.7%

Predicted      No Purchase    Purchase
Observed
No Purchase    91% (10785)    34% (151)
Purchase        9% (1124)     66% (297) 
```

## 10) Delelting the endpoints:
This step is necessary to free the resources.
```
sagemaker.Session().delete_endpoint(xgb_predictor.endpoint)
bucket_to_delete = boto3.resource('s3').Bucket("<Bucket Name>")
bucket_to_delete.objects.all().delete()
```


## License:

`MIT License`
### Learnings:
`Model Deployment using AWS Sagemaker` 






## References:
[AWS Sagemaker](https://aws.amazon.com/getting-started/hands-on/build-train-deploy-machine-learning-model-sagemaker/)

### Feedback

If you have any feedback, please reach out at pradnyapatil671@gmail.com


### 🚀 About Me
#### Hi, I'm Pradnya! 👋
I am an AI Enthusiast and  Data science & ML practitioner






[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]

