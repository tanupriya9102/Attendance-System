import numpy as np
import pandas as pd
import cv2
import redis
import insightface.app import FaceAnalysis
from sklearn.metrics import pairwise


# Connect to redis client
# redis-12025.c85.us-east-1-2.ec2.cloud.redislabs.com:12025
hostname='redis-12025.c85.us-east-1-2.ec2.cloud.redislabs.com'
port=12025
password='LhNFHs1KTyB9C1MWktmKlu2oP8iSr6C6'
r=redis.StrictRedis(host=hostname,
                    port=port,
                    password=password)

# configure face analysis
# configure model
faceapp=FaceAnalysis(name='buffalo_sc',
                     root='insightface_model',
                     providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0,det_size=(640,640),det_thresh=0.5)


# ml search algo
#     cosine similarity
def search_algo(df,feature_column,test_vector,name_role['Name','Role'],thresh=0.5):
    df=df.copy()
    X_list=df[feature_column].tolist()
    x=np.asarray(X_list)
    
    similar=pairwise.cosine_similarity(x,test_vector.reshape(1,-1))
    similar_arr=np.array(similar).flatten()
    df['cosine']=similar_arr
    
    data_filter=df.query(f'cosine>{thresh}')
#    
    if len(data_filter)>0:
        data_filter.reset_index(drop=True,inplace=True)
        argmax=data_filter['cosine'].argmax()
        person_name,person_role=data_filter.loc[argmax][name_role]
        
    else:
        person_name=person_role='Unknown'
        
    return person_name,person_role



def face_prediction(test_image,df,feature_column,name_role['Name','Role'],thresh=0.5):    
    results=faceapp.get(test_image)
    test_copy=test_image.copy()
    for res in results:
        x1,y1,x2,y2= res['bbox'].astype(int)
        embeddings=res['embedding']
        person_name,person_role=search_algo(df_compress,'Facial_Features',test_vector=embeddings,name_role=name_role,thresh=thresh)
    #     print(person_name,person_role)
        if person_name=="Unknown":
            color=(0,0,255) #bgr
        else:
            color=(0,255,0)

        cv2.rectangle(test_copy,(x1,y1),(x2,y2),color)
        text_gen=person_name
        cv2.putText(test_copy,text_gen,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.6,color,2)
        
        return test_copy


