from Home import st
from Home import face_rec
from streamlit_webrtc import webrtc_streamer
import av
import time



st.set_page_config(page_title='Predictions',layout='centered')

st.subheader("Real-Time Attendance System")

#Retrieve Data From Database
with st.spinner("Retrieving data from Redis..."):
    redis_face_db=face_rec.retrieve_data(name='academy:register')
    st.dataframe(redis_face_db)

st.success("Data retrieved successfully!")

waitTime=30 #sec
setTime=time.time()
realtimepred=face_rec.RealTimePred()

#Real time Prediction
def video_frame_callback(frame):
    global setTime
    img = frame.to_ndarray(format="bgr24") #3d np array
    pred_img=realtimepred.face_prediction(img,redis_face_db,'facial_features',['Name','Role'],thresh=0.5)
    # flipped = img[::-1,:,:]
    timenow=time.time()
    difftime=timenow-setTime
    if difftime>=waitTime:
        realtimepred.saveLogs_redis()
        setTime=time.time() #reset time
        print("save data to redis")

    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")


webrtc_streamer(key="realtimePrediction", video_frame_callback=video_frame_callback)