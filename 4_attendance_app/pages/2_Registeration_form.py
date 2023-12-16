from Home import st
from Home import face_rec
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
st.set_page_config(page_title='Registeration Form',layout='centered')

st.subheader("Registeration Form")

# init registration form 
registration_form=face_rec.Registrationform()

# collect person name and role
person_name=st.text_input(label='Name',placeholder='First & Last Name')
role=st.selectbox(label='Select your Role:',options=('Student','Teacher'))

# collect facial embeddings
#Real time Prediction
def video_frame_callback(frame): #can't store embeddings in redis directly
    img = frame.to_ndarray(format="bgr24") #3d np array
    reg_img,embedding=registration_form.get_embedding(img)
    # save embeddings on local computer
    if embedding is not None:
        with open('face_embedding.txt',mode='ab') as f:
            np.savetxt(f,embedding)


    return av.VideoFrame.from_ndarray(reg_img, format="bgr24")

webrtc_streamer(key="registeration", video_frame_callback=video_frame_callback)

# save data in redis
if st.button('Submit:'):
    return_val= registration_form.save_data_in_redis_db(person_name,role)
    if return_val==True:
        st.success(f"{person_name} registered successfully")
    elif return_val=='name_false':
        st.error("Name cannot be empty!!")
    elif return_val=='file_false':
        st.error("File not found... Please refresh the page and try again!")



    
    