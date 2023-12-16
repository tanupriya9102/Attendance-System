from Home import st
from Home import face_rec
st.set_page_config(page_title='Reporting',layout='wide')
st.subheader("Reporting")

# extract data from redis list 
name='attendance:logs'
def load_logs(name):
    logs_list=face_rec.r.lrange(name,start=0,end=-1)
    return logs_list

tab1,tab2=st.tabs(['Registered Data','Logs'])



with tab1:
    if st.button("Refresh data"):
        #Retrieve Data From Database
        with st.spinner("Retrieving data from Redis..."):
            redis_face_db=face_rec.retrieve_data(name='academy:register')
            st.dataframe(redis_face_db)

with tab2:
    if st.button('Refresh Logs'):
        st.write(load_logs(name=name))


