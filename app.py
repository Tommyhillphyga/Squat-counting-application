import tempfile
import cv2
import streamlit as st

from stream_detection import process_uploaded_file

def main():
   
    st.title("Squat Counting App")
    st.sidebar.title("Settings")
    st.markdown(
        """
    <style>
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    use_webcam = False
    # use_webcam = True
    st.sidebar.markdown('---')
    confidence = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value =0.25)
    st.sidebar.markdown('---')

    save_video = st.sidebar.checkbox("Save Video")
    enable_gpu = st.sidebar.checkbox("Enable GPU")
    
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov",'avi','asf', 'm4v'])

    Demo_video  = "squat1.mp4"
    tffile = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)

    if not video_file_buffer:
        if use_webcam:
            video = cv2.VideoCapture(0, cv2.CAP_ARAVIS)
            tffile.name = 0
        else:
            video = cv2.VideoCapture(Demo_video)
            tffile.name = Demo_video
            demo_video = open(tffile.name, 'rb')
            demo_bytes = demo_video.read()

            st.sidebar.text("Input video")
            st.sidebar.video(demo_bytes)

    else:
        tffile.write(video_file_buffer.read())
        demo_video = open(tffile.name, 'rb')
        demo_bytes = demo_video.read()

        st.sidebar.text("Input Video")
        st.sidebar.video(demo_bytes)

    print(tffile.name)

    stframe = st.empty()
    st.markdown("<hr/>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(spec=3, gap="large")

    with col1:
        st.markdown("** **")
        column1_text = st.markdown(" ")

    with col2:
        st.markdown("**Squat Count**")
        column2_text = st.markdown("0")

    with col3:
        st.markdown("** **")
        column3_text = st.markdown(" ")

    st.markdown("<hr/>", unsafe_allow_html=True)

    #load model and process frame
    process_uploaded_file('No', tffile.name, enable_gpu, save_video, confidence, column2_text, stframe, use_webcam)

    st.text('Video Processed')
    
if __name__== '__main__':
    try: 
        main()
    except SystemExit:
        pass


