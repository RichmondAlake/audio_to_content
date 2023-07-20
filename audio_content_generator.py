import logging
import traceback
from urllib.error import URLError
import streamlit as st
from deepgram import Deepgram
import io
from content import summarize_transcript, answer_queries, generate_therapy_session_report, generate_meeting_notes, generate_blog_post, generate_statement_of_work

# Initialize Deepgram client
deepgram_api = st.secrets['deepgram']
deepgram = Deepgram(deepgram_api)

st.title('Audio 🎶 to Content 📄')

# Create file upload button
uploaded_file = st.file_uploader("Choose an audio file", type=['mp3', 'wav'])

if uploaded_file is not None:
    # Load the content of the file into a variable
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
    st.write(file_details)

    # Convert the uploaded file to bytes
    audio_bytes = io.BytesIO(uploaded_file.getbuffer())

    source = {'buffer': audio_bytes, 'mimetype': file_details["FileType"]}
    options = { 
        "punctuate": True,
        "diarize": True,
        "model": "nova",
        "language": "en-US",
    }

    # Transcribe the audio file using deepgram
    with st.spinner('Transcribing...'):
        try:
            response = deepgram.transcription.sync_prerecorded(source, options)
            if response.get('results'):
                transcript = response['results']['channels'][0]['alternatives'][0]['transcript']

                st.markdown("<h2 style='text-align: center; color: lightblue;'>Transcript:</h2>", unsafe_allow_html=True)
                st.markdown(f'<div style="height: 200px; overflow-y: auto; border: 1px solid #f0f0f0; padding: 10px; border-radius: 5px;">{transcript}</div>', unsafe_allow_html=True)

                st.markdown("---")

            else:
                st.error("No transcription available for this audio.")
        except URLError as e:
            logging.error(f"Deepgram API: Error while making the request: {e.reason}")
            st.error("An error occurred during the transcription process. Please try again.")
        except Exception as e:
            logging.error(f"Deepgram API: Error while requesting transcript: {e}")
            st.error("An error occurred during the transcription process. Please try again.")
            logging.error(traceback.format_exc())

    # Content Generation
    if transcript:
        st.markdown("<h2 style='text-align: center; color: lightblue;'>Content Generation</h2>", unsafe_allow_html=True)
        content_type = st.selectbox('Choose the type of content:', ['Summary', 'Mental Health Report', 'Statement Of Work', 'Meeting Notes', 'Blog'])
        
        if content_type == 'Summary':
            with st.spinner('Generating summary...'):
                summary = summarize_transcript(transcript=transcript)
                st.write(summary)

        elif content_type == 'Mental Health Report':
            with st.spinner('Generating Mental Health Report...'):
                mental_health_report = generate_therapy_session_report(transcript=transcript)
                st.write(mental_health_report)

        elif content_type == 'Meeting Notes':
            with st.spinner(f'Generating {content_type.lower()}...'):
                meeting_notes = generate_meeting_notes(transcript=transcript)
                st.write(meeting_notes)

        elif content_type == 'Statement Of Work':
            with st.spinner(f'Generating {content_type.lower()}...'):
                sow = generate_statement_of_work(transcript=transcript)
                st.write(sow)

        elif content_type == 'Blog':
            with st.spinner(f'Generating {content_type.lower()}...'):
                blog = generate_blog_post(transcript=transcript)
                st.write(blog)
            
        st.markdown("---")

    if transcript:
        # Chat interface
        st.markdown("<h2 style='text-align: center; color: lightblue;'>Query Interface</h2>", unsafe_allow_html=True)
        with st.spinner('Chatting...'):

            user_input = st.text_input("Enter your query:")
            if st.button('Ask'):
                response = answer_queries(transcript, user_input)
                st.write(response)
