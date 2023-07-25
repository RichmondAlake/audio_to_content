import logging
import traceback
import requests
import streamlit as st
from urllib.error import URLError
from deepgram import Deepgram
import io
from content import summarize_transcript, answer_queries, generate_therapy_session_report, generate_meeting_notes, generate_blog_post, generate_statement_of_work

# Initialize Deepgram client
deepgram_api = st.secrets['deepgram']
deepgram = Deepgram(deepgram_api)

st.title('Audio 🎶 to Content 📄')

st.text("Use the preloaded audio below for specific use cases")
# Instructions
st.write("1. Tap 'Ted Talk' to load Conor Russomanno's audio on mind-controlled tech.")
st.write("2. Tap 'Client Meeting' to load a converstaion between a technical consultant and client.")
st.write("3. Tap 'Therapy Session' to load a conversation beteween a therapist and a patient.")

# Create three columns
col1, col2, col3 = st.columns(3)

# Create three buttons for the three audio files
if col1.button('Ted Talk'):
    filepath = 'audio_example/openbci.mp3'
    with open(filepath, 'rb') as f:
        st.session_state.audio_file = io.BytesIO(f.read())
    st.session_state.audio_file.name = 'openbci.mp3'
    st.session_state.audio_file.type = 'audio/mpeg'
    st.session_state.audio_file.size = st.session_state.audio_file.getbuffer().nbytes
    st.audio(st.session_state.audio_file, format='audio/mpeg')
elif col2.button('Client Meeting (coming soon...)'):
    url = 'https://example.com/audio2.mp3' 
    response = requests.get(url)
    st.session_state.audio_file = io.BytesIO(response.content)
    st.session_state.audio_file.name = 'audio2.mp3'
    st.session_state.audio_file.type = 'audio/mpeg'
    st.session_state.audio_file.size = len(response.content)
elif col3.button('Threapy Session (coming soon...)'):
    url = 'https://example.com/audio3.mp3' 
    response = requests.get(url)
    st.session_state.audio_file = io.BytesIO(response.content)
    st.session_state.audio_file.name = 'audio3.mp3'
    st.session_state.audio_file.type = 'audio/mpeg' 
    st.session_state.audio_file.size = len(response.content)
else:
    st.markdown("---")
    st.write("Or Upload your own audio below")
    st.session_state.audio_file = st.file_uploader("Choose an audio file", type=['mp3', 'wav'])

if st.session_state.audio_file is not None:
    # Load the content of the file into a variable
    file_details = {"FileName": st.session_state.audio_file.name, "FileType": st.session_state.audio_file.type, "FileSize": st.session_state.audio_file.size}
    st.write(file_details)

    # Convert the uploaded file to bytes
    audio_bytes = io.BytesIO(st.session_state.audio_file.getbuffer())

    source = {'buffer': audio_bytes, 'mimetype': file_details["FileType"]}
    options = { 
        "punctuate": True,
        "diarize": True,
        "model": "nova",
        "language": "en-US",
    }

    # Transcribe the audio file using deepgram
    if 'transcript' not in st.session_state:
        with st.spinner('Transcribing...'):
            try:
                response = deepgram.transcription.sync_prerecorded(source, options)
                if response.get('results'):
                    st.session_state.transcript = response['results']['channels'][0]['alternatives'][0]['transcript']
                else:
                    st.error("No transcription available for this audio.")
            except URLError as e:
                logging.error(f"Deepgram API: Error while making the request: {e.reason}")
                st.error("An error occurred during the transcription process. Please try again.")
            except Exception as e:
                logging.error(f"Deepgram API: Error while requesting transcript: {e}")
                st.error("An error occurred during the transcription process. Please try again.")
                logging.error(traceback.format_exc())

if 'transcript' in st.session_state:
    st.markdown("<h2 style='text-align: center; color: lightblue;'>Transcript:</h2>", unsafe_allow_html=True)
    st.markdown(f'<div style="height: 200px; overflow-y: auto; border: 1px solid #f0f0f0; padding: 10px; border-radius: 5px;">{st.session_state.transcript}</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Content Generation
    st.markdown("<h2 style='text-align: center; color: lightblue;'>Content Generation</h2>", unsafe_allow_html=True)
    st.session_state.content_type = st.selectbox('Choose the type of content:', ['Summary', 'Mental Health Report', 'Statement Of Work', 'Meeting Notes', 'Blog'], index=list(['Summary', 'Mental Health Report', 'Statement Of Work', 'Meeting Notes', 'Blog']).index(st.session_state.content_type) if 'content_type' in st.session_state else 0)

    # Introduce a new button for content generation
    if st.button('Generate Content'):
        if st.session_state.content_type == 'Summary':
            with st.spinner('Generating summary...'):
                summary = summarize_transcript(transcript=st.session_state.transcript)
                st.write(summary)

        elif st.session_state.content_type == 'Mental Health Report':
            with st.spinner('Generating Mental Health Report...'):
                mental_health_report = generate_therapy_session_report(transcript=st.session_state.transcript)
                st.write(mental_health_report)

        elif st.session_state.content_type == 'Meeting Notes':
            with st.spinner(f'Generating {st.session_state.content_type.lower()}...'):
                meeting_notes = generate_meeting_notes(transcript=st.session_state.transcript)
                st.write(meeting_notes)

        elif st.session_state.content_type == 'Statement Of Work':
            with st.spinner(f'Generating {st.session_state.content_type.lower()}...'):
                sow = generate_statement_of_work(transcript=st.session_state.transcript)
                st.write(sow)

        elif st.session_state.content_type == 'Blog':
            with st.spinner(f'Generating {st.session_state.content_type.lower()}...'):
                blog = generate_blog_post(transcript=st.session_state.transcript)
                st.write(blog)

    st.markdown("---")

    # Chat interface
    st.markdown("<h2 style='text-align: center; color: lightblue;'>Query Interface</h2>", unsafe_allow_html=True)
    st.session_state.user_input = st.text_input("Enter your query:", value=st.session_state.user_input if 'user_input' in st.session_state else '')
    if st.button('Ask'):
        with st.spinner('Processing your query...'):
            st.session_state.response = answer_queries(st.session_state.transcript, st.session_state.user_input)
            st.write(st.session_state.response)
