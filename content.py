
import openai
import tiktoken
from openai import OpenAIError
from collections import deque
import bisect
import timeit
import logging
import streamlit as st

# Initialize OpenAI client
openai.api_key = st.secrets['openai']

# Cache the encoding
ENCODING = tiktoken.get_encoding("gpt2")
GPT35_TOKEN_LENGTH = 16384
GPT4_TOKEN_LENGTH = 8192
GPT4_PLUS_TOKEN_LENGTH = 32768

MODEL = "gpt-3.5-turbo-16k"  # gpt-4 gpt-4-32k gpt-4-32k gpt-3.5-turbo-16k

if MODEL == "gpt-3.5-turbo-16k":
    TARGET_TOKEN_LENGTH = GPT35_TOKEN_LENGTH - 1000
elif MODEL == "gpt-4-32k":
    TARGET_TOKEN_LENGTH = GPT4_PLUS_TOKEN_LENGTH - 1000
else:
    TARGET_TOKEN_LENGTH = GPT4_TOKEN_LENGTH - 1000

# Update constants to uppercase names
OVERLAP = 100


def summarize_transcript(transcript: str, target_token_length=TARGET_TOKEN_LENGTH) -> str:
    system_prompt = "Produce a comprehensive summary of the transcript that is not only detailed but also retains key information. The summary should be substantive enough to serve as a foundation for creating various forms of written content, including but not limited to articles, social media posts, and other related materials. In the generated summary retain enough information from the original content to recreate the original content."
    chunks = break_transcript_into_chunks(transcript)

    prompt_response = []
    for i, chunk in enumerate(chunks):
        logging.info(f"Chunk {i}: {len(chunk)} tokens")
        prompt_request = ENCODING.decode(chunk)
        summary = generate_summary(prompt_request, system_prompt)
        prompt_response.append(summary)

    summarised_transcript = " ".join(prompt_response)

    # Adjust the length of the summarized transcript
    while count_tokens(summarised_transcript) > target_token_length:
        logging.info("Summarized transcript is still too long. Summarizing further.")
        chunks = break_transcript_into_chunks(summarised_transcript)
        prompt_response = []

        for i, chunk in enumerate(chunks):
            logging.info(f"Chunk {i}: {len(chunk)} tokens")
            prompt_request = ENCODING.decode(chunk)
            summary = generate_summary(prompt_request, "Remove redundant or repeated information")
            prompt_response.append(summary)

        summarised_transcript = " ".join(prompt_response)
        summarised_transcript_tokens = ENCODING.encode(summarised_transcript)

        if count_tokens(summarised_transcript) > target_token_length:
            summarised_transcript = ENCODING.decode(summarised_transcript_tokens[:bisect.bisect(summarised_transcript_tokens, target_token_length)])

    consolidated_summary = generate_summary(summarised_transcript, "Consolidate summary")

    return consolidated_summary


def break_transcript_into_chunks(transcript, chunk_size=TARGET_TOKEN_LENGTH, overlap=OVERLAP):
    """Break a transcript into chunks of a specified size with a specified overlap.

    Args:
        transcript (str): The transcript to break into chunks.
        chunk_size (int): The size of each chunk in number of tokens (default: 2000).
        overlap (int): The amount of overlap between each chunk in number of tokens (default: 100).

    Returns:
        list of lists: A list of chunks, where each chunk is a list of tokens.
    """
    # Count the number of tokens in the transcript
    tokens = ENCODING.encode(transcript)

    # Break the transcript into chunks with the specified size and overlap
    chunks = deque([tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size-overlap)])

    return chunks

def generate_summary(prompt_request: str, role_system_content: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": role_system_content},
                {"role": "user", "content": prompt_request},
            ],
            temperature=0.5,
            frequency_penalty=0,
            presence_penalty=0.6,
        )
        return response["choices"][0]["message"]["content"].strip()
    except OpenAIError as e:
        logging.error(f"Failed to generate summary with OpenAI: {str(e)}")
        raise

def count_tokens(transcript):
    """Count the number of tokens in a transcript."""
    return len(ENCODING.encode(transcript))


def generate_therapy_session_report(transcript):
    transcript_summary = summarize_transcript(transcript, 3500) if count_tokens(transcript) > 3500 else transcript
    
    instruction = """ Instruction: Using the provided therapy transcript: Generate a mental health report based on a therapy session with a patient. Include the following sections as headings and provide information under each section:
        Patient Information:
        Reason for Treatment:
        Assessment:
        Treatment Plan:
        Progress Notes:
        Medication:
        Referrals:
        Risk Assessment:
        Legal and Ethical Considerations:

        Generate a mental health report based on the above prompt, following the given sections above and providing relevant information under each section.
        Use the NICE(national institute for health and care excellence) Attention deficit hyperactivity disorder: diagnosis and management as your knowledge base and make references in each section of your response.
        If there is not enough information in transcript in to generate the report, always reply with "Not Enough Information".
        Do not make up information about the patient that is not in the transcript
        """
    report = openai.ChatCompletion.create(
        model= "gpt-4",  #"gpt-3.5-turbo", 
        messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": transcript_summary},
            ],
        temperature= 0.5,
        frequency_penalty= 0,
        presence_penalty= 0.6
    )

    return report["choices"][0]["message"]["content"]

def generate_blog(data, data_type, transcript_summarised=None) -> str:
    instruction = ("Revise the provided transcript into a comprehensive, captivating, and informative blog post by extracting "
                   "essential points, themes, and discussions. Compose a title, introduction, main body with well-structured headings, "
                   "and conclusion that effectively convey the content's depth and appeal.")
    return generate_social_content(data, data_type, instruction, transcript_summarised)

def generate_linkedIn_post_from_transcript(data, tone, length, data_type, transcript_summarised=None) -> str:
    instruction = "You convert transcripts into an engaging LinkedIn post. The post should have a {tone} tone and be {length} length. Add 3 relevant hashtags at the end of the post."
    return generate_social_content(data, data_type, instruction, transcript_summarised)



def generate_social_content(data, data_type, instruction, transcript_summarised=None):
    start_time = timeit.default_timer()
    token_length_check = TARGET_TOKEN_LENGTH
    transcript_summary = transcript_summarised

    if data_type != "transcript":
        logging.info("Generating content post from data analysis and directly from transcript")
        transcript_word, transcript_word_timestamp = transcript_combination_timestamp.transcript_combination_timestamp_for_content_generation(data)
        if not transcript_word:
            logging.error("Transcript data is empty")
            raise ValueError("Transcript cannot be empty.")
        
        summarize_start_time = timeit.default_timer()
        transcript_summary = summarize_transcript(transcript_word, token_length_check) if count_tokens(transcript_word) > token_length_check else transcript_word
        summarize_end_time = timeit.default_timer()

        # Save the summary in the database


        logging.info(f"Time taken for summarizing transcript: {summarize_end_time - summarize_start_time:.2f} seconds")
    else:
        logging.info("Generating content post from summary directly")

    try:
        social_post = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": transcript_summary},
            ],
            temperature=.5,
            frequency_penalty=0,
            presence_penalty=0.6
        )
    except OpenAIError as e:
        logging.error(f"Failed to generate content with OpenAI: {str(e)}")
        raise

    api_end_time = timeit.default_timer()
    end_time = timeit.default_timer()

    logging.info(f"Time taken for OPENAI_API call: {api_end_time - start_time:.2f} seconds")
    logging.info(f"Total time taken for process: {end_time - start_time:.2f} seconds")

    return social_post["choices"][0]["message"]["content"].strip()


def answer_queries(transcript, query):
    instruction = """
    Instruction: You are a knowledgeable assistant. Use the information provided in the given transcript to answer the user's questions. If the information needed to answer a question is not present in the transcript, respond with "I'm sorry, the information needed to answer your question is not in the provided transcript." Do not generate or assume any information that is not present in the transcript.
    """

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": transcript},
            {"role": "user", "content": query},
        ],
        temperature=0.5,
        frequency_penalty=0,
        presence_penalty=0.6
    )

    return response["choices"][0]["message"]["content"]
