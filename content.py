
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
    transcript_summary = summarize_transcript(transcript) if count_tokens(transcript) > GPT35_TOKEN_LENGTH else transcript
    
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
        model= MODEL,
        messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": transcript_summary},
            ],
        temperature= 0.5,
        frequency_penalty= 0,
        presence_penalty= 0.6
    )

    return report["choices"][0]["message"]["content"]

def generate_therapy_session_report(transcript):
    transcript_summary = summarize_transcript(transcript) if count_tokens(transcript) > GPT35_TOKEN_LENGTH else transcript
    
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
        model= MODEL,
        messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": transcript_summary},
            ],
        temperature= 0.5,
        frequency_penalty= 0,
        presence_penalty= 0.6
    )

    return report["choices"][0]["message"]["content"]

def generate_meeting_notes(transcript):
    transcript_summary = summarize_transcript(transcript) if count_tokens(transcript) > GPT35_TOKEN_LENGTH else transcript
    
    instruction = """ 
    Instruction: Using the provided meeting transcript, generate detailed meeting notes. Include the following sections as headings and provide information under each section:
        Meeting Date and Time:
        Attendees:
        Key Discussion Points:
        Decisions Made:
        Action Items:
        Next Steps:
        Any Other Business:

    Generate meeting notes based on the above prompt, following the given sections above and providing relevant information under each section.

    If there is not enough information in transcript to generate the meeting notes, always reply with "Not Enough Information".

    Do not make up information that is not in the transcript.
    """
    report = openai.ChatCompletion.create(
        model= MODEL,
        messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": transcript_summary},
            ],
        temperature= 0.5,
        frequency_penalty= 0,
        presence_penalty= 0.6
    )

    return report["choices"][0]["message"]["content"]

def generate_blog_post(transcript):
    transcript_summary = summarize_transcript(transcript) if count_tokens(transcript) > GPT35_TOKEN_LENGTH else transcript
    
    instruction = """ 
    Instruction: Using the provided transcript, generate a detailed and engaging blog post. Include the following sections as headings and provide information under each section:
        Introduction:
        Main Body:
        Conclusion:

    Generate a blog post based on the above prompt, following the given sections above and providing relevant information under each section. 

    The blog post should be engaging and easy to read, with a clear structure that includes an introduction, a main body with several key points or sections, and a conclusion.

    If there is not enough information in transcript to generate the blog post, always reply with "Not Enough Information".

    Do not make up information that is not in the transcript.
    """
    blog_post = openai.ChatCompletion.create(
        model= MODEL,
        messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": transcript_summary},
            ],
        temperature= 0.5,
        frequency_penalty= 0,
        presence_penalty= 0.6
    )

    return blog_post["choices"][0]["message"]["content"]

def generate_statement_of_work(transcript):
    transcript_summary = summarize_transcript(transcript) if count_tokens(transcript) > GPT35_TOKEN_LENGTH else transcript
    
    instruction = """ 
    Instruction: Using the provided consultation meeting transcript, generate a detailed Statement of Work (SoW). Include the following sections as headings and provide information under each section:
        Project Overview:
        Scope of Work:
        Deliverables:
        Timeline:
        Payment Schedule:
        Terms and Conditions:

    Generate a SoW based on the above prompt, following the given sections above and providing relevant information under each section. 

    The SoW should be clear and concise, outlining the work to be done, the deliverables, the timeline, the payment schedule, and any terms and conditions.

    If there is not enough information in the transcript to generate the SoW, always reply with "Not Enough Information".

    Do not make up information that is not in the transcript.
    """
    sow = openai.ChatCompletion.create(
        model= MODEL,
        messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": transcript_summary},
            ],
        temperature= 0.5,
        frequency_penalty= 0,
        presence_penalty= 0.6
    )

    return sow["choices"][0]["message"]["content"]

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
