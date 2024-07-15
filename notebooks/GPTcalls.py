import os
import openai
from dotenv import load_dotenv

"""
Here we have a series of functions with the help of LLMs and prompts for generating texts.

* question_gpt: Function formatted to generate questions based on a context. This function has two inputs,
context and previous. The input context passes on the information from the generated chunks and previous stores the generated questions 
by chunk to avoid repeating questions.

* answer_gpt: Function to generate responses. As input, this function receives the result generated from the previous function and the context
to adopt as a reference to answer the question.

* create_instruction: Function formatted to generate instructions about the generated datasets. One instruction is generated per document
and this is used for the entire document dataset.

* aux_processing: Function to optimize the context generated from the initial text processing. The functions of the DocProcessing class
use basic resources to partition documents, the partitions are used as input to the function in question and thus a 
new output is generated, taking into account the presented context.

"""

class GPTAssistant:
    def __init__(self, deployment):
        load_dotenv(override=True)
        self.endpoint = os.environ.get("OPENAI_URL")
        self.api_key = os.environ.get("OPEN_AI_KEY")
        self.deployment = deployment
        self.client = openai.AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version="2024-02-01",
        )

    def question_gpt(self,context,previous):
        response = self.client.chat.completions.create(model = self.deployment,
        messages=[
            {"role": "system", "content": f"""
                You operate as an assistant to create questions. Use this {context} to create a direct question.
                Follow the following instructions:
                1. Ensure each question is clear, concise and directly related to the content provided.
                2. Create just one English question.
                3. Avoid questions about tables, figures and quotations.
                4. Create a different question than these: {previous}.
                5. Use only the information described in {context}.
                """},
            {"role": "user", "content": f"Use this text {context}, generate a question that explores a specific aspect of the text provided."}])
        
        return response.choices[0].message.content
    
    def answer_gpt(self, context, question):
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": f"""
                    You act as an assistant to answer questions based on the text provided. Use this text ONLY this {context} to answer this question {question}.
                    Follow the following instructions:
                    1. Read the text.
                    2. Create an answer based solely on the text provided.
                    3. Do not use external information. Use ONLY this {context} to answer the question.
                    4. Create the answer in English.
                    5. Avoid mentioning figures, tables, or document metadata.
                """},
                {"role": "user", "content": f"For that question {question}, create a short answer. Use this information {context} as context to answer the question."}
            ]
        )
        return response.choices[0].message.content
    
    def create_instruction(self, context):
        full_response = ""
        stop = False
        max_attempts = 1

        for _ in range(max_attempts):
            if stop:
                break

            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": f"""
                    Be succinct and create a brief description based on the information presented. 
                    Provide general information, avoiding specifics like confidentiality levels, and avoid referencing authors, books, or similar sources.
                    Start with "Here we have information about..." and avoid referring to figures, tables, or metadata.
                    """},
                    {"role": "user", "content": f"Based on this information {context}, create an instruction."}
                ], temperature= 0.9,
                max_tokens=50,
                stop=["."]
            )

            partial_response = response.choices[0].message.content
            full_response += partial_response

            if full_response.endswith('.'):
                stop = True

        if not full_response.endswith('.'):
            full_response += '.'

        return full_response
    
    def aux_processing(self, previous, next, actual):
        response = self.client.chat.completions.create(
            model = self.deployment,
            messages=[
                {"role": "system", "content": f"""
                You serve as a tool for filtering, organizing and structuring texts.
                """},
                {"role": "user", "content": f"""
                A large document was divided into parts and the inputs are:
                Previous: Part of the previous text partition.
                Actual: Partition the current text.
                Next: Part of the following text partition.
                The strategy is to use the current partition with parts of the texts that surround it to create context.
                You must rewrite the text considering the information provided in order to filter the text in order to generate a consolidated and clean text output.
                Follow the following instructions:
                1. Try to maintain the structure of the text provided as much as possible.
                2. When identifying that this is information about the cover, back cover, summary, dedication, Epigraph, Preface and Presentation of a document, article or book, do not include this information.
                3. Please do not reference Figures, Tables, Sections and Images in the generated text (e.g.: "As seen in Figure x ...", "As seen in Table x ...", "This text presents ...")
                4. When checking unfinished information, remove it.
                5. Do not include information about authors.
                6. Prioritize maintaining a good context.
                Given this information:
                * Previous text: {previous}
                * Actual text: {actual}
                * Next text: {next}
                Create output text taking into account the guidelines above.
                """}
            ]
        )
        return response.choices[0].message.content