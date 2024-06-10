import openai
import os
from dotenv import load_dotenv
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import random

# Load the API key from .env file
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Define the LLM model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# Function to generate prompts
def generate_prompts(user_input, scenarios):
    template = """
    You are a creative assistant. Generate a marketing slogan for the following scenario:
    Task: {user_input}
    Scenario: {scenario}
    """
    prompts = []
    for scenario in scenarios:
        prompt_template = PromptTemplate(template=template, input_variables=["user_input", "scenario"])
        chain = LLMChain(llm=llm, prompt=prompt_template)
        result = chain.run(user_input=user_input, scenario=scenario)
        prompts.append(result)
    return prompts

# Function to evaluate prompts
def evaluate_prompts(user_input, prompts):
    vectorizer = TfidfVectorizer().fit_transform([user_input] + prompts)
    vectors = vectorizer.toarray()
    user_vector = vectors[0]
    prompt_vectors = vectors[1:]
    scores = cosine_similarity([user_vector], prompt_vectors)[0]
    return scores

# Monte Carlo matchmaking function
def monte_carlo_matchmaking(prompts, evaluation_scores, num_simulations=100):
    num_prompts = len(prompts)
    wins = np.zeros(num_prompts)

    for _ in range(num_simulations):
        for i in range(num_prompts):
            for j in range(i + 1, num_prompts):
                if random.random() < evaluation_scores[i] / (evaluation_scores[i] + evaluation_scores[j]):
                    wins[i] += 1
                else:
                    wins[j] += 1

    return wins

# ELO rating system functions
def expected_score(rating1, rating2):
    return 1 / (1 + 10 ** ((rating2 - rating1) / 400))

def update_elo_ratings(ratings, wins, num_simulations=100):
    K = 32
    for i in range(len(ratings)):
        for j in range(len(ratings)):
            if i != j:
                expected_i = expected_score(ratings[i], ratings[j])
                actual_i = wins[i] / num_simulations
                ratings[i] += K * (actual_i - expected_i)
    return ratings

# Streamlit UI
st.title("AI-Powered Prompt Generation System")

user_input = st.text_input("Enter your task or objective:", "Generate a marketing slogan for an eco-friendly product")
scenarios_input = st.text_input("Enter scenarios (comma-separated):", "short and catchy, emphasizes sustainability, targeted at young adults")

if st.button("Generate Prompts"):
    scenarios = [scenario.strip() for scenario in scenarios_input.split(',')]
    generated_prompts = generate_prompts(user_input, scenarios)
    evaluation_scores = evaluate_prompts(user_input, generated_prompts)
    num_simulations = 100
    wins = monte_carlo_matchmaking(generated_prompts, evaluation_scores, num_simulations)
    initial_ratings = np.full(len(generated_prompts), 1500)
    final_ratings = update_elo_ratings(initial_ratings, wins, num_simulations)

    st.subheader("Generated Prompts and Evaluations")
    for prompt, score, rating in zip(generated_prompts, evaluation_scores, final_ratings):
        st.write(f"**Prompt:** {prompt}")
        st.write(f"**Score:** {score:.2f}")
        st.write(f"**ELO Rating:** {rating:.2f}")
        st.write("---")

    best_prompt_index = np.argmax(final_ratings)
    best_prompt = generated_prompts[best_prompt_index]
    st.subheader(f"The best prompt is: {best_prompt}")

