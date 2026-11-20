import os
import re
import pandas as pd
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Nutrition Assistant", page_icon="🥗", layout="centered")

st.title("🥗 AI Nutrition Assistant")
st.caption("Ask about foods and quantities. Nutrition values are calculated from the included CSV dataset.")

DATA_PATH = os.path.join(os.path.dirname(__file__), "nutrition_dataset.csv")
NUTRIENTS = {
    "Calories": ("calories_kcal_per_100g", "kcal"),
    "Protein": ("protein_g_per_100g", "g"),
    "Carbohydrates": ("carbs_g_per_100g", "g"),
    "Fat": ("fat_g_per_100g", "g"),
    "Fiber": ("fiber_g_per_100g", "g"),
    "Sugar": ("sugar_g_per_100g", "g"),
}

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

try:
    df = load_data()
except Exception as exc:
    st.error(f"Could not load nutrition_dataset.csv: {exc}")
    st.stop()

def find_food(message):
    """Find a food name mentioned in the user's message (longest exact name first)."""
    text = message.lower()
    foods = sorted(df["food_name"].astype(str).tolist(), key=len, reverse=True)
    for food in foods:
        if re.search(r"(?<![a-z])" + re.escape(food.lower()) + r"(?![a-z])", text):
            return food
    return None

def find_quantity(message):
    """Read a quantity such as 150g, 150 g, or 150 grams."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:g|gram|grams)\b", message.lower())
    return float(match.group(1)) if match else None

def nutrition_result(food, grams):
    row = df[df["food_name"].str.lower() == food.lower()].iloc[0]
    result = {}
    for label, (column, unit) in NUTRIENTS.items():
        result[label] = (float(row[column]) * grams / 100, unit)
    return result

def format_report(food, grams, result):
    lines = [f"### Nutrition for {grams:g} g of {food}", "",
             "Calculated from the included dataset:"]
    for label, (value, unit) in result.items():
        lines.append(f"- **{label}:** {value:.2f} {unit}")
    lines.append("")
    lines.append("_Values are approximate and depend on food variety and preparation._")
    return "\n".join(lines)

def ask_groq(history, user_message, api_key):
    client = Groq(api_key=api_key)
    dataset_context = df.to_csv(index=False)
    system_message = {
        "role": "system",
        "content": (
            "You are a friendly nutrition education assistant. Be concise and clear. "
            "Use ONLY the dataset below for exact numeric food nutrition values. "
            "Values are per 100 grams. Never invent missing food values or claim this small "
            "sample dataset contains every food. If the user asks for a quantity, explain "
            "that the app can calculate it when they provide grams. You may explain general "
            "nutrition concepts, but do not diagnose, prescribe diets, or replace a registered "
            "dietitian or clinician. For allergies, pregnancy, disease, or medical diets, "
            "advise professional guidance.\n\nDATASET (CSV):\n" + dataset_context
        ),
    }
    messages = [system_message]
    messages.extend(history[-10:])
    messages.append({"role": "user", "content": user_message})
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.3,
        max_tokens=700,
    )
    return response.choices[0].message.content

with st.sidebar:
    st.header("Dataset foods")
    st.write(", ".join(df["food_name"].astype(str).tolist()))
    st.caption("Enter a food name and quantity, for example: 200g Banana")
    st.divider()
    st.write("**Dataset note:** Values are approximate educational sample values per 100g.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_text = st.chat_input("Try: How much nutrition is in 200g Banana?")
if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        food = find_food(user_text)
        grams = find_quantity(user_text)

        if food and grams is not None and grams > 0:
            result = nutrition_result(food, grams)
            answer = format_report(food, grams, result)
            st.markdown(answer)
        elif food and (grams is None or grams <= 0):
            answer = f"I found **{food}** in the dataset. Please tell me the quantity in grams, for example: `150g {food}`."
            st.markdown(answer)
        else:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                answer = (
                    "Groq API key not found. Add your NEW key to a `.env` file as "
                    "`GROQ_API_KEY=your_new_key_here`, then restart the app."
                )
                st.warning(answer)
            else:
                try:
                    history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                    ]
                    answer = ask_groq(history, user_text, api_key)
                    st.markdown(answer)
                except Exception as exc:
                    answer = f"Sorry, the Groq request failed: {exc}"
                    st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

if st.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()
