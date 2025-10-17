import streamlit as st
from PIL import Image
import io, json, os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

st.set_page_config(page_title="Smart Recipe Generator", layout="wide")

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")

MODEL = load_model()

# ------------------ DATA ------------------
RECIPE_DB = [
    {"id":1,"title":"Tomato Basil Pasta","cuisine":"Italian","ingredients":["pasta","tomato","basil","olive oil","garlic","salt","pepper"],
     "steps":["Boil pasta","Saute garlic & tomato","Toss pasta with tomato and basil","Serve"],"nutrition":{"calories":420,"protein":12},"difficulty":"easy","cook_time":20,"tags":["vegetarian"]},
    {"id":2,"title":"Chicken Stir Fry","cuisine":"Chinese","ingredients":["chicken","soy sauce","garlic","ginger","bell pepper","onion","oil"],
     "steps":["Slice chicken","Stir-fry veggies","Add chicken & sauce","Cook until done"],"nutrition":{"calories":380,"protein":28},"difficulty":"medium","cook_time":25,"tags":[]},
    {"id":3,"title":"Veggie Omelette","cuisine":"French","ingredients":["eggs","milk","bell pepper","onion","salt","pepper","butter"],
     "steps":["Beat eggs with milk","Cook veggies","Pour eggs and fold"],"nutrition":{"calories":260,"protein":18},"difficulty":"easy","cook_time":10,"tags":["vegetarian","gluten-free"]},
    {"id":4,"title":"Mango Smoothie","cuisine":"Indian","ingredients":["mango","yogurt","milk","honey","ice"],
     "steps":["Blend all ingredients until smooth","Serve chilled"],"nutrition":{"calories":210,"protein":6},"difficulty":"easy","cook_time":5,"tags":["vegetarian"]},
    {"id":5,"title":"Chickpea Curry","cuisine":"Indian","ingredients":["chickpeas","tomato","onion","garlic","ginger","cumin","coriander","oil"],
     "steps":["Saute onions & spices","Add tomato & chickpeas","Simmer"],"nutrition":{"calories":350,"protein":14},"difficulty":"medium","cook_time":35,"tags":["vegetarian","vegan"]},
    {"id":6,"title":"Greek Salad","cuisine":"Greek","ingredients":["cucumber","tomato","feta","olive oil","olives","onion","oregano"],
     "steps":["Chop veggies","Toss with feta, oil & oregano"],"nutrition":{"calories":220,"protein":6},"difficulty":"easy","cook_time":10,"tags":["vegetarian","gluten-free"]},
    {"id":7,"title":"Pancakes","cuisine":"American","ingredients":["flour","milk","egg","sugar","baking powder","butter"],
     "steps":["Mix batter","Cook on griddle","Serve with syrup"],"nutrition":{"calories":350,"protein":8},"difficulty":"easy","cook_time":20,"tags":["vegetarian"]},
    {"id":8,"title":"Lentil Soup","cuisine":"Middle Eastern","ingredients":["lentils","carrot","onion","garlic","cumin","salt","pepper"],
     "steps":["Saute veggies","Add lentils & water","Simmer until tender"],"nutrition":{"calories":250,"protein":12},"difficulty":"easy","cook_time":40,"tags":["vegetarian","vegan","gluten-free"]},
    {"id":9,"title":"Baked Salmon","cuisine":"International","ingredients":["salmon","lemon","olive oil","salt","pepper","dill"],
     "steps":["Season salmon","Bake at 200C until cooked"],"nutrition":{"calories":360,"protein":30},"difficulty":"medium","cook_time":20,"tags":["gluten-free"]},
    {"id":10,"title":"Avocado Toast","cuisine":"American","ingredients":["bread","avocado","salt","pepper","lemon"],
     "steps":["Toast bread","Mash avocado with lemon & salt","Spread on toast"],"nutrition":{"calories":270,"protein":6},"difficulty":"easy","cook_time":5,"tags":["vegetarian"]}
]

SUBSTITUTIONS = {
    "milk":["soy milk","almond milk"],
    "yogurt":["greek yogurt","plant yogurt"],
    "butter":["margarine","olive oil"],
    "egg":["flaxseed meal (1 tbsp) + water (3 tbsp)"],
    "chicken":["tofu","mushroom"],
    "beef":["mushroom","jackfruit (young)"]
}

IMAGENET_TO_ING = {
    "banana":"banana","orange":"orange","lemon":"lemon","pineapple":"pineapple",
    "broccoli":"broccoli","mushroom":"mushroom","bell_pepper":"bell pepper",
    "strawberry":"berries","gar":"garlic","granny_smith":"apple","custard_apple":"apple"
}

# ------------------ FUNCTIONS ------------------
def recognize_ingredients(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224,224))
    x = np.expand_dims(np.array(img), axis=0)
    x = preprocess_input(x)
    preds = MODEL.predict(x)
    decoded = decode_predictions(preds, top=5)[0]
    ingredients = set()
    for (_, label, _) in decoded:
        label = label.lower().replace(" ","_")
        if label in IMAGENET_TO_ING:
            ingredients.add(IMAGENET_TO_ING[label])
    return list(ingredients)

def match_recipes(provided_ings, dietary=None, max_time=None, difficulty=None):
    provided = set(i.strip().lower() for i in provided_ings)
    results = []
    for r in RECIPE_DB:
        if dietary and dietary not in r["tags"]:
            continue
        if max_time and r["cook_time"] > max_time:
            continue
        if difficulty and r["difficulty"] != difficulty:
            continue
        matched = provided & set(r["ingredients"])
        if matched:
            score = len(matched) / len(r["ingredients"])
            results.append((score, matched, r))
    results.sort(reverse=True, key=lambda x: x[0])
    return results[:5]

def suggest_subs(missing):
    return {m: SUBSTITUTIONS.get(m, []) for m in missing if m in SUBSTITUTIONS}

# ------------------ STREAMLIT UI ------------------
st.title("🥣 Smart Recipe Generator")
st.write("Upload ingredient photos or enter ingredients manually to get recipe suggestions!")

with st.sidebar:
    st.header("Filters & Preferences")
    dietary = st.selectbox("Dietary preference", ["None","vegetarian","vegan","gluten-free"])
    if dietary == "None": dietary = None
    difficulty = st.selectbox("Difficulty", ["Any","easy","medium"])
    if difficulty == "Any": difficulty = None
    max_time = st.number_input("Max cooking time (minutes)", min_value=0, value=0)
    if max_time == 0: max_time = None

# Image upload
uploaded_files = st.file_uploader("Upload ingredient photos (optional)", type=["jpg","png","jpeg"], accept_multiple_files=True)
recognized = []
if uploaded_files:
    for f in uploaded_files:
        bytes_data = f.read()
        rec = recognize_ingredients(bytes_data)
        recognized.extend(rec)
        st.image(f, caption=f.name, width=150)
    st.success(f"Recognized ingredients: {', '.join(set(recognized))}")

# Text input
typed_ings = st.text_input("Enter ingredients (comma-separated)", "")
typed_ings_list = [i.strip() for i in typed_ings.split(",") if i.strip()]
all_ings = list(set(typed_ings_list + recognized))

if st.button("Generate Recipes"):
    if not all_ings:
        st.warning("Please provide ingredients or upload images.")
    else:
        st.info(f"Searching recipes using: {', '.join(all_ings)}")
        matches = match_recipes(all_ings, dietary, max_time, difficulty)
        if not matches:
            st.error("No matches found. Try adding more ingredients.")
        else:
            for score, matched, r in matches:
                with st.expander(f"🍽️ {r['title']} ({r['cuisine']}) — {r['difficulty'].title()}, {r['cook_time']} min"):
                    st.markdown(f"**Matched Ingredients:** {', '.join(matched)}")
                    missing = set(r['ingredients']) - set(all_ings)
                    if missing:
                        st.markdown(f"**Missing:** {', '.join(missing)}")
                        subs = suggest_subs(missing)
                        if subs:
                            st.markdown("**Substitution suggestions:**")
                            for k,v in subs.items():
                                st.markdown(f"- {k}: {', '.join(v)}")
                    st.subheader("Ingredients")
                    st.write(", ".join(r["ingredients"]))
                    st.subheader("Steps")
                    for i, step in enumerate(r["steps"], 1):
                        st.write(f"{i}. {step}")
                    st.subheader("Nutrition Info")
                    st.write(r["nutrition"])

st.markdown("---")
st.caption("Built with ❤️ using Streamlit & TensorFlow | Smart Recipe Generator Demo")
