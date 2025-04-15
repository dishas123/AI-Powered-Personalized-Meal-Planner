import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import requests
from io import BytesIO
import re

# Set page config
st.set_page_config(
    page_title="NutriSmart",
    page_icon="🥗",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #1E88E5;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .stat-card {
        background-color: #f5f5f5;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .food-card {
        background-color: #f5f5f5;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .highlight {
        color: #1E88E5;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load and preprocess the dataset
@st.cache_data
def load_data():
    # Try different encodings to load the dataset
    encodings_to_try = ['latin-1', 'ISO-8859-1', 'cp1252', 'utf-8-sig']
    
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv("https://raw.githubusercontent.com/chandan232002/Health-based-food-recommendation-system-/main/IndianFoodDatasetXLSFinal%20(3).csv", 
                            encoding=encoding)
            st.success(f"Successfully loaded dataset with encoding: {encoding}")
            
            # Clean the data
            df = df.dropna()
            
            # Convert numerical columns to appropriate data types
            numerical_cols = ['Calories', 'Protein', 'Fat', 'Carbohydrates', 'Fiber', 'Sugar']
            for col in numerical_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Replace NaN values with column means
            df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
            
            # Clean the state column
            df['State'] = df['State'].str.strip()
            
            return df
        except Exception as e:
            continue
    
    # If all encoding attempts fail, try to download the raw file
    try:
        import io
        import requests
        
        url = "https://raw.githubusercontent.com/chandan232002/Health-based-food-recommendation-system-/main/IndianFoodDatasetXLSFinal%20(3).csv"
        s = requests.get(url).content
        
        # Try to detect encoding
        try:
            import chardet
            detected = chardet.detect(s)
            encoding = detected['encoding']
            st.info(f"Detected encoding: {encoding}")
        except:
            encoding = 'latin-1'  # Fallback encoding
            
        df = pd.read_csv(io.StringIO(s.decode(encoding)))
        
        # Clean the data
        df = df.dropna()
        
        # Convert numerical columns to appropriate data types
        numerical_cols = ['Calories', 'Protein', 'Fat', 'Carbohydrates', 'Fiber', 'Sugar']
        for col in numerical_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Replace NaN values with column means
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
        
        # Clean the state column
        df['State'] = df['State'].str.strip()
        
        return df
    except Exception as e:
        st.error(f"All encoding attempts failed: {e}")
        
        # Last resort: Use a hardcoded sample dataset
        st.warning("Using a sample dataset instead.")
        return create_sample_dataset()

def create_sample_dataset():
    """Create a sample dataset when the original cannot be loaded"""
    data = {
        'Name': ['Butter Chicken', 'Palak Paneer', 'Idli Sambar', 'Masala Dosa', 'Biryani', 
                'Chole Bhature', 'Tandoori Chicken', 'Rajma Chawal', 'Aloo Gobi', 'Pav Bhaji'],
        'Calories': [350, 280, 180, 250, 400, 450, 320, 330, 200, 350],
        'Protein': [25, 15, 6, 8, 20, 12, 30, 15, 6, 10],
        'Fat': [22, 18, 2, 10, 15, 25, 12, 8, 10, 18],
        'Carbohydrates': [10, 12, 35, 40, 50, 60, 5, 45, 25, 45],
        'Fiber': [2, 5, 4, 3, 3, 6, 0, 12, 5, 4],
        'Sugar': [3, 4, 2, 2, 2, 3, 1, 2, 3, 6],
        'Diet': ['non vegetarian', 'vegetarian', 'vegetarian', 'vegetarian', 'non vegetarian',
                'vegetarian', 'non vegetarian', 'vegetarian', 'vegetarian', 'vegetarian'],
        'Course': ['Main Course', 'Main Course', 'Breakfast', 'Breakfast', 'Main Course',
                  'Breakfast', 'Main Course', 'Main Course', 'Side Dish', 'Snack'],
        'State': ['Punjab', 'Punjab', 'Tamil Nadu', 'Karnataka', 'Hyderabad',
                 'Delhi', 'Punjab', 'Punjab', 'Punjab', 'Maharashtra'],
        'Ingredients': ['Chicken, Butter, Cream, Tomato, Spices', 'Spinach, Paneer, Cream, Spices',
                       'Rice flour, Black gram, Sambar', 'Rice, Potato, Spices', 'Rice, Meat, Spices',
                       'Chickpeas, Flour, Spices', 'Chicken, Yogurt, Spices', 'Kidney beans, Rice, Spices',
                       'Potato, Cauliflower, Spices', 'Potato, Vegetables, Butter, Bread']
    }
    
    return pd.DataFrame(data)

# Get recommendations based on nutritional goals
def get_recommendations(df, user_profile, dietary_preferences, excluded_ingredients, n_recommendations=5):
    # Copy the dataset to avoid modifying the original
    dataset = df.copy()
    
    # Filter based on dietary preferences
    if 'Vegetarian' in dietary_preferences and 'Non-Vegetarian' not in dietary_preferences:
        dataset = dataset[dataset['Diet'] == 'vegetarian']
    elif 'Non-Vegetarian' in dietary_preferences and 'Vegetarian' not in dietary_preferences:
        dataset = dataset[dataset['Diet'] == 'non vegetarian']
    
    # Filter out excluded ingredients
    if excluded_ingredients:
        for ingredient in excluded_ingredients:
            dataset = dataset[~dataset['Ingredients'].str.lower().str.contains(ingredient.lower())]
    
    # If dataset is empty after filtering
    if dataset.empty:
        return None
    
    # Features to consider for recommendation
    features = ['Calories', 'Protein', 'Fat', 'Carbohydrates', 'Fiber', 'Sugar']
    
    # Standardize the features
    scaler = StandardScaler()
    dataset_scaled = scaler.fit_transform(dataset[features])
    
    # Define ideal nutritional profile based on user input
    ideal_profile = np.array([
        user_profile['calorie_goal'],
        user_profile['protein_goal'],
        user_profile['fat_goal'],
        user_profile['carb_goal'],
        user_profile['fiber_goal'],
        user_profile['sugar_goal']
    ]).reshape(1, -1)
    
    # Standardize the ideal profile
    ideal_profile_scaled = scaler.transform(ideal_profile)
    
    # Calculate similarity scores
    similarity_scores = cosine_similarity(dataset_scaled, ideal_profile_scaled)
    dataset['similarity'] = similarity_scores
    
    # Get top recommendations
    recommendations = dataset.sort_values('similarity', ascending=False).head(n_recommendations)
    
    return recommendations

# Create meal plan from recommendations
def create_meal_plan(recommendations, days=3):
    # If not enough recommendations, adjust days
    if len(recommendations) < days * 3:  # Assuming 3 meals per day
        days = len(recommendations) // 3
        if days == 0:
            days = 1
    
    meal_plan = {}
    food_items = recommendations.sample(days * 3) if len(recommendations) >= days * 3 else recommendations
    
    food_counter = 0
    for day in range(1, days + 1):
        meal_plan[f"Day {day}"] = {
            "Breakfast": food_items.iloc[food_counter].to_dict() if food_counter < len(food_items) else None,
            "Lunch": food_items.iloc[food_counter + 1].to_dict() if food_counter + 1 < len(food_items) else None,
            "Dinner": food_items.iloc[food_counter + 2].to_dict() if food_counter + 2 < len(food_items) else None
        }
        food_counter += 3
    
    return meal_plan

# Calculate nutritional stats
def calculate_stats(meal_plan):
    stats = {
        "Total Calories": 0,
        "Total Protein": 0,
        "Total Fat": 0,
        "Total Carbs": 0,
        "Total Fiber": 0,
        "Total Sugar": 0
    }
    
    for day, meals in meal_plan.items():
        for meal_type, meal in meals.items():
            if meal:
                stats["Total Calories"] += meal["Calories"]
                stats["Total Protein"] += meal["Protein"]
                stats["Total Fat"] += meal["Fat"]
                stats["Total Carbs"] += meal["Carbohydrates"]
                stats["Total Fiber"] += meal["Fiber"]
                stats["Total Sugar"] += meal["Sugar"]
    
    # Calculate per day averages
    days = len(meal_plan)
    for stat in stats:
        stats[stat] = stats[stat] / days
    
    return stats

# Function to display meal plan
def display_meal_plan(meal_plan, user_profile):
    st.markdown("<h2 class='section-header'>Your Personalized Meal Plan</h2>", unsafe_allow_html=True)
    
    for day, meals in meal_plan.items():
        with st.expander(f"{day}", expanded=True):
            cols = st.columns(3)
            
            for i, (meal_type, meal) in enumerate(meals.items()):
                if meal:
                    with cols[i]:
                        st.markdown(f"<h3>{meal_type}</h3>", unsafe_allow_html=True)
                        st.markdown(f"<div class='food-card'>", unsafe_allow_html=True)
                        st.markdown(f"<h4>{meal['Name']}</h4>", unsafe_allow_html=True)
                        st.markdown(f"<p><span class='highlight'>Calories:</span> {meal['Calories']:.1f} kcal</p>", unsafe_allow_html=True)
                        st.markdown(f"<p><span class='highlight'>Protein:</span> {meal['Protein']:.1f} g</p>", unsafe_allow_html=True)
                        st.markdown(f"<p><span class='highlight'>Carbs:</span> {meal['Carbohydrates']:.1f} g</p>", unsafe_allow_html=True)
                        st.markdown(f"<p><span class='highlight'>Fat:</span> {meal['Fat']:.1f} g</p>", unsafe_allow_html=True)
                        st.markdown(f"<p><span class='highlight'>Ingredients:</span> {meal['Ingredients']}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p><span class='highlight'>Course:</span> {meal['Course']}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p><span class='highlight'>Region:</span> {meal['State']}</p>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
    
    # Display nutritional summary
    stats = calculate_stats(meal_plan)
    st.markdown("<h2 class='section-header'>Nutritional Summary (Daily Average)</h2>", unsafe_allow_html=True)
    
    cols = st.columns(3)
    with cols[0]:
        st.markdown("<div class='stat-card'>", unsafe_allow_html=True)
        st.markdown(f"<p><span class='highlight'>Calories:</span> {stats['Total Calories']:.1f} kcal</p>", unsafe_allow_html=True)
        st.markdown(f"<p><span class='highlight'>Protein:</span> {stats['Total Protein']:.1f} g</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown("<div class='stat-card'>", unsafe_allow_html=True)
        st.markdown(f"<p><span class='highlight'>Carbs:</span> {stats['Total Carbs']:.1f} g</p>", unsafe_allow_html=True)
        st.markdown(f"<p><span class='highlight'>Fat:</span> {stats['Total Fat']:.1f} g</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown("<div class='stat-card'>", unsafe_allow_html=True)
        st.markdown(f"<p><span class='highlight'>Fiber:</span> {stats['Total Fiber']:.1f} g</p>", unsafe_allow_html=True)
        st.markdown(f"<p><span class='highlight'>Sugar:</span> {stats['Total Sugar']:.1f} g</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Display macronutrient distribution chart
    st.markdown("<h2 class='section-header'>Macronutrient Distribution</h2>", unsafe_allow_html=True)
    
    # Calculate macronutrient percentages
    protein_cals = stats["Total Protein"] * 4
    carb_cals = stats["Total Carbs"] * 4
    fat_cals = stats["Total Fat"] * 9
    
    total_cals = protein_cals + carb_cals + fat_cals
    
    if total_cals > 0:
        protein_pct = (protein_cals / total_cals) * 100
        carb_pct = (carb_cals / total_cals) * 100
        fat_pct = (fat_cals / total_cals) * 100
        
        # Create a pie chart
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = ['Protein', 'Carbohydrates', 'Fat']
        sizes = [protein_pct, carb_pct, fat_pct]
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        plt.title('Macronutrient Distribution (% of Calories)')
        
        st.pyplot(fig)
    
    # Display recommendations based on user's goals
    st.markdown("<h2 class='section-header'>Recommendations Based on Your Goals</h2>", unsafe_allow_html=True)
    
    if user_profile['goal'] == 'Weight Loss':
        st.info("""
        🔹 Focus on high-protein, fiber-rich meals to maintain satiety.
        🔹 Adjust your meal timing to optimize metabolism.
        🔹 Consider intermittent fasting or smaller, more frequent meals based on your preference.
        🔹 Stay hydrated - aim for at least 8 glasses of water daily.
        🔹 Complement your diet with regular exercise - aim for 150+ minutes of moderate activity per week.
        """)
    elif user_profile['goal'] == 'Muscle Gain':
        st.info("""
        🔹 Ensure you're consuming enough protein (1.6-2.2g per kg of bodyweight).
        🔹 Don't neglect carbohydrates - they fuel your workouts and recovery.
        🔹 Time your meals around your workouts for optimal performance and recovery.
        🔹 Consider a pre-workout meal rich in carbs and a post-workout meal with protein and carbs.
        🔹 Progressive overload in your strength training is key to muscle growth.
        """)
    elif user_profile['goal'] == 'Maintenance':
        st.info("""
        🔹 Focus on nutrient density rather than calorie restriction.
        🔹 Aim for variety in your diet to ensure adequate micronutrient intake.
        🔹 Regular physical activity helps maintain muscle mass and metabolic health.
        🔹 Listen to your body's hunger and fullness cues.
        🔹 Prioritize sleep and stress management for overall health.
        """)
    elif user_profile['goal'] == 'Improve Health':
        st.info("""
        🔹 Focus on whole foods and minimize ultra-processed foods.
        🔹 Include a variety of colorful fruits and vegetables daily.
        🔹 Prioritize anti-inflammatory foods like fatty fish, nuts, seeds, and spices.
        🔹 Consider reducing sodium intake if you have high blood pressure.
        🔹 Regular physical activity, quality sleep, and stress management are key components of health.
        """)

# Main application
def main():
    # Header
    st.markdown("<h1 class='main-header'>NutriSmart</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Your Personalized Indian Cuisine Nutrition Planner</p>", unsafe_allow_html=True)
    
    # Load the dataset
    df = load_data()
    
    if df is None:
        st.error("Failed to load the dataset. Please try again later.")
        return
    
    # Sidebar for user inputs
    st.sidebar.markdown("<h2>Your Information</h2>", unsafe_allow_html=True)
    
    # User profile
    age = st.sidebar.slider("Age", min_value=18, max_value=80, value=30)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
    weight = st.sidebar.number_input("Weight (kg)", min_value=40.0, max_value=150.0, value=70.0, step=0.1)
    height = st.sidebar.number_input("Height (cm)", min_value=140.0, max_value=220.0, value=170.0, step=0.1)
    activity_level = st.sidebar.select_slider(
        "Activity Level",
        options=["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"],
        value="Moderately Active"
    )
    
    # Health goals
    st.sidebar.markdown("<h2>Health Goals</h2>", unsafe_allow_html=True)
    goal = st.sidebar.selectbox("Primary Goal", ["Weight Loss", "Muscle Gain", "Maintenance", "Improve Health"])
    
    # Calculate BMR and daily calorie needs
    if gender == "Male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    
    activity_factors = {
        "Sedentary": 1.2,
        "Lightly Active": 1.375,
        "Moderately Active": 1.55,
        "Very Active": 1.725,
        "Extremely Active": 1.9
    }
    
    maintenance_calories = bmr * activity_factors[activity_level]
    
    # Adjust calories based on goal
    if goal == "Weight Loss":
        recommended_calories = maintenance_calories * 0.8
    elif goal == "Muscle Gain":
        recommended_calories = maintenance_calories * 1.1
    else:
        recommended_calories = maintenance_calories
    
    # Calculate macronutrient goals
    if goal == "Weight Loss":
        protein_pct = 0.35
        fat_pct = 0.3
        carb_pct = 0.35
    elif goal == "Muscle Gain":
        protein_pct = 0.3
        fat_pct = 0.25
        carb_pct = 0.45
    else:
        protein_pct = 0.25
        fat_pct = 0.3
        carb_pct = 0.45
    
    protein_goal = (recommended_calories * protein_pct) / 4  # 4 calories per gram of protein
    fat_goal = (recommended_calories * fat_pct) / 9  # 9 calories per gram of fat
    carb_goal = (recommended_calories * carb_pct) / 4  # 4 calories per gram of carbs
    fiber_goal = 25  # General recommendation
    sugar_goal = 25  # General recommendation
    
    # Display recommendations
    st.sidebar.markdown("<h3>Recommended Daily Intake</h3>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<p><span class='highlight'>Calories:</span> {recommended_calories:.0f} kcal</p>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<p><span class='highlight'>Protein:</span> {protein_goal:.0f} g</p>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<p><span class='highlight'>Fat:</span> {fat_goal:.0f} g</p>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<p><span class='highlight'>Carbs:</span> {carb_goal:.0f} g</p>", unsafe_allow_html=True)
    
    # Advanced nutrition settings
    show_advanced = st.sidebar.checkbox("Show Advanced Nutrition Settings")
    
    if show_advanced:
        st.sidebar.markdown("<h3>Advanced Nutrition Settings</h3>", unsafe_allow_html=True)
        protein_goal = st.sidebar.number_input("Protein (g)", min_value=0.0, max_value=300.0, value=protein_goal, step=5.0)
        fat_goal = st.sidebar.number_input("Fat (g)", min_value=0.0, max_value=200.0, value=fat_goal, step=5.0)
        carb_goal = st.sidebar.number_input("Carbohydrates (g)", min_value=0.0, max_value=500.0, value=carb_goal, step=5.0)
        fiber_goal = st.sidebar.number_input("Fiber (g)", min_value=0.0, max_value=100.0, value=fiber_goal, step=1.0)
        sugar_goal = st.sidebar.number_input("Sugar (g)", min_value=0.0, max_value=100.0, value=sugar_goal, step=1.0)
    
    # Dietary preferences
    st.sidebar.markdown("<h2>Dietary Preferences</h2>", unsafe_allow_html=True)
    dietary_preferences = st.sidebar.multiselect(
        "Select your dietary preferences",
        ["Vegetarian", "Non-Vegetarian"],
        default=["Vegetarian", "Non-Vegetarian"]
    )
    
    # Food exclusions
    st.sidebar.markdown("<h2>Food Exclusions</h2>", unsafe_allow_html=True)
    excluded_ingredients = st.sidebar.text_input("Ingredients to exclude (comma-separated)", "")
    excluded_ingredients = [item.strip() for item in excluded_ingredients.split(",") if item.strip()]
    
    # Number of days for meal plan
    days = st.sidebar.slider("Number of Days for Meal Plan", min_value=1, max_value=7, value=3)
    
    # User profile for recommendation
    user_profile = {
        "age": age,
        "gender": gender,
        "weight": weight,
        "height": height,
        "activity_level": activity_level,
        "goal": goal,
        "calorie_goal": recommended_calories / 3,  # Per meal
        "protein_goal": protein_goal / 3,  # Per meal
        "fat_goal": fat_goal / 3,  # Per meal
        "carb_goal": carb_goal / 3,  # Per meal
        "fiber_goal": fiber_goal / 3,  # Per meal
        "sugar_goal": sugar_goal / 3  # Per meal
    }
    
    # Generate recommendations button
    if st.sidebar.button("Generate Meal Plan"):
        with st.spinner("Generating your personalized meal plan..."):
            # Get recommendations
            recommendations = get_recommendations(
                df, user_profile, dietary_preferences, excluded_ingredients, n_recommendations=days * 4
            )
            
            if recommendations is None or len(recommendations) == 0:
                st.error("No recommendations found based on your preferences. Please adjust your filters and try again.")
            else:
                # Create meal plan
                meal_plan = create_meal_plan(recommendations, days=days)
                
                # Display meal plan
                display_meal_plan(meal_plan, user_profile)
    else:
        # Show app intro
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            ## Welcome to NutriSmart!
            
            NutriSmart is your personalized nutrition planner specialized in Indian cuisine. Our AI-powered system provides customized meal recommendations based on your health goals, dietary preferences, and nutritional requirements.
            
            ### How it works:
            
            1. Enter your personal details in the sidebar
            2. Set your health goals and dietary preferences
            3. Specify any ingredients you want to exclude
            4. Click "Generate Meal Plan" to receive your personalized recommendations
            
            Our recommendations are based on a comprehensive database of Indian dishes with detailed nutritional information.
            
            ### Benefits:
            
            - Personalized meal plans tailored to your specific needs
            - Detailed nutritional analysis of your meal plan
            - Discover new and delicious Indian dishes
            - Track your macronutrient intake with visual charts
            - Get expert advice based on your health goals
            
            Fill in your information in the sidebar and click "Generate Meal Plan" to get started!
            """)
        
        with col2:
            st.markdown("""
            ### Features:
            
            - 🍲 **Personalized Meal Plans**
            - 📊 **Nutrition Analysis**
            - 🥗 **Indian Cuisine Focus**
            - 🏋️ **Goal-Based Recommendations**
            - 🔍 **Ingredient Exclusions**
            - 📈 **Visual Nutrition Breakdown**
            
            ### Dataset Overview:
            """)
            
            # Display dataset information
            st.write(f"Total Dishes: {len(df)}")
            st.write(f"Vegetarian Dishes: {len(df[df['Diet'] == 'vegetarian'])}")
            st.write(f"Non-Vegetarian Dishes: {len(df[df['Diet'] == 'non vegetarian'])}")
            
            # Display sample dishes
            st.markdown("### Sample Dishes:")
            for i, row in df.sample(3).iterrows():
                st.write(f"- {row['Name']} ({row['State']})")

if __name__ == "__main__":
    main()
