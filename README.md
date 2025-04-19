# 🥗 NutriSmart<br>
NutriSmart is a personalized meal planning application built with Streamlit that recommends Indian dishes based on your nutritional goals and dietary preferences. <br>
It helps you plan balanced meals while considering ingredient exclusions and provides a nutritional summary of your meal plan.<br>
<br>
<br>
# 🚀 Features<br>
🍛 Personalized Meal Recommendations<br>
Get food suggestions based on your health goals, dietary preferences, and nutritional requirements.<br>
<br>
🥗Dietary Preferences<br>
Choose between vegetarian and non-vegetarian options.<br>
<br>
🚫 Ingredient Exclusions<br>
Exclude any ingredients you don’t want in your meals (e.g., dairy, gluten, nuts).<br>
<br>
📅 Meal Plan Generator<br>
Automatically generate a multi-day meal plan with breakfast, lunch, and dinner.<br>
<br>
📊 Nutritional Summary<br>
View a clean summary of your average daily intake from the meal plan.<br>
<br>
🌍 Indian Cuisine Focus<br>
Includes a diverse list of Indian dishes.<br>
<br>
# 🛠 Tech Stack<br>
Frontend & App: Streamlit<br>

# Backend & Logic: <br>
Python (Pandas, NumPy, Scikit-learn)<br>

# Data: <br>
Sample dataset of Indian food items with nutritional values<br>

# ⚙️ How It Works<br>
Input Your Goals:<br>
Enter your details(gender,age,height and weight) and your health goals(weight loss,muscle bulding maintenance etc).<br>

Set Preferences:<br>
Select vegetarian or non-vegetarian and list any ingredients to avoid.<br>

Get Recommendations:<br>
The app uses cosine similarity on standardized nutritional data to recommend the best matching dishes.<br>

Generate Meal Plan:<br>
Based on the recommendations, a balanced 3-day meal plan is created with visual summaries.<br>
<br>
<br>
<br>
# FOLDER STRUCTURE: 
nutrismart/ 
├── app.py              # Main Streamlit app
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation

# STEPS TO RUN: <br>
1. Clone this repository. <br>
In order to clone this repository open a terminal and type <br>
git clone https://github.com/dishas123/AI-Powered-Personalized-Meal-Planner.git <br>
Navigate to the cloned folder on the terminal in your system.  <br>
Install the dependencies <br>
pip install streamlit pandas numpy scikit-learn matplotlib seaborn Pillow <br>
Then type :  <br>
streamlit run foodnew.py  <br>
Output screenshots of the website :<br>
![image](https://github.com/user-attachments/assets/90a436a0-16cd-4e47-8637-2472834570b1)
![image](https://github.com/user-attachments/assets/90ae44ff-382c-4a87-9976-efb61daf1de7)
![image](https://github.com/user-attachments/assets/8454e8e0-111e-4309-90b7-bd3ecc7308ca)
![image](https://github.com/user-attachments/assets/d46def80-4a49-4818-a393-35277e1177f2)
