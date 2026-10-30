
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="CareerPath AI",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}

h1 {
    color: #172b4d;
}

h2, h3 {
    color: #243b5a;
}

.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    height: 45px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# =====================================
# HEADER
# =====================================

st.title("🎓 CareerPath AI")

st.subheader(
    "Student Placement Prediction & Company Eligibility System"
)

st.write(
    "Analyze student profiles, predict placement outcomes "
    "and discover companies matching your qualifications."
)

st.divider()


# =====================================
# COMPANY DATABASE
# =====================================

companies = [
    {
        "name": "TechNova Solutions",
        "cgpa": 7.0,
        "tenth": 60,
        "twelfth": 60,
        "backlogs": 0,
        "internships": 0,
        "skills": 3,
        "mini": 1,
        "major": 0,
        "departments": ["CSE", "IT", "ECE"],
        "education": ["12th", "Diploma"],
        "description": "Software development and IT services",
        "package": "3.5 - 5 LPA"
    },
    {
        "name": "DataSphere Technologies",
        "cgpa": 8.0,
        "tenth": 70,
        "twelfth": 70,
        "backlogs": 0,
        "internships": 1,
        "skills": 5,
        "mini": 2,
        "major": 1,
        "departments": ["CSE", "IT", "AIML"],
        "education": ["12th", "Diploma"],
        "description": "Data science and analytics",
        "package": "5 - 8 LPA"
    },
    {
        "name": "WebCraft Systems",
        "cgpa": 6.5,
        "tenth": 50,
        "twelfth": 50,
        "backlogs": 1,
        "internships": 0,
        "skills": 2,
        "mini": 1,
        "major": 0,
        "departments": ["CSE", "IT", "ECE", "EEE"],
        "education": ["12th", "Diploma"],
        "description": "Web development and software",
        "package": "2.5 - 4 LPA"
    },
    {
        "name": "CloudBridge",
        "cgpa": 7.5,
        "tenth": 65,
        "twelfth": 65,
        "backlogs": 0,
        "internships": 1,
        "skills": 4,
        "mini": 2,
        "major": 1,
        "departments": ["CSE", "IT", "AIML", "ECE"],
        "education": ["12th", "Diploma"],
        "description": "Cloud computing and networking",
        "package": "4 - 7 LPA"
    },
    {
        "name": "InnovateAI Labs",
        "cgpa": 8.5,
        "tenth": 75,
        "twelfth": 75,
        "backlogs": 0,
        "internships": 2,
        "skills": 5,
        "mini": 2,
        "major": 2,
        "departments": ["CSE", "IT", "AIML"],
        "education": ["12th", "Diploma"],
        "description": "Artificial intelligence and ML",
        "package": "6 - 10 LPA"
    }
]


# =====================================
# MACHINE LEARNING MODEL
# =====================================

@st.cache_resource
def train_model():

    rng = np.random.default_rng(42)

    n = 1500

    data = pd.DataFrame({
        "tenth": rng.uniform(40, 100, n),
        "twelfth": rng.uniform(40, 100, n),
        "cgpa": rng.uniform(4, 10, n),
        "communication": rng.uniform(1, 10, n),
        "backlogs": rng.integers(0, 5, n),
        "internships": rng.integers(0, 5, n),
        "skills": rng.integers(1, 11, n),
        "mini": rng.integers(0, 6, n),
        "major": rng.integers(0, 5, n)
    })

    score = (
        data["tenth"] * 0.08
        + data["twelfth"] * 0.10
        + data["cgpa"] * 5
        + data["communication"] * 1.5
        + data["internships"] * 3
        + data["skills"] * 1.5
        + data["mini"] * 1
        + data["major"] * 2
        - data["backlogs"] * 10
        + rng.normal(0, 8, n)
    )

    data["placed"] = (score >= 90).astype(int)

    X = data.drop(columns=["placed"])
    y = data["placed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy


model, accuracy = train_model()


# =====================================
# NAVIGATION
# =====================================

page = st.radio(
    "Navigation",
    [
        "🏠 Student Dashboard",
        "🏢 Company Requirements"
    ],
    horizontal=True
)

st.divider()


# =====================================
# STUDENT DASHBOARD
# =====================================

if page == "🏠 Student Dashboard":

    st.header("📝 Student Information")

    with st.form("student_form"):

        col1, col2 = st.columns(2)

        with col1:

            gender = st.selectbox(
                "Gender",
                [
                    "Choose...",
                    "Male",
                    "Female",
                    "Other",
                    "Prefer not to say"
                ]
            )

            education = st.selectbox(
                "Education Type",
                ["12th", "Diploma"]
            )

            department = st.selectbox(
                "Department",
                [
                    "CSE", "IT", "AIML", "ECE",
                    "EEE", "Mechanical", "Civil"
                ]
            )

            communication = st.number_input(
                "Communication Skills (1-10)",
                min_value=1,
                max_value=10,
                value=7
            )

            internships = st.number_input(
                "Number of Internships",
                min_value=0,
                max_value=20,
                value=1
            )

            skills = st.number_input(
                "Number of Skills",
                min_value=0,
                max_value=30,
                value=4
            )

        with col2:

            tenth = st.number_input(
                "10th Marks (%)",
                min_value=0.0,
                max_value=100.0,
                value=80.0
            )

            twelfth = st.number_input(
                "12th/Diploma Marks (%)",
                min_value=0.0,
                max_value=100.0,
                value=75.0
            )

            cgpa = st.number_input(
                "CGPA",
                min_value=0.0,
                max_value=10.0,
                value=8.0
            )

            backlogs = st.number_input(
                "Number of Backlogs",
                min_value=0,
                max_value=20,
                value=0
            )

            mini = st.number_input(
                "Number of Mini Projects",
                min_value=0,
                max_value=20,
                value=2
            )

            major = st.number_input(
                "Number of Major Projects",
                min_value=0,
                max_value=10,
                value=1
            )

        st.caption(
            "Gender is optional and is not used for "
            "prediction or company eligibility."
        )

        submit = st.form_submit_button(
            "🚀 PREDICT PLACEMENT & FIND COMPANIES",
            use_container_width=True
        )

    if submit:

        student = {
            "tenth": tenth,
            "twelfth": twelfth,
            "cgpa": cgpa,
            "communication": communication,
            "backlogs": backlogs,
            "internships": internships,
            "skills": skills,
            "mini": mini,
            "major": major
        }

        input_data = pd.DataFrame([student])

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0][1]

        st.session_state["student"] = student
        st.session_state["department"] = department
        st.session_state["education"] = education
        st.session_state["prediction"] = prediction
        st.session_state["probability"] = probability

        st.success(
            "Student information analyzed successfully!"
        )

    # =================================
    # PREDICTION RESULT
    # =================================

    if "student" in st.session_state:

        st.divider()

        st.header("📊 Placement Prediction Result")

        student = st.session_state["student"]

        probability = st.session_state["probability"]

        prediction = st.session_state["prediction"]

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Placement Probability",
                f"{probability * 100:.1f}%"
            )

        with c2:
            st.metric(
                "CGPA",
                student["cgpa"]
            )

        with c3:
            st.metric(
                "Model Test Accuracy",
                f"{accuracy * 100:.1f}%"
            )

        if prediction == 1:

            st.success(
                "Model prediction: Likely to be placed"
            )

        else:

            st.warning(
                "Model prediction: Less likely to be placed"
            )

        st.caption(
            "Educational demonstration using synthetic training "
            "data. Predictions are not validated against actual "
            "placement outcomes."
        )

        # =================================
        # COMPANY MATCHING
        # =================================

        st.divider()

        st.header("🏢 Your Company Eligibility Results")

        eligible = []

        not_eligible = []

        department = st.session_state["department"]

        education = st.session_state["education"]

        for company in companies:

            reasons = []

            if student["cgpa"] < company["cgpa"]:
                reasons.append(
                    f"CGPA must be at least {company['cgpa']}"
                )

            if student["tenth"] < company["tenth"]:
                reasons.append(
                    f"10th marks must be at least {company['tenth']}%"
                )

            if student["twelfth"] < company["twelfth"]:
                reasons.append(
                    f"12th/Diploma marks must be at least {company['twelfth']}%"
                )

            if student["backlogs"] > company["backlogs"]:
                reasons.append(
                    f"Maximum allowed backlogs: {company['backlogs']}"
                )

            if student["internships"] < company["internships"]:
                reasons.append(
                    f"Minimum internships: {company['internships']}"
                )

            if student["skills"] < company["skills"]:
                reasons.append(
                    f"Minimum skills required: {company['skills']}"
                )

            if student["mini"] < company["mini"]:
                reasons.append(
                    f"Minimum mini projects: {company['mini']}"
                )

            if student["major"] < company["major"]:
                reasons.append(
                    f"Minimum major projects: {company['major']}"
                )

            if department not in company["departments"]:
                reasons.append(
                    "Department not accepted"
                )

            if education not in company["education"]:
                reasons.append(
                    "Education type not accepted"
                )

            if not reasons:

                eligible.append(company)

            else:

                not_eligible.append(
                    (company, reasons)
                )

        st.subheader(
            f"✅ Eligible Companies: {len(eligible)}"
        )

        if eligible:

            for company in eligible:

                with st.container(border=True):

                    st.success(company["name"])

                    st.write(company["description"])

                    st.write(
                        f"**Package:** {company['package']}"
                    )

                    st.write(
                        "All configured demonstration "
                        "requirements satisfied."
                    )

        else:

            st.info(
                "No companies currently match all "
                "configured criteria."
            )

        st.subheader(
            f"⚠️ Companies with Unmet Requirements: "
            f"{len(not_eligible)}"
        )

        if not_eligible:

            for company, reasons in not_eligible:

                with st.expander(
                    f"{company['name']} - Requirements not met"
                ):

                    st.write(company["description"])

                    st.write(
                        f"**Package:** {company['package']}"
                    )

                    st.write("**Requirements to review:**")

                    for reason in reasons:

                        st.write("❌ " + reason)


# =====================================
# COMPANY REQUIREMENTS PAGE
# =====================================

elif page == "🏢 Company Requirements":

    st.header("🏢 Company Requirements Database")

    st.write(
        "Explore the eligibility requirements "
        "configured for each demonstration company."
    )

    for company in companies:

        with st.expander(company["name"]):

            st.write(company["description"])

            st.write(
                f"**Expected package:** {company['package']}"
            )

            st.write(
                f"**Minimum CGPA:** {company['cgpa']}"
            )

            st.write(
                f"**10th marks:** {company['tenth']}%"
            )

            st.write(
                f"**12th/Diploma marks:** {company['twelfth']}%"
            )

            st.write(
                f"**Maximum backlogs:** {company['backlogs']}"
            )

            st.write(
                f"**Internships required:** {company['internships']}"
            )

            st.write(
                f"**Skills required:** {company['skills']}"
            )

            st.write(
                f"**Mini projects:** {company['mini']}"
            )

            st.write(
                f"**Major projects:** {company['major']}"
            )

            st.write(
                f"**Eligible departments:** "
                f"{', '.join(company['departments'])}"
            )

            st.write(
                f"**Education types:** "
                f"{', '.join(company['education'])}"
            )


# =====================================
# ABOUT SECTION AT BOTTOM
# =====================================

st.divider()

with st.expander("ℹ️ About CareerPath AI"):

    st.header("About Our Project")

    st.write("""
    CareerPath AI is a Machine Learning based
    Student Placement Prediction and Company
    Eligibility System.

    This application helps students understand
    their placement opportunities based on their
    academic performance and technical skills.

    It also identifies companies whose configured
    recruitment requirements match the student's profile.
    """)

    st.subheader("🎯 Project Objective")

    st.write("""
    To develop an intelligent web-based system
    that helps students understand placement
    opportunities and identify companies matching
    their qualifications.
    """)

    st.subheader("💻 Technologies Used")

    st.write("""
    - Python
    - Streamlit
    - Pandas
    - NumPy
    - Scikit-learn
    - Logistic Regression
    """)

    st.subheader("📌 Project Modules")

    st.write("""
    1. Student Profile Management
    2. Placement Prediction
    3. Company Requirements Database
    4. Company Eligibility Matching
    5. Prediction Result Display
    """)

    st.subheader("🤖 Machine Learning Algorithm")

    st.write("""
    Logistic Regression is used for classification
    to demonstrate student placement outcome prediction.
    The current training dataset is synthetically generated
    for educational demonstration.
    """)

    st.info(
        "Company requirements are sample demonstration "
        "criteria. They should be replaced with verified "
        "recruitment information before being treated as "
        "actual company eligibility rules."
    )


# =====================================
# FOOTER
# =====================================

st.caption(
    "🎓 CareerPath AI | Student Placement Prediction "
    "& Company Eligibility System | AIML Mini Project"
)
