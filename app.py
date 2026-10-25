import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

MODEL_PATH = "student_performance_model_actionable.pkl"
model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
.main {
    padding-top: 1.5rem;
}

/* Hero */
.hero {
    padding: 3.5rem 3rem;
    border-radius: 24px;
    text-align: center;
    background: linear-gradient(135deg, #eef4ff, #f8faff);
    border: 1px solid #dce6f5;
    margin-bottom: 2rem;
}

.hero h1 {
    font-size: 46px;
    color: #17365d;
    margin-bottom: 15px;
}

.hero p {
    font-size: 19px;
    color: #4a5568;
    max-width: 750px;
    margin: auto;
    line-height: 1.7;
}

/* Feature cards */
.feature-card {
    padding: 25px;
    border-radius: 18px;
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    min-height: 170px;
}

.feature-card h3 {
    color: #17365d;
    margin-bottom: 10px;
}

.feature-card p {
    color: #5b6573;
    line-height: 1.6;
}

/* Section title */
.section-title {
    font-size: 27px;
    font-weight: 700;
    color: #FFFFFF;
    margin-top: 30px;
    margin-bottom: 15px;
}

/* Helper boxes */
.helper-box {
    background-color: #f7f9fc;
    border-left: 4px solid #4f81bd;
    padding: 12px 16px;
    border-radius: 8px;
    margin-top: 5px;
    margin-bottom: 18px;
    color: #4a5568;
    font-size: 14px;
    line-height: 1.6;
}

.helper-box ul {
    margin-top: 4px;
    margin-bottom: 4px;
}

.helper-box li {
    margin-bottom: 3px;
}


/* ==================================================
   RESULT SECTION
   ================================================== */

/* Score result */
.score-card {
    padding: 22px;
    border-radius: 18px;
    text-align: center;
    background: linear-gradient(135deg, #eef4ff, #ffffff);
    border: 1px solid #d5e1f2;
    margin-top: 18px;
    margin-bottom: 15px;
}

.score-label {
    font-size: 12px;
    font-weight: 700;
    color: #53657a;
    letter-spacing: 1px;
}

.score-number {
    font-size: 46px;
    font-weight: 800;
    color: #17365d;
    margin: 4px 0;
}

.score-subtitle {
    font-size: 14px;
    color: #64748b;
}


/* Result section headings */
.result-heading {
    font-size: 19px;
    font-weight: 700;
    margin-top: 24px;
    margin-bottom: 10px;
    color: #253858;
}


/* Green Strength Cards */
.strength-card {
    padding: 13px 15px;
    border-radius: 11px;
    background-color: #ecfdf3;
    border: 1px solid #bbebcf;
    border-left: 4px solid #22a060;
    margin-bottom: 9px;
}

.strength-title {
    color: #167447;
    font-size: 14px;
    font-weight: 700;
}

.strength-text {
    color: #4b6357;
    font-size: 12px;
    margin-top: 3px;
}


/* Red Priority Cards */
.priority-card {
    padding: 13px 15px;
    border-radius: 11px;
    background-color: #fef2f2;
    border: 1px solid #f4caca;
    border-left: 4px solid #dc3b3b;
    margin-bottom: 9px;
}

.priority-number {
    color: #b42318;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.5px;
}

.priority-title {
    color: #9f2020;
    font-size: 14px;
    font-weight: 700;
    margin-top: 2px;
}

.priority-text {
    color: #704545;
    font-size: 12px;
    margin-top: 3px;
}


/* Orange Improvement Cards */
.improvement-card {
    padding: 12px 15px;
    border-radius: 11px;
    background-color: #fff7ed;
    border: 1px solid #fed7aa;
    border-left: 4px solid #ed8b24;
    margin-bottom: 9px;
}

.improvement-title {
    color: #a85d0b;
    font-size: 14px;
    font-weight: 700;
}


/* Blue Action Cards */
.action-card {
    padding: 13px 15px;
    border-radius: 11px;
    background-color: #eff6ff;
    border: 1px solid #c9defb;
    border-left: 4px solid #4285d4;
    margin-bottom: 9px;
}

.action-number {
    color: #3066a8;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.5px;
}

.action-text {
    color: #50647d;
    font-size: 13px;
    line-height: 1.5;
    margin-top: 3px;
}


/* Small note */
.small-note {
    color: #6b7280;
    font-size: 12px;
    line-height: 1.5;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "home"

if "prediction" not in st.session_state:
    st.session_state.prediction = None


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def helper(title, points):
    bullet_html = "".join([f"<li>{point}</li>" for point in points])

    st.markdown(
        f"""
        <div class="helper-box">
            <strong>{title}</strong>
            <ul>
                {bullet_html}
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )


def get_performance_level(score):

    if score < 40:
        return "Needs Significant Improvement"
    elif score < 50:
        return "Needs Improvement"
    elif score < 60:
        return "Developing"
    elif score < 75:
        return "Good"
    elif score < 90:
        return "Very Good"
    else:
        return "Excellent"


def generate_recommendations(data):

    priorities = []
    improvements = []
    strengths = []
    actions = []

    # -------------------------
    # Study hours
    # -------------------------

    if data["study_hours"] >= 4:
        strengths.append("Focused study routine")
    else:
        improvements.append("Focused study time")
        priorities.append("Increase focused study time")
        actions.append(
            "Try gradually increasing focused self-study time while keeping sessions realistic and consistent."
        )

    # -------------------------
    # Attendance
    # -------------------------

    if data["attendance"] >= 85:
        strengths.append("Attendance")
    else:
        improvements.append("Attendance")
        priorities.append("Improve attendance")
        actions.append(
            "Aim to attend classes consistently so you have regular exposure to lectures, discussions, and course material."
        )

    # -------------------------
    # Previous grade
    # -------------------------

    if data["previous_grade"] >= 70:
        strengths.append("Previous academic performance")
    else:
        improvements.append("Previous academic performance")
        priorities.append("Strengthen academic foundations")
        actions.append(
            "Review topics from previous assessments where your understanding was weaker and use them as a foundation for current subjects."
        )

    # -------------------------
    # Assignments
    # -------------------------

    if data["assignments_completed"] >= 7:
        strengths.append("Assignment completion")
    else:
        improvements.append("Assignment completion")
        priorities.append("Complete more assignments")
        actions.append(
            "Try to complete assignments regularly instead of allowing several tasks to accumulate near deadlines."
        )

    # -------------------------
    # Practice tests
    # -------------------------

    if data["practice_tests_taken"] >= 5:
        strengths.append("Practice testing")
    else:
        improvements.append("Practice testing")
        actions.append(
            "Include more practice tests to check your understanding and identify topics that still need revision."
        )

    # -------------------------
    # Time management
    # -------------------------

    if data["time_management_score"] >= 7:
        strengths.append("Time management")
    else:
        improvements.append("Time management")
        priorities.append("Improve time management")
        actions.append(
            "Use a simple daily or weekly study plan and divide larger academic tasks into smaller deadlines."
        )

    # -------------------------
    # Motivation
    # -------------------------

    if data["motivation_level"] >= 7:
        strengths.append("Motivation")
    else:
        improvements.append("Motivation")
        actions.append(
            "Set small, specific study goals so progress is easier to track and maintain."
        )

    # -------------------------
    # Notes
    # -------------------------

    if data["notes_quality_score"] >= 7:
        strengths.append("Quality of notes")
    else:
        improvements.append("Quality of notes")
        actions.append(
            "Make notes more structured by highlighting key concepts, formulas, examples, and topics that need revision."
        )

    # -------------------------
    # Sleep
    # -------------------------

    if 7 <= data["sleep_hours"] <= 9:
        strengths.append("Sleep routine")
    else:
        improvements.append("Sleep routine")
        actions.append(
            "Try to maintain a consistent sleep routine that gives you enough time to recover between study days."
        )

    # -------------------------
    # Digital habits
    # -------------------------

    if data["social_media_hours"] <= 2:
        strengths.append("Social media usage")
    else:
        improvements.append("Social media usage")
        priorities.append("Reduce social media time")
        actions.append(
            "Consider setting specific periods for social media so it does not overlap with planned study sessions."
        )

    # -------------------------
    # Ensure priorities exist
    # -------------------------

    if not priorities:
        priorities = improvements[:3]

    if not actions:
        actions.append(
            "Continue maintaining your current academic and study habits while monitoring your performance regularly."
        )

    return priorities[:4], improvements[:5],strengths[:5],actions[:5]


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

if st.session_state.page == "home":

    st.markdown(
        """
        <div class="hero">
        <h1>🎓 Student Performance Predictor</h1>
        <p>
        Understand your academic profile, estimate your final performance,
        and discover areas where you can focus your efforts.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Main CTA
    st.markdown("### Ready to check your academic profile?")

    if st.button(
        "📝 Give Assessment & Predict Your Score",
        type="primary",
        use_container_width=True
    ):
        st.session_state.page = "assessment"
        st.rerun()

    st.divider()

    st.markdown("## What can you do here?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
            <h3>🎯 Predict</h3>
            <p>
            Enter information about your academic performance,
            study habits, well-being, and digital habits to receive
            an estimated final grade.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="feature-card">
            <h3>📊 Understand</h3>
            <p>
            See which parts of your current academic profile appear
            strong and which areas may need more attention.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="feature-card">
            <h3>🚀 Improve</h3>
            <p>
            Receive practical recommendations based on the information
            you provide and identify areas you can work on.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    st.markdown("## How it works")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### 01")
        st.write("Enter your information")

    with col2:
        st.markdown("### 02")
        st.write("Get your predicted score")

    with col3:
        st.markdown("### 03")
        st.write("Understand your profile")

    with col4:
        st.markdown("### 04")
        st.write("Get areas to focus on")

    st.info(
        "The predicted grade is an estimate produced by a machine learning model. "
        "It should not be treated as a guarantee of your actual final grade."
    )


# --------------------------------------------------
# ASSESSMENT PAGE
# --------------------------------------------------

elif st.session_state.page == "assessment":

    st.title("📝 Student Assessment")

    st.write(
        "Enter your information below. Each section includes a short guide "
        "to help you understand what to enter."
    )

    st.divider()

    # -------------------------
    # Academic Performance
    # -------------------------

    st.markdown(
        '<div class="section-title">📚 Academic Performance</div>',
        unsafe_allow_html=True
    )

    previous_grade = st.number_input(
        "Previous Academic Score (%)",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=0.5
    )

    helper(
        "What should you enter?",
        [
            "Enter your most recent previous academic score.",
            "Use a percentage between 0 and 100.",
            "Example: If your previous score was 72%, enter 72."
        ]
    )

    attendance = st.number_input(
        "Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=1.0
    )

    helper(
        "What should you enter?",
        [
            "Enter your approximate overall class attendance.",
            "Use a percentage between 0 and 100.",
            "Example: If your attendance is 84%, enter 84."
        ]
    )

    col1, col2 = st.columns(2)

    with col1:

        assignments_completed = st.number_input(
            "Assignments Completed",
            min_value=0.0,
            value=5.0,
            step=1.0
        )

        helper(
            "What should you enter?",
            [
                "Enter the approximate number of assignments completed.",
                "Use the number of completed assignments, not the percentage."
            ]
        )

    with col2:

        practice_tests_taken = st.number_input(
            "Practice Tests Taken",
            min_value=0.0,
            value=3.0,
            step=1.0
        )

        helper(
            "What should you enter?",
            [
                "Enter the approximate number of practice tests you have taken.",
                "Include mock tests or subject-level practice tests."
            ]
        )

    # -------------------------
    # Study Habits
    # -------------------------

    st.markdown(
        '<div class="section-title">📖 Study Habits</div>',
        unsafe_allow_html=True
    )

    study_hours = st.number_input(
        "Focused Study Hours / Day",
        min_value=0.0,
        value=3.0,
        step=0.5
    )

    helper(
        "Focused study means:",
        [
            "Time spent actively studying on your own.",
            "Do not include regular classroom or lecture hours.",
            "Count time spent solving questions, revising, reading, or practicing.",
            "The training data mostly contains values between 0 and 12 hours."
        ]
    )

    group_study_hours = st.number_input(
        "Group Study Hours / Day",
        min_value=0.0,
        value=1.0,
        step=0.5
    )

    helper(
        "Group study means:",
        [
            "Average time spent studying with classmates or in a study group.",
            "Enter hours per day.",
            "If you normally study alone, you can enter 0."
        ]
    )

    st.markdown("#### 🗒️ Notes Quality")

    notes_quality_score = st.slider(
        "Rate the quality of your study notes",
        min_value=1,
        max_value=10,
        value=6
    )

    helper(
        "Use this scale:",
        [
            "1–2 → Very incomplete, difficult to use later.",
            "3–4 → Some useful content, but several important gaps.",
            "5–6 → Average notes covering the main material.",
            "7–8 → Well-organized and mostly complete.",
            "9–10 → Very comprehensive, organized, and useful for revision."
        ]
    )

    st.markdown("#### ⏱️ Time Management")

    time_management_score = st.slider(
        "Rate your time-management ability",
        min_value=1,
        max_value=10,
        value=6
    )

    helper(
        "Use this scale:",
        [
            "1–2 → Frequently procrastinate or miss planned work.",
            "3–4 → Often struggle to follow a schedule.",
            "5–6 → Manage routine academic tasks reasonably.",
            "7–8 → Usually follow plans and meet deadlines.",
            "9–10 → Consistently plan, prioritize, and manage time effectively."
        ]
    )

    # -------------------------
    # Well-being
    # -------------------------

    st.markdown(
        '<div class="section-title">🧠 Well-being</div>',
        unsafe_allow_html=True
    )

    sleep_hours = st.number_input(
        "Sleep Hours / Day",
        min_value=0.0,
        value=7.0,
        step=0.5
    )

    helper(
        "What should you enter?",
        [
            "Enter your average amount of sleep per day/night.",
            "The training data mostly contains values between 3 and 10 hours.",
            "If your value falls outside this range, the app will show a reliability warning."
        ]
    )

    motivation_level = st.slider(
        "Motivation Level",
        min_value=1,
        max_value=10,
        value=7
    )

    helper(
        "Use this scale:",
        [
            "1–2 → Very low motivation.",
            "3–4 → Low motivation.",
            "5–6 → Moderate motivation.",
            "7–8 → Good motivation.",
            "9–10 → Very high motivation."
        ]
    )

    mental_health_score = st.slider(
        "Academic Well-being Score",
        min_value=1,
        max_value=10,
        value=7
    )

    helper(
        "Think about your academic well-being:",
        [
            "1–2 → Frequently feel overwhelmed by academic demands.",
            "3–4 → Often experience significant academic stress.",
            "5–6 → Generally manage academic demands with occasional difficulty.",
            "7–8 → Usually feel balanced while handling academic responsibilities.",
            "9–10 → Feel very well-balanced and confident managing academic demands."
        ]
    )

    st.caption(
        "This is a self-reported academic well-being indicator, not a medical or psychological assessment."
    )

    # -------------------------
    # Digital Habits
    # -------------------------

    st.markdown(
        '<div class="section-title">📱 Digital Habits</div>',
        unsafe_allow_html=True
    )

    screen_time = st.number_input(
        "Total Screen Time / Day",
        min_value=0.0,
        value=5.0,
        step=0.5
    )

    helper(
        "What should you enter?",
        [
            "Enter your total daily screen/device usage.",
            "Include study-related and non-study-related device use.",
            "The training data mostly contains values between 0 and 12 hours."
        ]
    )

    social_media_hours = st.number_input(
        "Social Media Hours / Day",
        min_value=0.0,
        value=2.0,
        step=0.5
    )

    helper(
        "What should you enter?",
        [
            "Enter only the time spent on social media.",
            "This should be a portion of your total screen time.",
            "Example: 5 hours total screen time and 2 hours social media."
        ]
    )

    # -------------------------
    # Validation
    # -------------------------

    if social_media_hours > screen_time:

        st.error(
            "Social media hours cannot be greater than total screen time."
        )

    if study_hours > 12:

        st.warning(
            "Study hours are outside the range commonly observed in the "
            "training data. The prediction may be less reliable."
        )

    if sleep_hours > 10:

        st.warning(
            "Sleep hours are outside the range commonly observed in the "
            "training data. The prediction may be less reliable."
        )

    # -------------------------
    # Prediction
    # -------------------------

    st.divider()

    if social_media_hours <= screen_time:

        predict_button = st.button(
            "🎯 Predict My Performance",
            type="primary",
            use_container_width=True
        )

        if predict_button:

            input_data = pd.DataFrame([{
                "study_hours": study_hours,
                "attendance": attendance,
                "sleep_hours": sleep_hours,
                "previous_grade": previous_grade,
                "assignments_completed": assignments_completed,
                "practice_tests_taken": practice_tests_taken,
                "group_study_hours": group_study_hours,
                "notes_quality_score": notes_quality_score,
                "time_management_score": time_management_score,
                "motivation_level": motivation_level,
                "mental_health_score": mental_health_score,
                "screen_time": screen_time,
                "social_media_hours": social_media_hours
            }])

            prediction = model.predict(input_data)[0]
            prediction = max(0, min(100, prediction))

            st.session_state.prediction = prediction
            st.session_state.input_data = input_data

    # ==================================================
    # RESULTS
    # ==================================================

    if st.session_state.prediction is not None:

        prediction = st.session_state.prediction
        data = st.session_state.input_data.iloc[0].to_dict()

        st.divider()

        st.markdown("## 📊 Your Performance Profile")

        # --------------------------------------------------
        # SCORE
        # --------------------------------------------------

        st.markdown(
            f"""
            <div class="score-card">
                <div class="score-label">PREDICTED FINAL GRADE</div>
                <div class="score-number">{prediction:.1f} / 100</div>
                <div class="score-subtitle">
                    {get_performance_level(prediction)}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(prediction / 100)

        st.caption(
            "This is a model-based estimate, not a guaranteed final grade."
        )

        # --------------------------------------------------
        # RECOMMENDATIONS
        # --------------------------------------------------

        priorities,improvements,strengths,actions = generate_recommendations(data)

        

        # --------------------------------------------------
        # PRIORITY AREAS
        # --------------------------------------------------

        st.markdown(
            '<div class="result-heading">🔴 Priority Areas</div>',
            unsafe_allow_html=True
        )

        if priorities:

            for i, item in enumerate(priorities, 1):

                st.markdown(
                    f"""
                    <div class="priority-card">
                        <div class="priority-number">
                            PRIORITY {i}
                        </div>
                        <div class="priority-title">
                            {item}
                        </div>
                        <div class="priority-text">
                            This is an area worth giving additional attention.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # --------------------------------------------------
        # AREAS TO IMPROVE
        # --------------------------------------------------

        st.markdown(
            '<div class="result-heading">🟠 Areas to Improve</div>',
            unsafe_allow_html=True
        )

        if improvements:

            cols = st.columns(2)

            for i, item in enumerate(improvements):

                with cols[i % 2]:

                    st.markdown(
                        f"""
                        <div class="improvement-card">
                            <div class="improvement-title">
                                • {item}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        else:

            st.success(
                "Your current profile does not show any obvious area requiring additional attention."
            )


        # --------------------------------------------------
        # STRENGTHS
        # --------------------------------------------------
        
        st.markdown(
            '<div class="result-heading">🟢 Strengths</div>',
            unsafe_allow_html=True
        )
        
        if strengths:
        
           
            cols = st.columns(min(len(strengths), 3))
        
            for i, strength in enumerate(strengths):
        
                with cols[i % len(cols)]:
        
                    st.markdown(
                        f"""
                        <div class="strength-card">
                            <div class="strength-title">
                                ✓ {strength}
                            </div>
                        <div class="strength-text">
                            A relatively strong area in your current profile.
                        </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        
        else:
            st.info(
                "Your current inputs do not show a clearly strong area."
            )

        # --------------------------------------------------
        # ACTION PLAN
        # --------------------------------------------------

        st.markdown(
            '<div class="result-heading">🔵 Personalized Action Plan</div>',
            unsafe_allow_html=True
        )

        for i, action in enumerate(actions, 1):

            st.markdown(
                f"""
                <div class="action-card">
                    <div class="action-number">
                        STEP {i}
                    </div>
                    <div class="action-text">
                        {action}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.divider()

        st.markdown(
            """
            <div class="small-note">
                The recommendations are based on your entered profile and
                are intended to highlight areas for attention. They do not
                establish that a particular factor causes your academic result.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("")

        if st.button("🔄 Take Assessment Again"):

            st.session_state.prediction = None
            st.rerun()