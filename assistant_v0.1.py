import gradio as gr
from datetime import datetime, timedelta
import pandas as pd
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv('.env')

core_beginner_materials = pd.read_csv('beginner_materials.csv',index_col='course')

supported_languages = list(core_beginner_materials.columns)
recommend_texts = {
    "english": "# We recommend the following learning materials:",
    "swedish": "# Vi rekommenderar följande läromaterial:",
    "estonian": "# Soovitame järgmisi õppematerjale:",
    "russian": "# Мы рекомендуем следующие учебные материалы:",
    "finnish": "# Suosittelemme seuraavia oppimateriaaleja:"
}
no_recommendations_text = {
    "english": "On the basis of your input, you are at least basic level on all core skills. No additional learning is necessary at the moment.",
    "swedish": "Baserat på din input är du minst på grundläggande nivå i alla kärnkompetenser. Inget ytterligare lärande behövs för tillfället.",
    "estonian": "Teie sisendi põhjal olete kõigis põhioskustes vähemalt algtasemel. Hetkel pole täiendavat õppimist vajalik.",
    "russian": "На основании вашего ввода вы находитесь как минимум на базовом уровне во всех ключевых навыках. Дополнительное обучение в данный момент не требуется.",
    "finnish": "Antamiesi tietojen perusteella olet vähintään perustasolla kaikissa ydintaidoissa. Lisäoppimista ei tällä hetkellä tarvita."
}
learning_plan_text = {
    "english": "# Your learning plan:",
    "swedish": "# Din lärandeplan:",
    "estonian": "# Teie õppeplaan:",
    "russian": "# Ваш план обучения:",
    "finnish": "# Sinun oppimissuunnitelmasi:"
}

training_period = pd.DataFrame(data={
    'date':[datetime(year=2025,day=15,month=1)+timedelta(days=1) for i in range(len(core_beginner_materials))],
    'content':[x.split('\n')[0] for x in core_beginner_materials['english']]}
) # start,end
current_date = datetime(year=2024,day=18,month=11) #

skills_levels = ["Beginner", "Basic", "Advanced"]
skill_level_score_map = {x:(k+1) for k,x in enumerate(skills_levels)}

client = OpenAI(max_retries=2)
llm_config = {
    "temperature": 0,
    "model": 'gpt-4o-mini',
    "timeout": 20
}

learning_plans = []
learning_plan_dropbox_vals = {}
current_study_plan=''
recommended_materials=''

def get_llm_response(system_prompt,prompt):
    # return "DUMMY RESPONSE"

    failed_count = 0
    response = None
    while failed_count < 3:
        try:
            print('submitting request to LLM...',end='')
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user","content": prompt}
                ],
                model=llm_config['model'],
                temperature=llm_config['temperature'],
            )
            print(' success!')
            break
        except:
            failed_count+=1
            print(f'\nFailed to get LLM response, retrying (failed_count {failed_count})')

    if response is None:
        raise(Exception('Failed to call LLM'))

    return response.choices[0].message.content

def get_llm_timetable(training_materials,background_info,language):

    startdate = training_period.iloc[0]['date']
    days_until_training = int((startdate - current_date).days)
    if days_until_training>0:
        system_prompt = ('You are a helpful learning assistant who designs learning timetables for students. You help students to prepare for the upcoming face-to-face (f2f) teaching.'
    'You are given a list of suggested learning materials for the student. Each of them takes at least 1 day to complete. You are also given current date and startdate of the f2f teaching.'
    'Your task: Given suggested learning materials (up to 8), plan a detailed learning timetable for the student so that all materials are studied before the f2f teaching begins.'
    'Build a coherent and logical timetable that takes into account optimal tempo and order of learnign materials. Do not include more than 2 days for any material. No repeats or reviewing, just one-time learning per material. Give your timetable as a concise list of dates and contents, for example:'
    '\n\nStudy day 1 (Mon 2 Feb 2024): [recommended learning materials and rationale]'
    '\nStudy day 2 (Tue 3 Mar 2024): [recommended learning materials and rationale]'
    '\n...'                         
    '\nStudy day 8 (Fri 4 Mar 2024): [recommended learning materials and rationale]\n\n'           
    'In your response, address the student personally and make your response concise and simple using markdown format. You are also given additional background information of the student, which you can leverage in you response if its relevant.')
    else:
        system_prompt = ('You are a helpful learning assistant who designs learning timetables for students. You help students to prepare for the upcoming face-to-face (f2f) teaching.'
    'You are given a list of suggested learning materials for the student. Each of them takes 1 day to complete. You are also given current date and startdate of the f2f teaching.'
    'Your task: Given suggested learning materials (from 1 to 8), plan a student a detailed learning timetable so that all materials are studied before the f2f teaching begins.'
    'Build a coherent and logical timetable that takes into account optimal tempo and order of learnign materials. Give your timetable as a concise list of dates and contents, such as'
    '\n\nStudy day 1 (Mon 2 Feb 2024): [recommended learning materials and rationale]'
    '\nStudy day 2 (Tue 3 Mar 2024): [recommended learning materials and rationale]'
    '\n...'                         
    '\nStudy day 8 (Fri 4 Mar 2024): [recommended learning materials and rationale]\n\n'    
    'In your response, address the student personally and make your response concise and simple using markdown format.')

    if language != 'english':
        prompt = f'Information of myself:\n{background_info}\n\nToday is {current_date.strftime("%B %d, %Y")} and face-to-face training starts at {startdate.strftime("%B %d, %Y")}, there are {days_until_training} days for learning the following materials:\n{training_materials}\n\n. Give me an optimized learning timetable in {language} language.'
    else:
        prompt = f'Information of myself:\n{background_info}\n\nToday is {current_date.strftime("%B %d, %Y")} and face-to-face training starts at {startdate.strftime("%B %d, %Y")}, there are {days_until_training} days for learning the following materials:\n{training_materials}\n\n. Give me an optimized learning timetable.'

    response = get_llm_response(system_prompt,prompt)
    return response

def translate_text(input_text,language):
    if language != 'english':
        print(f'translating text into {language}')
        output_text = get_llm_response(f'You are a professional text translator from English to {language}. You are specialized in technical translations and learning materials. You return only the translated text, nothing else.',f'Translate the following texts into {language} while retain the same formatting:\n\n{input_text}')
        return output_text
    else:
        return input_text

def process_survey(education, experience,native_language,q0, q1, q2, q3, q4, q5,q6,q7,q8):
    global learning_plans
    global recommended_materials
    global learning_plan_dropbox_vals

    # Calculate results
    module_levels = [q1, q2, q3, q4, q5,q6,q7,q8,q0]

    information =''
    if len(education)>0:
        information += f'Education: {education}.\n'
    if len(experience)>0:
        information += f'Experience: {experience}.\n'

    count = 0
    recommended_materials = ''
    for k in range(9):
        if skill_level_score_map[module_levels[k]]==1:
            count += 1
            recommended_materials+=f'{count}. ' + core_beginner_materials.iloc[k][native_language]+'\n'

    recommended_materials = f'{recommend_texts[native_language]}\n\n' + recommended_materials + '\n\n'

    if count==0:
        response = no_recommendations_text[native_language]
    else:
        timetable = get_llm_timetable(recommended_materials,information,native_language)
        response = recommended_materials + f'{learning_plan_text[native_language]}\n\n {timetable}'

    learning_plans = [response]
    learning_plan_dropbox_vals = {'Plan 1':response}

    return response

def create_chat_response(user_input,language):
    global current_study_plan
    global learning_plans
    global learning_plan_dropbox_vals

    # Use the same OpenAI client and configuration from the original script
    system_prompt = (
        "You are a helpful learning assistant. The user has received a personalized learning recommendation and plan. You can only discuss about the students' plan and update it as requested."
        "You task: Student wants to update or revise his/her existing personalized learning plan. Read the current student learning plan carefully. Then provide an updated plan as requested by the student. You can modify dates and order of the plan, e.g., to only include specific days or drop some of the content."
        "You may not change the recommended materials, current date or the starting date of the face-to-face trainings. You must always provide a new personalized study plan, either an old one or updated one."
    )

    prompt = f"Here is all relevant information and the current study plan that you may update:\n\n{current_study_plan}\n\nUser request related to plan: {user_input}"
    response = get_llm_response(system_prompt, prompt)

    learning_plan = recommended_materials + translate_text(f'# Your learning plan: \n\n'+ response,language)

    learning_plans.append(learning_plan)
    learning_plan_dropbox_vals.update({('Plan '+str(len(learning_plans))):learning_plan})

    print('New study plan created')
    print(f'Total {len(learning_plans)} learning plans in list')

    return learning_plan

def update_plan(output_text):
    global current_study_plan
    current_study_plan = output_text
    print('Current study plan changed')

def create_survey_interface():
    with gr.Blocks() as survey_app:
        # Store previous output for chat functionality

        gr.Markdown("# UPBEAT LEARNING ASSISTANT ")
        gr.Markdown(
            "I am your learning assistant. Please answer all questions below to receive recommendations for you learning journey.")

        # Debug Button to Prefill Form
        with gr.Row():
            debug_btn = gr.Button("prefill (debug)",scale=0.05)

        # Background information section
        with gr.Group():
            gr.Markdown("### Background Information")
            education = gr.Textbox(label="Highest Level of Education",
                                   placeholder="e.g., Bachelor's in Computer Science",
                                   lines=1)
            experience = gr.Textbox(label="Relevant Work Experience in Entrepreneurship (if any)",
                                    placeholder="Describe your entrepreneurial experience",
                                    lines=3)

            native_language = gr.Dropdown(label=f"Choose language ({', '.join(supported_languages)})",
                                        choices=supported_languages,
                                        interactive=True,
                                        visible=True,
                                        allow_custom_value=False)

        # Survey questions section
        with gr.Group():
            gr.Markdown("### Core skills assessment")
            # Create radio button groups (existing code)
            q0 = gr.Radio(choices=skills_levels, label="Your general computer skills", info="Select one option")
            q1 = gr.Radio(choices=skills_levels, label="1. Generative AI skills  ", info="Select one option")
            q2 = gr.Radio(choices=skills_levels, label="2. Market Analysis and Customer Understanding ",info="Select one option")
            q3 = gr.Radio(choices=skills_levels, label="3. Creating and Testing Business Ideas", info="Select one option")
            q4 = gr.Radio(choices=skills_levels, label="4. Making a Business Plan", info="Select one option")
            q5 = gr.Radio(choices=skills_levels, label="5. Running a Business", info="Select one option")
            q6 = gr.Radio(choices=skills_levels, label="6. Marketing and Advertising", info="Select one option")
            q7 = gr.Radio(choices=skills_levels, label="7. Sales and Customer Service", info="Select one option")
            q8 = gr.Radio(choices=skills_levels, label="8. HR Management", info="Select one option")

        # Create submit button and output section
        submit_btn = gr.Button("Submit Survey",scale=0.05)
        gr.Markdown("### Recommendations for learning")

        output = gr.Markdown()

        # Add history dropdown
        with gr.Row():
            history_dropdown = gr.Dropdown(
                choices=[],
                label="View Previous Plans",
                interactive=True,
                visible = False,
                allow_custom_value=False,
            )

        # Chat input and submit after recommendations
        with gr.Group(visible=False) as chat_group:
            chat_input = gr.Textbox(label="Ask for updates for your learning plan",lines=3)
            chat_submit = gr.Button("Request plan update")

        def update_history_dropdown():
            return gr.Dropdown(
                visible=True,allow_custom_value=False,interactive=True,
                choices=list(learning_plan_dropbox_vals.keys()),
                value=list(learning_plan_dropbox_vals.keys())[-1] if learning_plans else None
            )

        def process_selection(displayed_value):
            # Map the displayed value back to the stored value
            stored_value = learning_plan_dropbox_vals[displayed_value]
            return stored_value

        submit_btn.click(
            fn=process_survey,
            inputs=[education, experience, native_language, q0, q1, q2, q3, q4, q5, q6, q7, q8],
            outputs=[output]
        ).then(
            fn=update_plan,
            inputs=[output],
            outputs=[]
        ).then(
            fn=update_history_dropdown,
            outputs=[history_dropdown]
        ).then(
            fn=lambda *args: gr.update(visible=True),
            outputs=[chat_group]
        )

        # Debug Button functionality
        debug_btn.click(
            fn=lambda: (
                "Master's in Business Administration",  # education
                "I have a small retail business where I sell items online. I have general knowledge how to run a small business, but poor technical skills.",
                # experience
                "english",  # native_language
                skills_levels[1],  # additional_language_level
                skills_levels[1],  # q0 computer skills
                skills_levels[0],  # q1 generative AI
                skills_levels[1],  # q2 market analysis
                skills_levels[1],  # q3 business ideas
                skills_levels[0],  # q4 business plan
                skills_levels[1],  # q5 running business
                skills_levels[1],  # q6 marketing
                skills_levels[0],  # q7 sales
                skills_levels[1]  # q8 HR
            ),
            outputs=[
                education, experience, native_language,
                q0, q1, q2, q3, q4, q5, q6, q7, q8
            ]
        )

        # Chat submit functionality
        chat_submit.click(
            fn=create_chat_response,
            inputs=[chat_input,native_language],
            outputs=[output]
        ).then(
            fn=update_plan,
            inputs=[output],
            outputs=[]
        ).then(
            fn=update_history_dropdown,
            outputs=[history_dropdown]
        ).then(
            fn=lambda: "",  # This returns an empty string to clear the input
            outputs=[chat_input]
        )

        # Connect history dropdown to output
        history_dropdown.change(
            fn=process_selection,
            inputs=[history_dropdown],
            outputs=[output]
        ).then(
            fn=update_plan,
            inputs=[output],
            outputs=[]
        )

    return survey_app

# Replace the existing launch code with this
if __name__ == "__main__":
    survey_interface = create_survey_interface()
    survey_interface.launch(share=False)
