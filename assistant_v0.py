import gradio as gr
from datetime import datetime, timedelta
import pandas as pd
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv('.env')

core_beginner_materials = [
'''Generative AI Skills for Young Entrepreneurs  
To be able to utilize the commonly used generative AI tools, you will need some basic knowledge and training on them. 
Learning Resources:  Explore materials on "Generative AI for Beginners" and LinkedIn Learning's AI Fundamentals.
Practice Tip: Choose one generative AI model (ChatGPT, Copilot, Gemini or Claude), sign into its free version and familiarize yourself with creating content using AI.
''',
'''Market Analysis and Customer Understanding  
To be able to perform market analysis and utilize customer understanding, you’ll need some basic knowledge and training on them. 
Learning Resources: Explore materials on consumer behavior and look into introductory YouTube tutorials on market analysis.
Practice Tip: Choose a product or service that you commonly use, define its usual customers and report how and why are they using the product. Also find out about the competition for that product. You can use generative AI to assist you.
''',
'''Creating and Testing Business Ideas  
To be able to create and test business Ideas perform market analysis and utilize customer understanding, you’ll need some basic knowledge and training on them. 
Learning Resources: Udemy's "Business Idea Creation" course or articles on the Lean Startup method can be useful.
Practice Tip: Develop and refine a simple business idea. Gather feedback from friends or online communities to start testing and iterating on the concept. You can use generative AI to assist you.
''',
'''Making a Business Plan  
To be able to make an effective business plan, you’ll need some basic knowledge and training on them. 
Learning Resources: Try HubSpot's free "Business Plan Template" and review tutorials on creating business plans on LinkedIn Learning.
Practice Tip: Draft a mini-business plan, focusing on basic sections such as vision, goals, target audience, and financial projections. You can use generative AI to assist you.
''',
'''Running a Business  
To be able to consider all aspects of running a successful business, you’ll need some basic knowledge and training on them. 
Learning Resources: SBA.gov and Small Business Development Centers (SBDCs) offer beginner resources for business management.
Practice Tip: Study case studies on small business operations and identify essential operational processes. You can use generative AI to assist you.
''',
'''Branding and Marketing  
To be able to start successfully marketing your  business, products and services, you’ll need some basic knowledge and training on marketing essentials.
Learning Resources: Coursera’s "Marketing in a Digital World" or free Google Ads and Facebook Blueprint courses are great starting points.
Practice Tip: Develop a simple ad for a hypothetical product and run it on a small budget to understand basics in digital advertising. You can use generative AI to assist you.
''',
'''Sales and Customer Service  
To be able to start closing deals, make effective sales efforts and provide successful customer service, you’ll need some basic knowledge and training on marketing essentials.
Learning Resources: HubSpot Academy has free courses on sales fundamentals and customer service.
Practice Tip: Practice a basic sales pitch or handle a mock customer inquiry to build confidence in communication and service skills. You can use generative AI to assist you.
''',
'''HR Management  
To be able to understand business from human resources perspective, you’ll need some basic knowledge and training on HR management.
Learning Resources: Check out LinkedIn Learning’s "Human Resources Foundations" and free articles from SHRM (Society for Human Resource Management).
Practice Tip: Learn about simple HR tasks, like drafting a job description, to get familiar with HR basics. You can use generative AI to assist you.
''',
]
computer_literacy='''General Resources for Computer Literacy  
If you're looking to improve your computer skills, Microsoft Learn and Google’s Applied Digital Skills offer free tutorials on a range of software and digital tools.
'''

training_period = pd.DataFrame(data={
    'date':[datetime(year=2025,day=15,month=1)+timedelta(days=1) for i in range(len(core_beginner_materials))],
    'content':[x.split('\n')[0] for x in core_beginner_materials]}
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
current_study_plan=''
recommended_materials=''

def get_llm_response(system_prompt,prompt):
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
    '\n\nStudy day 1 (Mo xx.xx.202x): [recommended learning materials and rationale]'
    '\nStudy day 2 (Tu xx.xx.20xx): [recommended learning materials and rationale]'
    '\n...'                         
    '\nStudy day 2 (Tu xx.xx.20xx): [recommended learning materials and rationale]\n\n'           
    'In your response, address the student personally and make your response concise and simple using markdown format. You are also given additional background information of the student, which you can leverage in you response if its relevant.')
    else:
        system_prompt = ('You are a helpful learning assistant who designs learning timetables for students. You help students to prepare for the upcoming face-to-face (f2f) teaching.'
    'You are given a list of suggested learning materials for the student. Each of them takes 1 day to complete. You are also given current date and startdate of the f2f teaching.'
    'Your task: Given suggested learning materials (from 1 to 8), plan a student a detailed learning timetable so that all materials are studied before the f2f teaching begins.'
    'Build a coherent and logical timetable that takes into account optimal tempo and order of learnign materials. Give your timetable as a concise list of dates and contents, such as'
    '\nDay xx.xx.xx: [recommended learning materials and rationale]\n'
    '\nDay xx.xx.xx: [recommended learning materials and rationale]\n'
    'In your response, address the student personally and make your response concise and simple using markdown format.')

    if language != 'english':
        prompt = f'Information of myself:\n{background_info}\n\nToday is {current_date.strftime("%B %d, %Y")} and face-to-face training starts at {startdate.strftime("%B %d, %Y")}, there are {days_until_training} days for learning the following materials:\n{training_materials}\n\n. Give me an optimized learning timetable in {language} language.'
    else:
        prompt = f'Information of myself:\n{background_info}\n\nToday is {current_date.strftime("%B %d, %Y")} and face-to-face training starts at {startdate.strftime("%B %d, %Y")}, there are {days_until_training} days for learning the following materials:\n{training_materials}\n\n. Give me an optimized learning timetable.'

    response = get_llm_response(system_prompt,prompt)
    return response

def translate_text(input_text,language):
    if language != 'english':
        output_text = get_llm_response(f'You are a professional text translator from English to {language}. You are specialized in technical translations and learning materials. You return only the translated text, nothing else.',f'Translate the following texts into {language} while retain the same formatting:\n\n{input_text}')
        return output_text
    else:
        return input_text

def process_survey(education, experience,native_language,additional_language,q0, q1, q2, q3, q4, q5,q6,q7,q8):
    global learning_plans
    global recommended_materials

    # Calculate results
    module_levels = [q1, q2, q3, q4, q5,q6,q7,q8]
    computer_level = q0

    information =''
    if len(education)>0:
        information += f'Education: {education}.\n'
    if len(experience)>0:
        information += f'Experience: {experience}.\n'

    count = 0
    recommended_materials = ''
    for k in range(8):
        if skill_level_score_map[module_levels[k]]==1:
            count += 1
            recommended_materials+=f'{count}. ' + core_beginner_materials[k]+'\n'

    if computer_level==skills_levels[0]:
        count += 1
        recommended_materials += f'{count}. ' + computer_literacy + '\n'

    recommended_materials = translate_text(f'# We recommend the following {count} learning materials:\n\n' + recommended_materials + '\n\n',native_language)

    if count==0:
        response = translate_text(f'On the basis of your input, you are at least basic level on all core skills. No additional learning is necessary at the moment.\n',native_language)
    else:
        timetable = get_llm_timetable(recommended_materials,information,native_language)
        response = recommended_materials + translate_text(f'# Here is suggested learning timetable for you:\n\n {timetable}',native_language)

    learning_plans = [response]

    return response

def create_chat_response(user_input,language):
    global current_study_plan
    global learning_plans

    # Use the same OpenAI client and configuration from the original script
    system_prompt = (
        "You are a helpful learning assistant. The user has received a personalized learning recommendation and plan. You can only discuss about the students' plan and update it as requested."
        "You task: Student wants to update or revise his/her existing personalized learning plan. Read the current student learning plan carefully. Then provide an updated plan as requested by the student. You can modify dates and order of the plan, e.g., to only include specific days or drop some of the content."
        "You may not change the recommended materials, current date or the starting date of the face-to-face trainings. You must always provide a new personalized study plan, either an old one or updated one."
    )

    prompt = f"Here is all relevant information and the current study plan that you may update:\n\n{current_study_plan}\n\nUser question related to plan: {user_input}"
    response = get_llm_response(system_prompt, prompt)

    learning_plan = recommended_materials + translate_text(f'# You learning plan: \n\n'+ response,language)

    learning_plans.append(learning_plan)

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
            debug_btn = gr.Button("🐞 DEBUG PREFILL",scale=0.05)

        # Background information section
        with gr.Group():
            gr.Markdown("### Background Information")
            education = gr.Textbox(label="Highest Level of Education",
                                   placeholder="e.g., Bachelor's in Computer Science",
                                   lines=1)
            experience = gr.Textbox(label="Relevant Work Experience in Entrepreneurship (if any)",
                                    placeholder="Describe your entrepreneurial experience",
                                    lines=3)

            native_language = gr.Textbox(label="Native languages",
                                         placeholder="Which is/are your native language",
                                         lines=1)

            # Language section with side-by-side layout
            with gr.Row():
                gr.Markdown("Other Languages")
                additional_language = gr.Textbox(label="Language",
                                                 placeholder="",
                                                 lines=1)
                additional_language_level = gr.Radio(choices=skills_levels,
                                                     label="Level",
                                                     info="Select one option")

        # Survey questions section
        with gr.Group():
            gr.Markdown("### Core skills assessment")
            # Create radio button groups (existing code)
            q0 = gr.Radio(choices=skills_levels, label="Your general computer skills", info="Select one option")
            q1 = gr.Radio(choices=skills_levels, label="1. Generative AI skills  ", info="Select one option")
            q2 = gr.Radio(choices=skills_levels, label="2. Market Analysis and Customer Understanding ",
                          info="Select one option")
            q3 = gr.Radio(choices=skills_levels, label="3. Creating and Testing Business Ideas",
                          info="Select one option")
            q4 = gr.Radio(choices=skills_levels, label="4. Making a Business Plan", info="Select one option")
            q5 = gr.Radio(choices=skills_levels, label="5. Running a Business", info="Select one option")
            q6 = gr.Radio(choices=skills_levels, label="6. Marketing and Advertising", info="Select one option")
            q7 = gr.Radio(choices=skills_levels, label="7. Sales and Customer Service", info="Select one option")
            q8 = gr.Radio(choices=skills_levels, label="8. HR Management", info="Select one option")

        # Create submit button and output section
        submit_btn = gr.Button("Submit Survey",scale=0.05)
        gr.Markdown("### Recommendations for preparations")

        output = gr.Markdown()

        # Add history dropdown
        with gr.Row():
            history_dropdown = gr.Dropdown(
                choices=[],
                label="View Previous Plans",
                type="index",
                interactive=True
            )

        # Chat input and submit after recommendations
        with gr.Group(visible=False) as chat_group:
            chat_input = gr.Textbox(label="Ask for updates in your learning plan",lines=3)
            chat_submit = gr.Button("Request plan update")

        def update_history_dropdown():
            return gr.Dropdown(choices=learning_plans, value=learning_plans[-1] if learning_plans else None)

        def select_history(idx):
            if idx is not None and 0 <= idx < len(learning_plans):
                return learning_plans[idx]
            return ""

        submit_btn.click(
            fn=process_survey,
            inputs=[education, experience, native_language, additional_language, q0, q1, q2, q3, q4, q5, q6, q7, q8],
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
                "swedish",  # additional_language
                skills_levels[1],  # additional_language_level
                skills_levels[1],  # q0 computer skills
                skills_levels[0],  # q1 generative AI
                skills_levels[1],  # q2 market analysis
                skills_levels[1],  # q3 business ideas
                skills_levels[1],  # q4 business plan
                skills_levels[0],  # q5 running business
                skills_levels[0],  # q6 marketing
                skills_levels[0],  # q7 sales
                skills_levels[1]  # q8 HR
            ),
            outputs=[
                education, experience, native_language, additional_language, additional_language_level,
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
            fn=select_history,
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
    survey_interface.launch()