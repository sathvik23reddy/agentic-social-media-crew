from crewai import Agent, Task, Crew
from crewai import LLM

#Using locally run llama3.2
llm = LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434"
)

#Three agents to perform each task
trend_watcher = Agent(
    name="Trend Watcher",
    role="Social Media Analyst",
    goal="Find trending topics and hashtags on social media.",
    backstory="An AI expert in analyzing social media trends and identifying viral content.",
    llm=llm,
    verbose=True,
)

content_creator = Agent(
    name="Content Creator",
    role="Social Media Copywriter",
    goal="Write engaging social media posts based on trends.",
    backstory="A skilled AI writer who crafts attention-grabbing social media content.",
    llm=llm,
    verbose=True,
)

post_scheduler = Agent(
    name="Post Scheduler",
    role="Social Media Manager",
    goal="Schedule and optimize posts for engagement.",
    backstory="An AI-powered social media strategist who knows the best times to post.",
    llm = llm,
    verbose=True,
)

#Three agent have respective tasks
trend_task = Task(
    agent=trend_watcher,
    description="Analyze trending topics on social media and summarize findings.",
    expected_output="List of top 10 most trending topics in social media with a 2-line description for each."
)

#Output of trend analysis is piped to content generation task
content_task = Task(
    agent=content_creator,
    description="Pick 3 random topics from the trending topics list which cater to teenagers and young adults and generate engaging social media posts for each topic chosen",
    expected_output="3 engaging social media posts with hashtags and emojis.", 
    context=[trend_task],
)

schedule_task = Task(
    agent=post_scheduler,
    description="Suggest the best posting schedule for the generated posts.",
    expected_output="An optimized posting schedule with recommended time slots."
)

#Run agents
social_media_crew = Crew(
    agents=[trend_watcher, content_creator, post_scheduler],
    tasks=[trend_task, content_task, schedule_task],
    verbose=True,
)

social_media_crew.kickoff()
