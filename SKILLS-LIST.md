# Skills

A skill teaches what the model doesn't already know: a widget's format, how to build a file, or
Uflo's own output structure. Persona prompts that ask for what models do unprompted are not
skills. The tools they use are in TOOLS.md.

Each skill is an Agent Skills folder: SKILL.md, whose frontmatter gives its `name` and
`description`, then the instructions, with optional scripts and references beside it. The model
loads one through the `skill` tool.

## Building now

| skill | kind | covers |
|---|---|---|
| docx | file | Word documents in the sandbox, from Anthropic's office skill |
| pptx | file | PowerPoint decks in the sandbox, from Anthropic's office skill |
| xlsx | file | Excel workbooks in the sandbox, from Anthropic's office skill |
| pdf | file | PDFs in the sandbox, from Anthropic's office skill |
| data-analysis | file | analysis and charts in the sandbox, counting included |
| question writer | teaching | quizzes, practice questions, DOK assignments, AI-resistant tasks, learning outcomes, surveys |
| lesson-design | teaching | lessons, units, syllabi, PD sessions, UDL choice boards, activities, group work, simulations, standards sequencing |
| differentiation | teaching | rewriting material for a target reading level, English learners and common IEP accommodations, reading level measured in the sandbox; combines with the file skills to turn one deck or doc into differentiated versions |
| rubric writer | teaching | writing grading rubrics the Uflo way ([rubric refinement](https://docs.uflo.io/instructors/grading/rubric-refinement)): exact answers, partial-credit tiers, every accepted form listed, no dominant criterion, criteria in answer order, no overlap |
| research | research | many searches and deep searches until the data is in hand, then the sandbox and other skills to turn it into an output |

## Later

The widget skills wait for the widget tool, and the rest read through our MCP server.

| skill | kind | covers |
|---|---|---|
| desmos | widget | graphs on the graphing calculator |
| molecule | widget | 2D and 3D molecular structures |
| periodic-table | widget | a periodic table highlighting chosen elements |
| diagram | widget | labeled visuals: SVG code for anything that needs exact labels (flowcharts, timelines, Venn diagrams, cycles, mindmaps), `generate_image` for maps, infographics and illustrations |
| flashcards | widget | flashcard decks |
| quiz-question | widget | interactive quiz questions |
| solution | widget | step-by-step solutions |
| phet | widget | an existing PhET simulation, picked and set up for the concept |
| interactive-scene | widget | self-contained HTML built in the sandbox, checked by screenshot, then delivered: activities and mini-games, and simulations that visualize a process with controls |
| analytics | teaching | course analytics, assessment analytics, question analytics (difficulty, discrimination, distractors, questions to fix); reads through our MCP server |
| rubric writer | teaching | refining a rubric from 5–10 submissions across the score range: find the criterion that misfires, edit, regrade all, and aim for 95% agreement. Reads submissions and manual regrades through our MCP server |
