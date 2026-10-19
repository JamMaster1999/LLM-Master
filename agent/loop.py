"""The loop tools: todo is Codex's update_plan; ask_user is Claude Code's AskUserQuestion."""
from typing import Annotated, Literal

from pydantic import Field
from typing_extensions import TypedDict  # pydantic takes typing's only from Python 3.12

from agent.result import Result

TODO = {"type": "function", "function": {
    "name": "todo",
    "description": "Updates the task plan.\nProvide an optional explanation and a list of plan items, each with a step "
                   "and status.\nAt most one step can be in_progress at a time.\n",
    "parameters": {"type": "object", "properties": {
        "explanation": {"type": "string", "description": "Optional explanation for this plan update."},
        "plan": {"type": "array", "description": "The list of steps", "items": {"type": "object", "properties": {
            "step": {"type": "string", "description": "Task step text."},
            "status": {"type": "string", "enum": ["pending", "in_progress", "completed"], "description": "Step status."},
        }, "required": ["step", "status"], "additionalProperties": False}},
    }, "required": ["plan"], "additionalProperties": False}}}

ASK_USER = {"type": "function", "function": {
    "name": "ask_user",
    "description": "Use this tool when you need to ask the user questions during execution. This allows you to:\n"
                   "1. Gather user preferences or requirements\n2. Clarify ambiguous instructions\n"
                   "3. Get decisions on choices as you work\n4. Offer choices to the user about what direction to take.\n"
                   "\nUsage notes:\n- Users will always be able to select \"Other\" to provide custom text input\n"
                   "- Use multiSelect: true to allow multiple answers to be selected for a question\n"
                   "- If you recommend a specific option, make that the first option in the list and add "
                   "\"(Recommended)\" at the end of the label",
    "parameters": {"type": "object", "properties": {
        "questions": {"type": "array", "minItems": 1, "maxItems": 4, "description": "Questions to ask the user (1-4 questions)",
                      "items": {"type": "object", "properties": {
                          "question": {"type": "string", "description": "The complete question to ask the user. Should "
                                       "be clear, specific, and end with a question mark. Example: \"Which grade level "
                                       "is this for?\" If multiSelect is true, phrase it accordingly, e.g. \"Which "
                                       "sections should the quiz cover?\" Each question's text must be unique."},
                          "header": {"type": "string", "description": "Very short label displayed as a chip/tag (max "
                                     "12 chars). Examples: \"Grade\", \"Format\", \"Length\"."},
                          "options": {"type": "array", "minItems": 2, "maxItems": 4, "description": "The available "
                                      "choices for this question. Must have 2-4 options. Each option should be a "
                                      "distinct, mutually exclusive choice (unless multiSelect is enabled). There "
                                      "should be no 'Other' option, that will be provided automatically.",
                                      "items": {"type": "object", "properties": {
                                          "label": {"type": "string", "description": "The display text for this option "
                                                    "that the user will see and select. Should be concise (1-5 words) "
                                                    "and clearly describe the choice."},
                                          "description": {"type": "string", "description": "Explanation of what this "
                                                          "option means or what will happen if chosen."},
                                      }, "required": ["label", "description"]}},
                          "multiSelect": {"type": "boolean", "description": "Set to true to allow the user to select "
                                          "multiple options instead of just one. Use when choices are not mutually "
                                          "exclusive."},
                      }, "required": ["question", "header", "options", "multiSelect"]}},
    }, "required": ["questions"]}}}


class Step(TypedDict):
    step: str
    status: Literal["pending", "in_progress", "completed"]


class Option(TypedDict):
    label: str
    description: str


class Question(TypedDict):
    question: str
    header: str
    options: Annotated[list[Option], Field(min_length=2, max_length=4)]
    multiSelect: bool


def todo(plan: list[Step], explanation: str | None = None) -> str:
    return "Plan updated"  # the UI draws the plan from the call's arguments


def ask_user(questions: Annotated[list[Question], Field(min_length=1, max_length=4)]) -> Result:
    return Result("", waiting="question")


def answered(answers: dict[str, str] | None) -> str:
    """Claude Code's text for the user's answers, keyed by question; multi-select answers joined by commas."""
    if not answers:
        return "User declined to answer questions"
    pairs = ", ".join(f'"{question}"="{answer}"' for question, answer in answers.items())
    return f"User has answered your questions: {pairs}. You can now continue with the user's answers in mind."
