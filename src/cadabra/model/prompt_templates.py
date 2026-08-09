# templeton aut prompt templates
TEMPLETON_AUT_CREATE_TEMPLATE = "Think of creative uses for {stimuli}. Report the most original idea. Keep thinking of ideas during the thinking period. Quality is more important than quantity."
TEMPLETON_AUT_OBJECT_TEMPLATE = "Think of the physical properties of {stimuli}. Report the most prominent characteristic. Keep thinking of ideas during the thinking period."

TEMPLETON_AUT_CREATE_SHORT_TEMPLATE = "Think of creative uses for {stimuli}. Report only the most original idea in a few words."
TEMPLETON_AUT_OBJECT_SHORT_TEMPLATE = "Think of the physical properties of {stimuli}. Report only the most prominent characteristic in a few words."

TEMPLETON_AUT_EMPTY_TEMPLATE = ""
TEMPLETON_AUT_NOLANG_TEMPLATE = "#" * len(TEMPLETON_AUT_CREATE_TEMPLATE)

LLM_AUT_SCORING_TEMPLATE = (
    "You are an expert in evaluating the creativity of responses in Alternative Uses Task (AUT). "
    "The AUT is a test where participants are given a common object and asked to think of a creative use for that object. "
    "Your task is to evaluate the creativity of the given response. Please rate the creativity of the response on a scale from 1 to 5, where 1 indicates a response that is not creative at all, and 5 indicates a response that is highly creative. "
    "Consider factors such as originality, uniqueness, and how well the response deviates from typical uses of the object when assigning your rating.\n"
    "Output only the creativity rating as a single integer between 1 and 5 and nothing else.\n"
    "Object: {stimuli}\n"
    "Response: {output}\n"
    "Creativity Rating: "
)