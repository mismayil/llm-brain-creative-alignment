import argparse
import pathlib
from dotenv import load_dotenv
import pandas as pd

from cadabra.utils import write_json, remainder_args_to_dict, convert_nan_to_none, find_files
from cadabra.textgrid import get_transcript_from_textgrid

load_dotenv()


def prepare_templeton_aut_data(input_path, **kwargs):
    if input_path.endswith('.csv'):
        input_data = pd.read_csv(input_path)
    elif input_path.endswith('.xlsx'):
        input_data = pd.read_excel(input_path)
    else:
        raise ValueError("Input path must point to a .csv or .xlsx file.")
    
    task_condition = None

    if "condition" in kwargs:
        task_condition = kwargs["condition"]
        print(f"Filtering data for task condition: {task_condition}")
        input_data = input_data[input_data['condition'] == task_condition]

    stimuli = input_data["stimuli"].unique().tolist()
    print(f"Found {len(stimuli)} unique stimuli.")
    task_data = []

    id_prefix = "templeton_aut"
    if "condition" in kwargs and kwargs["condition"]:
        id_prefix += f"-{kwargs['condition']}"

    for stimulus in stimuli:
        stimulus_id = stimulus.replace(" ", "_")
        task_data.append({
            "id": f"{id_prefix}-{stimulus_id}",
            "stimuli_id": stimulus_id,
            "stimuli": stimulus,
            "condition": task_condition,
        })

    return task_data

def prepare_templeton_aut_data_with_subjects(input_path, **kwargs):
    if input_path.endswith('.csv'):
        input_data = pd.read_csv(input_path)
    elif input_path.endswith('.xlsx'):
        input_data = pd.read_excel(input_path)
    else:
        raise ValueError("Input path must point to a .csv or .xlsx file.")
    
    task_condition = None

    if "condition" in kwargs:
        task_condition = kwargs["condition"]
        print(f"Filtering data for task condition: {task_condition}")
        input_data = input_data[input_data['condition'] == task_condition]

    task_data = []

    id_prefix = "templeton_aut"
    if "condition" in kwargs and kwargs["condition"]:
        id_prefix += f"-{kwargs['condition']}"

    for sample in input_data.itertuples():
        stimuli_id = sample.stimuli.replace(" ", "_")
        response = convert_nan_to_none(sample.response)
        if response is not None:
            task_data.append({
                "id": f"{id_prefix}-{sample.id}-{stimuli_id}",
                "stimuli_id": stimuli_id,
                "stimuli": sample.stimuli,
                "condition": task_condition,
                "response": response,
                "subject_id": sample.id,
            })

    return task_data

def prepare_lebel2023_transcript_data(input_path, **kwargs):
    textgrid_files = find_files(input_path, "TextGrid")
    
    task_data = []

    for textgrid_path in textgrid_files:
        textgrid_path = pathlib.Path(textgrid_path)
        transcript = get_transcript_from_textgrid(textgrid_path)
        task_data.append({
            "id": f"lebel-{textgrid_path.stem}",
            "stimuli_id": textgrid_path.stem,
            "stimuli": transcript
        })

    return task_data

def prepare_five_sent_story_data(input_path, **kwargs):
    txt_files = find_files(input_path, "txt")
    
    task_data = []

    for txt_path in txt_files:
        txt_path = pathlib.Path(txt_path)
        with open(txt_path, 'r') as f:
            items = f.readlines()
        
        for line in items:
            if line.strip():
                item = line.strip()
                task_data.append({
                    "id": f"five-sent-story-item-{item}",
                    "stimuli_id": item,
                    "stimuli": ", ".join(item.split("-")),
                    "semantic_dist": "high" if "high" in str(txt_path) else "low"
                })
    return task_data

TASK_MAP = {
    "templeton_aut": prepare_templeton_aut_data,
    "templeton_aut_with_subjects": prepare_templeton_aut_data_with_subjects,
    "lebel2023_transcript": prepare_lebel2023_transcript_data,
    "five_sent_story": prepare_five_sent_story_data,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input-path", type=str, help="Input file path", required=True
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        required=True,
        help="Output file path.",
    )
    parser.add_argument("-t", "--task", type=str, help="Task name", required=True)
    parser.add_argument("rest", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    task_args = remainder_args_to_dict(args.rest)
    task_data = TASK_MAP[args.task](args.input_path, **task_args)
    output_path = pathlib.Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "metadata": {
            "source": args.input_path,
            "size": len(task_data),
            "task": args.task,
            "task_args": task_args,
        },
        "data": task_data,
    }
    write_json(output_data, output_path)
    print(f"Output data saved to {output_path}")


if __name__ == "__main__":
    main()
