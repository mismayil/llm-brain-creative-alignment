import argparse
import pathlib
from tqdm import tqdm

from cadabra.utils import read_json, write_json, get_template_keys, find_files
from cadabra.model.prompt_templates import *

USER_INSTRUCTION_TEMPLATES = {
    "default": "{prompt}",
    "templeton_aut_create": TEMPLETON_AUT_CREATE_TEMPLATE,
    "templeton_aut_object": TEMPLETON_AUT_OBJECT_TEMPLATE,
    "templeton_aut_create_short": TEMPLETON_AUT_CREATE_SHORT_TEMPLATE,
    "templeton_aut_object_short": TEMPLETON_AUT_OBJECT_SHORT_TEMPLATE,
    "llm_aut_scoring": LLM_AUT_SCORING_TEMPLATE,
    "templeton_aut_empty": TEMPLETON_AUT_EMPTY_TEMPLATE,
    "templeton_aut_nolang": TEMPLETON_AUT_NOLANG_TEMPLATE
}


def prepare_template_value(value):
    if isinstance(value, list):
        return ", ".join(value)
    return value


def prepare_template(sample, template):
    template_keys = get_template_keys(template)
    format_args = {
        k: prepare_template_value(sample[k])
        for k in template_keys
        if sample.get(k) is not None
    }
    return template.format(**format_args)


def prepare_user_instruction(sample, template):
    instruction_template = USER_INSTRUCTION_TEMPLATES[template]
    if callable(instruction_template):
        instruction_template = instruction_template(sample)
    return prepare_template(sample, instruction_template)

USER_INSTRUCTION_PROCESSORS = {
    "default": prepare_user_instruction,
}

SHOT_PROCESSORS = {
    "default": lambda *args, **kwargs: "",
}


def prepare_sample_for_eval(sample, template, num_shots=1, shot_data=None):
    user_instr_processor = USER_INSTRUCTION_PROCESSORS.get(
        template, USER_INSTRUCTION_PROCESSORS["default"]
    )
    shot_processor = SHOT_PROCESSORS.get(template, SHOT_PROCESSORS["default"])

    eval_data = []

    user_prompt = user_instr_processor(sample, template)
    shot_prompt = shot_processor(
        sample, template, num_shots=num_shots, shot_data=shot_data
    )

    if shot_prompt:
        if isinstance(user_prompt, list):
            user_prompt += ["\n\n" + shot_prompt]
        else:
            user_prompt += "\n\n" + shot_prompt

    if isinstance(user_prompt, list):
        user_prompt = [p for p in user_prompt]

    eval_data.append(
        {**sample, "id": sample["id"], "user_prompt": user_prompt}
    )

    return eval_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-path",
        type=str,
        help="Path to input task data in json or directory",
        required=True,
    )
    parser.add_argument(
        "-t", "--template", type=str, default="default", help="Template name"
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        required=True,
        help="Output file path.",
    )
    parser.add_argument(
        "-sp",
        "--shot-path",
        type=str,
        default=None,
        help="Path to shot examples in json",
    )
    parser.add_argument(
        "-n",
        "--num-shots",
        type=int,
        default=1,
        help="Number of shot examples to include",
    )

    args = parser.parse_args()

    datapaths = []

    datapath = pathlib.Path(args.input_path)

    if datapath.is_file():
        datapaths.append(args.input_path)
    else:
        datapaths.extend(find_files(args.input_path, "json"))

    for datapath in datapaths:
        input_data = read_json(datapath)
        shot_data = read_json(args.shot_path) if args.shot_path is not None else None

        eval_data = []

        for sample in tqdm(
            input_data["data"], desc=f"Preparing {datapath} for evaluation"
        ):
            eval_data.extend(
                prepare_sample_for_eval(
                    sample,
                    template=args.template,
                    num_shots=args.num_shots,
                    shot_data=shot_data,
                )
            )

        datapath = pathlib.Path(datapath)
        output_path = pathlib.Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "metadata": {
                "source": str(datapath),
                "template": args.template,
                "size": len(eval_data),
                "shot_path": args.shot_path,
                "output_path": str(output_path),
            },
            "data": eval_data,
        }
        write_json(output_data, output_path)

        print(f"Output data saved to {output_path}")


if __name__ == "__main__":
    main()
