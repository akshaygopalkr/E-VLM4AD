import argparse
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import os
from peft import LoraConfig, get_peft_model
from tqdm import tqdm as progress_bar
from transformers import BlipForQuestionAnswering, BlipProcessor
from single_frame_dataset import SingleFrameDataset
from torch.utils.data import DataLoader
from multi_frame_model import DriveVLMT5
import json
import pandas as pd
import torch
from fvcore.nn import FlopCountAnalysis
from thop import profile
from torchprofile import profile_macs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def val_model(dloader):

    model.eval()
    ids_answered = set()
    test_data_greedy = []
    test_data_beam = []

    for idx, (batch, img_paths, q_texts, orig_encodings) in progress_bar(enumerate(dloader), total=len(dloader)):


        outputs_beam = model.generate(**batch, max_length=args.max_len, num_beams=args.beam_size)
        outputs_greedy = model.generate(**batch, max_length=args.max_len)

        # Run the model and profile the forward pass

        # Profile the model to estimate FLOPs
        flop_input = (batch['input_ids'], batch['pixel_values'], batch['attention_mask'])

        flops = FlopCountAnalysis(model, flop_input)
        print(flops.total())

        # Get the text output
        text_outputs_beam = [processor.decode(output, skip_special_tokens=True) for output in outputs_beam]
        text_outputs_greedy = [processor.decode(output, skip_special_tokens=True) for output in outputs_greedy]
        print('Beam search outputs: ')
        print(text_outputs_beam)
        print('Greedy outputs:')
        print(text_outputs_greedy)

        for image_path, q_text, text_output_beam, text_output_greedy in zip(img_paths, q_texts, text_outputs_beam, text_outputs_greedy):

            # Skip duplicate questions
            if image_id_dict[image_path + ' ' + q_text] in ids_answered:
                continue

            ids_answered.add(image_id_dict[image_path + ' ' + q_text])
            test_data_greedy.append({'image_id': image_id_dict[image_path + ' ' + q_text], 'caption': text_output_greedy})
            test_data_beam.append({'image_id': image_id_dict[image_path + ' ' + q_text], 'caption': text_output_beam})

    # Save test output to file
    with open(os.path.join('results', args.model_name, 'predictions_beam_2.json'), 'w') as f:
        json.dump(test_data_beam, f)
    with open(os.path.join('results', args.model_name, 'predictions_greedy_2.json'), 'w') as f:
        json.dump(test_data_greedy, f)


def save_experiment(stats, g_type):
    """
    Saves the experiment results to a csv
    :param args: The hyperparameters used
    :param statistics: The accuracies for the training, validation, and test sets
    """

    trial_dict = {}

    # Add metrics to dictionary
    for metric, score in stats.eval.items():
        trial_dict[metric] = [score]
    
    fname = 'metrics_' + g_type + '.csv'

    trial_dict = pd.DataFrame(trial_dict)
    trial_dict.to_csv(os.path.join('results', args.model_name, fname), index=False, header=True)



def params():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='20240215-044630')
    parser.add_argument('--beam-size', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lora-dim', default=256, type=int)
    parser.add_argument('--lora-enabled', action='store_true')
    parser.add_argument('--lora-alpha', default=512, type=int)
    parser.add_argument('--lora-dropout', default=0.05, type=float)
    parser.add_argument('--max-len', default=512, type=int)
    args = parser.parse_args()

    return args



if __name__ == '__main__':

    # Load in the arguments
    args = params()

    # Load processors and models
    processor = BlipProcessor.from_pretrained('Salesforce/blip-vqa-base')
    model = BlipForQuestionAnswering.from_pretrained('Salesforce/blip-vqa-base')
    
    # Create LoRA model
    if args.lora_enabled:
        lora_config = LoraConfig(
            r = args.lora_dim,
            lora_alpha = args.lora_alpha,
            lora_dropout = args.lora_dropout,
            bias='none',
            target_modules=['query', 'value']
        )
        model = get_peft_model(model, lora_config)
    
    model.load_state_dict(torch.load(os.path.join('results', args.model_name, args.model_name + '.pth')))
    model.to(device)
    

    # Load dataset and dataloader
    test_dset = SingleFrameDataset(
        input_file=os.path.join('data', 'single_frame',
                                'single_frame_test.json'),
        processor=processor,
        custom_train=True
    )
    test_dloader = DataLoader(test_dset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=test_dset.test_collate_fn, drop_last=True)


    # Load in image ids
    with open(os.path.join('data', 'single_frame', 'image_id.json')) as f:
        image_id_dict = json.load(f)

    # Get the loss and predictions from the model
    val_model(test_dloader)

    annotation_file = os.path.join('data', 'single_frame', 'single_frame_test_coco.json')
    results_file_beam = os.path.join('results', args.model_name, 'predictions_beam_2.json')
    results_file_greedy = os.path.join('results', args.model_name, 'predictions_greedy_2.json')

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result_beam = coco.loadRes(results_file_beam)
    coco_result_greedy = coco.loadRes(results_file_greedy)
    

    # create coco_eval object by taking coco and coco_result
    coco_eval_beam = COCOEvalCap(coco, coco_result_beam)
    coco_eval_greedy = COCOEvalCap(coco, coco_result_greedy)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    coco_eval_beam.params['image_id'] = coco_result_beam.getImgIds()
    coco_eval_greedy.params['image_id'] = coco_result_greedy.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval_beam.evaluate()
    coco_eval_greedy.evaluate()

    # Save the experiment results
    save_experiment(coco_eval_beam, 'beam')
    save_experiment(coco_eval_greedy, 'greedy')
