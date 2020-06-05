import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import util

from args import get_test_args
from collections import OrderedDict
from json import dumps
from models import BiDAF
from os.path import join
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD
from collections import Counter
from setup import get_embedding, process_file, build_features
import json

def test_model(questions, context, use_squad_v2= True):
    # Set up logging
    #args.save_dir = util.get_save_dir(args.save_dir, args.name, training=False)
    #log = util.get_logger(args.save_dir, args.name)
    #log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    #args = get_test_args()
    device, gpu_ids = util.get_available_devices()
    batch_size = 64 * max(1, len(gpu_ids))

    # Get embeddings
    #print('Loading embeddings...')
    word_vectors = util.torch_from_json('../data/word_emb.json')

    # Get model
    #print('Building model...')
    model = BiDAF(word_vectors=word_vectors,
                  hidden_size=100)
    model = nn.DataParallel(model, gpu_ids)
    model_path = "../save/training-02/best.pth.tar"
    #print(f'Loading checkpoint from {args.load_path}...')
    model = util.load_model(model, model_path, gpu_ids, return_step=False)
    model = model.to(device)
    model.eval()

    # Get data loader
    #print('Building dataset...')
    #record_file = vars(args)[f'{args.split}_record_file']
    # my code start here
    # this is a simple approch when dealing with the user date
    # according to your approch of creating the interface you can change this code
    # and also you have to check the function "process_file" in the setup.py file 
    processed_questions = []
    for index,question in enumerate(questions):
        processed_question = {"question":question ,
                              "id": index,
                              "answers": []
                             }
        processed_questions.append(processed_question)
    source = {
        "paragraphs" : [{
                        "qas" : processed_questions,
                        "context" : context
                       }]   
              }
    word_counter, char_counter = Counter(), Counter()
    with open("../data/word2idx.json","r") as f1:
        word2idx_dict = json.load(f1)
    with open("../data/char2idx.json","r") as f2:
        char2idx_dict = json.load(f2)
    my_test_examples, my_test_eval = process_file(source, "my_test", word_counter, char_counter)
    npz = build_features(my_test_examples, "my_test",
                               word2idx_dict, char2idx_dict, is_test=True)
    #my code end here
    dataset = SQuAD(npz, use_squad_v2)
    data_loader = data.DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=4,
                                  collate_fn=collate_fn)

    # Evaluate
    #print(f'Evaluating on {args.split} split...')
    nll_meter = util.AverageMeter()
    pred_dict = {}  # Predictions for TensorBoard
    sub_dict = {}   # Predictions for submission
    #eval_file = vars(args)[f'{args.split}_eval_file']    
    gold_dict = my_test_eval
    #print("gold_dict", gold_dict)
    #print("data_loader", data_loader)
    with torch.no_grad(), \
            tqdm(total=len(dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            batch_size = cw_idxs.size(0)
            # Forward
            log_p1, log_p2 = model(cw_idxs, qw_idxs)
            y1, y2 = y1.to(device), y2.to(device)
            loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
            nll_meter.update(loss.item(), batch_size)

            # Get F1 and EM scores
            p1, p2 = log_p1.exp(), log_p2.exp()
            starts, ends = util.discretize(p1, p2, 15, use_squad_v2)
            print("starts ",starts," ends ", ends)

            # Log info
            progress_bar.update(batch_size)
            #if args.split != 'test':
                # No labels for the test set, so NLL would be invalid
                #progress_bar.set_postfix(NLL=nll_meter.avg)

            idx2pred, uuid2pred = util.convert_tokens(gold_dict,
                                                      ids.tolist(),
                                                      starts.tolist(),
                                                      ends.tolist(),
                                                      use_squad_v2)
            pred_dict.update(idx2pred)
            sub_dict.update(uuid2pred)
            
    #print("my evaluation ....")
    
    #for el in pred_dict:
        #print(el, pred_dict[el])

    #for el in sub_dict:
        #print(el, sub_dict[el])
    return pred_dict

def main():
    questions = ["how old is ahmed ?", "with whom does ahmed live ?", "who killed kenedy"]
    context = "ahmed is 22 years old . he lives in ariana with his family and friends"
    pred_dict = test_model(questions, context)
    
    for el in pred_dict:
        print(el, pred_dict[el])
    


if __name__ == '__main__':
    main()