import run_model
import torch
import argparse
import json
import os
from evaluate import do_evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--program_generator', default="../models/CLEVR/program_generator_700k.pt")
parser.add_argument('--execution_engine', default="../models/CLEVR/execution_engine_700k_strong.pt")
parser.add_argument('--baseline_model', default=None)
parser.add_argument('--use_gpu', default=1, type=int)

# For running on a preprocessed dataset
parser.add_argument('--input_question_h5', default='data/val_questions.h5')
parser.add_argument('--input_features_h5', default='data-ssd/val_features.h5')
parser.add_argument('--use_gt_programs', default=0, type=int)

# This will override the vocab stored in the checkpoint;
# we need this to run CLEVR models on human data
parser.add_argument('--vocab_json', default=None)

# For running on a single example
parser.add_argument('--question', default=None)
parser.add_argument('--image', default="../img/CLEVR_val_000013.png")
parser.add_argument('--cnn_model', default='resnet101')
parser.add_argument('--cnn_model_stage', default=3, type=int)
parser.add_argument('--image_width', default=224, type=int)
parser.add_argument('--image_height', default=224, type=int)

parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_samples', default=None, type=int)
parser.add_argument('--family_split_file', default=None)

parser.add_argument('--sample_argmax', type=int, default=1)
parser.add_argument('--temperature', default=1.0, type=float)

# If this is passed, then save all predictions to this file
parser.add_argument('--output_h5', default=None)

parser.add_argument('--focus_data', default=None)
parser.add_argument('--focus_img', default=None)

parser.add_argument('--question_family', default=0)

parser.add_argument('--benchmark_questions', default='../benchmark/reduced_questions.json')
parser.add_argument('--benchmark_images', default='../benchmark/images/')
parser.add_argument('--benchmark_results', default='../benchmark/results/')

args = parser.parse_args()

question_family = int(args.question_family)
model = run_model.init_model(args)

dtype = torch.FloatTensor
if args.use_gpu == 1:
    dtype = torch.cuda.FloatTensor
cnn = run_model.build_cnn(args, dtype)

raw=open(args.benchmark_questions,encoding='utf-8')
data=json.load(raw)
scores=[]
question_idx=0
scoref=open('./scores_%d.txt'%(question_family),'w')
questionf=open('./ques.txt','w')

RESULTS = args.benchmark_results
if not os.path.exists(RESULTS):
    os.mkdir(RESULTS)
for q in data['questions']:
    question_idx+=1
    print("Now %d"%(question_idx))
    if len(q['answer_objects'])==0: continue  
    if q['question_family_index']!=question_family: continue
    imageName=q['image']
    focus_imgName='%s_%d'%(imageName.replace('CLEVR_new_','segment'),question_idx)
    args.focus_img=RESULTS + '/%s.png'%(focus_imgName)
    args.question = q['question']
    args.image=args.benchmark_images + '/images/%s.png'%(imageName)
    run_model.run_single_example(args, model, cnn)
    score=do_evaluate(args.benchmark_images + '/segments/%s.seg.png'%(imageName),RESULTS+'/%s.png'%(focus_imgName),q['answer_objects'], args.benchmark_images + '/scenes/%s.json'%(imageName))
    print(score,file=scoref)
    print(q['question'],file=questionf)
    scores=scores+score
auc_data=sorted(scores,key=lambda s:-s[0])
#print(auc_data)
now=0
final=0
for (s,g) in auc_data:
    if(g): now+=1
    final+=now

final=final/(now*len(auc_data))
print("FAMILY", args.question_family, " AUC=", final)

