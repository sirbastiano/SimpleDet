from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from pathlib import Path
import matplotlib.pyplot as plt
import json

import pandas as pd 

from style import set_style

set_style()


def find_json(folder, mode='test'):
    jsons = list(folder.glob('**/**/.json'))
    if mode == 'train':
        # filter out the vis_data folder
        jsons = [j for j in jsons if 'vis_data' in str(j)]
        # not pick the scalars json
        jsons = [j for j in jsons if 'scalars' not in str(j)]
    else:
        # pick the test_results forlder
        jsons = [j for j in jsons if j.name.startswith('coco_metrics')]

    assert len(jsons) == 1, f"Found {len(jsons)} json files. Expected 1. \n {jsons}"
    return jsons[0]


def evaluate_object_detector(res_file, ann_file):
    # Load the ground truth annotations
    coco_gt = COCO(ann_file)
    
    # Load the detection results
    coco_dt = coco_gt.loadRes(res_file)
    
    # Create COCOeval object
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    
    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Extract serializable results
    eval_results = {
        'params': {k: v for k, v in vars(coco_eval.params).items() if k != 'imgIds'},
        'counts': [int(i) for i in coco_eval.eval['counts']],
        'precision': coco_eval.eval['precision'].tolist(),
        'recall': coco_eval.eval['recall'].tolist(),
    }
    
    
    # Return the detailed evaluation results
    return coco_eval.stats, eval_results, coco_eval

       
def parse_and_plot(json_file_path, verbose=False, visualize=True, mode='test'):
    # Load the JSON file
    
    if mode == 'test':
        df = pd.read_json(json_file_path, orient='index')
        try:
            res = df.to_dict()[0]
            return {'name': json_file_path.parent.parent.stem, 'coco_bbox_mAP': res['coco/bbox_mAP'], 'coco_bbox_mAP_50': res['coco/bbox_mAP_50'], 'coco_bbox_mAP_75': res['coco/bbox_mAP_75']}
        except KeyError:
            # handling zero-valued tests:
            return {'name': json_file_path.parent.parent.stem, 'coco_bbox_mAP': 0, 'coco_bbox_mAP_50': 0, 'coco_bbox_mAP_75': 0}
            
    else:
        with open(json_file_path, 'r') as f:
            data = [json.loads(line) for line in f]

        # Initialize storage for different types of data
        epochs = []
        iterations = []
        lr = []
        loss = []
        loss_cls = []
        loss_bbox = []
        loss_bbox_rf = []
        data_time = []
        time = []
        memory = []
        step = []
        
        coco_bbox_mAP = []
        coco_bbox_mAP_50 = []
        coco_bbox_mAP_75 = []
        coco_bbox_mAP_s = []
        coco_bbox_mAP_m = []
        coco_bbox_mAP_l = []
        coco_iterations = []

        # Extract data
        for entry in data:
            epochs.append(entry.get('epoch'))
            iterations.append(entry.get('iter'))
            lr.append(entry.get('lr'))
            loss.append(entry.get('loss'))
            loss_cls.append(entry.get('loss_cls'))
            loss_bbox.append(entry.get('loss_bbox'))
            loss_bbox_rf.append(entry.get('loss_bbox_rf'))
            data_time.append(entry.get('data_time'))
            time.append(entry.get('time'))
            memory.append(entry.get('memory'))
            step.append(entry.get('step'))

            # Check for COCO metrics
            if 'coco/bbox_mAP' in entry:
                coco_bbox_mAP.append(entry.get('coco/bbox_mAP'))
                coco_bbox_mAP_50.append(entry.get('coco/bbox_mAP_50'))
                coco_bbox_mAP_75.append(entry.get('coco/bbox_mAP_75'))
                coco_bbox_mAP_s.append(entry.get('coco/bbox_mAP_s'))
                coco_bbox_mAP_m.append(entry.get('coco/bbox_mAP_m'))
                coco_bbox_mAP_l.append(entry.get('coco/bbox_mAP_l'))
                coco_iterations.append(entry.get('iter'))

        if verbose:
            print(f"Total iterations: {len(iterations)}")
            print(f"Coco iterations: {len(coco_iterations)}")
            # Print COCO values
            print(f"COCO bbox mAP: {coco_bbox_mAP}")
            print(f"COCO bbox mAP 50: {coco_bbox_mAP_50}")
            print(f"COCO bbox mAP 75: {coco_bbox_mAP_75}")
            print(f"COCO bbox mAP Small: {coco_bbox_mAP_s}")
            print(f"COCO bbox mAP Medium: {coco_bbox_mAP_m}")
            print(f"COCO bbox mAP Large: {coco_bbox_mAP_l}")

        if visualize:
            # Plotting
            plt.figure(figsize=(20, 20))

            # Learning Rate
            plt.subplot(3, 2, 1)
            plt.plot(iterations, lr, label='Learning Rate')
            plt.xlabel('Iteration')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate over Iterations')
            plt.grid(True)

            # Loss
            plt.subplot(3, 2, 2)
            plt.plot(iterations, loss, label='Total Loss')
            plt.plot(iterations, loss_cls, label='Classification Loss')
            plt.plot(iterations, loss_bbox, label='BBox Loss')
            plt.plot(iterations, loss_bbox_rf, label='BBox RF Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Loss over Iterations')
            plt.legend()
            plt.grid(True)

            # Data and Processing Time
            plt.subplot(3, 2, 3)
            plt.plot(iterations, data_time, label='Data Time')
            plt.plot(iterations, time, label='Processing Time')
            plt.xlabel('Iteration')
            plt.ylabel('Time (seconds)')
            plt.title('Data and Processing Time over Iterations')
            plt.legend()
            plt.grid(True)


            # COCO mAP metrics
            if coco_bbox_mAP:
                plt.subplot(3, 2, 4)
                plt.plot(range(len(coco_iterations)), coco_bbox_mAP, label='COCO bbox mAP')
                plt.plot(range(len(coco_iterations)), coco_bbox_mAP_50, label='COCO bbox mAP 50')
                plt.plot(range(len(coco_iterations)), coco_bbox_mAP_75, label='COCO bbox mAP 75')
                # plt.plot(range(len(coco_iterations)), coco_bbox_mAP_s, label='COCO bbox mAP Small')
                # plt.plot(range(len(coco_iterations)), coco_bbox_mAP_m, label='COCO bbox mAP Medium')
                # plt.plot(range(len(coco_iterations)), coco_bbox_mAP_l, label='COCO bbox mAP Large')
                plt.xlabel('Iteration')
                plt.ylabel('COCO mAP')
                plt.title('COCO bbox mAP Metrics over Iterations')
                plt.ylim([0,1.])
                plt.legend()
                plt.grid(True)

            plt.tight_layout()
            plt.show()
        
        return {'name': json_file_path.parent.parent.stem, 'coco_bbox_mAP': coco_bbox_mAP, 'coco_bbox_mAP_50': coco_bbox_mAP_50, 'coco_bbox_mAP_75': coco_bbox_mAP_75}
    
    
def parse_json_test(json_file_path):
    partial = {
        'Seed': Path(json_file_path).parent.parent.name.split('_')[0],
        'BS': Path(json_file_path).parent.parent.name.split('_')[2],
        'LR': Path(json_file_path).parent.parent.name.split('_')[4],
        'ME': Path(json_file_path).parent.parent.name.split('_')[6],
        'OPT': Path(json_file_path).parent.parent.name.split('_')[8],
        'Band': Path(json_file_path).parent.parent.parent.name.split('perfect_')[-1],
    }

    # Read the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)


    # merge data and partial 
    data = {**data, **partial}
    return data