from turtle import shape
import numpy as np
import scipy.linalg
import scipy.io as io
import sys
import os
import motmetrics as mm
from yolox.evaluators import MOTEvaluator

from yolox.tracker.byte_tracker import BYTETracker
import numpy as np
sys.path.append(r'C:\Users\lzt\Desktop\CV\ByteTrack-main\yolox')
pathgt=r'C:\Users\lzt\Desktop\CV\MOTChallenge\datasets\MOT17\train\MOT17-05-FRCNN\gt\gt.txt'
path=r'C:\Users\lzt\Desktop\CV\MOTChallenge\datasets\MOT17\train\MOT17-05-FRCNN\det\det.txt'
def evaluate(pathgt,ts_file):
    metrics = list(mm.metrics.motchallenge_metrics)
    gt=mm.io.loadtxt(pathgt, fmt="mot16", min_confidence=1)
    ts=mm.io.loadtxt(ts_file, fmt="mot16")
    name=os.path.splitext(os.path.basename(ts_file))[0]
    acc=mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=metrics, name=name)
    print(mm.io.render_summary(summary, formatters=mh.formatters,namemap=mm.io.motchallenge_metric_names))

def tlwh_to_tlbr(tlwh):
    ret = np.asarray(tlwh).copy()
    ret[2:] += ret[:2]
    return ret

def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
def write_results_no_score(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1))
                f.write(line)
    
class arg:
    def __init__(self):
        self.track_thresh = 0.65
        self.track_buffer = 14
        self.match_thresh = 0.71
        self.mot20=False
        self.min_box_area=0



max_frame=832

arg=arg()

img_size=[]
results = []
img_size.append(1920)
img_size.append(1080)
tracker = BYTETracker(arg)

for k in range(1,max_frame+1):
    f= open(pathgt, "r")
    dets=np.zeros(shape=(0,5),dtype=np.float64)
    cnt=0
    for line in f:
        linedata=line.split(',')
        if int(linedata[0])==k:  
            ins=np.array(linedata[2:6],dtype=np.float64).T
            ins=tlwh_to_tlbr(ins)
            ins=np.append(ins,float(linedata[7]))
            dets=np.insert(dets,cnt,ins,axis=0)
            cnt+=1
    f.close()
    online_targets = tracker.update(dets, img_size, img_size)   
# print("tracking output ",k)
    online_tlwhs = []
    online_ids = []
    online_scores = []

    for t in online_targets:
        tlwh = t.tlwh
        tid = t.track_id
        vertical = tlwh[2] / tlwh[3] > 1.6
        if tlwh[2] * tlwh[3] > arg.min_box_area and not vertical:
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_scores.append(t.score)
    results.append((k, online_tlwhs, online_ids, online_scores))

write_results('result_test1.txt', results)
evaluate(pathgt,'result_test1.txt')
evaluate(pathgt,pathgt)