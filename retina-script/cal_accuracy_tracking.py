import numpy as np
import pandas as pd

ssd_pred = pd.read_csv("retina_track_out.csv")
ssd_real = pd.read_csv("tracking_label.csv")
     
import torch          
def intersect(box_a, box_b):

    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
  
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]
    
    
names = set(ssd_real['name'].tolist())
names =  sorted(names)

track_id_pred = []
track_id_real = []
iou_threshold=0.5
all_=  []
for id,name in enumerate(names):    
    
    #print(name)
    
    pred_label = ssd_pred.loc[ssd_pred['name'] == name].to_numpy()
    real_label = ssd_real.loc[ssd_real['name'] == name].to_numpy()
    
  
    detected_annotations = []
    
    for label in pred_label:
      
        pred_id= label[1]
        

        pred_box = torch.from_numpy(label[2:].reshape(1,4).astype(float))
        real_boxes = torch.from_numpy(real_label[:,2:].astype(float))
        
        
        overlaps = jaccard(pred_box, real_boxes)
        assigned_annotation = np.argmax(overlaps, axis=1)
        max_overlap = overlaps[0, assigned_annotation]
        
        
        
        if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
            detected_annotations.append(assigned_annotation)              
            if(pred_id not in track_id_pred):

                real_id = real_label[assigned_annotation,1]
                
                all_real_id = ssd_real.loc[ssd_real['id'] == real_id]#get all real frame that have this vehicle id
                all_pred_id = ssd_pred.loc[ssd_pred['label'] == pred_id]#get all pred frame that have this vehicle id
                
                #print(all_real_id.shape, real_id,' ', all_pred_id.shape,pred_id)
                frame_names = set(all_real_id['name'].tolist())
                
                
                track_id_pred.append(pred_id)

                all_.append([all_real_id.shape[0],real_id,all_pred_id.shape[0],pred_id])
         
   

#print(all_)
all_df = pd.DataFrame(all_,columns=["all_real_num" ,"real_id" ,"all_pred_num", "pred_id"])

unique_real = set(all_df['real_id'].tolist())
all_correct = 0
for id in unique_real:

	rows_by_id = all_df.loc[all_df['real_id'] == id]
	all_correct += rows_by_id.all_pred_num.max()

      
print("acc: ")
print(all_correct/ssd_real.shape[0])

