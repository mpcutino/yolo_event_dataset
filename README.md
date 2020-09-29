### **Pipeline of work**

In *bag_related.py* file there is the code that reads images from a ros bag, 
writes them in disk and get the bounding boxes according to YOLO detection. 
The output from YOLO should be reviewed. If there are some missing detections,
then use https://www.makesense.ai/ for manually put the bounding boxes. 
To check the missing detections I use *check_missing_annot* from *util.py* 
file. If we use this tool, then to update the dataframe with this new data we
must use *update_yolo_data* in *manual/update.py*.

I also build an image labeling tool (instead of using this website). It is not 
the best image labeling tool, but it fits well for the purposes of this work. 
I can remove specific tags in image, make new bounding boxes and remove all 
appearance of some tag from all the dataframe. This tool save results in a 
new dataframe named *edited_annotate.txt*.

