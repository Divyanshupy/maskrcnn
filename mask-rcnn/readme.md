**Mask-RCNN**

**Instance segmentation**

**vs.**

**Semantic segmentation**

![mask_rcnn_segmentation_types](https://user-images.githubusercontent.com/36807028/53910869-f315f100-407a-11e9-8e1b-e894c9262603.jpg)

**Explaining the differences between traditional image classification, object
detection, semantic segmentation, and instance segmentation is best done
visually.**

**When performing traditional image classification our goal is to predict a set
of labels to characterize the contents of an input image (top-left).**

**Object detection builds on image classification, but this time allows us to
localize each object in an image. The image is now characterized by:**

1.  **Bounding box (x, y)-coordinates for each object**

2.  **An associated class label for each bounding box**

**An example of semantic segmentation can be seen in bottom-left. Semantic
segmentation algorithms require us to associate every pixel in an input image
with a class label (including a class label for the background).**

**Pay close attention to our semantic segmentation visualization — notice how
each object is indeed segmented but each “cube” object has the same color.**

**While semantic segmentation algorithms are capable of labeling every object in
an image they cannot differentiate between two objects of the same class.**

**This behavior is especially problematic if two objects of the same class are
partially occluding each other — we have no idea where the boundaries of one
object ends and the next one begins, as demonstrated by the two purple cubes, we
cannot tell where one cube starts and the other ends.**

**Instance segmentation algorithms, on the other hand, compute a pixel-wise mask
for every object in the image, even if the objects are of the same class label
(bottom-right). Here you can see that each of the cubes has their own unique
color, implying that our instance segmentation algorithm not only localized each
individual cube but predicted their boundaries as well.**

**The Mask R-CNN architecture we’ll be discussing in this tutorial is an example
of an instance segmentation algorithm.**

### What is Mask R-CNN?

**The Mask R-CNN algorithm was introduced by He et al. in their 2017
paper, **[Mask R-CNN](https://arxiv.org/abs/1703.06870)**.**

**Mask R-CNN builds on the previous object detection work
of **[R-CNN](https://arxiv.org/abs/1311.2524)** (2013), **[Fast
R-CNN](https://arxiv.org/abs/1504.08083)** (2015), and **[Faster
R-CNN](https://arxiv.org/abs/1506.01497)** (2015), all by Girshick et al.**

**In order to understand Mask R-CNN let’s briefly review the R-CNN variants,
starting with the original R-CNN:**

**![mask_rcnn_rcn_orig](https://user-images.githubusercontent.com/36807028/53911046-6586d100-407b-11e9-9470-95ea9b672180.jpg)**

**The original R-CNN algorithm is a four-step process:**

-   **Step \#1: Input an image to the network.**

-   **Step \#2: Extract region proposals (i.e., regions of an image that
    potentially contain objects) using an algorithm such as **[Selective
    Search](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf)**.**

-   **Step \#3: Use transfer learning, specifically feature extraction, to
    compute features for each proposal (which is effectively an ROI) using the
    pre-trained CNN.**

-   **Step \#4: Classify each proposal using the extracted features with a
    Support Vector Machine (SVM).**

**The reason this method works is due to the robust, discriminative features
learned by the CNN.**

**However, the problem with the R-CNN method is it’s incredibly slow. And
furthermore, we’re not actually learning to localize via a deep neural network,
we’re effectively just building a more advanced **[HOG + Linear SVM
detector](https://www.pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/)**.**

**To improve upon the original R-CNN, Girshick et al. published the Fast
R-CNN algorithm:**

**![mask_rcnn_fast_rcnn](https://user-images.githubusercontent.com/36807028/53911185-bbf40f80-407b-11e9-8659-0e5a263b873e.jpg)**

**Similar to the original R-CNN, Fast R-CNN still utilizes Selective Search to
obtain region proposals; however, the novel contribution from the paper
was Region of Interest (ROI) Pooling module.**

**ROI Pooling works by extracting a fixed-size window from the feature map and
using these features to obtain the final class label and bounding box. The
primary benefit here is that the network is now, effectively, end-to-end
trainable:**

1.  **We input an image and associated ground-truth bounding boxes**

2.  **Extract the feature map**

3.  **Apply ROI pooling and obtain the ROI feature vector**

4.  **And finally, use the two sets of fully-connected layers to obtain (1) the
    class label predictions and (2) the bounding box locations for each
    proposal.**

**While the network is now end-to-end trainable, performance suffered
dramatically at inference (i.e., prediction) by being dependent on Selective
Search.**

**To make the R-CNN architecture even faster we need to incorporate the region
proposal directly into the R-CNN:**

**![mask_rcnns_faster_rcnn](https://user-images.githubusercontent.com/36807028/53911240-e5ad3680-407b-11e9-9cf7-27b92dd24811.jpg)**

**The Faster R-CNN paper by Girshick et al. introduced the Region Proposal
Network (RPN)that bakes region proposal directly into the architecture,
alleviating the need for the Selective Search algorithm.**

**As a whole, the Faster R-CNN architecture is capable of running at
approximately 7-10 FPS, a huge step towards making real-time object detection
with deep learning a reality.**

**The Mask R-CNN algorithm builds on the Faster R-CNN architecture with two
major contributions:**

1.  **Replacing the ROI Pooling module with a more accurate ROI Align module**

2.  **Inserting an additional branch out of the ROI Align module**

**This additional branch accepts the output of the ROI Align and then feeds it
into two CONV layers.**

**The output of the CONV layers is the mask itself.**

**We can visualize the Mask R-CNN architecture in the following figure:**

**![mask_rcnn_arch](https://user-images.githubusercontent.com/36807028/53911302-0d040380-407c-11e9-8894-0581b49014a7.png)**

**Figure 5: The Mask R-CNN work by He et al. replaces the ROI Polling module
with a more accurate ROI Align module. The output of the ROI module is then fed
into two CONV layers. The output of the CONV layers is the mask itself.**

**Notice the branch of two CONV layers coming out of the ROI Align module — this
is where our mask is actually generated.**

**As we know, the Faster R-CNN/Mask R-CNN architectures leverage a Region
Proposal Network (RPN) to generate regions of an image that potentially contain
an object.**

**Each of these regions is ranked based on their “objectness score” (i.e., how
likely it is that a given region could potentially contain an object) and then
the top N most confident objectness regions are kept.**

**In the original Faster R-CNN publication Girshick et al. set N=2,000, but in
practice, we can get away with a much smaller N, such as N={10, 100, 200,
300} and still obtain good results.**

**He et al. set N=300 in **[their
publication](https://arxiv.org/abs/1703.06870)** which is the value we’ll use
here as well.**

**Each of the 300 selected ROIs go through three parallel branches of the
network:**

1.  **Label prediction**

2.  **Bounding box prediction**

3.  **Mask prediction**

**Figure 5 above above visualizes these branches.**

**During prediction, each of the 300 ROIs go through **[non-maxima
suppression](https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/)** and
the top 100 detection boxes are kept, resulting in a 4D tensor of 100 x L x 15 x
15 where L is the number of class labels in the dataset and 15 x 15 is the size
of each of the L masks.**

**The Mask R-CNN we’re using here today was trained on the **[COCO
dataset](http://cocodataset.org/#home)**, which has L=90classes, thus the
resulting volume size from the mask module of the Mask R CNN is 100 x 90 x 15 x
15.**

**To visualize the Mask R-CNN process take a look at the figure below:**

**![mask_rcnn_mask_resizing](https://user-images.githubusercontent.com/36807028/53911395-43418300-407c-11e9-86eb-edd275d7894f.jpg)**

**Here you can see that we start with our input image and feed it through our
Mask R-CNN network to obtain our mask prediction.**

**The predicted mask is only 15 x 15 pixels so we resize the mask back to the
original input image dimensions.**

**Finally, the resized mask can be overlaid on the original input image. For a
more thorough discussion on how Mask R-CNN works be sure to refer to:**

1.  **The original **[Mask
    R-CNN](https://arxiv.org/abs/1703.06870)** publication by He et al.**

2.  **Adrian Rosebrock and his best selling book Deep Learning For Computer
    Vision.**
