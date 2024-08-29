from pycocotools.cocoeval import COCOeval
import numpy as np


class COCOevalSIoU(COCOeval):
    def __init__(self, cocoGt=None, cocoDt=None, iouType='bbox', gamma=0.5, kappa=8):
        super().__init__(cocoGt, cocoDt, iouType)
        self.gamma = gamma
        self.kappa = kappa

    @staticmethod
    def calculate_siou(b1, b2, gamma, kappa):
        """
        Calculate the Scaled Intersection over Union (SIoU) between two bounding boxes.

        Parameters:
        -----------
        b1 : list of float
            The first bounding box in the format [x, y, w, h], where (x, y) is the top-left corner.
        b2 : list of float
            The second bounding box in the format [x, y, w, h], where (x, y) is the top-left corner.
        gamma : float, optional
            Scaling factor that controls the impact of the size of the bounding boxes. Default is 0.5.
        kappa : float, optional
            Parameter related to the scale of the bounding boxes. Default is 1.0.

        Returns:
        --------
        float
            The Scaled Intersection over Union (SIoU) between the two bounding boxes.
        """
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2

        x1_max, y1_max = x1 + w1, y1 + h1
        x2_max, y2_max = x2 + w2, y2 + h2

        inter_xmin = max(x1, x2)
        inter_ymin = max(y1, y2)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        inter_width = max(0, inter_xmax - inter_xmin)
        inter_height = max(0, inter_ymax - inter_ymin)
        inter_area = inter_width * inter_height

        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        iou = inter_area / union_area if union_area > 0 else 0

        size_factor = np.sqrt(w1 * h1 + w2 * h2)
        p = 1 - gamma * np.exp(-size_factor / np.sqrt(2 * kappa))

        siou = iou ** p

        return siou


    def computeIoU(self, imgId, catId):
        """
        Compute the SIoU between ground truth and detection bounding boxes.
        This method overrides the default IoU computation in COCOeval.
        """
        p = self.params
        gt = self._gts[imgId, catId]
        dt = self._dts[imgId, catId]
        if len(gt) == 0 or len(dt) == 0:
            return []

        # Sort detections by score
        dt = sorted(dt, key=lambda x: -x['score'])
        if len(dt) > p.maxDets[-1]:
            dt = dt[0:p.maxDets[-1]]

        # Prepare list to store IoUs
        ious = np.zeros((len(dt), len(gt)))

        for i, d in enumerate(dt):
            dbox = d['bbox']  # [x, y, w, h]
            for j, g in enumerate(gt):
                gbox = g['bbox']  # [x, y, w, h]
                ious[i, j] = self.calculate_siou(dbox, gbox, self.gamma, self.kappa)

        return ious