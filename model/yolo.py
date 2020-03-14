# Copyright 2020 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Any, List, Tuple, Optional
import numpy as np
import tensorflow as tf
from model.base import KerasObjectDetectionBase


class YoloConvBlock(tf.keras.layers.Layer):
    """Yolo convolution block implementaton."""

    def __init__(
            self,
            filters: int,
            kernel: int,
            stride: int,
            weight_decay: float,
            activation_cls: Optional[Any] = tf.keras.layers.LeakyReLU) -> None:
        """Initialize yolo convolution block.

        Args:
            filters (int): convolution output filter size.
            kernel (int): convolution kernel size.
            stride (int): convolution stride size.
            weight_decay (float): weight decay's weight.
            activation_cls (Optional[str]): activation class.

        """
        super(YoloConvBlock, self).__init__()

        self._conv = tf.keras.layers.Conv2D(
                        filters,
                        kernel_size=kernel,
                        strides=stride,
                        padding="same",
                        kernel_initializer="he_normal",
                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                        use_bias=False)
        self._bn = tf.keras.layers.BatchNormalization()
        self._activation = None
        if activation_cls is not None:
            self._activation = activation_cls()

    def call(
            self,
            inputs: tf.Variable) -> tf.Variable:
        """Layer forward process.

        Args:
            inputs (tf.Variable): input array whose shape is (BatchSize, Height, Width, Channel)

        Return:
            outputs (tf.Variable): output array whose shape is (BatchSize, Height, Width, Channel)

        """
        x = self._conv(inputs)
        x = self._bn(x)
        if self._activation is not None:
            x = self._activation(x)
        return x


class YoloBottleneckBlock(tf.keras.layers.Layer):
    """Yolo convolution block implementaton."""

    def __init__(
            self,
            filters: int,
            bottleneck_filters: int,
            weight_decay: float) -> None:
        """Initialize yolo convolution block.

        Args:
            filters (int): convolution output filter size.
            bottleneck_filters (int): bottleneck convolution output filter size.
            weight_decay (float): weight decay's weight.

        """
        super(YoloBottleneckBlock, self).__init__()

        self._conv1 = YoloConvBlock(
                            filters=filters,
                            kernel=3,
                            stride=1,
                            weight_decay=weight_decay)

        self._conv2 = YoloConvBlock(
                            filters=bottleneck_filters,
                            kernel=1,
                            stride=1,
                            weight_decay=weight_decay)

        self._conv3 = YoloConvBlock(
                            filters=filters,
                            kernel=3,
                            stride=1,
                            weight_decay=weight_decay)

    def call(
            self,
            inputs: tf.Variable) -> tf.Variable:
        """Layer forward process.

        Args:
            inputs (tf.Variable): input array whose shape is (BatchSize, Height, Width, Channel)

        Return:
            outputs (tf.Variable): output array whose shape is (BatchSize, Height, Width, Channel)

        """
        x = self._conv1(inputs)
        x = self._conv2(x)
        x = self._conv3(x)
        return x


class YoloV2(KerasObjectDetectionBase):
    """YOLO v2 implementation with darknet-19.

    reference code is here https://github.com/allanzelener/YAD2K.

    Args:
        weight_decay (float): weight decay's weight.
        resize_shape (Tuple[int]): model input shape.
        iou_threshold (float): detection iou threshold.

    """

    def __init__(
            self,
            weight_decay: float = 5e-4,
            resize_shape: Tuple[int, int] = (416, 416),
            iou_threshold: float = 0.6,
            **kwargs: Any) -> None:
        """Intialize parameter and build model."""
        super(YoloV2, self).__init__(**kwargs)

        self.resize_shape = resize_shape
        self.iou_threshold = iou_threshold
        self.anchors = [(0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434), (7.88282, 3.52778), (9.77052, 9.16828)]
        filters = [32, 64, 128, 256, 512, 1024]

        inputs = tf.keras.layers.Input(self.dataset.input_shape)
        x = tf.image.resize(inputs, resize_shape, method=tf.image.ResizeMethod.BILINEAR)

        # darknet body
        for i in range(2):
            x = YoloConvBlock(
                    filters=filters[i],
                    kernel=3,
                    stride=1,
                    weight_decay=weight_decay)(x)
            x = tf.keras.layers.MaxPool2D()(x)

        for i in range(2, 4):
            x = YoloBottleneckBlock(
                    filters=filters[i],
                    bottleneck_filters=filters[i-1],
                    weight_decay=weight_decay)(x)
            x = tf.keras.layers.MaxPool2D()(x)

        for i in range(4, 6):
            x = YoloBottleneckBlock(
                    filters=filters[i],
                    bottleneck_filters=filters[i-1],
                    weight_decay=weight_decay)(x)
            x = YoloConvBlock(
                    filters=filters[i-1],
                    kernel=1,
                    stride=1,
                    weight_decay=weight_decay)(x)
            x = YoloConvBlock(
                    filters=filters[i],
                    kernel=3,
                    stride=1,
                    weight_decay=weight_decay)(x)

            if i == 4:
                _fine_grained = x
                x = tf.keras.layers.MaxPool2D()(x)

        # normal path
        x = YoloConvBlock(
                filters=filters[-1],
                kernel=3,
                stride=1,
                weight_decay=weight_decay)(x)
        x = YoloConvBlock(
                filters=filters[-1],
                kernel=3,
                stride=1,
                weight_decay=weight_decay)(x)

        # fine grained path
        _fine_grained = YoloConvBlock(
                filters=filters[1],
                kernel=1,
                stride=1,
                weight_decay=weight_decay)(_fine_grained)
        _fine_grained = tf.nn.space_to_depth(_fine_grained, 2)

        x = tf.concat([x, _fine_grained], axis=-1)
        x = YoloConvBlock(
                filters=filters[-1],
                kernel=3,
                stride=1,
                weight_decay=weight_decay)(x)
        outputs = tf.keras.layers.Conv2D(
                filters=len(self.anchors) * (self.dataset.category_nums + 5),
                kernel_size=1,
                strides=1,
                padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                use_bias=True)(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.setup()

    @tf.function
    def _iou(
            self,
            box1: np.array,
            box2: np.array,
            mode: str = 'np') -> np.array:
        """Calculate IoUs between box1 and box2.

        Args:
            box1 (List): array with shape (None, 5) containing center_x, cneter_y, width and height array.
            box2 (List): array with shape (None, 5) containing center_x, cneter_y, width and height array.

        Return:
            iou (List): iou array.

        """
        if mode == 'np':
            xp = np
            concat = np.concatenate
        elif mode == 'tf':
            xp = tf
            concat = tf.concat

        x, y, w, h = box1
        _x, _y, _w, _h = box2

        left_top = concat([xp.expand_dims(x - w / 2., axis=-1), xp.expand_dims(y - h / 2., axis=-1)], axis=-1)
        right_under = concat([xp.expand_dims(x + w / 2., axis=-1), xp.expand_dims(y + h / 2., axis=-1)], axis=-1)
        _left_top = concat([xp.expand_dims(_x - _w / 2., axis=-1), xp.expand_dims(_y - _h / 2., axis=-1)], axis=-1)
        _right_under = concat([xp.expand_dims(_x + _w / 2., axis=-1), xp.expand_dims(_y + _h / 2., axis=-1)], axis=-1)

        intersect_left_top = xp.maximum(left_top, _left_top,)
        intersect_right_under = xp.minimum(right_under, _right_under)
        intersect_wh = xp.maximum(intersect_right_under - intersect_left_top, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        box1_areas = w * h
        box2_areas = _w * _h
        iou = intersect_areas / (box1_areas + box2_areas - intersect_areas)
        return iou

    @tf.function
    def _head(
            self,
            outputs: tf.Variable) -> tf.Variable:
        """Convert final layer features to bounding box parameters.

        Args:
            outputs (List): Final convolutional layer features. shape is (B, f_H, f_W, C).

        Return:
            x (List): prediction box center x. shape is (B, f_H, f_W, A, 1). value is in [0, 1].
            y (List): prediction box center y.
            w (List): prediction box width.
            h (List): prediction box height.
            confidence (List): probability estimate for whether each box contains any object.
            class_prob (List): probability distribution estimate for each box over class labels.

        """
        # Create grid index tensor
        f_height, f_width = outputs.shape[1:3]
        conv_height_index = tf.range(0, f_height, dtype=tf.float32)
        conv_width_index = tf.range(0, f_width, dtype=tf.float32)
        conv_height_index = tf.tile(conv_height_index, [f_width])

        conv_width_index = tf.tile(tf.expand_dims(conv_width_index, 0), [f_height, 1])
        conv_width_index = tf.reshape(tf.transpose(conv_width_index), (-1, ))
        conv_index = tf.transpose(tf.stack([conv_height_index, conv_width_index]))
        conv_index = tf.reshape(conv_index, [1, f_height, f_width, 1, 2])
        conv_index = tf.cast(conv_index, outputs.dtype)

        num_anchors = len(self.anchors)
        anchors_tensor = tf.reshape(np.array(self.anchors, dtype=np.float32), [1, 1, 1, num_anchors, 2])
        outputs = tf.reshape(outputs, (-1, f_height, f_width, num_anchors, 5 + self.dataset.category_nums))
        conv_dims = tf.cast(tf.reshape([f_height, f_width], [1, 1, 1, 1, 2]), outputs.dtype)

        box_xy = tf.sigmoid(outputs[..., :2])
        box_wh = tf.exp(outputs[..., 2:4])
        confidence = tf.sigmoid(outputs[..., 4:5])
        class_proba = tf.nn.softmax(outputs[..., 5:])

        # Adjust preditions to each spatial grid point and anchor size.
        # Note: YOLO iterates over height index before width index.
        box_xy = (box_xy + conv_index) / conv_dims
        box_wh = box_wh * anchors_tensor / conv_dims

        return box_xy[..., 0:1], box_xy[..., 1:2], box_wh[..., 0:1], box_wh[..., 1:2], confidence, class_proba

    @tf.function
    def preprocess_gt_boxes(
            self,
            batch_boxes: np.array) -> Tuple[np.array, np.array]:
        """Find detector in YOLO where ground truth box should appear.

        Args:
            batch_boxes (List): List of ground truth boxes in form of relative x, y, w, h, class.
                                Relative coordinates are in the range [0, 1] indicating a percentage of the original image dimensions.

        Return:
            detect_masks (List): 0/1 mask for detectors in [B, conv_height, conv_width, num_anchors, 1]
                                 that should be compared with a matching ground truth box.
            matching_true_boxes (List): Same shape as detect_masks with the corresponding ground truth box
                                        adjusted for comparison with predicted parameters at training time.

        """
        height, width = self.resize_shape
        num_anchors = len(self.anchors)
        anchors = tf.constant(self.anchors, dtype=batch_boxes.dtype)

        # Downsampling factor of 5x 2-stride max_pools == 32.
        assert height % 32 == 0, 'Image sizes in Yolov2 must be multiples of 32.'
        assert width % 32 == 0, 'Image sizes in Yolov2 must be multiples of 32.'
        conv_height = height // 32
        conv_width = width // 32
        batch_size = batch_boxes.shape[0]
        num_box_size = batch_boxes.shape[1]
        num_box_params = batch_boxes.shape[2]
        detect_masks = tf.zeros((batch_size, conv_height, conv_width, num_anchors, 1), dtype=tf.float32)
        matching_true_boxes = tf.zeros((batch_size, conv_height, conv_width, num_anchors, num_box_params), dtype=tf.float32)

        # Calculate base
        batch_box_classes = batch_boxes[..., 4:5]
        batch_boxes = batch_boxes[..., 0:4] * [conv_width, conv_height, conv_width, conv_height]
        i = tf.cast(tf.math.floor(batch_boxes[..., 1:2]), dtype=tf.int32)
        j = tf.cast(tf.math.floor(batch_boxes[..., 0:1]), dtype=tf.int32)

        # Calculate IoU
        box = tf.tile(tf.expand_dims(tf.transpose(batch_boxes, (2, 0, 1)), axis=-1), (1, 1, 1, num_anchors))  # (4, B, None, A)
        batch_anchors = tf.tile(tf.expand_dims(tf.expand_dims(tf.transpose(anchors), axis=1), axis=1), (1, batch_size, num_box_size, 1))
        ious = self._iou(
            tf.split(box, [1] * 4),
            tf.split(tf.concat([box[0:2], batch_anchors], axis=0), [1] * 4),
            mode='tf')[0]
        best_iou = tf.reduce_max(ious, axis=2)
        best_anchor = tf.argmax(ious, axis=2, output_type=tf.int32)

        # Calculate mask
        masks = tf.expand_dims(
            tf.cast((tf.reduce_all(batch_boxes != 0, axis=-1) & (best_iou > 0)), dtype=tf.float32), axis=-1)
        batches = tf.tile(tf.expand_dims(tf.range(batch_size, dtype=tf.int32), axis=1), (1, num_box_size))
        indices = tf.stack([batches, i[..., 0], j[..., 0], best_anchor], axis=-1)
        detect_masks = tf.tensor_scatter_nd_update(detect_masks, indices, (1 * masks))

        best_acnhors = tf.gather_nd(anchors, tf.expand_dims(best_anchor, axis=-1))
        adjusted_box = tf.cast(tf.concat(
            [
                batch_boxes[..., 0:1] - tf.cast(j, dtype=batch_boxes.dtype),
                batch_boxes[..., 1:2] - tf.cast(i, dtype=batch_boxes.dtype),
                tf.math.log(batch_boxes[..., 2:3] / best_acnhors[..., 0:1] + 1e-7),
                tf.math.log(batch_boxes[..., 3:4] / best_acnhors[..., 1:2] + 1e-7),
                batch_box_classes
            ],
            axis=-1), dtype=tf.float32)
        indices = tf.stack([batches, i[..., 0], j[..., 0], best_anchor], axis=-1)
        matching_true_boxes = tf.tensor_scatter_nd_update(matching_true_boxes, indices, (adjusted_box * masks))
        return detect_masks, matching_true_boxes

    @tf.function
    def _loss(
            self,
            label: tf.Variable,
            pred: tf.Variable) -> float:
        """Calculate loss.

        Args:
            label (Tuple[List]): Ground truth boxes tensor with shape (B, num_true_boxes, 5)
                                 containing box relative x_center, y_center, width, height, and class.
            pred (Tuple[List]): last layer output features.

        Return:
            loss (float): los value.

        """
        # For keras setup code
        if len(label.shape) != 3:
            label = tf.constant([[[0.5, 0.5, 0.5, 0.5, 0.5]]], dtype=np.float32)
        else:
            label = tf.ensure_shape(label, (self.dataset.batch_size, self.dataset.max_boxes, 5))

        # Create gt boxes
        detect_masks, matching_true_boxes = self.preprocess_gt_boxes(label)

        # Reshape to corresponding for (B, 1, 1, 1, num_true_boxes, 5) array
        true_boxes = tf.expand_dims(tf.expand_dims(tf.expand_dims(label, axis=1), axis=1), axis=1)
        x, y, w, h, _ = tf.split(true_boxes, [1] * 5, axis=-1)
        pred_x, pred_y, pred_w, pred_h, pred_confidence, pred_class_prob = self._head(pred)
        pred_x = tf.expand_dims(pred_x, 4)
        pred_y = tf.expand_dims(pred_y, 4)
        pred_w = tf.expand_dims(pred_w, 4)
        pred_h = tf.expand_dims(pred_h, 4)

        # calc confidence_loss
        iou_scores = self._iou((x, y, w, h), (pred_x, pred_y, pred_w, pred_h), mode='tf')
        best_ious = tf.math.reduce_max(iou_scores, axis=4)
        pred_detect_masks = tf.cast(best_ious > self.iou_threshold, dtype=best_ious.dtype)
        no_object_loss = (1 - pred_detect_masks) * (1 - detect_masks) * tf.math.square(-pred_confidence)
        objects_loss = detect_masks * tf.math.square(1 - pred_confidence)
        confidence_loss = tf.math.reduce_mean(5 * objects_loss + no_object_loss)

        # calc classification_loss
        matching_classes = tf.cast(matching_true_boxes[..., 4], dtype=tf.int32)
        matching_classes = tf.one_hot(matching_classes, self.dataset.category_nums)
        classification_loss = tf.math.reduce_mean(detect_masks * tf.math.square(matching_classes - pred_class_prob))

        # Calc coordinates_loss
        # x and y are in the range [0, 1] that mean difference ratio to grid left corner.
        # widht and height are in the range (0, ∞) that mean difference log ratio to anchor box.
        matching_boxes = matching_true_boxes[..., 0:4]
        features = tf.reshape(pred, (-1, pred.shape[1], pred.shape[2], len(self.anchors), 5 + self.dataset.category_nums))
        pred_boxes = tf.concat([tf.sigmoid(features[..., 0:2]), features[..., 2:4]], axis=-1)
        coordinates_loss = tf.math.reduce_mean(detect_masks * tf.math.square(matching_boxes - pred_boxes))

        loss = confidence_loss + classification_loss + coordinates_loss
        return loss

    def filter_boxes(self, boxes, box_confidence, box_class_probs, threshold=0.6):
        """Filter YOLO boxes based on object and class confidence."""
        box_scores = box_confidence * box_class_probs
        box_classes = tf.argmax(box_scores, axis=-1)
        box_class_scores = tf.reduce_max(box_scores, axis=-1)
        prediction_mask = box_class_scores >= threshold

        boxes = tf.boolean_mask(boxes, prediction_mask)
        scores = tf.boolean_mask(box_class_scores, prediction_mask)
        classes = tf.boolean_mask(box_classes, prediction_mask)
        return boxes, scores, classes

    def inference(self) -> Tuple[List[Any], List[List[Any]], List[Any]]:
        """Inference model.

        Return:
            target (List[Any]): inference target.
            predicts (List[List[float]]): inference result. shape is data size x category_nums.
            gt (List[Any]): ground truth data.

        """
        (x_test, y_test) = next(self.dataset.eval_data_generator())
        predicts = self.model.predict(x_test)

        suppresed_predicts = []
        for pred in predicts:
            pred_x, pred_y, pred_w, pred_h, pred_confidence, pred_class_prob = self._head(np.array([pred]))
            boxes = tf.reshape(tf.concat([pred_x, pred_y, pred_w, pred_h], axis=-1), (-1, 4))
            pred_confidence = tf.reshape(pred_confidence, (-1, 1))
            pred_class_prob = tf.reshape(pred_class_prob, (-1, self.dataset.category_nums))
            boxes, scores, classes = self.filter_boxes(boxes, pred_confidence, pred_class_prob)
            suppresed_indices = tf.image.non_max_suppression(
                                    boxes,
                                    scores,
                                    max_output_size=5,
                                    iou_threshold=0.5)

            boxes = tf.gather(boxes, suppresed_indices)
            scores = tf.expand_dims(tf.gather(scores, suppresed_indices), axis=-1)
            classes = tf.expand_dims(tf.gather(classes, suppresed_indices), axis=-1)
            preds = tf.concat([boxes, scores, tf.cast(classes, dtype=boxes.dtype)], axis=-1)
            suppresed_predicts.append(preds.numpy())
        return x_test, np.array(suppresed_predicts), y_test
