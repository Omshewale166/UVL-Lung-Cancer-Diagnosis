# Research paper implementation mapping

This document records how the software maps to the attached paper and where engineering choices were needed.

## Problem, objectives, and scope

The paper targets slow, variable manual interpretation of lung CT images. Its stated objectives are (1) classify slices as Normal, Benign, or Malignant with a custom CNN, (2) automate report generation, (3) explore vision-language fusion, and (4) improve scalability and efficiency. The current executable methodology specifies the CNN and image-processing pipeline. GPT-based report generation, feature-to-language fusion, and language-model fine-tuning are described as a future stage without model weights, corpus, implementation details, or evaluation results. The software fully implements the specified classifier workflow and reports only classifier results; it does not claim that future stage has been completed.

## End-to-end pipeline

1. Read a PNG/JPEG image or one single-frame DICOM image.
2. Convert color data from OpenCV BGR to RGB.
3. Resize to 256 × 256.
4. Convert RGB to grayscale.
5. Apply a 5 × 5 Gaussian blur.
6. Apply binary Otsu thresholding.
7. Divide pixel values by 255 to normalize to [0, 1].
8. Erode and then dilate using the documented 3 × 3/one-iteration assumption.
9. Add one channel dimension and pass float32 image tensors to the CNN.
10. Return three softmax probabilities in class order Normal, Benign, Malignant.
11. Format the output as a limited summary that does not add lesion findings.

The source methodology's prose first says resize then grayscale, which is implemented. The architecture table and class count are used as the authoritative CNN geometry.

## Model geometry and parameter equations

For a valid convolution, the output spatial size is `n_out = n_in - k + 1`. A 256 input with 3 × 3 kernels becomes 254, then 2 × 2 stride-2 pooling produces 127. The next convolution becomes 125, then pooling produces 62. Flattening produces `62 × 62 × 64 = 246,016` values.

With a one-channel input, Conv2D parameters are `(k_h × k_w × input_channels + 1 bias) × filters`:

- First convolution: `(3 × 3 × 1 + 1) × 64 = 640`.
- Second convolution: `(3 × 3 × 64 + 1) × 64 = 36,928`.
- Dense hidden layer: `(246,016 + 1) × 16 = 3,936,272`.
- Softmax classifier: `(16 + 1) × 3 = 51`.
- Total: 3,973,891 parameters.

Otsu's threshold `t` is selected to maximize between-class variance. For histogram class probabilities `w_0(t)` and `w_1(t)` and class means `μ_0(t)` and `μ_1(t)`, the criterion is `σ²_between(t) = w_0(t) w_1(t) [μ_0(t) - μ_1(t)]²`. OpenCV's `THRESH_OTSU` computes the threshold.

## Training procedure and evaluation

The paper specifies sparse categorical cross-entropy, Adam, and accuracy. The implementation uses those exact choices. Keras' default Adam learning rate is 0.001. Since the paper leaves out batch size, epochs, split, early stopping, and class imbalance handling, software defaults are batch size 32, maximum 50 epochs, seed 42, stratified 70/15/15 train/validation/test, balanced class weights, and early stopping after 8 validation-loss epochs without improvement. No data augmentation is applied. Patient-group splitting is used when all manifest rows have patient IDs; otherwise, stratification is at image level.

Evaluation provides accuracy, macro F1, per-class precision/recall/F1/support, confusion matrix, and macro one-vs-rest ROC AUC when defined. These metrics exceed the paper's training accuracy metric to expose class-specific behavior. The paper's 90.8% result cannot be independently verified without its exact dataset split and checkpoint.

## Data requirements and input/output contract

Training data must have integer class semantics mapped as Normal=0, Benign=1, Malignant=2. Class-folder datasets use `Normal/`, `Benign/`, and `Malignant/`; manifests use `image_path,label` with optional `patient_id`. A model input is `(256, 256, 1)` float32. Model output is a normalized 3-vector. The HTTP and CLI outputs contain the selected class, per-class probabilities, maximum score, generation time, checkpoint label, report text, and research-use disclaimer.

## Dependencies

TensorFlow/Keras implements the model and training; OpenCV implements the paper's image-processing operations; Pillow decodes common image files; pydicom reads DICOM; NumPy supplies arrays; scikit-learn provides split and evaluation metrics; FastAPI/Uvicorn provide the API; Matplotlib renders plots; pytest and HTTPX verify the application.

## Safety and limitations

The paper describes 2-D slices, so the code does not combine slices into a CT volume or make longitudinal comparisons. It does not infer lesion location, size, morphology, diagnosis text, or treatment advice. DICOM min-max intensity scaling is an assumption because the paper specifies no CT windowing. The generated result must be reviewed by a radiologist and is not validated for clinical use.
