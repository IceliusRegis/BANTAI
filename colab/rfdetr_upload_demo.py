#@title 8A. Upload road photos - validation-selected model
# Paste this entire cell after section 7 and before section 9 in the running Colab notebook.
# Section 8 may remain disabled; this demo never evaluates the reserved dataset.
# It requires the corrected section 2 helpers and the existing notebook configuration.
from pathlib import Path
import gc
import hashlib
import io
from google.colab import files
from IPython.display import display
from PIL import Image, ImageDraw, ImageOps

# Read the recorded validation selection, without requiring final-test results.
validation_path = OUT / 'validation_comparison.json'
if not validation_path.exists() and (BACKUP / 'reports/validation_comparison.json').exists():
    atomic_copy(BACKUP / 'reports/validation_comparison.json', validation_path)
assert validation_path.exists(), 'Section 7 validation comparison is missing.'
validation_report = read_json(validation_path)
selected_candidate = validation_report['selected_by_validation']
assert selected_candidate in ('baseline', 'fine_tuned'), 'Unknown validation-selected candidate.'
print('Section 8A | Validation-selected candidate:', selected_candidate)

# Explicitly reload the validation-selected checkpoint. Do not reuse section 8's detector.
if selected_candidate == 'fine_tuned':
    demo_weights = TRAIN_DIR / 'checkpoint_best_total.pth'
    expected_sha256 = validation_report['fine_tuned']['model_sha256']
else:
    demo_weights = Path(WEIGHTS)
    expected_sha256 = validation_report['baseline']['model_sha256']
assert demo_weights.exists(), 'Selected checkpoint is missing. Restore it before running the demo.'
assert sha256(demo_weights) == expected_sha256, 'Selected checkpoint hash differs from validation.'
for model_variable in ('detector', 'demo_detector'):
    if model_variable in globals():
        del globals()[model_variable]
gc.collect()
if DEVICE == 'cuda':
    torch.cuda.empty_cache()
demo_detector = load_detector(demo_weights)
print('Demo model:', selected_candidate, '| confidence:', CONFIDENCE)
print('Exploratory demonstration only; these images do not change evaluation scores or select the model.')

uploaded = files.upload()
assert uploaded, 'No image uploaded. Rerun this cell to choose a JPG, PNG, or WebP image.'
demo_dir = OUT / 'upload_demo'
demo_dir.mkdir(parents=True, exist_ok=True)
colors = ['#00bfff', '#ffb000', '#d65dff', '#ff4242']
for filename, image_bytes in uploaded.items():
    try:
        with Image.open(io.BytesIO(image_bytes)) as source_image:
            road_image = ImageOps.exif_transpose(source_image).convert('RGB')
    except (OSError, ValueError):
        print('Skipped unreadable image:', filename)
        continue
    detections = demo_detector.predict(road_image, threshold=CONFIDENCE)
    predictions, background_filtered = foreground_predictions(detections, {'file_name': filename, 'id': 'upload-demo'})
    annotated = road_image.copy()
    draw = ImageDraw.Draw(annotated)
    line_width = max(2, round(min(annotated.size) / 300))
    for prediction in predictions:
        category = prediction['category_id']
        x1, y1, x2, y2 = prediction['xyxy']
        label = f"{CLASSES[category].replace('_', ' ')} {prediction['score']:.1%}"
        color = colors[category]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        text_position = (max(0, x1), max(0, y1 - 14))
        text_box = draw.textbbox(text_position, label)
        draw.rectangle(text_box, fill=color)
        draw.text(text_position, label, fill='black')
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    image_path = demo_dir / (image_hash[:16] + '_detections.jpg')
    annotated.save(image_path)
    write_json(image_path.with_suffix('.json'), {
        'source_filename': filename, 'source_sha256': image_hash,
        'selected_by_validation': selected_candidate, 'checkpoint_sha256': expected_sha256,
        'confidence': CONFIDENCE, 'predictions': predictions,
        'background_filtered': background_filtered, 'purpose': 'exploratory_upload_demo'})
    print(filename, '|', len(predictions), 'road-damage detections')
    if not predictions:
        print('No road damage detected at this confidence; this does not prove the road is defect-free.')
        low_detections = demo_detector.predict(road_image, threshold=0.01)
        low_preds, _ = foreground_predictions(low_detections, {'file_name': filename, 'id': 'upload-demo-low'})
        if low_preds:
            top = max(low_preds, key=lambda p: p['score'])
            print(f"  Top low-confidence candidate: {CLASSES[top['category_id']]} @ {top['score']:.1%}")
    for prediction in predictions:
        print(' ', CLASSES[prediction['category_id']], f"{prediction['score']:.1%}")
    display(ImageOps.contain(annotated, (1400, 1000)))
backup_results(demo_dir, BACKUP / 'reports/upload_demo')
print('Demo saved. Run section 9 next to export the results ZIP, including these examples.')
