# Paste as one code cell under Section 8A, before Section 9. No retraining.
from pathlib import Path
import gc
import hashlib
import io
from google.colab import files
from IPython.display import display
from PIL import Image, ImageDraw, ImageOps

# Reuse the original uploaded image bytes when available, not the screenshot/overlay.
diagnostic_uploads = globals().get('uploaded')
if not diagnostic_uploads:
    diagnostic_uploads = files.upload()
assert diagnostic_uploads, 'Upload the same original road photo used in the missed detection.'
validation_file = OUT / 'validation_comparison.json'
if not validation_file.exists() and (BACKUP / 'reports/validation_comparison.json').exists():
    atomic_copy(BACKUP / 'reports/validation_comparison.json', validation_file)
assert validation_file.exists(), 'Restore the Section 7 validation comparison first.'
diagnostic_validation = read_json(validation_file)
diagnostic_pairs = [('baseline', Path(WEIGHTS)),
                    ('fine_tuned', TRAIN_DIR / 'checkpoint_best_total.pth')]
for model_name, path in diagnostic_pairs:
    assert path.exists(), f'Missing {model_name} checkpoint: restore it before comparing.'
    assert model_name in diagnostic_validation, f'Missing {model_name} validation report.'
    assert sha256(path) == diagnostic_validation[model_name]['model_sha256'], 'Checkpoint differs from evaluated weights.'

print('DIAGNOSTIC ONLY: evaluation confidence and final-test scores remain unchanged.')
print('Low-score boxes are uncertain hypotheses, not confirmed road defects.')
diagnostic_dir = OUT / 'diagnostic_demo'
diagnostic_dir.mkdir(parents=True, exist_ok=True)
diagnostic_records = {}
diagnostic_colors = ['#00bfff', '#ffb000', '#d65dff', '#ff4242']

for model_name, path in diagnostic_pairs:
    for variable in ('detector', 'demo_detector', 'diagnostic_detector'):
        if variable in globals():
            del globals()[variable]
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    diagnostic_detector = load_detector(path)
    for filename, raw_image in diagnostic_uploads.items():
        try:
            with Image.open(io.BytesIO(raw_image)) as source:
                original_image = ImageOps.exif_transpose(source).convert('RGB')
        except (OSError, ValueError):
            print('Skipped unreadable image:', filename)
            continue
        detection = diagnostic_detector.predict(original_image, threshold=0.001)
        predictions, filtered = foreground_predictions(detection, {'id': 'diagnostic', 'file_name': filename})
        predictions = sorted(predictions, key=lambda item: item['score'], reverse=True)
        image_hash = hashlib.sha256(raw_image).hexdigest()
        diagnostic_records.setdefault(image_hash, {'filename': filename, 'models': {}})
        diagnostic_records[image_hash]['models'][model_name] = {
            'checkpoint_sha256': sha256(path), 'predictions_above_0_001': predictions,
            'background_filtered': filtered}
        print('\n', model_name.upper(), '|', filename)
        for category, class_name in enumerate(CLASSES):
            ranked = [p for p in predictions if p['category_id'] == category]
            maximum = f"{ranked[0]['score']:.2%}" if ranked else 'none at or above 0.001'
            print(class_name, '| strongest score:', maximum,
                  '| boxes at 0.25:', sum(p['score'] >= .25 for p in ranked))

        # Same photo at three display thresholds; max five boxes per panel avoids clutter.
        sheet = Image.new('RGB', (1500, 470), 'white')
        for column, display_threshold in enumerate((0.25, 0.10, 0.05)):
            annotated = original_image.copy()
            draw = ImageDraw.Draw(annotated)
            shown = [p for p in predictions if p['score'] >= display_threshold][:5]
            for p in shown:
                label = f"{CLASSES[p['category_id']]} {p['score']:.1%}"
                color = diagnostic_colors[p['category_id']]
                draw.rectangle(p['xyxy'], outline=color, width=max(2, min(annotated.size)//250))
                position = (max(0, p['xyxy'][0]), max(0, p['xyxy'][1]-14))
                draw.rectangle(draw.textbbox(position, label), fill=color)
                draw.text(position, label, fill='black')
            thumbnail = ImageOps.contain(annotated, (490, 420))
            sheet.paste(thumbnail, (column*500, 40))
            ImageDraw.Draw(sheet).text((column*500+5, 10),
                f'{model_name}: score >= {display_threshold:.2f} (top 5)', fill='black')
        sheet.save(diagnostic_dir / f'{image_hash[:16]}_{model_name}.jpg')
        display(sheet)

write_json(diagnostic_dir / 'comparison.json', {
    'purpose': 'exploratory missed-detection investigation; not evaluation or threshold selection',
    'inference_floor': .001, 'display_thresholds': [.25, .10, .05],
    'images': diagnostic_records})
backup_results(diagnostic_dir, BACKUP / 'reports/diagnostic_demo')
print('Saved diagnostic comparison. Run Section 9 and send the ZIP plus these panels here.')
