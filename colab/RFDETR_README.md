# RF-DETR Nano baseline and optional fine-tuning

Upload `BANTAI_RFDETR_Nano_Baseline_Finetuning.ipynb` to Google Colab. The notebook
is self-contained: no helper upload or repository clone is needed. Older notebooks
are unchanged. There is no app integration in this deliverable.

1. Select a T4 GPU and run sections 1–5 for setup, downloads and a validation baseline.
2. Set `RUN_FINE_TUNING=True` in section 6 to train for at most 10 epochs, with early
   stopping. GPU is required for this optional stage.
3. Section 7 compares validation results. Finish development before setting
   `RUN_FINAL_TEST=True` in section 8 for paired final evaluation.
4. Section 8A runs after section 7 and explicitly loads the validation-selected
   weights for an uploaded road-photo demo, with boxes, class names and confidence
   scores. Section 8 can remain disabled; no final-test reports are required.
5. Section 9 exports the report ZIP, including demo images. Send it back for interpretation.

For an already-running Colab notebook, paste `rfdetr_upload_demo.py` as one new code
cell after section 7 and before section 9. Replace the old 8A cell; do not restart or
rerun training. The local
workflow includes the confirmed RF-DETR background fix: only ID 4 explicitly labeled
`__background__` is excluded; other unknown IDs still stop evaluation. Compatible
prediction cache signatures are unchanged.

For an unexpected missed photo, paste `rfdetr_diagnostic_demo.py` under section 8A.
It reuses uploaded image bytes, compares the original and fine-tuned checkpoints,
and shows uncertain predictions at several display thresholds. It leaves evaluation
settings, model selection and test reports unchanged. Run section 9 afterward to
include diagnostic output in the results ZIP. Lower display thresholds are not a
validated deployment configuration.

The initial checkpoint is `dronefreak/rdd2022-rfdetr-nano`, pinned to commit
`66429f39b916e56c6a33243d776718088f5f43ad`. The matching publisher dataset export is
`dronefreak/RDD2022`, pinned to `72b01f94c2e5bfe430861afd73995f2d412b8855`.
The notebook downloads small metadata files and 950 selected India images (600
train, 150 validation, 200 test) to `/content`. Samples preserve publisher split
assignments and contain half pothole images per split. They are diagnostic samples,
not representative estimates of everyday road conditions. All four class labels
are retained. Exact duplicates in the sample cause an explicit error.

Reports and per-image predictions are backed up to Drive. Training writes its
checkpoints directly to the Drive run directory, including `last.ckpt` after each
completed epoch. Resume by keeping RUN_NAME/settings unchanged and rerunning setup,
data preparation and the enabled training section. An incomplete epoch repeats.
Drive space and reliability are required. During an interrupted final test, skip
training and resume section 8 after setup/validation comparison.

Sections have named troubleshooting slots. Add a temporary repair cell under the
affected section; rerun that section and dependents. After verification, incorporate
the fix into the original section and clear the repair cell. Change RUN_NAME when
changing experimental settings. Never tune on the final test results.

The checkpoint was already trained on RDD2022: arbitrary resplitting would leak
upstream training images into evaluation. Publisher split correspondence is based
on their documented provenance, not independently proven. Route independence and
near duplicates remain limitations; local camera evaluation is still necessary.
Adapting to a subset of already-seen training data may fail to improve performance.

Dependencies are installed only when the user runs section 2 in Colab:
`rfdetr[train]==1.11.2`, `huggingface-hub==2.3.0`, `pycocotools==2.0.11`.
Colab PyTorch and transitive dependencies are recorded with `pip freeze`; they are
not a cross-platform lockfile. No new local dependencies are needed to rebuild/test.

For maintenance, edit `rfdetr_workflow.py` for data/scoring helpers or
`build_rfdetr_notebook.py` for notebook sections, then run:

```powershell
python -B colab/build_rfdetr_notebook.py
python -B colab/test_rfdetr_workflow.py
```

Offline tests cover syntax, deterministic selection, COCO label conversion,
split preservation, duplicate rejection, fixed-threshold matching and immutable
run metadata. They do not establish checkpoint compatibility, GPU performance or
model accuracy; those require a real Colab run. No training result is bundled.
