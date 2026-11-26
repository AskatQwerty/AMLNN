# Partial fine-tuning evaluation on 100 validation images

The complete original joint model at epoch0 and the latest complete last.pt were evaluated on identical100validationIDs from the resolution pilot (seed42). Both CNN states and the caption decoder were loaded from each checkpoint. Latestcheckpoint:epoch1,22,688trainingimages,709successfuloptimizerupdates.

Generation:beam5,length penalty0,max33,batch1,original224/800resolution. Loss uses the same fixed first human caption, at most32contenttokens, and averages over all1,226nonpaddingtargettokens. Caption metrics use all raw human references and the original project tokenizer/pycocoevalcap1.2implementation.

CIDEr-D decreased from0.9980559to0.9873869. BLEU-4 increased from0.3144995to0.3212847. Validation loss changed from2.0373489to2.0380197. This is mixed evidence after only part of the first epoch, not a full-validation conclusion.

No training or optimizer update was performed. Existing checkpoints, optimizer state and training history were preserved. New metrics must not be inserted as a completed epoch in the original training history.
