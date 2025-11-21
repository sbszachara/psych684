import sys
import os
import numpy as np
import pandas as pd
import torch
# from torch.utils.data import Dataset
from datasets import Dataset, Audio
import torchaudio
from peft import LoraConfig, PeftModel
# from huggingface_hub import login
import evaluate
import argparse

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
)

# >>> GRAFN import <<<
from grafn import GRAFNNormalizer

hf_token = 'INSERT_HF_TOKEN'
model_id = "openai/whisper-tiny"
wer_metric = evaluate.load("wer")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------
# Helper: normalize CommonVoice gender labels for GRAFN
# ---------------------------------------------------------
def normalize_gender_label(g):
    """
    Map dataset gender strings to { 'male', 'female' } for GRAFN.

    Examples that become 'male':
        'male', 'male_masculine', 'masculine', 'M', etc.

    Examples that become 'female':
        'female', 'female_feminine', 'feminine', 'F', etc.

    Returns None if gender is unknown/other.
    """
    if g is None:
        return None
    g_low = str(g).strip().lower()
    if g_low.startswith("male") or g_low.startswith("masc"):
        return "male"
    if g_low.startswith("female") or g_low.startswith("fem"):
        return "female"
    return None


# Custom Data Collator for ASR Data. Based on transformers DataCollatorSpeechSeq2SeqWithPadding
class CustomDataCollator:
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, features):
        # Pad input features (spectrograms)
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad labels (tokens)
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore when computing the loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Prevent loss being calculated on the starting token
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    


# evaluates wer based on given group (gender, age, accent):
def group_wer(group, dataset, model, processor, batch_size=8, subgroup=[]):
    groups = {}
    
    # get all the categories
    set_groups = set(dataset[group])
    
    for currGroup in set_groups:
        # gets subset of data that is in that group
        subset = dataset.filter(lambda x: x[group] == currGroup)
        if len(subset) == 0:
            continue # shouldnt be empty but just in case
        
        # need to run predictions
        preds = []
        refs  = []

        # loop thru subset by batch size
        for i in range(0, len(subset), batch_size):
            batch = subset[i : i + batch_size]

            input_features = torch.tensor(batch["input_features"]).to(model.device)
            
            with torch.no_grad():
                predicted_ids = model.generate(input_features)
            
            pred_str = processor.batch_decode(predicted_ids, skip_special_tokens=True)

            label_ids = []

            # should return the labels we need
            for seq in batch["labels"]:
                label_ids.append([
                    processor.tokenizer.pad_token_id if t == -100 else t
                    for t in seq
                ])

            # decodes
            label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

            preds.extend(pred_str)
            refs.extend(label_str)
        # get WER of this particular group
        wer = wer_metric.compute(predictions=preds, references=refs)
        groups[currGroup] = wer
    
    return groups
    

# also need a custom class for the training callback for WER parity.
class WERParityCallback(TrainerCallback):
    def __init__(self, eval_dataset, model, processor, group, subgroups = []):
        self.eval_dataset = eval_dataset
        self.model = model
        self.processor = processor
        self.group   = group
        self.subgroups = subgroups

    # thiis is returned in the seq2seq trainer object as a custom eval metric
    def on_evaluate(self, args, state, control, **kwargs):
        
        results = group_wer(
            self.group,
            self.eval_dataset,
            self.model,
            self.processor,
        )
        # prints WER for each category
        for category, wer in results.items():
            print(f"  {category}: {wer:.4f}")

        # helps look at gender parity
        if self.group == 'gender':
            # print WER parity between male and female
            print("Gender WER parity:")
            try:
                print(results.get('male_masculine') - results.get('female_feminine'))
            except:
                print("Missing WER metric for this entry")
        
        


# Returns predicted token IDs as a PyTorch tensor, allowing Trainer's internal utilities to handle final conversion to numpy.
def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
        
    # Apply argmax to get the predicted token IDs
    pred_ids = torch.argmax(logits, dim=-1)
    
    # Move the tensor to CPU memory if it's on a device (DML/CUDA/GPU). Trainer will then handle the final conversion to NumPy inside its loop.
    pred_ids = pred_ids.cpu()
    
    # Returns a tuple of predicted IDs, original labels
    return pred_ids, labels


def load_whisper():
    # processor = WhisperProcessor.from_pretrained(model_id)

    # For AMD: Load model on CPU/default first, then manually move to DML
    # For NVIDIA: Replace with CUDA implementation
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id
    )
    model.to(device)
    return model




if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="path to MVC folder containing the 'en' folder")
    args = ap.parse_args()

    mozillaPath = f'{args.path}/en/validated.tsv'
    audioPath   = f'{args.path}/en/clips'
    cvDelta = pd.read_csv(mozillaPath, sep='\t')


    # print(cvDelta.head)
    # print(cvDelta.columns)
    # print(len(cvDelta))
    cvDelta = cvDelta.dropna(subset=['age', 'gender', 'accents'])
    # print(len(cvDelta))

    print("Demographic info:")
    print(cvDelta['gender'].value_counts())
    print((cvDelta['gender'].value_counts(normalize=True) * 100).round(2))


    cvDelta['path'] = cvDelta['path'].apply(lambda x: os.path.join(audioPath, x))

    # ---------------------------------------------------------
    # GRAFN: fit gender-responsive normalization on corpus
    # ---------------------------------------------------------
    sampling_rate = 16000

    # Build training lists for GRAFN
    grafn_train_paths = []
    grafn_train_genders = []

    for _, row in cvDelta.iterrows():
        g_norm = normalize_gender_label(row['gender'])
        if g_norm is None:
            continue
        grafn_train_paths.append(row['path'])
        grafn_train_genders.append(g_norm)

    print(f"GRAFN training set size: {len(grafn_train_paths)} (male/female only)")

    grafn = GRAFNNormalizer(sr=sampling_rate)
    grafn.fit(grafn_train_paths, grafn_train_genders)
    print("GRAFN: learned gender-specific spectral filters.")

    # ---------------------------------------------------------
    # HuggingFace Dataset + Whisper pipeline
    # ---------------------------------------------------------
    cvDelta = cvDelta.rename(columns={'sentence': 'text'})
    cvDelta = Dataset.from_pandas(cvDelta)

    # casts path column to type that can be automatically loaded and resampled
    cvDelta = cvDelta.cast_column("path", Audio(sampling_rate=sampling_rate))
    cvDelta = cvDelta.train_test_split(test_size=0.1)

    # curious about demographic distribution after split
    train_demo = cvDelta["train"].to_pandas()
    test_demo  = cvDelta["test"].to_pandas()

    print("Train distribution:")
    print(train_demo["gender"].value_counts())

    print("Test distribution:")
    print(test_demo["gender"].value_counts())

    # Prepare dataset for training
    def prepare_dataset(batch, processor, grafn=None):
        # Load audio from HF Audio feature
        audio = batch["path"]  # dict: { 'array': np.ndarray, 'sampling_rate': 16000 }

        waveform = audio["array"]
        sr = audio["sampling_rate"]

        # Apply GRAFN normalization if available and gender is known
        if grafn is not None:
            g_norm = normalize_gender_label(batch.get("gender"))
            if g_norm is not None:
                waveform = grafn.transform_waveform(waveform, gender=g_norm)

        # Compute log-Mel spectrogram and input features (Whisper front-end)
        batch["input_features"] = processor.feature_extractor(
            waveform, sampling_rate=sr
        ).input_features[0]

        # Encode transcripts
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    processor = WhisperProcessor.from_pretrained(model_id)
    model = load_whisper()

    # Apply preprocessing to all splits concurrently
    cvDelta =  cvDelta.map(
        prepare_dataset,
        #remove_columns= cvDelta.column_names["train"],
        remove_columns = ['path', 'text'], # keep demographic features for now
        num_proc=4,
        fn_kwargs={"processor": processor, "grafn": grafn}
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    

    # Apply LoRA to model
    model.add_adapter(lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params} \n All parameters: {total_params} \n Trainable %: {100 * trainable_params / total_params:.2f}")

    # Set evaluation metric to Word Error Rate
    metric = evaluate.load("wer")

    # Altered to avoid CPU/memory bottleneck!
    def compute_metrics(pred):
        # Assumes preprocess_logits_for_metrics returns a PyTorch tensor, converting it to a NumPy array.
        pred_ids = pred.predictions[0] 
        label_ids = pred.label_ids

        # Filter out unexpected negative values
        pred_ids[pred_ids < 0] = 0
        
        # Explicitly cast to 32-bit integer type to satisfy tokenizer backend.
        pred_ids = pred_ids.astype(np.int32)
        
        # replace -100 with the pad token ID
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        
        # Decode predictions and labels
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        # Compute WER
        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    # Training arguments
    # For NVIDIA: Several arguments can be changed to improve performance
    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-mozilla-finetuned",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        num_train_epochs=10,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        do_train=True,
        do_eval=True,
        fp16=True, # for nvidia set to true
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False
    )

    # Initialize trainer and collator
    data_collator = CustomDataCollator(processor=processor)

    # create gender WER training callback for bias detection

    gender_WER_parity = WERParityCallback(
        eval_dataset = cvDelta["test"],
        model = model,
        processor = processor,
        group = 'gender'
    )

    trainer = Seq2SeqTrainer(
        model = model,
        args = training_args,
        compute_metrics = compute_metrics,
        train_dataset = cvDelta["train"],
        eval_dataset = cvDelta["test"],
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        data_collator = data_collator,
        # can add as many callbacks as we want here
        # maybe todo: detect group arguments and add them
        callbacks = [gender_WER_parity]
    )

    # Perform training
    trainer.train()

    # Save adapter weights
    trainer.save_model("./whisper-mozilla-finetuned-adapter")

    # Load base model fully
    base_model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)

    # Load adapter weights
    lora_model = PeftModel.from_pretrained(base_model, "./whisper-mozilla-finetuned-adapter")

    # Merge adapter weights into base model for deployment and further evaluation
    merged_model = lora_model.merge_and_unload()
    
    # Move the model to CPU
    merged_model = merged_model.to('cpu') 
    
    merged_model.save_pretrained("./whisper-mozilla-finetuned-adapter")
    
    processor.save_pretrained("./whisper-mozilla-finetuned-adapter")

    print("Fine-tuning complete. Merged model should be saved in: ./whisper-mozilla-finetuned-adapter")
