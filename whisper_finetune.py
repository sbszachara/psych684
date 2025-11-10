import sys
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
from peft import LoraConfig, PeftModel
from huggingface_hub import login
import evaluate

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)


hf_token = 'INSERT_HF_TOKEN'
model_id = "openai/whisper-tiny"
wer_metric = evaluate.load("wer")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    mozillaPath = 'cv_22_delta/en/validated.tsv'
    cvDelta = pd.read_csv(mozillaPath, sep='\t')
    # print(cvDelta.head)
    # print(cvDelta.columns)
    # print(len(cvDelta))
    cleaned_cv = cvDelta.dropna(subset=['age', 'gender', 'accents'])
    # print(len(cvDelta))
    sampling_rate = 16000

    # Prepare dataset for training
    def prepare_dataset(batch, processor):
        # Load audio
        audio = batch["audio"]

        # Compute log-Mel spectrogram and input features
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        # Encode transcripts
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    processor = WhisperProcessor.from_pretrained(model_id)
    model = load_whisper()

    # Apply preprocessing to all splits concurrently
    dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset.column_names["train"],
        num_proc=4,
        fn_kwargs={"processor": processor}
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
        output_dir="./whisper-medical-finetuned",
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
        fp16=False, # For NVIDIA: Set to true
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False
    )

    # Initialize trainer and collator
    data_collator = CustomDataCollator(processor=processor)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        data_collator=data_collator,
    )

