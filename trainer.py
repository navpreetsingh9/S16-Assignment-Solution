from pytorch_lightning import LightningModule, seed_everything

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchmetrics

from train import get_model, get_ds, greedy_decode
from config import get_config

class LitTransfomer(LightningModule):
    def __init__(self, data_dir="."):
        super().__init__()

        config = get_config()
        self.config = config
        train_loader, val_loader, tokenizer_src, tokenizer_tgt = get_ds(config)
        self.steps_per_epoch = len(train_loader)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.max_len = config['seq_len']

        self.model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1) 

    def forward(self, encoder_input,  encoder_mask, decoder_input, decoder_mask):
        encoder_output = self.model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
        decoder_output = self.model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # 
        proj_output = self.model.project(decoder_output) # (B, seq_len, vocab_size)
        return proj_output
    
    def training_step(self, batch, batch_idx):
        encoder_input = batch['encoder_input'] # (b, seq_len)
        decoder_input = batch['decoder_input'] # (B, seq_len)
        encoder_mask = batch['encoder_mask'] # (B, 1, 1, seq_len)
        decoder_mask = batch['decoder_mask']# (B, 1, seq_len, seq_len)

        proj_output = self(encoder_input,  encoder_mask, decoder_input, decoder_mask)
        label = batch['label'] # (B, seq_len)

        loss = self.loss_fn(proj_output.view(-1, self.tokenizer_tgt.get_vocab_size()), label.view(-1))
        
        self.log("train_loss", loss.item(), prog_bar=True, logger=True)

        return loss
    
    def on_train_epoch_end(self):
        pass

    def on_validation_start(self):
        self.source_texts = []
        self.expected = []
        self.predicted = []


    def validation_step(self, batch, batch_idx):
        encoder_input = batch["encoder_input"] # (b, seq_length)
        encoder_mask = batch["encoder_mask"] # (b, 1, 1, seq_len)

        # check that the batch size is 1
        assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

        model_out = greedy_decode(self.model, encoder_input, encoder_mask, self.tokenizer_src, self.tokenizer_tgt, self.max_len, self.device)

        source_text = batch["src_text"][0]
        target_text = batch["tgt_text"][0]
        model_out_text = self.tokenizer_tgt.decode(model_out.detach().cpu().numpy())

        self.source_texts.append(source_text)
        self.expected.append(target_text)
        self.predicted.append(model_out_text)

    def on_validation_epoch_end(self):
        cer = torchmetrics.CharErrorRate()(self.predicted, self.expected)
        wer = torchmetrics.WordErrorRate()(self.predicted, self.expected)
        bleu = torchmetrics.BLEUScore()(self.predicted, self.expected)

        self.log("cer", cer, prog_bar=True)
        self.log("wer", wer, prog_bar=True)
        self.log("bleu", bleu, prog_bar=True)

    def test_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'], eps=1e-9)
        sched = torch.optim.lr_scheduler.OneCycleLR(
                                            optimizer,
                                            max_lr=self.config['max_lr'],
                                            steps_per_epoch=3213,
                                            epochs=self.config['num_epochs'],
                                            pct_start=1/10,
                                            div_factor=10,
                                            three_phase=True,
                                            final_div_factor=10,
                                            anneal_strategy='linear'
                                        )
        scheduler = {
            'scheduler': sched,
            'interval': 'step'
        }
        return [optimizer], [scheduler]
        # return optimizer
    
    ####################
    # DATA RELATED HOOKS
    ####################

    def setup(self, stage=None):
        seed_everything(1, workers=True)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.val_loader
