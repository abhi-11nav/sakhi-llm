import json
import tempfile
import os
import torch
from unittest.mock import Mock

from sakhilabs.pipelines.train.pretrain.dataset import SakhiPreTrainDataset


def test_sakhi_pretrain_dataset():
    mock_tokenizer = Mock()
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    mock_tokenizer.pad_token_id = 0
    

    test_data = [
        {"text": "Hello world"},
        {"text": "This is a test"}
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        temp_file = f.name
    
    try:
        dataset = SakhiPreTrainDataset(temp_file, chunk_length=8, tokenizer=mock_tokenizer)
        
        assert len(dataset) == 2
        
        item = dataset[0]
        assert "input_ids" in item
        assert "labels" in item
        
        assert item["input_ids"].shape == (8,)
        assert item["labels"].shape == (8,)
        
        assert item["input_ids"].dtype == torch.long
        assert item["labels"].dtype == torch.long
        
        expected_labels = torch.tensor([2, 3, 4, 5, 0, 0, 0, 0], dtype=torch.long)
        assert torch.equal(item["labels"], expected_labels)
        
        expected_input_ids = torch.tensor([1, 2, 3, 4, 5, 0, 0, 0], dtype=torch.long)
        assert torch.equal(item["input_ids"], expected_input_ids)
        
        print("All tests passed!")
        
    finally:
        os.unlink(temp_file)


def test_sakhi_pretrain_dataset_truncation():
    mock_tokenizer = Mock()
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    mock_tokenizer.pad_token_id = 0
    
    test_data = [{"text": "Long text that will be truncated"}]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        temp_file = f.name
    
    try:
        dataset = SakhiPreTrainDataset(temp_file, chunk_length=5, tokenizer=mock_tokenizer)
        item = dataset[0]
        
        assert item["input_ids"].shape == (5,)
        assert item["labels"].shape == (5,)
        
        expected_input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        expected_labels = torch.tensor([2, 3, 4, 5, 0], dtype=torch.long)
        
        assert torch.equal(item["input_ids"], expected_input_ids)
        assert torch.equal(item["labels"], expected_labels)
        
        print("Truncation test passed!")
        
    finally:
        os.unlink(temp_file)