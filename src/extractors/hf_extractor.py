import os 
import logging
from typing import Optional, List
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from huggingface_hub import login
from src.models import Entity, Label, Source
from src.extractors.base import BaseExtractor

logger = logging.getLogger(__name__)

class HuggingFaceExtractor(BaseExtractor):
    """Extractor for HuggingFace models like medbert.de, gbert, etc."""
    
    def __init__(self, model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", 
                 device: str = "cpu"):
        """
        Initialize the HuggingFace extractor.
        
        Args:
            model_name: Name of the HuggingFace model to use
            device: Device to run the model on ("cpu" or "cuda")
        """
        self.model_name = model_name
        self.device = device

        try:
            login(token=os.getenv("HF_TOKEN"))
            login.info("Huggingface Login successful with environment variable HF_TOKEN")
        except Exception as e: 
            logger.error(f"Huggingface needs a valid authorization e.g. with env variable 'HF_TOKEN' for model {self.model_name}: {e}")

        logger.info(f"Loading HuggingFace model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            self.ner_pipeline = pipeline(
                "ner", 
                model=self.model, 
                tokenizer=self.tokenizer, 
                aggregation_strategy="average",
                device=0 if device == "cuda" and torch.cuda.is_available() else -1
            )
            logger.info(f"Successfully loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def extract(self, text: str, exclude: Optional[list[str]] = None) -> List[Entity]:
        """
        Extract entities from text using the HuggingFace model.
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            List of Entity objects
        """
        entities = []
        
        try:
            # Run NER pipeline
            ner_results = self.ner_pipeline(text)
            for result in ner_results:
                label_str = result["entity_group"] if "entity_group" in result else result["entity"]

                if label_str in Label.__members__:
                    label = Label[label_str]
                else:
                    label = Label.MISCELLANEOUS
                logger.debug(f"Entity found: {result['word']} ({result['entity_group']} -> {label})")
                
                # Create Entity object
                entity = Entity(
                    label=label,
                    text=result["word"],
                    start=result["start"],
                    end=result["end"],
                    confidence=str(result["score"]),
                    source=Source.LLM,  # Using LLM as source for now
                    search_term=result["word"]  # Using the entity text as search term
                )
                entities.append(entity)
                                           
        except Exception as e:
            logger.error(f"Error during entity extraction: {e}")
            
        logger.info(f"Extracted {len(entities)} entities using {self.model_name}")
        return entities