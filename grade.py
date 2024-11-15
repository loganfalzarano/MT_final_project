#!/usr/bin/env python3
import argparse
from nltk.translate.bleu_score import corpus_bleu
import logging

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s %(levelname)s %(message)s')

def clean(strx):
    """
    Remove BPE markers and EOS tokens from text
    Args:
        strx: String with potential BPE markers and EOS tokens
    Returns:
        Cleaned string
    """
    return ' '.join(strx.replace('@@ ', '').replace('<EOS>', '').strip().split())

def read_translations(filepath):
    """
    Read translations from file, one per line
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def read_references(filepath):
    """
    Read references from file, splitting on |||
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip().split('|||')[1].strip() for line in f]

def main():
    parser = argparse.ArgumentParser(description='Calculate BLEU score for translations')
    parser.add_argument('--translations', default='translations',
                        help='path to file containing translations')
    parser.add_argument('--references', required=True,
                        help='path to file containing references')
    
    args = parser.parse_args()
    
    # Read and clean translations
    translations = read_translations(args.translations)
    translations = [clean(t).split() for t in translations]
    
    # Read and clean references
    references = read_references(args.references)
    references = [[clean(r).split()] for r in references]
    
    # Calculate BLEU score
    bleu = corpus_bleu(references[:len(translations)], translations)
    logging.info('BLEU score: %.2f', bleu)

if __name__ == '__main__':
    main()