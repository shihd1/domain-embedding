import re
import nltk
from pdfminer.high_level import extract_text
from nltk.tokenize import word_tokenize

"""
Function for document processing.
Initially the document is read using the extract_text function, as a result we have a reasonable processing
but with noise and loss of context.
NLTK is used to count tokens per text and thus generate document partitions.
The output is obtained with the execute function and returns three results:
 - Previous: Part before the current one
 - Current: Current part
 - Next: Part after the current one

 These outputs are used with the help of LLMs to optimize the extracted text by combining the generated parts. As a consequence, 
 Some information may be repeated, however, it will not be lost.
"""

class DocProcessing:
    def __init__(self, filename):
        self.filename = filename

    def filter_lines(self, text):
        filtered_lines = []
        for line in text.splitlines():
            stripped_line = line.strip()
            if len(stripped_line) >= 15 and len(re.findall(r'[a-zA-Z]', stripped_line)) >= 7:
                filtered_lines.append(stripped_line)
        return "\n".join(filtered_lines)

    def tokenize_text(self, text):
        return word_tokenize(text)

    def split_into_parts(self, tokens, current_part_size=700, context_size=250):
        parts = []
        total_tokens = len(tokens)
        index = 0
        
        while index < total_tokens:
            current_end = min(index + current_part_size, total_tokens)
            previous_start = max(index - context_size, 0)
            next_end = min(current_end + context_size, total_tokens)
            
            previous = tokens[previous_start:index]
            actual = tokens[index:current_end]
            next = tokens[current_end:next_end]
            
            parts.append({
                'previous': previous,
                'actual': actual,
                'next': next
            })
            
            index = current_end
        
        return parts
    
    def extractt_text(self):
        return extract_text(self.filename)

    def execute(self):
        try:
            extracted_text = extract_text(self.filename)

            filtered_text = self.filter_lines(extracted_text)

            tokens = self.tokenize_text(filtered_text)

            parts = self.split_into_parts(tokens)

            parts_dict = {
                f"part {i+1}": {
                    "previous": ' '.join(part['previous']),
                    "actual": ' '.join(part['actual']),
                    "next": ' '.join(part['next'])
                } for i, part in enumerate(parts)
            }
            return parts_dict
        except Exception as e:
            print(e)