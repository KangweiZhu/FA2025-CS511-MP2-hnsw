import pytest
import numpy as np
# from starter_code_HNSW import evaluate_hnsw

def test_evaluate_hnsw():
    
    expected = [932085, 934876, 561813, 708177, 706771, 695756, 435345, 701258, 455537, 872728]
   
    with open("output.txt", "r") as output_file:
        output_lines = output_file.readlines()
    
    output_lines = [int(index.strip()) for index in output_lines]
    assert output_lines == expected, "Output does not match expected output"
    
