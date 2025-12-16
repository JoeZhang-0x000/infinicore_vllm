## TODO
- [x] Add support for GEMM.
- [x] Add support for SiluAndMul.
- [x] Add support for RMSNorm.
- [ ] Add support for attention.
- [ ] Add support for ccl

## Install
```
pip install -e .
```
## Example
``` 
python examples/basic.py 
```
Excepted Output:
```
Successfully patched vllm.model_executor.layers.utils.dispatch_unquantized_gemm.
Successfully patched vllm.model_executor.layers.activation.SiluAndMul.forward.
Successfully patched vllm.model_executor.layers.layernorm.RMSNorm.forward.
...
Successfully patched vllm.model_executor.layers.utils.dispatch_unquantized_gemm.
Successfully patched vllm.model_executor.layers.activation.SiluAndMul.forward.
Successfully patched vllm.model_executor.layers.layernorm.RMSNorm.forward.
...
(EngineCore_DP0 pid=2378270) Using Custom from torch
(EngineCore_DP0 pid=2378270) Infini GEMM is enabled.
...

Generated Outputs:
------------------------------------------------------------
Prompt:    'Hello, my name is'
Output:    " Josh and I'm in the middle of a project to develop a hybrid mobile app. I'm looking for guidance on how to go about using modular frameworks. I want to use React and Vue. I need to decide on the framework to use. Can you help me choose the right framework and suggest some best practices for using them?\n\nAdditionally, I want to know what are the best practices for using a modular framework in the context of a web application? Also, what are the best practices for using a"
------------------------------------------------------------
Prompt:    'The president of the United States is'
Output:    ' the most powerful individual in the world, but what about the most powerful individual in the world who is also a woman? The most powerful individual in the world who is also a woman, especially in the second half of the 20th century. The president of the United States is the most powerful individual in the world. What about the other female leaders who are also on the list of powerful individuals in the world?\n\nAnswer: The President of the United States is the most powerful individual in the world'
------------------------------------------------------------
Prompt:    'The capital of France is'
Output:    ' Paris. Therefore, the capital of Paris is Paris. This is an example of __________.\n\nThe correct answer is: [list]\n\nThe question is about a logical fallacy. The correct answer is: [list]\nAnswer:\n\nThe correct answer is: [list]\n\nThe question is about a logical fallacy. The correct answer is: [list]\n\nThe question is about a logical fallacy. The correct answer is: [list]\n\nThe question is about a logical fallacy. The correct answer'
------------------------------------------------------------
Prompt:    'The future of AI is'
Output:    ' becoming more interesting and plausible, especially in the areas of the fusion of AI with other technologies. The study of AI and its development is evolving rapidly, and researchers are exploring new ways to implement AI in existing systems, including AI systems that use machine learning to integrate with human thought and emotions. This new type of AI is called neural networks, which are designed to learn and adapt with their environment. In addition, there are areas of research that focus on the integration of AI with other systems, such as'
------------------------------------------------------------
```