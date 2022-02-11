# FlexGPT
The project aims to optimize the tradeoff between large language model inference runtime and RAN memory usage. The fully autoregressive language model takes all previous tokens into the logits for sampling, thus resulting in a O(N^2) runtime. Particularly, the self-attention forward pass is the main overhead, taking up over 98% of the runtime. In this project, we implemented a cached self-attention layer to reduce the theoretical runtime from O(N^2) to O(N) and established the benchmark for trade off between runtime (mem_selfattn's cache length) and RAM usage.

![Screen Shot 2022-02-10 at 22 16 39](https://user-images.githubusercontent.com/37657480/153545622-92c5d1c7-d10d-4c5a-b9c4-87a82ae9773d.png)
![Screen Shot 2022-02-10 at 22 16 44](https://user-images.githubusercontent.com/37657480/153545624-e6de4f8f-0d46-40a4-81bc-22e21a3f1ff3.png)




To set up python packages and dependencies:
```
python3 -m setup.py install
```

To run the cached self-attention layer: 
```
cd ./minGPT/memgpt
python3 mem_selfattn_separate.py 
```
The benchmark results are printed on the console and saved as a csv file to the logs directory.
