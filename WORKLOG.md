2022-09-23
----------
One gap previous implementation has is the excessive transpose traffic. It is only needed in one place (alternative is to have 8 smaller heads for that one place). In other places, these transposes can be replaced with permutation. Once replaced with permutation, we need to update GEMM implementation to take into account varied strides for batches. These are the work I did in the past 4 days.

When permute introduced, it removed about 2 seconds from previous 17s. The rest are from not calling the most optimal GEMM from CUBLAS and also not the most optimized layer norm / group norm.

Thus, after this change, I should start to work on img2img and integrate with MPSGraph. Further optimizations on CUDA won't help to promote efficiencies of other implementations (unless some obvious issue discovered).
