# Swift Diffusion

This is a single-file re-implementation of [Stable Diffusion](https://github.com/CompVis/stable-diffusion) model. It includes the models for CLIP text encoder, UNet diffusion model and the decoder model. It also includes PLMS inference implementation. The implementation tries to match the Stable Diffusion outputs layer-by-layer, thus, given the same start point `x_T`, this implementation and Stable Diffusion will output the same image.

## Rationale

This re-implementation serves and an education for me to understand diffusion models. It is also necessary for my follow-up work to enable Stable Diffusion on mobile devices such as iPad / iPhone. Without a Swift re-implementation, doing mobile-focused optimization with Python would be difficult and impossible to ship in App Store. It is possible to do this differently, such as exporting to ONNX runtime and use that as the driver on mobile devices. That does limit what kind of optimizations you can apply though. As you can tell, running models that totals about 8GiB in-memory and 4GiB at-rest with full floating-point precision is not trivial on mobile devices. It might requires some non-conventional optimizations that may not be available through existing frameworks. Using something I am familiar with (a framework I built) would be a good starting point.

## Where We Are

CLIP text model, UNet diffusion model and the decoder has been ported. The `examples:txt2img` target is useful with some path changesinside `examples/txt2img/main.swift`. Need to port the encoder over to enable `img2img`. Other targets, such as `examples:unet`, `examples:clip`, `examples:autoencoder` are the example programs to convert PyTorch weights to the one s4nnc uses.

## What's Next

The next on my list is to implement the tokenizer. Thanks to PythonKit, right now, I am using the tokenizer from Hugging Face. After tokenizer implemented, the whole thing should be able to run without Python dependencies.

After that, I should change the convolution layout from NCHW to NHWC. That will enable bunch of optimizations in attention layer, mostly to avoid some of the transpose traffic. I can enable CPU mode either by converting convolution layout to NHWC, or implement NCHW convolution in s4nnc. The latter is long overdue, but doing former would be helpful for performance on CPU.

Right now, at run time, UNet model uses ~1.5GiB memory in additional to its 3.3GiB weights. A big chunk of that 1.5GiB is due to the dot product in attention layer. I already optimized away about 1GiB because previously, softmax doesn't run in-place properly (due to complex reasons relating to aliases and reshapes). I believe this is still a case for PyTorch code because there is no in-place softmax method. That dot product can be split further into smaller batches to save peak memory usage (along the token dimension of k). If these are done correctly, we should be able to reduce UNet memory usage to somewhere around 3.8GiB full floating-point. Another idea I have to further reduce the memory usage is to compress shortcut activations in UNet (these shortcut activations will be saved along downsample path and used in upsample path, thus, occupying for long time). But I am less sure how much memory that can save.

Converting the model to FP16 would save memory footprint instantly, but this will be close-to-the-last thing to do. Just by using FP16, UNet should use around 1.9GiB memory, which is very manageable on mobile devices now. Given that we can unload UNet model and load decoder from disk when it is done, this combined can, hopefully, finally, run stable diffusion on mobile. We can further quantize weights to int8 with the `LLM.int8()` transformers trick: https://arxiv.org/pdf/2208.07339.pdf.

## Is It Comparable

Right now, I didn't run any specific optimizations. Further, the model loading as of today for s4nnc requires executing the model once, and we have some optimization runs (find the most efficient kernels etc.) that are not saved. That has been said, we can compare the execution time of txt2img from Swift v.s. the one from CompVis (there are more optimized forks available, but going through them to find the best would take time) of the diffusion process + decoding process. The Swift txt2img on GPU took about 17s while the CompVis took about 11s (both with one 2080 Ti). Cursory look at `nvprof` output shows that transpose and not using cublasLt the leading cause for the extra 6s spent.

## How to Run This

There are quite a bit of setup right now. As I get all bits moved to Swift and start CPU / Metal work, it should be easier. It would also help if I move to support SwiftPM on s4nnc side. But that is not as high priority as the other two.

First, you need to install Bazel and various dependencies for s4nnc. To install Bazel, follow: https://bazel.build/install.

Other dependencies include Swift compiler, CUDA (10.2 and above) and clang. For former two, you have to install yourself. For the others, if you are on Debian-like system, you can install with:

```
sudo apt install clang llvm libicu-dev libpng-dev libjpeg-dev libatlas-base-dev libblas-dev libgsl-dev libdispatch-dev libomp-dev libfftw3-dev
```

For now, you need to install `transformers` for the tokenizer.

```
virtualenv -p python3 _env
source _env/bin/activate
pip install transformers
```

You also need to download the model. I put the Stable Diffusion v1.4 model in http://static.libccv.org/sd-v1.4.ckpt. Note that this is a s4nnc-compatible file, not PyTorch one you download elsewhere. You can check related examples for how this file is generated.

With these, you can run:

```
bazel run examples:txt2img --compilation_mode=opt -- /home/the-absolute-work-directory-that-contains-sd-v1.4.ckpt-file "a photograph of an astronaut riding a horse"
```

The image will be generated under the given directory with name `txt2img.png`.
