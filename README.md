# Swift Diffusion

This is a single-file re-implementation of [Stable Diffusion](https://github.com/CompVis/stable-diffusion) model. It includes CLIP text tokenizer, the models for CLIP text encoder, UNet diffusion model and the decoder model. It also includes PLMS inference implementation. The implementation tries to match the Stable Diffusion outputs layer-by-layer, thus, given the same start point `x_T`, this implementation and Stable Diffusion will output the same image.

## Rationale

This re-implementation serves as an education for me to understand diffusion models. It is also necessary for my follow-up work to enable Stable Diffusion on mobile devices such as iPad / iPhone. Without a Swift re-implementation, doing mobile-focused optimization with Python would be difficult and impossible to ship in App Store. It is possible to do this differently, such as exporting to ONNX runtime and use that as the driver on mobile devices. That does limit what kind of optimizations you can apply though. As you can tell, running models that totals about 8GiB in-memory and 4GiB at-rest with full floating-point precision is not trivial on mobile devices. It might requires some non-conventional optimizations that may not be available through existing frameworks. Using something I am familiar with (a framework I built) would be a good starting point.

## Where We Are

CLIP text tokenizer, image model, text model, UNet diffusion model and the autoencoders has been ported. The `examples:txt2img`, `examples:img2img` and `examples:inpainting` target is useful. Other targets, such as `examples:unet`, `examples:clip`, `examples:decoder`, `examples:encoder` and `examples:vit` are the example programs to convert PyTorch weights to the one s4nnc uses.

## What's Next

The next is to evaluate Int8 convolution kernels and Int8 + Float16 GEMM kernels for their viability. These should help to reduce memory usage down to somewhere around 2GiB at runtime.

Right now, at run time, UNet model uses ~1.5GiB memory in additional to its 3.3GiB weights. A big chunk of that 1.5GiB is due to the dot product in attention layer. I already optimized away about 1GiB because previously, softmax doesn't run in-place properly (due to complex reasons relating to aliases and reshapes). I believe this is still a case for PyTorch code because there is no in-place softmax method. That dot product can be split further into smaller batches to save peak memory usage (along the token dimension of k). If these are done correctly, we should be able to reduce UNet memory usage to somewhere around 3.8GiB full floating-point. Another idea I have to further reduce the memory usage is to compress shortcut activations in UNet (these shortcut activations will be saved along downsample path and used in upsample path, thus, occupying for long time). But I am less sure how much memory that can save.

Converting the model to FP16 would save memory footprint instantly. A switch was implemented in `txt2img`, `img2img` and `inpainting` such that you can switch between Float16 or Float32. Just by using FP16, UNet should use around 1.9GiB memory. If we can unload UNet model and load decoder from disk when it is done, this combined can, hopefully, finally, run stable diffusion on mobile. We can further quantize weights to int8 with the `LLM.int8()` transformers trick: https://arxiv.org/pdf/2208.07339.pdf. See more about these tricks: https://github.com/TimDettmers/bitsandbytes.

## Is It Comparable

I've reduced the transpose traffic by implementing permute operator in s4nnc. When we compare the execution time of `txt2img` from Swift v.s. the one from CompVis (there are more optimized forks available, but going through them to find the best would take time) of the diffusion process + decoding process. The Swift txt2img on GPU took about 15s while the CompVis took about 11s (both with one 2080 Ti). There are some inefficiencies in the LayerNorm / GroupNorm kernels I use, as well as some mysteries on why certain low-performance GEMM kernels are selected. I am going to switch to MPS implementation though. Optimizing further on CUDA end won't translate to gains on MPS end.

The MPS backend took about 95s to finish with peak memory around 4GiB in FP16 on a M1 Mac Mini. Comparing with PyTorch's ~120s, this seems to be quite reasonable.

## How to Run This

There are quite a bit of setup right now. It would help if I move to support SwiftPM on s4nnc side. But that is not a high priority.

First, after checking out this repository to your local storage, you need to install Bazel and various dependencies for s4nnc. To install Bazel, follow: https://bazel.build/install.

You also need to download the model. I put the Stable Diffusion v1.4 model in http://static.libccv.org/sd-v1.4.ckpt. Note that this is a s4nnc-compatible file, not PyTorch one you download elsewhere. You can check related examples for how this file is generated.

### Linux

Other dependencies include Swift compiler, CUDA (10.2 and above) and clang. For former two, you have to install yourself. For the others, if you are on Debian-like system, you can install with:

```
sudo apt install clang llvm libicu-dev libpng-dev libjpeg-dev libatlas-base-dev libblas-dev libgsl-dev libdispatch-dev libomp-dev libfftw3-dev
```

Finally, setup Bazel properly on Linux:

```
./bazel/setup_clang.sh /usr/local
```

With these, you can run:

```
bazel run examples:txt2img --compilation_mode=opt -- /home/the-absolute-work-directory-that-contains-sd-v1.4.ckpt-file "a photograph of an astronaut riding a horse"
```

### macOS

Once Bazel is installed, you should modify `WORKSPACE` file under this repository. In particular, this line about `ccv_setting` should be modified to the following:

```python
ccv_setting(
    name = "local_config_ccv",
    have_accelerate_framework = True,
    have_pthread = True,
)
```

You also need to add a new `.bazelrc.local` file under this repository with one line `build --config=mps`.

Afterwards, you should be able to run:

```
bazel run examples:txt2img --compilation_mode=opt -- /Users/the-absolute-work-directory-that-contains-sd-v1.4.ckpt-file "a photograph of an astronaut riding a horse"
```

It took about 95s on a M1 Mac Mini. Be patient.

The image will be generated under the given directory with name `txt2img.png`.

For `img2img`, it looks for `init_img.png` under the work directory you provided. For `inpainting`, it looks for `init_inpainting.png` under the work directory. As of now, it expects the inpainting parts to be green. You also need to provide prompt guidance for `inpainting` otherwise it won't work.

## Existing Issues

I cannot seem to have inpainting without prompt guidance to work. I tried to use CLIP image model for guidance. However, because Stable Diffusion model uses the text embedding prior to text projection, making the image model's embedding not comparable to the text embedding one.
