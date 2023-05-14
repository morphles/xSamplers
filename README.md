# xSamplers
[ComfyUI](https://github.com/comfyanonymous/ComfyUI) alternative high resolution sampling option.

Couple nodes that allow for alternative to "hi-res fix", they sample at multiple resolutions, combining different resolution latient images, this way reducing distortion in high resolution renderings. That's the idea at least, intention was to have higher details rendering capability than high res fix usually allows, and also with less need to fiddle with denoise, between - not enough, no real details added; too much - artifacts and crap appears. I do not think I managed to achieve as good result as initial envisioned, but it is at least interesting alternative that I believe some times gives nicer results. But now you get to fiddle with mixing curve params instead of denoising...

Some comparison/sample images (not the best, probably cause I'm quite tired from long renders and tweaks, but should convey it somewhat; maybe will update the showcases some time in future).  

Prompt:  
foreground - young woman walking on old fortress wall; she has a staff. background - forest with lake. (wide angle shot:1.05), front view. consistent extremely detailed perfectly focused image. she is medieval fantasy chaos necromancer, clothes with skull, bone and darkness motifs. detailed face  
Negative:  
draft, sketch, unfinished, low quality, deformed, disfigured, distorted, blurry, noisy, hazy, fuzzy, low quality. low contrast, sitting. (close up, narrow angle shot:1.05)  

(You should be able to load images in ComfyUI to see the full workflow)

Usual size image:  
![Usual size image](https://raw.githubusercontent.com/morphles/xSamplers/main/img/usual_scale.png)  


Double sized renderings:  
Usual upscale flow:  
![Usual upscale flow](https://raw.githubusercontent.com/morphles/xSamplers/main/img/usuale_upscale.png)  
Rerender using x2Sampler:  
![Rerender using x2Sampler](https://raw.githubusercontent.com/morphles/xSamplers/main/img/rerender_x2.png)  
Direct x2Sampler render (no original image):  
![Direct x2Sampler render (no original image)](https://raw.githubusercontent.com/morphles/xSamplers/main/img/direct_x2.png)  
  
Quadruple size renderings:
Direct x4Sampler render (no original image):  
![Direct x4Sampler render (no original image)](https://raw.githubusercontent.com/morphles/xSamplers/main/img/direct_x4.png)  
Rerender using x4Sampler:  
![Rerender using x4Sampler](https://raw.githubusercontent.com/morphles/xSamplers/main/img/rerender_x4.png)  
Upscale and rerender using x2Sampler:  
![Upscale and rerender using x2Sampler](https://raw.githubusercontent.com/morphles/xSamplers/main/img/upscale_rerender_x2.png)  

X8 size rendering, really not tweaked, as it just takes very long to render, most likely can be made to be much much better
Upscale and rerender using x4Sampler:  
![Upscale and rerender using x4Sampler](https://raw.githubusercontent.com/morphles/xSamplers/main/img/upscale_rerender_x4.png)  


Notes on use:
 - x2Sampler is quite good, and as I said I consider in some cases to be better than usual upscale
 - x4Sampler (it does sampling at 3 scales and combines results) is much less so, as it is much less table, seems to have much harder problems on more complex prompts. But can work ok on some.
 - they can be combined with regular upscale/image to image flow. Instead of denoise adjust start step, to not start at 0, the higher the number the less "denoise". This can produce quite nice results, at least with 2x.
 - ancestral samplers completely do not work
 - if doing normal sampling and x2 or x4 sampling, do not expect same/similar images for same seeds - x2/x4 use/generate noise at findal resolution so of course it can't be similar to low rest sampling
 - this mixing of different scaled noise/images already produces benefits of ancestral samplers - namely not many steps needed; which is nice since high res images take quite some time to render...
 - so I strongly suggest just sticking to fastest sampler - dpmpp_2m
 - you generally do not need to many steps, even 7 can work for some cases for ok results, but I'd say 17 is sorta sweet spot to start experimenting from
 - any scheduler can work depending on prompt/model, but in general normal and simple are the best, with karras being works (which is my general experience overal).
 - upscale method, almost certainly has to be nearest-exact (this is how noises/images are scaled during combination steps), *BUT* bilinear dos some weird stuff, that maybe needs to be looked at by someone smarter than me, see example image (this is not the best sample, but the gist is - it sorta removes background, and manages to render subject quite well; well in this case not particularly well), so it might have some uses, and maybe help with better understanding of models/technology:  
<img src="https://raw.githubusercontent.com/morphles/xSamplers/main/img/bilinear.png" alt="Bilinear sample image" width="1024">

Use mixing curve "widget" (click on image)  
[<img src="https://raw.githubusercontent.com/morphles/xSamplers/main/img/curve_helper.png" alt="Mixing curve" width="640">](https://www.geogebra.org/calculator/rtcp5qgt)  
to help with tuning nodes parameters. Move green point/circle to controll curves mid point and slope, read out pos and slope values in lower right to enter them in node. Also you can use steps slider to show where steps intersect mixing line. Shaded area under curve represents how strong high resolution image (that we want to be final) should be mixed in. And ussualy we want last steps it to be maximum. And we want to strat with high proportions of low resolution (unshaded area). Exact params can vary and for best results depends on model and prompt complexity. Though sometimes/some prompts, mostly for x4 sampler I was unable to find good values.


Original idea by me, and initial implementation by [BlenderNeko](https://github.com/BlenderNeko), with some fine tunning and tweeks by me.
