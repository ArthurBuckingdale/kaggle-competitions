function image=half_scale(path)
image=imread(path);
image=imresize(image,0.5);
image=rgb2gray(image);
image=rescale(image,0,1);
