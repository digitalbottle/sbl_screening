%demo

image1=imread('image1.jpg');
image2=imread('image2.jpg');
psnr_=getPSNR(image1,image2);
ssim_=getMSSIM(image1,image2);
fprintf('PSNR= %f - SSIM= %f\n',psnr_,ssim_);