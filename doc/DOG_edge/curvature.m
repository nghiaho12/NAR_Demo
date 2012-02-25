img = imread("edge.png");

sigma1 = 1.6;
sigma2 = 1.6*1.6;

n1 = int32(sigma1*3*2)+1;
n2 = int32(sigma2*3*2)+1;

k1 = fspecial("gaussian", [n1 n1], sigma1);
k2 = fspecial("gaussian", [n2 n2], sigma2);

img1 = conv2(img, k1, 'same');
img2 = conv2(img, k2, 'same');

dog = img1 - img2;

[rows,cols] = size(img);

cx = cols/2;
cy = rows/2;

r = 1;

sub = dog(cy-r:cy+r, cx-r:cx+r)

%{
sub = [-1 -1 -1; -1 20 -1; -1 -1 -1]
a = sub(1,1);
b = sub(1,2);
c = sub(1,3);
d = sub(2,1);
e = sub(2,2);
f = sub(2,3);
g = sub(3,1);
h = sub(3,2);
i = sub(3,3);
%}

Dxx = d - 2*e + f;
Dyy = b - 2*e + h;

Dx1 = (c-a)*0.5;
Dx2 = (i-g)*0.5;
Dxy = (Dx2-Dx1)*0.5

trace = Dxx + Dyy
det = Dxx*Dyy - Dxy*Dxy
ratio = trace^2/det


a = min(dog(:));
b = max(dog(:));

dog2 = (dog - a)/(b-a);
imwrite(dog2,"edge-dog.png");
imshow(img2);
