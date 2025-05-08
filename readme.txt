Name of my venv: ME

1.
Skew_kurt works ok on example images, and a sample can be created that kind of 
resembles it. However, the tail is not large enough.

--> idea: get higher (even) power as extra constraint

2. Rayleigh doesn't work yet. Looks like a missing constant/wrong exponent

3. skew-kurt works on example images, in differing quality, which depends
on characteristic of image. 
See 19 resulting images. The p-vales of KS test is between e-80 and 0.17.

Next step: use correlation function to create image. Don't know how yet.
Ideas to improve algorithm: 
- choosing l,u in a good way (to maximize p-value for example? But this would be
VERY computationally intensive, since rejection sampling takes long)
- Improve speed of rejection sampling (in C?)
- 
