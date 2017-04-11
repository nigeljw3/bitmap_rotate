# Fast Bilinear Bitmap Rotation

Bitmap rotation algorithm which uses cache aware optimizations to minimize cache miss ratio.

Compiled with fused multiplication/addition and also advanced vector extensions for Intel architectures.

Compilation for using profie guided optimizations also included.

Source and output files for bitmaps are in ASCII PGM format.

Usage:
./rotate_bitmap [filename] [angle]

Todo:
Finish Bresenham integer only solution
