/**
 * Copyright (C) 2016 Nigel Williams
 *
 * This program is free software:
 * you can redistribute it and/or modify it under the terms of the
 * GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <fstream>
#include <istream>
#include <sstream>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <chrono>
#include <cmath>
#include <cstdalign>
#include <cassert>
#include <algorithm>
#include <cstring>
#include <smmintrin.h>
#include <immintrin.h>
#include <xmmintrin.h>

// Using purely for convenience as only the standard library is included
using namespace std;

// global constants
const uint32_t cacheline_size = 64;

// Prototypes
void rotate_bitmap_bilinear(const uint8_t* bitmap, const uint32_t width, const uint32_t height, const float dx, const float dy, uint8_t* output, const bool interchange);
void rotate_bitmap_bilinear_simd(const uint8_t* bitmap, const uint32_t width, const uint32_t height, const float dx, const float dy, uint8_t* output, const bool interchange);
bool load_bitmap(const char* filename, uint8_t*& bitmap, uint32_t& width, uint32_t& height, uint32_t& max_val);
void generate_bitmap(const uint8_t* bitmap, const uint32_t& width, const uint32_t& height, const uint32_t& max_val, const bool interchange);

inline __m256 fblerp(const __m256 x1, const __m256 x2, const __m256 y1, const __m256 y2, const __m256 dx, const __m256 dy) {
    __m256 m1 = _mm256_sub_ps(x2, x1);
    m1 = _mm256_fmadd_ps(m1, dx, x1);
    __m256 m2 = _mm256_sub_ps(y2, y1);
    m2 = _mm256_fmadd_ps(m2, dx, y1);
    m1 = _mm256_sub_ps(m1, m2);
    return _mm256_fmadd_ps(m1, dy, m2);
}

// Implementation

// Optimization: Hint to the compiler that the small low complexity functions should be inlined
inline float flerp(const float value1, const float value2, const float delta) {
    // Optimization: Minimize multiplications in linear interpolation
    return value1 + (value2 - value1)*delta;
}

inline float lerp(const uint8_t value1, const uint8_t value2, const float delta) {
    // Optimization: Minimize floating point operations in lintear interpolation
    return value1 + static_cast<float>(value2 - value1)*delta;
}

// Optimal float clamping function
inline float fclamp(const float value, const float min_val, const float max_val) {
    const float min_result = (value > min_val) ? value : min_val;
    return (min_result < max_val) ? min_result : max_val;
}

// Optimized round implementation: Not exact for all inputs, but more than sufficient for color intensities
inline uint8_t fround(const float value) {
    return static_cast<uint8_t>(value + 0.5f);
}

// Load source bitmap from PGM file
bool load_bitmap(const char* filename, uint8_t*& bitmap, uint32_t& width, uint32_t& height, uint32_t& max_val) {
    bool result = false;
    ifstream input_file(filename);

    if (input_file.is_open()) {
        input_file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); 
        
        while (input_file.peek() == '#') {
            input_file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); 
        }
        input_file >> width;
        input_file >> height;
        input_file >> max_val;

        // Allocate bitmap array
        size_t size = static_cast<size_t>(width*height);
        
        // Allocate enough storage for the orignal and rotated bitmap images to be held in the same allocation
        bitmap = new uint8_t[size << 1];

        if (bitmap != nullptr) {
            for (size_t i = 0; i < size; ++i) {
                uint16_t value;
                input_file >> value;
                bitmap[i] = static_cast<uint8_t>(value);
            }

            result = true;
        }

        input_file.close();
    }

    return result;
} 

// Write output bitmap into PGM file
void generate_bitmap(const uint8_t* bitmap, const uint32_t& width, const uint32_t& height, const uint32_t& max_val, const bool interchange) {
    ofstream file("output.pgm");
    
    if (file.is_open()) {
        file << "P2" << endl;
        file << width << " " << height << endl;
        file << max_val << endl;
        file << dec;

        if (interchange) {
            for (uint32_t i = 0; i < width; ++i) {
                for (uint32_t j = 0; j < height; ++j) {
                    uint16_t value = static_cast<uint16_t>(bitmap[j*height + i]);
                    file << value << " ";
                }
                file << endl;
            }
        }else {
            for (uint32_t i = 0; i < height; ++i) {
                for (uint32_t j = 0; j < width; ++j) {
                    uint16_t value = static_cast<uint16_t>(bitmap[i*width + j]);
                    file << value << " ";
                }
                file << endl;
            }
        }
        file << endl;
        file.close();
    }
}

void rotate_bitmap_bilinear_simd(const uint8_t* bitmap, const uint32_t width, const uint32_t height, const float dx, const float dy, uint8_t* output, const bool interchange) {
    const uint32_t tile_size = cacheline_size;
    const uint32_t origin_x = width >> 1;
    const uint32_t origin_y = height >> 1;
    uint32_t columns;
    uint32_t rows;
    const float delta = 1.f - dy;
    float init_x = origin_x*delta - origin_y*dx;
    float init_y = origin_y*delta + origin_x*dx;
    init_x = (init_x > width - 1) ? width - 1 : init_x;
    init_y = (init_y > height - 1) ? height - 1 : init_y;

    if (interchange) {
        columns = height;
        rows = width;
    }else {
        columns = width;
        rows = height;
    }
    
    const __m256 m_init = _mm256_setr_ps(0.f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f);
    const __m256i mask8bit = _mm256_set1_epi32(0xff);
    const __m256 m_one = _mm256_set1_ps(1);
    const __m256 m_zero = _mm256_setzero_ps();
    const __m256 m_height = _mm256_set1_ps(height);
    const __m256 m_width = _mm256_set1_ps(width);
    const __m256 m_ceils_x = _mm256_set1_ps(columns - 1);
    const __m256 m_ceils_y = _mm256_set1_ps(rows - 1);
    const __m256i m_columnsi = _mm256_set1_epi32(static_cast<int32_t>(columns));
    const __m256 m_dx1 = _mm256_set1_ps(dy); 
    const __m256 m_dy1 = _mm256_set1_ps(dx); 

    for (uint32_t i = 0; i < width; i += tile_size) {
        for (uint32_t j = 0; j < height; j += tile_size) {
            for (uint32_t k = i; k < i + tile_size; ++k) {
                const uint64_t row = k*width;
                const float start_x = init_x + k*dx;
                const float start_y = init_y + k*dy;
                const uint32_t offset = sizeof(__m256)/sizeof(float);

                for (uint32_t l = j; l < j + tile_size; l += offset) {
                    float x = start_x + l*dy;
                    float y = start_y - l*dx;
                    
                    __m256 m_startx = _mm256_set1_ps(x); 
                    __m256 m_starty = _mm256_set1_ps(y); 
                    __m256 m_x1, m_y1;

                    // Interchange the array accesses if they are orthogonal to the memory layout
                    if (interchange) {
                        m_y1 = _mm256_fmadd_ps(m_init, m_dx1, m_startx);
                        m_x1 = _mm256_fnmadd_ps(m_init, m_dy1, m_starty);
                    }else {
                        m_x1 = _mm256_fmadd_ps(m_init, m_dx1, m_startx);
                        m_y1 = _mm256_fnmadd_ps(m_init, m_dy1, m_starty);
                    }

                    // Clamp values to min and max range of width and height
                    m_x1 = _mm256_min_ps(m_x1, m_width);
                    m_x1 = _mm256_max_ps(m_x1, m_zero);
                    m_y1 = _mm256_min_ps(m_y1, m_height);
                    m_y1 = _mm256_max_ps(m_y1, m_zero);
                    
                    // Floor to get the lower image offset coordinates
                    __m256 m_x_floor = _mm256_floor_ps(m_x1);
                    __m256 m_y_floor = _mm256_floor_ps(m_y1);
                    
                    // Find the delta values for the blerp operation
                    __m256 m_dx = _mm256_sub_ps(m_x1, m_x_floor);
                    __m256 m_dy = _mm256_sub_ps(m_y1, m_y_floor);
                    
                    // Either increment the floored value or adjust the
                    // tiled offset to the start of the next cacheline
                    __m256 mask = _mm256_cmp_ps(m_x1, m_ceils_x, _CMP_LT_OS);
                    __m256 m_x_ceiling = _mm256_and_ps(mask, m_one);
                    m_x_ceiling = _mm256_add_ps(m_x1, m_x_ceiling);
                    
                    mask = _mm256_cmp_ps(m_y1, m_ceils_y, _CMP_LT_OS);
                    __m256 m_y_ceiling = _mm256_and_ps(mask, m_one);
                    m_y_ceiling = _mm256_add_ps(m_y1, m_y_ceiling);

                    m_x_floor = _mm256_floor_ps(m_x_floor);
                    m_x_ceiling = _mm256_floor_ps(m_x_ceiling);
                    m_y_floor = _mm256_floor_ps(m_y_floor);
                    m_y_ceiling = _mm256_floor_ps(m_y_ceiling);
                    
                    // Convert the rotated index values into 32 bit integers
                    __m256i m_x1i =_mm256_cvtps_epi32(m_x_floor);
                    __m256i m_x2i =_mm256_cvtps_epi32(m_x_ceiling);
                    __m256i m_y1i =_mm256_cvtps_epi32(m_y_floor);
                    __m256i m_y2i =_mm256_cvtps_epi32(m_y_ceiling);

                    // Calculate the memory offsets for the tiled memory accesses
                    // Sadly there isn't a madd avx instruction for 32 bit integers
                    __m256i m_intermediate = _mm256_mullo_epi32(m_y2i, m_columnsi);
                    __m256i m_index1 = _mm256_add_epi32(m_intermediate, m_x1i);
                    __m256i m_index2 = _mm256_add_epi32(m_intermediate, m_x2i);
                    m_intermediate = _mm256_mullo_epi32(m_y1i, m_columnsi);
                    __m256i m_index3 = _mm256_add_epi32(m_intermediate, m_x1i);
                    __m256i m_index4 = _mm256_add_epi32(m_intermediate, m_x2i);

                    const int32_t* input = reinterpret_cast<const int32_t*>(bitmap);
                    
                    // Gather 32 bits of source input values from the calculated offsets
                    // This is a bit of a waste, but the gather avx instructions are limited to 32 bits
                    // and this is a lot quicker that storing out the offsets and loading the memory
                    m_index1 = _mm256_i32gather_epi32(input, m_index1, 1);
                    m_index2 = _mm256_i32gather_epi32(input, m_index2, 1);
                    m_index3 = _mm256_i32gather_epi32(input, m_index3, 1);
                    m_index4 = _mm256_i32gather_epi32(input, m_index4, 1);
                    
                    // Mask off the the rest of the 8 bit values in the 32 bit integer
                    m_index1 = _mm256_and_si256(m_index1, mask8bit);
                    m_index2 = _mm256_and_si256(m_index2, mask8bit);
                    m_index3 = _mm256_and_si256(m_index3, mask8bit);
                    m_index4 = _mm256_and_si256(m_index4, mask8bit);
                    
                    // Convert the source image to floating point images
                    __m256 m_f1 = _mm256_cvtepi32_ps(m_index1);
                    __m256 m_f2 = _mm256_cvtepi32_ps(m_index2);
                    __m256 m_f3 = _mm256_cvtepi32_ps(m_index3);
                    __m256 m_f4 = _mm256_cvtepi32_ps(m_index4);
                    
                    // Bilinear interpolation
                    __m256 m_result = fblerp(m_f1, m_f2, m_f3, m_f4, m_dx, m_dy);
                    m_result = _mm256_round_ps(m_result, _MM_FROUND_TO_NEAREST_INT);
                    
                    // Convert the result to integers and saturate it down to 8 bits
                    m_result = _mm256_cvtps_epi32(m_result);
                    __m128i m_low = _mm256_castsi256_si128(m_result);
                    __m128i m_high = _mm256_extracti128_si256(m_result, 1);
                    __m128i m_saturate = _mm_packus_epi32(m_low, m_high);
                    
                    // A slight waste of vector operations, but it is better
                    // than doing a convert and shuffle using _m64
                    m_saturate = _mm_packus_epi16(m_saturate, m_saturate);

                    // Store out the result to 8 bit integer allocation
                    __m128i* m_ptr = reinterpret_cast<__m128i*>(&output[row + l]);
                    _mm_storel_epi64(m_ptr, m_saturate);
                }
            }
        }
    }
}

void rotate_bitmap_bilinear(const uint8_t* bitmap, const uint32_t width, const uint32_t height, const float dx, const float dy, uint8_t* output, const bool interchange) {
    const uint32_t tile_size = cacheline_size;
    const uint32_t origin_x = width >> 1;
    const uint32_t origin_y = height >> 1;
    uint32_t columns = width;
    uint32_t rows = height;
    const float delta = 1.f - dy;
    float init_x = origin_x*delta - origin_y*dx;
    float init_y = origin_y*delta + origin_x*dx;
    init_x = (init_x > width - 1) ? width - 1 : init_x;
    init_y = (init_y > height - 1) ? height - 1 : init_y;

    // Cache Oblivious Optimization: Loop Interchange
    // When rotating bitmap around 90 or 270 degrees, array accesses are othogonal
    // Interchange the indexing so that the cache miss ratio is reduced at these angles
    if (interchange) {
        columns = height;
        rows = width;
    }

    // Cache Aware Optimization: Loop Tiling/Blocking
    // Transform the loop iteration by nesting inner loops based on the chacheline size
    // Improves data locality and increases reuse of cached data for the stride
    for (uint32_t i = 0; i < width; i += tile_size) {
        for (uint32_t j = 0; j < height; j += tile_size) {
            for (uint32_t k = i; k < i + tile_size; ++k) {
                const uint64_t row = k*width;
                const float start_x = init_x + k*dx;
                const float start_y = init_y + k*dy;

                for (uint32_t l = j; l < j + tile_size; ++l) {
                    float x = start_x + l*dy;
                    float y = start_y - l*dx;
                    
                    if ((x > -1.f) && (y > -1.f) && (x < width) && (y < height)) {
                        float clampX = fclamp(x, 0.f, static_cast<float>(width-1));
                        float clampY = fclamp(y, 0.f, static_cast<float>(height-1));

                        if (interchange) {
                            const float temp = clampX;
                            clampX = clampY;
                            clampY = temp;
                        }

                        const uint32_t floorX = static_cast<uint32_t>(clampX);
                        const uint32_t floorY = static_cast<uint32_t>(clampY);
                        const uint32_t ceilX = (floorX < (columns - 1)) ? floorX + 1 : floorX;
                        const uint32_t ceilY = (floorY < (rows - 1)) ? floorY + 1 : floorY;

                        const uint64_t top_offset = ceilY*columns;
                        const uint64_t bottom_offset = floorY*columns;

                        const float deltaX = clampX - static_cast<float>(floorX);

                        // Perform bilinear filtering on the source to determine the output pixel value
                        const float top = lerp(bitmap[top_offset + floorX], bitmap[top_offset + ceilX], deltaX);
                        const float bottom = lerp(bitmap[bottom_offset + floorX], bitmap[bottom_offset + ceilX], deltaX);
                        output[row + l] = fround(flerp(bottom, top, clampY - static_cast<float>(floorY)));
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
    chrono::microseconds delta;
    
    if (argc < 3) {
        cout << "Usage: ./rotate_bitmap filename angle [scalar]" << endl;
        return -1;
    }

    const char* filename = argv[1];
    float angle = fmod(stof(argv[2]), 360.f);

    // if scalar is specified in the command line, then use scalar instructions
    // insead of vectorized simd operations
    bool vectorize = true;
    if ((argc == 4) && (strncmp(argv[3], "scalar", 6) == 0)) {
        vectorize = false;
    }

    cout << "Rotate Bitmap " << angle << " Degrees" <<  endl;
    
    // Determine quadrant for trig optimization
    int8_t quadrant = 1;
    const float absAngle = fabs(angle);
    if (absAngle > 90.f && absAngle < 270.f) {
        quadrant = -1;
    }

    bool interchange = false;
    if (((absAngle > 45) && (absAngle < 135)) || ((absAngle > 225 && absAngle < 315))) {
        interchange = true;
        angle = -angle;
    } 
    
    // convert degress to radians
    const float pi = acos(-1.f);
    const float radians = angle*pi/180.f;
    
    // Potential optimization: sin calculation could be optimized using a lookup table
    const float dx = sin(radians);
    const float dy = static_cast<float>(quadrant)*sqrt(1.f - dx*dx);

    uint8_t* bitmap;
    uint32_t width;
    uint32_t height;
    uint32_t max_val;

    cout << "Loading Image: " << filename << endl;
 
    // Optimization: Always compare against zero/null/false instead of the opposite whenever possible
    // There are intrinsic assembly instructions for comparing against zero which are highly optimal
    // The compiler is smart enough to compare against false here when there is no explicit comparison
    if (load_bitmap(filename, bitmap, width, height, max_val)) {
        cout << "Width: " << width << endl;
        cout << "Height: " << height << endl;
        cout << "Max Value: " << max_val << endl;

        assert(max_val <= 255);
    
        const size_t size = static_cast<size_t>(width*height);
        
        uint8_t* output = bitmap + size;

        if (output != nullptr) {
            cout << "Rotating..." << endl;
            
            chrono::high_resolution_clock::time_point start_rotate = chrono::high_resolution_clock::now();
            
            if (vectorize) {
                rotate_bitmap_bilinear_simd(bitmap, width, height, dx, dy, output, interchange);
            } else {
                rotate_bitmap_bilinear(bitmap, width, height, dx, dy, output, interchange);
            }

            chrono::high_resolution_clock::time_point end_rotate = chrono::high_resolution_clock::now();

            delta = chrono::duration_cast<chrono::microseconds>(end_rotate - start_rotate);
            cout << "Rotate Run-time: " << delta.count() << endl;;
        
            cout << "Generate Output Bitmap" << endl;
            generate_bitmap(output, width, height, max_val, interchange);
        }
        
        // load_bitmap result is only true if bitmap is not null
        delete[] bitmap;
    }

    chrono::high_resolution_clock::time_point end = chrono::high_resolution_clock::now();
    delta = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Total Run-time (includes timing and IO operations): " << delta.count() << endl;;

    return 0;
}
