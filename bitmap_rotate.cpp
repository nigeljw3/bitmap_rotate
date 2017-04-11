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

// Using purely for convenience as only the standard library is included
using namespace std;

// global constants
const uint32_t cacheline_size = 64;

// Prototypes
void rotate_bitmap_bilinear(const uint8_t* bitmap, const uint32_t width, const uint32_t height, const float dx, const float dy, uint8_t* output, const bool interchange);
bool load_bitmap(const char* filename, uint8_t*& bitmap, uint32_t& width, uint32_t& height, uint32_t& max_val);
void generate_bitmap(const uint8_t* bitmap, const uint32_t& width, const uint32_t& height, const uint32_t& max_val, const bool interchange);

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
        cout << "Usage: ./rotate_bitmap [filename] [angle]" << endl;
        return -1;
    }

    const char* filename = argv[1];
    float angle = fmod(stof(argv[2]), 360.f);

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
            rotate_bitmap_bilinear(bitmap, width, height, dx, dy, output, interchange);
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
