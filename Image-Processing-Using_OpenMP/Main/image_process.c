#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>       
#include <string.h>     
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Parameters:
#define WIDTH_FALLBACK 512
#define HEIGHT_FALLBACK 512
#define BRIGHTNESS_OFFSET 50 // fixed brightness offset
#define IMG_FILENAME "original.png"

// aaplying the range for the pixel values : 
unsigned char clamp(int value) {
    if (value > 255) return 255;
    if (value < 0) return 0;
    return (unsigned char)value;
}


// Genrting a geradient image incase there is no inpuy image found :
unsigned char* generate_img(int* width, int* height, int* channels) {
    *width = WIDTH_FALLBACK;
    *height = HEIGHT_FALLBACK;
    *channels = 3; 
    
    printf("Generating %dx%d 3-channel fallback image.\n", *width, *height);
    
    unsigned char* image = (unsigned char*)malloc((*width) * (*height) * (*channels));
    if (!image) return NULL;

    for (int y = 0; y < *height; y++) {
        for (int x = 0; x < *width; x++) {

            int idx = (y * (*width) + x) * (*channels);
            image[idx + 0] = (unsigned char)((float)x / *width * 255.0f); // Red gradient
            image[idx + 1] = (unsigned char)((float)y / *height * 255.0f); // Green gradient
            image[idx + 2] = 128; // Blue constant
        }
    }
    return image;
}

// Serial Processing:
void process_serial(unsigned char* in, unsigned char* out_bright, 
                    unsigned char* out_filtered, int w, int h, int c) 
{
    //  Brightness (Serial):
    for (int i = 0; i < w * h * c; i++) {
        out_bright[i] = clamp((int)in[i] + BRIGHTNESS_OFFSET);
    }

    //  3x3 Mean Filter (Serial) :
    // compying the bright image to filtered image output :
    memcpy(out_filtered, out_bright, w * h * c);

    // applying over inner pixels (skipping 1-pixel border):
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            for (int ch = 0; ch < c; ch++) {
                int sum = 0;
                
                // 3x3 kernel :
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        // Get pixel from the bright image
                        int pixel_idx = ((y + ky) * w + (x + kx)) * c + ch;
                        sum += out_bright[pixel_idx];
                    }
                }
                
                // writing the avergaed images to the filtered output imge:
                int out_idx = (y * w + x) * c + ch;
                out_filtered[out_idx] = (unsigned char)(sum / 9);
            }
        }
    }
}


// Parallel OpenMP Processing:
void process_parallel(unsigned char* in, unsigned char* out_bright, 
                      unsigned char* out_filtered, int w, int h, int c) 
{
    //  Brightness (Serial):

    #pragma omp parallel for schedule(static) // using static scheduling as the task already has enough ILP
    for (int i = 0; i < w * h * c; i++) {
        out_bright[i] = clamp((int)in[i] + BRIGHTNESS_OFFSET);
    }

    //  3x3 Mean Filter (Serial) :
    memcpy(out_filtered, out_bright, w * h * c);

    // Parallelize the outer 'y' and 'x' loops.
    // 'collapse(2)' tells OpenMP to parallelize *both* loops,
    // creating a single large iteration space (h-2) * (w-2).

// combining the two loops into one using collapse :
    #pragma omp parallel for schedule(static) collapse(2)
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            for (int ch = 0; ch < c; ch++) {
                int sum = 0;
                
                // 3x3 kernel :
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        int pixel_idx = ((y + ky) * w + (x + kx)) * c + ch;
                        sum += out_bright[pixel_idx];
                    }
                }
                
                int out_idx = (y * w + x) * c + ch;
                out_filtered[out_idx] = (unsigned char)(sum / 9);
            }
        }
    }
}

void print(const char* title, unsigned char* image, int w, int c) {
    printf("\n--- %s (5x5 Sample, Top-Left) ---\n", title);
    for (int y = 0; y < 5; y++) {
        for (int x = 0; x < 5; x++) {
            printf("%3d ", (int)image[(y * w + x) * c]); // Print first channel
        }
        printf("\n");
    }
}

// =============Driver function ==============
int main() {
    int width, height, channels;
    double start, end;

    printf("== Image Brightness + 3x3 Smoothing (OpenMP) ==\n");
    unsigned char *image = stbi_load(IMG_FILENAME, &width, &height, &channels, 0);

    if (!image) {
        printf("Error loading '%s'.\n", IMG_FILENAME);
        image = generate_img(&width, &height, &channels);
        if (!image) {
            printf("Error generating a random gradient image!\n");
            return 1;
        }
    }

    printf("Loaded image: %dx%d, Channels: %d\n", width, height, channels);
    size_t img_size = width * height * channels;

    unsigned char *image_bright = (unsigned char *)malloc(img_size);
    unsigned char *image_filtered = (unsigned char *)malloc(img_size);

    if (!image_bright || !image_filtered) {
        printf("Memory allocation failed!\n");
        stbi_image_free(image);
        return 1;
    }

    // --- Serial Baseline ---
    printf("\nRunning Serial baseline...");
    fflush(stdout);
    start = omp_get_wtime();
    process_serial(image, image_bright, image_filtered, width, height, channels);
    end = omp_get_wtime();
    double serial_time = end - start;
    printf(" Done in %.6f sec\n", serial_time);

    stbi_write_png("output_bright_serial.png", width, height, channels, image_bright, width * channels);
    stbi_write_png("output_filtered_serial.png", width, height, channels, image_filtered, width * channels);
    printf("Saved 'output_bright_serial.png' and 'output_filtered_serial.png'\n");


    // Saving the performance data into CSV:
    FILE *csv_file = fopen("image_performance.csv", "w");
    if (csv_file == NULL) {
        perror("fopen image_performance.csv");
        return 1;
    }
    fprintf(csv_file, "Threads,Parallel_Time(s),Speedup,Efficiency,Serial_Time(s)\n");
    printf("\n--- Performance Test ---\n");
    printf("Threads\tParallel_Time(s)\tSpeedup\t\tEfficiency\n");
    printf("----------------------------------------------------------\n");

    int max_threads = omp_get_max_threads();
    
    // --- Parallel Execution Loop ---
    for (int threads = 1; threads <= max_threads; threads *= 2) {
        omp_set_num_threads(threads);

        start = omp_get_wtime();
        process_parallel(image, image_bright, image_filtered, width, height, channels);
        end = omp_get_wtime();
        
        double parallel_time = end - start;
        double speedup = serial_time / parallel_time;
        double efficiency = speedup / (double)threads;

        printf("%2d\t%.6f\t\t%.2fx\t\t%.4f\n", 
               threads, parallel_time, speedup, efficiency);
        
        fprintf(csv_file, "%d,%.6f,%.2f,%.4f,%.6f\n",
                threads, parallel_time, speedup, efficiency, serial_time);
    }
    
    fclose(csv_file);
    printf("Saved performance data to 'image_performance.csv'\n");

    stbi_write_png("output_bright_parallel.png", width, height, channels, image_bright, width * channels);
    stbi_write_png("output_filtered_parallel.png", width, height, channels, image_filtered, width * channels);
    printf("Saved final parallel results to 'output_bright_parallel.png' and 'output_filtered_parallel.png'\n");

    print("Original Image", image, width, channels);
    print("Brightened Image", image_bright, width, channels);
    print("Filtered (Smoothed) Image", image_filtered, width, channels);

    // freein thje memory:
    free(image_bright);
    free(image_filtered);
    stbi_image_free(image);

    return 0;
}

